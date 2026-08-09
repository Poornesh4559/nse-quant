"""Phase 3 sentiment pipeline orchestrator: fetch -> map -> score -> store.

CLI (run from repo root)::

    .venv/bin/python -m collector.news.pipeline today
    .venv/bin/python -m collector.news.pipeline backfill --days 60
    .venv/bin/python -m collector.news.pipeline status

Commands
--------
today     Fetch Google News for every DB symbol + all market RSS feeds +
          last-24h Reddit (pullpush) submissions, map each article to a
          symbol, dual-model score it and upsert into ``news_sentiment``.
backfill  Per-symbol GDELT article backfill over the last N days (default
          60). GDELT is rate-limited to 1 req / 5 s, so 52 symbols take
          ~5 minutes; progress is logged per symbol and stored incrementally.
status    Aggregate stats for ``news_sentiment`` (totals, source/symbol
          breakdown, date range) printed as a table.

Robustness: every fetch, per-article map/score and each store call is
wrapped so a failure never crashes the run (sources already never raise;
scorer degrades FinBERT->VADER; store upserts dedupe by URL).

Logging: INFO to console + append to ``data/news.log``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from collector.config import DATA_DIR, settings
from collector.db import get_conn, list_symbols

logger = logging.getLogger("collector.news.pipeline")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging() -> None:
    """INFO to console + append to data/news.log."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(DATA_DIR / "news.log", mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _article_text(article: dict) -> str:
    """Title (+ optional description/snippet when a fetcher provides one)."""
    parts = [str(article.get("title") or "")]
    snippet = article.get("description") or article.get("snippet") or ""
    if snippet:
        parts.append(str(snippet))
    return " ".join(parts)


def _process(article: dict, mapper, scorer) -> dict | None:
    """Map + dual-model score a single article into a store-ready row.

    Never raises: mapper/scorer failures log and skip the article.
    symbol = first mapped symbol, or None (market-wide) when nothing maps.
    """
    try:
        text = _article_text(article)
        symbols = mapper.map_symbols(text)
        score = scorer.score_text(text)  # scores title (+short snippet)
        return {
            "source": article.get("source"),
            "symbol": symbols[0] if symbols else None,
            "title": str(article.get("title") or "")[:500],
            "url": article.get("url"),
            "published_at": article.get("published"),
            "sentiment_compound": score.get("compound"),
            "sentiment_label": score.get("label"),
        }
    except Exception:
        logger.exception("failed to map/score article: %r", article.get("title"))
        return None


def _safe_store(store_mod, rows: list[dict]) -> int:
    """Dedupe rows by URL (keep first) then upsert; returns stored count.

    Google News returns the same story URL under many symbol queries, and a
    single ON CONFLICT DO UPDATE batch cannot contain duplicate keys — so the
    orchestrator dedupes before handing rows to store.upsert_news.
    """
    if not rows:
        return 0
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        key = r.get("url") or r.get("title") or ""
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(r)
    if len(unique) < len(rows):
        logger.info("dropped %d duplicate-url rows before upsert", len(rows) - len(unique))
    try:
        return store_mod.upsert_news(unique) or len(unique)
    except Exception:
        logger.exception("upsert_news failed for %d rows", len(unique))
        return 0


def _print_summary(fetched: int, per_source: dict[str, int], stored: int,
                   symbols_covered: set[str], compounds: list[float]) -> None:
    avg = sum(compounds) / len(compounds) if compounds else 0.0
    print(
        f"\nSUMMARY: {fetched} articles fetched "
        f"({per_source.get('google', 0)} google, {per_source.get('feeds', 0)} feeds, "
        f"{per_source.get('reddit', 0)} reddit) | {stored} stored | "
        f"{len(symbols_covered)} unique symbols covered | avg compound {avg:.3f}\n"
    )


# ---------------------------------------------------------------------------
# today
# ---------------------------------------------------------------------------
def cmd_today(mapper, scorer, store_mod, sources_mod, config_mod) -> int:
    symbols = [s["symbol"] for s in list_symbols()]
    logger.info("today: %d symbols from DB", len(symbols))

    google_articles: list[dict] = []
    for sym in symbols:
        try:
            google_articles.extend(sources_mod.fetch_google_news(sym) or [])
        except Exception:
            logger.exception("google fetch failed for %s", sym)
    logger.info("google news: %d articles across %d symbols", len(google_articles), len(symbols))

    feed_articles: list[dict] = []
    try:
        feeds = sources_mod.fetch_market_feeds() or []
        feed_articles = list(feeds.values())[0] if isinstance(feeds, dict) else list(feeds)
    except Exception:
        logger.exception("market feed fetch failed")
    logger.info("market feeds: %d articles", len(feed_articles))

    reddit_articles: list[dict] = []
    after = datetime.now(timezone.utc) - timedelta(seconds=86400)
    for sub in config_mod.REDDIT_SUBREDDITS:
        try:
            reddit_articles.extend(sources_mod.fetch_reddit(sub, after=after.timestamp()) or [])
        except Exception:
            logger.exception("reddit fetch failed for %s", sub)
    logger.info("reddit: %d articles across %d subreddits", len(reddit_articles),
                len(config_mod.REDDIT_SUBREDDITS))

    all_articles = google_articles + feed_articles + reddit_articles
    rows = [r for r in (_process(a, mapper, scorer) for a in all_articles) if r]
    stored = _safe_store(store_mod, rows)

    symbols_covered = {r["symbol"] for r in rows if r["symbol"]}
    compounds = [r["sentiment_compound"] for r in rows if r["sentiment_compound"] is not None]
    _print_summary(
        len(all_articles),
        {"google": len(google_articles), "feeds": len(feed_articles), "reddit": len(reddit_articles)},
        stored, symbols_covered, compounds,
    )
    return stored


# ---------------------------------------------------------------------------
# premarket — the 8:30 IST market-direction call (Phase 4)
# Same fetch/score/store as `today` PLUS a market-wide aggregate written to
# market_sentiment(date). The paper bot reads this as its risk gate.
# ---------------------------------------------------------------------------
def cmd_premarket(mapper, scorer, store_mod, sources_mod, config_mod) -> int:
    symbols = [s["symbol"] for s in list_symbols()]
    logger.info("premarket: %d symbols, 24h news window", len(symbols))

    google_articles: list[dict] = []
    for sym in symbols:
        try:
            google_articles.extend(sources_mod.fetch_google_news(sym) or [])
        except Exception:
            logger.exception("google fetch failed for %s", sym)
    feed_articles: list[dict] = []
    try:
        feeds = sources_mod.fetch_market_feeds() or []
        feed_articles = list(feeds.values())[0] if isinstance(feeds, dict) else list(feeds)
    except Exception:
        logger.exception("market feed fetch failed")
    reddit_articles: list[dict] = []
    after = datetime.now(timezone.utc) - timedelta(seconds=86400)
    for sub in config_mod.REDDIT_SUBREDDITS:
        try:
            reddit_articles.extend(sources_mod.fetch_reddit(sub, after=after.timestamp()) or [])
        except Exception:
            logger.exception("reddit fetch failed for %s", sub)

    all_articles = google_articles + feed_articles + reddit_articles
    rows = [r for r in (_process(a, mapper, scorer) for a in all_articles) if r]
    stored = _safe_store(store_mod, rows)

    compounds = [r["sentiment_compound"] for r in rows if r["sentiment_compound"] is not None]
    n_pos = sum(1 for c in compounds if c and c > 0.1)
    n_neg = sum(1 for c in compounds if c and c < -0.1)
    avg = (sum(compounds) / len(compounds)) if compounds else 0.0
    direction = "BULLISH" if avg >= 0.1 else ("BEARISH" if avg <= -0.1 else "NEUTRAL")

    # per-symbol aggregates (for the top-mover callout)
    sym_avg: dict[str, float] = {}
    for r in rows:
        if r.get("symbol") and r["sentiment_compound"] is not None:
            sym_avg.setdefault(r["symbol"], []).append(r["sentiment_compound"])  # type: ignore[arg-type]
    sym_avg = {k: sum(v) / len(v) for k, v in sym_avg.items()}  # type: ignore[arg-type]

    today = datetime.now(timezone.utc).date()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO market_sentiment (date, avg_compound, n_articles, n_positive, n_negative, direction)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (date) DO UPDATE SET avg_compound=EXCLUDED.avg_compound,
                 n_articles=EXCLUDED.n_articles, n_positive=EXCLUDED.n_positive,
                 n_negative=EXCLUDED.n_negative, direction=EXCLUDED.direction,
                 created_at=now()""",
            (today, round(avg, 4), len(compounds), n_pos, n_neg, direction),
        )
        conn.commit()

    top = sorted(sym_avg.items(), key=lambda kv: kv[1], reverse=True)[:5]
    bottom = sorted(sym_avg.items(), key=lambda kv: kv[1])[:5]
    print("\n===== PRE-MARKET SENTIMENT CALL =====")
    print(f"date        : {today} (IST)")
    print(f"articles    : {len(compounds)} scored | pos {n_pos} / neg {n_neg}")
    print(f"avg compound: {avg:+.3f}  ->  {direction}")
    print("most bullish:", ", ".join(f"{s} ({v:+.2f})" for s, v in top) or "—")
    print("most bearish:", ", ".join(f"{s} ({v:+.2f})" for s, v in bottom) or "—")
    print("===================================\n")
    return stored


# ---------------------------------------------------------------------------
# cues — global/geopolitical sentiment (SEPARATE attribute from market news).
# Bucketed by theme (crude/fed/us_markets/usd_inr/geo) into global_cues; the
# paper bot weights this against market_sentiment for its risk gate.
# ---------------------------------------------------------------------------
def cmd_cues(mapper, scorer, store_mod, sources_mod, config_mod) -> int:
    articles = sources_mod.fetch_global_cues() or []
    logger.info("cues: %d global-cue articles fetched", len(articles))

    rows: list[dict] = []
    for a in articles:
        row = _process(a, mapper, scorer)
        if row:
            row["symbol"] = None            # cues are market-wide, never per-symbol
            row["theme"] = a.get("theme")
            rows.append(row)
    stored = _safe_store(store_mod, rows)

    theme_buckets: dict[str, dict] = {}
    for r in rows:
        theme = r.get("theme") or "other"
        c = r["sentiment_compound"]
        if c is None:
            continue
        b = theme_buckets.setdefault(theme, {"n": 0, "sum": 0.0})
        b["n"] += 1
        b["sum"] += c

    all_c = [r["sentiment_compound"] for r in rows if r["sentiment_compound"] is not None]
    n_pos = sum(1 for c in all_c if c > 0.1)
    n_neg = sum(1 for c in all_c if c < -0.1)
    avg = (sum(all_c) / len(all_c)) if all_c else 0.0
    direction = "BULLISH" if avg >= 0.1 else ("BEARISH" if avg <= -0.1 else "NEUTRAL")

    import json as _json
    themes_json = _json.dumps(
        {k: {"avg": round(v["sum"] / v["n"], 4), "n": v["n"]} for k, v in theme_buckets.items()}
    )
    today = datetime.now(timezone.utc).date()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO global_cues (date, avg_compound, n_articles, n_positive, n_negative, direction, themes)
               VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
               ON CONFLICT (date) DO UPDATE SET avg_compound=EXCLUDED.avg_compound,
                 n_articles=EXCLUDED.n_articles, n_positive=EXCLUDED.n_positive,
                 n_negative=EXCLUDED.n_negative, direction=EXCLUDED.direction,
                 themes=EXCLUDED.themes, created_at=now()""",
            (today, round(avg, 4), len(all_c), n_pos, n_neg, direction, themes_json),
        )
        conn.commit()

    print("\n===== GLOBAL CUES CALL =====")
    print(f"date        : {today} (IST)")
    print(f"articles    : {len(all_c)} scored | pos {n_pos} / neg {n_neg}")
    print(f"avg compound: {avg:+.3f}  ->  {direction}")
    for theme, b in theme_buckets.items():
        t_avg = b["sum"] / b["n"]
        print(f"  {theme:10s}: {t_avg:+.3f}  ({b['n']} articles)")
    print("=============================\n")
    return stored


# ---------------------------------------------------------------------------
# backfill (GDELT)
# ---------------------------------------------------------------------------
def cmd_backfill(args, mapper, scorer, store_mod, sources_mod) -> int:
    days = args.days
    symbols = [s["symbol"] for s in list_symbols()]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    logger.info("backfill: %d symbols, %s -> %s (gdelt rate-limited 5.5s/req)", len(symbols), start, end)

    total_fetched, total_stored = 0, 0
    for i, sym in enumerate(symbols, 1):
        try:
            arts = sources_mod.fetch_gdelt(sym, start_dt=start, end_dt=end) or []
        except Exception:
            logger.exception("gdelt fetch failed for %s", sym)
            arts = []
        rows = []
        for art in arts:
            row = _process(art, mapper, scorer)
            if row:
                # GDELT queries are symbol-specific: keep the first mapped
                # symbol, else attribute the article to the queried symbol.
                row["symbol"] = row["symbol"] or sym
                rows.append(row)
        n = _safe_store(store_mod, rows)
        total_fetched += len(rows)
        total_stored += n
        logger.info("gdelt %s (%d/%d): %d fetched, %d stored", sym, i, len(symbols), len(rows), n)

    print(f"\nBACKFILL SUMMARY: {total_fetched} articles fetched, {total_stored} stored "
          f"across {len(symbols)} symbols ({days}d window)\n")
    return total_stored


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def cmd_status() -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM news_sentiment")
            row = cur.fetchone()
            total = row[0] if row else 0

            cur.execute(
                "SELECT source, count(*) AS n FROM news_sentiment "
                "GROUP BY source ORDER BY n DESC"
            )
            by_source = cur.fetchall()

            cur.execute(
                "SELECT symbol, count(*) AS n FROM news_sentiment "
                "WHERE symbol IS NOT NULL GROUP BY symbol ORDER BY n DESC, symbol LIMIT 10"
            )
            by_symbol = cur.fetchall()

            cur.execute("SELECT min(published_at), max(published_at) FROM news_sentiment")
            rng = cur.fetchone()
            dmin, dmax = (rng[0], rng[1]) if rng else (None, None)

    def table(rows, headers):
        widths = [max([len(str(h))] + [len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]
        def line(vals):
            return "  " + "  ".join(f"{str(v).ljust(w)}" for v, w in zip(vals, widths))
        print(line(headers))
        print("  " + "-" * (sum(widths) + 2 * (len(headers) - 1)))
        for r in rows:
            print(line(r))

    print(f"\nnews_sentiment STATUS — {total} total rows")
    print(f"date range: {dmin}  ->  {dmax}")
    print("\nrows by source:")
    table([(r[0] or "(null)", str(r[1])) for r in by_source], ["source", "rows"])
    print("\ntop symbols (by row count):")
    table([(r[0] or "(null)", str(r[1])) for r in by_symbol], ["symbol", "rows"])
    print()
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collector.news.pipeline",
        description="Phase 3 sentiment pipeline: fetch -> map -> score -> store.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("today", help="google news + market feeds + reddit (24h), score, store")
    sub.add_parser("premarket", help="today + market-wide aggregate -> market_sentiment (8:30 IST call)")
    sub.add_parser("cues", help="global/geopolitical news (crude/fed/us_markets/usd_inr/geo) -> global_cues")
    bf = sub.add_parser("backfill", help="GDELT per-symbol backfill")
    bf.add_argument("--days", type=int, default=60, help="window length in days (default 60)")
    sub.add_parser("status", help="news_sentiment stats table")

    args = parser.parse_args(argv)
    setup_logging()

    if args.cmd == "status":
        cmd_status()
        return 0

    # Lazy import so `status` works even if a sibling module is broken.
    from collector.news import config as config_mod
    from collector.news import mapper, scorer, sources, store

    if args.cmd == "today":
        cmd_today(mapper, scorer, store, sources, config_mod)
        return 0
    if args.cmd == "premarket":
        cmd_premarket(mapper, scorer, store, sources, config_mod)
        return 0
    if args.cmd == "cues":
        cmd_cues(mapper, scorer, store, sources, config_mod)
        return 0
    if args.cmd == "backfill":
        cmd_backfill(args, mapper, scorer, store, sources)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
