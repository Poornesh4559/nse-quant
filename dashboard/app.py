"""Phase 3 dashboard backend — read-only JSON API over the TimescaleDB data.

Serves OHLCV candles and pandas-computed technical indicators (SMA, EMA,
RSI, MACD, Bollinger bands, volume-SMA), per-symbol quotes (latest close,
prev close, change %), plus stub endpoints for trades / portfolio / sentiment
(those tables are populated in later phases — the API returns empty lists and
zeroed summaries gracefully until then).

All DB access is read-only, parameterized SQL only. Config comes from the
repo-root ``.env`` (see collector/config.py for the same convention).
"""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Single source of truth is the repo-root .env file (same convention as collector/config.py).
load_dotenv(REPO_ROOT / ".env")

IST = ZoneInfo("Asia/Kolkata")

DB_URL = (
    f"host={os.getenv('POSTGRES_HOST', '127.0.0.1')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'nse_quant')} "
    f"user={os.getenv('POSTGRES_USER', 'nse')} "
    f"password={os.getenv('POSTGRES_PASSWORD', '')}"
)

VALID_TIMEFRAMES = ("1d", "5m", "15m")

app = FastAPI(title="NSE Quant Dashboard API", version="0.3.0")


@contextmanager
def _get_conn() -> Iterator[psycopg2.extensions.connection]:
    """Yield a short-lived read-only psycopg2 connection."""
    conn = psycopg2.connect(DB_URL)
    try:
        yield conn
    finally:
        conn.close()


def _iso(dt) -> str:
    """Serialize a tz-aware datetime (or plain date) as ISO in Asia/Kolkata."""
    if isinstance(dt, datetime):
        return dt.astimezone(IST).isoformat()
    return dt.isoformat()  # datetime.date / date


def _is_market_open(now: datetime | None = None) -> bool:
    """True Mon-Fri between 09:15 and 15:30 IST (NSE cash market hours)."""
    now = now or datetime.now(IST)
    if now.weekday() >= 5:  # Saturday / Sunday
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)


def _json_num(v: Any) -> Any:
    """JSON-safe scalar: None for NaN/None, otherwise the value unchanged."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


# Windowed query: latest 1d candle per symbol (ROW_NUMBER) + previous close
# (LAG over ts). change_pct is computed + sorted in Python (Postgres lacks
# round(double precision, integer)).
_LATEST_SQL = """
WITH ranked AS (
    SELECT symbol, ts, close,
           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn,
           LAG(close) OVER (PARTITION BY symbol ORDER BY ts) AS prev_close
    FROM candles
    WHERE timeframe = '1d'
)
SELECT symbol, ts, close, prev_close
FROM ranked
WHERE rn = 1
"""


def _change_pct(close: float | None, prev_close: float | None) -> float | None:
    """Percent change from prev close to close, rounded to 2dp (None when incalculable)."""
    if close is None or prev_close is None or prev_close == 0:
        return None
    return round((close - prev_close) / prev_close * 100, 2)


def _latest_1d(order: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Latest 1d close + prev close + change_pct per symbol, ordered by change_pct.

    ``order`` must be "DESC" (gainers) or "ASC" (losers); rows with a NULL
    change_pct (no prior candle) always sort last. ``limit`` truncates the
    result when given.
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_LATEST_SQL)
            rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["ts"] = _iso(r["ts"])
        r["change_pct"] = _change_pct(r["close"], r["prev_close"])
    valid = [r for r in rows if r["change_pct"] is not None]
    invalid = [r for r in rows if r["change_pct"] is None]
    valid.sort(key=lambda r: r["change_pct"], reverse=(order == "DESC"))
    if limit is not None:
        rows = (valid + invalid)[:limit]
    else:
        rows = valid + invalid
    return rows


def _parse_date(value: str | None, end_of_day: bool = False) -> datetime | None:
    """Parse a YYYY-MM-DD query param into an IST-aware datetime.

    ``end_of_day=True`` pins the time to 23:59:59 IST (inclusive range end).
    Returns None when the value is missing; raises 400 on malformed input.
    """
    if value is None:
        return None
    try:
        d = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid date '{value}', expected YYYY-MM-DD")
    if end_of_day:
        return datetime.combine(d, dtime(23, 59, 59), tzinfo=IST)
    return datetime.combine(d, dtime(0, 0, 0), tzinfo=IST)


def _fetch_candle_rows(
    symbol: str,
    timeframe: str,
    days: int | None,
    limit: int,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> list[dict[str, Any]]:
    """Validate symbol/timeframe and fetch OHLCV rows ordered ts ASC.

    Precedence: explicit ``from_dt``/``to_dt`` (either one) wins over ``days``,
    which wins over ``limit``. Raises 404 for an unknown symbol and 400 for an
    invalid timeframe. Returns raw rows with ts as tz-aware datetime.
    """
    if timeframe not in VALID_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid timeframe '{timeframe}', must be one of {', '.join(VALID_TIMEFRAMES)}",
        )
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT 1 FROM symbols WHERE symbol = %s", (symbol,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail=f"symbol not found: {symbol}")
            if from_dt is not None or to_dt is not None:
                clauses: list[str] = ["symbol = %s", "timeframe = %s"]
                params: list[Any] = [symbol, timeframe]
                if from_dt is not None:
                    clauses.append("ts >= %s")
                    params.append(from_dt)
                if to_dt is not None:
                    clauses.append("ts <= %s")
                    params.append(to_dt)
                cur.execute(
                    "SELECT ts, open, high, low, close, volume FROM candles "
                    f"WHERE {' AND '.join(clauses)} ORDER BY ts ASC",
                    params,
                )
            elif days is not None:
                cutoff = datetime.now(IST) - timedelta(days=days)
                cur.execute(
                    "SELECT ts, open, high, low, close, volume FROM candles "
                    "WHERE symbol = %s AND timeframe = %s AND ts >= %s ORDER BY ts ASC",
                    (symbol, timeframe, cutoff),
                )
            else:
                cur.execute(
                    "SELECT ts, open, high, low, close, volume FROM candles "
                    "WHERE symbol = %s AND timeframe = %s ORDER BY ts DESC LIMIT %s",
                    (symbol, timeframe, limit),
                )
                return [dict(r) for r in reversed(cur.fetchall())]  # back to oldest-first
            rows = [dict(r) for r in cur.fetchall()]
    return rows


# Indicator columns rounded to 2dp (volume stays int, vol_sma20 included below).
_INDICATOR_COLS = (
    "open", "high", "low", "close",
    "sma20", "sma50", "ema12", "ema26", "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower",
)


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA/EMA/RSI/MACD/Bollinger/volume-SMA columns to an OHLCV frame.

    ``df`` must be indexed by ts (ascending) with close/volume columns. All
    windows use the standard 20/50 SMA, 12/26 EMA, Wilder RSI(14), MACD
    (12,26,9) and 20-period Bollinger bands. Warmup rows carry NaN — callers
    keep them so the frontend can render partial indicators.
    """
    close = df["close"]

    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()
    df["ema12"] = close.ewm(span=12, adjust=False).mean()
    df["ema26"] = close.ewm(span=26, adjust=False).mean()

    # Wilder RSI: ewm with alpha=1/period on gains/losses (adjust=False).
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        df["rsi14"] = 100 - 100 / (1 + rs)

    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    std20 = close.rolling(20).std()
    df["bb_upper"] = df["sma20"] + 2 * std20
    df["bb_mid"] = df["sma20"]
    df["bb_lower"] = df["sma20"] - 2 * std20

    df["vol_sma20"] = df["volume"].rolling(20).mean()
    return df


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness + DB connectivity + market-hours probe."""
    db_ok = False
    candles_total = 0
    try:
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute("SELECT count(*) FROM candles")
            row = cur.fetchone()
            candles_total = int(row[0]) if row else 0
        db_ok = True
    except Exception as exc:  # noqa: BLE001 - health must not 500 when DB is down
        logger.warning("health check: db unreachable: %s", exc)
    return {
        "status": "ok",
        "db_ok": db_ok,
        "market_open": _is_market_open(),
        "candles_total": candles_total,
    }


@app.get("/api/symbols")
def list_symbols() -> dict[str, list[dict[str, Any]]]:
    """Reference list of all instruments in the `symbols` table (incl. sector)."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT symbol, name, sector, instrument_type FROM symbols ORDER BY symbol")
            rows = [dict(r) for r in cur.fetchall()]
    return {"symbols": rows}


@app.get("/api/candles")
def get_candles(
    symbol: str = Query(..., min_length=1, description="Ticker symbol, e.g. TCS"),
    timeframe: str = Query("1d", description="Candle timeframe: 1d, 5m or 15m"),
    days: int | None = Query(None, ge=1, le=365, description="Return candles with ts >= now_IST - days (wins over limit)"),
    from_date: str | None = Query(None, alias="from", description="Exclusive start date YYYY-MM-DD (wins over days)"),
    to_date: str | None = Query(None, alias="to", description="Inclusive end date YYYY-MM-DD (wins over days)"),
    limit: int = Query(250, ge=1, le=5000, description="Newest N candles to return when neither days nor from/to is given"),
) -> dict[str, Any]:
    """OHLCV candles for one symbol+timeframe, oldest-first.

    Precedence: explicit ``from``/``to`` range > trailing ``days`` window >
    newest ``limit`` candles.
    """
    from_dt = _parse_date(from_date)
    to_dt = _parse_date(to_date, end_of_day=True)
    rows = _fetch_candle_rows(symbol, timeframe, days, limit, from_dt, to_dt)
    for c in rows:
        c["ts"] = _iso(c["ts"])
    return {"symbol": symbol, "timeframe": timeframe, "candles": rows}


@app.get("/api/indicators")
def get_indicators(
    symbol: str = Query(..., min_length=1, description="Ticker symbol, e.g. TCS"),
    timeframe: str = Query("1d", description="Candle timeframe: 1d, 5m or 15m"),
    days: int | None = Query(None, ge=1, le=365, description="Return candles with ts >= now_IST - days (wins over limit)"),
    from_date: str | None = Query(None, alias="from", description="Exclusive start date YYYY-MM-DD (wins over days)"),
    to_date: str | None = Query(None, alias="to", description="Inclusive end date YYYY-MM-DD (wins over days)"),
    limit: int = Query(250, ge=1, le=5000, description="Newest N candles to use when neither days nor from/to is given"),
) -> dict[str, Any]:
    """OHLCV candles augmented with SMA/EMA/RSI/MACD/Bollinger/volume-SMA.

    Same row-selection semantics as /api/candles (from/to > days > limit),
    oldest-first. Warmup rows keep null indicator values — they are not
    dropped so the frontend can render partial curves. Floats rounded to 2dp.
    """
    from_dt = _parse_date(from_date)
    to_dt = _parse_date(to_date, end_of_day=True)
    rows = _fetch_candle_rows(symbol, timeframe, days, limit, from_dt, to_dt)
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    computed = _compute_indicators(df)
    indicator_cols = list(_INDICATOR_COLS)
    computed[indicator_cols] = computed[indicator_cols].round(2)
    computed["vol_sma20"] = computed["vol_sma20"].round(2)

    candles: list[dict[str, Any]] = []
    for ts, r in computed.iterrows():
        candle: dict[str, Any] = {"ts": _iso(ts.to_pydatetime())}
        for col in (*_INDICATOR_COLS, "volume", "vol_sma20"):
            candle[col] = _json_num(r[col])
        candle["volume"] = int(candle["volume"]) if candle["volume"] is not None else None
        candles.append(candle)
    return {"symbol": symbol, "timeframe": timeframe, "candles": candles}


@app.get("/api/trades")
def get_trades(
    limit: int = Query(50, ge=1, le=500, description="Newest N trades to return"),
) -> dict[str, Any]:
    """Paper trades, newest first. Returns an empty list until trades exist."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, symbol, side, qty, price, ts, strategy, status "
                "FROM trades ORDER BY ts DESC LIMIT %s",
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
    for t in rows:
        t["ts"] = _iso(t["ts"])
        t["qty"] = int(t["qty"])
    return {"trades": rows}


@app.get("/api/portfolio")
def portfolio() -> dict[str, Any]:
    """Paper-trading portfolio summary + per-symbol positions.

    Trades have no entry/exit linkage yet, so P&L fields are zeroed and
    win_rate is null. Positions aggregate net qty and total invested per
    symbol from the trades table (empty list when no trades exist).
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT count(*) AS n FROM trades")
            row = cur.fetchone()
            trade_count = int(row["n"]) if row else 0
            cur.execute(
                "SELECT symbol, sum(qty) AS qty, sum(qty * price) AS invested "
                "FROM trades GROUP BY symbol ORDER BY symbol"
            )
            pos_rows = [dict(r) for r in cur.fetchall()]

    positions = [
        {
            "symbol": r["symbol"],
            "qty": int(r["qty"] or 0),
            "invested": round(float(r["invested"] or 0.0), 2),
        }
        for r in pos_rows
    ]
    total_invested = round(float(sum(p["invested"] for p in positions)), 2)
    summary = {
        "open_positions": len(positions),
        "total_invested": total_invested,
        "current_value": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "win_rate": None,
        "trade_count": trade_count,
    }
    return {"summary": summary, "positions": positions}


@app.get("/api/recent-trades")
def recent_trades(
    limit: int = Query(10, ge=1, le=50, description="Newest N trades with full decision context"),
) -> dict[str, Any]:
    """Last N trades joined with their full decision log (composite, ML prob,
    sentiment, regime, technicals, LLM rating/reason) — the complete trace."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT t.id AS trade_id, t.symbol, t.side, t.qty, t.price,
                          t.ts, t.fees, t.pnl, t.pnl_pct, t.exit_reason, t.position_id,
                          d.composite_score, d.mom_rank, d.ml_p_up, d.sent_3d,
                          d.market_sentiment, d.global_cues, d.regime_score, d.regime_risk_on,
                          d.rsi14, d.macd, d.bb_pos, d.atr14, d.ret_1, d.ret_5, d.ret_21,
                          d.llm_rating, d.llm_reason, d.llm_model
                   FROM trades t
                   LEFT JOIN trade_decisions d ON d.trade_id = t.id
                   ORDER BY t.ts DESC LIMIT %s""",
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
    for t in rows:
        t["ts"] = _iso(t["ts"])
        t["qty"] = int(t["qty"])
    return {"trades": rows}


@app.get("/api/market")
def market_panel() -> dict[str, Any]:
    """Landing-panel data: latest market sentiment call, global cues (with
    per-theme breakdown) and the latest equity snapshot vs benchmark."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT date, avg_compound, n_articles, n_positive, n_negative, direction "
                "FROM market_sentiment ORDER BY date DESC LIMIT 1"
            )
            row = cur.fetchone()
            ms = dict(row) if row else None
            cur.execute(
                "SELECT date, avg_compound, n_articles, n_positive, n_negative, direction, themes "
                "FROM global_cues ORDER BY date DESC LIMIT 1"
            )
            row = cur.fetchone()
            gc = dict(row) if row else None
            if gc:
                gc["themes"] = json.loads(gc["themes"]) if isinstance(gc["themes"], str) else gc["themes"]
            cur.execute(
                "SELECT date, equity, cash, benchmark, strategy FROM equity_curve "
                "ORDER BY date DESC LIMIT 1"
            )
            row = cur.fetchone()
            eq = dict(row) if row else None
    return {"market_sentiment": ms, "global_cues": gc, "equity": eq}


@app.get("/api/equity")
def equity_series(
    limit: int = Query(200, ge=2, le=1000, description="Equity curve rows (oldest first)"),
) -> dict[str, Any]:
    """Daily paper equity vs benchmark (NIFTY 50 buy-hold), oldest first."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT date, equity, cash, benchmark, strategy FROM equity_curve "
                "ORDER BY date DESC LIMIT %s",
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
    rows.reverse()
    for r in rows:
        r["date"] = _iso(r["date"])
    return {"points": rows}


@app.get("/api/decisions")
def decisions(
    limit: int = Query(20, ge=1, le=200, description="Newest N decision-log rows"),
) -> dict[str, Any]:
    """Raw trade_decisions rows (the training log), newest first."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM trade_decisions ORDER BY decision_ts DESC LIMIT %s", (limit,)
            )
            rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["decision_ts"] = _iso(r["decision_ts"])
    return {"decisions": rows}


RAW_TABLES = {
    "trades": "trades",
    "trade_decisions": "trade_decisions",
    "news_sentiment": "news_sentiment",
    "market_sentiment": "market_sentiment",
    "global_cues": "global_cues",
    "equity_curve": "equity_curve",
    "symbols": "symbols",
    "candles_1d": "candles",
}


@app.get("/api/raw")
def raw_table(
    table: str = Query(..., description="Whitelisted table key: " + ", ".join(RAW_TABLES)),
    limit: int = Query(50, ge=1, le=500, description="Max rows"),
) -> dict[str, Any]:
    """Raw table viewer over a whitelist. candles_1d maps to candles filtered
    to timeframe='1d' (and capped at 500 rows). Never exposes anything else."""
    real = RAW_TABLES.get(table)
    if not real:
        raise HTTPException(status_code=404, detail=f"unknown table key '{table}'")
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if real == "candles":
                cur.execute(
                    "SELECT * FROM candles WHERE timeframe='1d' ORDER BY ts DESC LIMIT %s", (limit,)
                )
            else:
                cur.execute(f"SELECT * FROM {real} ORDER BY 1 DESC LIMIT %s", (limit,))
            rows = [dict(r) for r in cur.fetchall()]
            cur.execute(
                """SELECT column_name, data_type FROM information_schema.columns
                   WHERE table_name = %s ORDER BY ordinal_position""",
                (real,),
            )
            cols = [dict(r) for r in cur.fetchall()]
    for r in rows:
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = _iso(v)
    return {"table": real, "columns": cols, "rows": rows}


@app.get("/api/sentiment")
def sentiment(
    symbol: str | None = Query(None, min_length=1, description="Optional ticker filter (LIKE match on symbol column)"),
    limit: int = Query(20, ge=1, le=100, description="Newest N sentiment rows to return"),
) -> dict[str, Any]:
    """News sentiment rows, newest first. Empty list until the pipeline runs.

    When ``symbol`` is given, rows are filtered with a LIKE match on the
    symbol column (news may store e.g. 'TCS' or 'TATAMOTORS').
    """
    sql = (
        "SELECT id, symbol, source, title, url, published_at, "
        "sentiment_compound, sentiment_label FROM news_sentiment"
    )
    params: list[Any] = []
    if symbol:
        sql += " WHERE symbol LIKE %s"
        params.append(f"%{symbol}%")
    sql += " ORDER BY published_at DESC LIMIT %s"
    params.append(limit)
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["published_at"] = _iso(r["published_at"])
    return {"rows": rows}


@app.get("/api/latest")
def latest() -> dict[str, Any]:
    """Latest 1d close, prev close and change % for every symbol, best gainers first."""
    rows = _latest_1d("DESC")
    return {"as_of": _iso(datetime.now(IST)), "rows": rows}


@app.get("/api/movers")
def movers(n: int = Query(10, ge=1, le=100, description="How many gainers/losers to return")) -> dict[str, Any]:
    """Top N gainers (change_pct desc) and top N losers (change_pct asc) on 1d close."""
    return {"gainers": _latest_1d("DESC", n), "losers": _latest_1d("ASC", n)}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the dashboard frontend entry point."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/ingest/reddit")
async def ingest_reddit(request: Request) -> dict[str, Any]:
    """Ingest endpoint for the HOME-SERVER reddit scraper (live subreddit feed).

    The home LXC (home IP — not reddit-blocked) scrapes subreddit JSON and
    POSTs here. Bearer token protects the write. Posts are mapped to symbols,
    dual-model scored (VADER+FinBERT) and deduped by permalink — the same
    pipeline quality as every other news source.

    Body: {"posts": [{"title", "permalink", "created_utc", "subreddit", "selftext"}]}
    """
    token = (request.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
    if not token or token != os.getenv("REDDIT_INGEST_TOKEN", ""):
        raise HTTPException(status_code=401, detail="invalid ingest token")
    body = await request.json()
    posts = body.get("posts") or []
    if not posts:
        return {"stored": 0, "posts": 0}

    # lazy import so the dashboard doesn't pay torch/FinBERT at startup
    from collector.news import mapper, scorer, store as news_store

    rows: list[dict] = []
    for p in posts:
        try:
            title = str(p.get("title") or "")[:500]
            selftext = str(p.get("selftext") or "")[:400]
            text = (title + " " + selftext).strip()
            if not text:
                continue
            symbols = mapper.map_symbols(text)
            score = scorer.score_text(text)
            rows.append({
                "source": f"reddit_live:{p.get('subreddit') or '?'}",
                "symbol": symbols[0] if symbols else None,
                "title": title,
                "url": f"https://www.reddit.com{p.get('permalink') or ''}",
                "published_at": datetime.fromtimestamp(int(p.get("created_utc", 0)), tz=timezone.utc)
                                if p.get("created_utc") else None,
                "sentiment_compound": score.get("compound"),
                "sentiment_label": score.get("label"),
            })
        except Exception:  # noqa: BLE001
            logger.exception("ingest failed for reddit post %r", p.get("title"))
    stored = news_store.upsert_news(rows) if rows else 0
    logger.info("reddit ingest: %d posts -> %d stored", len(posts), stored)
    return {"posts": len(posts), "stored": stored}


# Everything under /static/ (plus the API routes registered above, which take
# precedence) is served from the static directory.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
