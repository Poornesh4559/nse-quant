"""nse-quant MCP server — exposes the whole pipeline to any LLM client
(Claude Desktop, OpenCode, Cursor, ...) as callable tools.

Run:  .venv/bin/python -m mcp_nse.server        (stdio transport)
Test: .venv/bin/python -m mcp_nse.client "get_sentiment" '{"symbol":"TCS"}'

All tools are READ-ONLY: they query the TimescaleDB / signal files the
pipeline already produces. No order placement, no writes to the trading
tables (paper or real) — keep it safe by design.
"""
from __future__ import annotations

import json
from pathlib import Path

import psycopg2
from fastmcp import FastMCP

from collector.config import settings

REPO_ROOT = Path(__file__).resolve().parent.parent
PICKS_PATH = REPO_ROOT / "data" / "signals" / "nextday_picks.json"

mcp = FastMCP("nse-quant", instructions=(
    "You are connected to Poornesh's NSE quant stack. You can query live "
    "candles, indicators, sentiment scores, raw news, the market regime "
    "(market sentiment + global cues + Asian markets), the paper portfolio "
    "and today's model picks. All data is as of the last completed trading "
    "session. Prefer get_sentiment (fast, deterministic score) over raw news "
    "unless the user wants headlines."
))


def _conn():
    return psycopg2.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        dbname=settings.postgres_db, user=settings.postgres_user,
        password=settings.postgres_password,
    )


def _iso(v):
    from datetime import date, datetime
    if isinstance(v, datetime):
        return v.astimezone().isoformat(timespec="seconds")
    if isinstance(v, date):
        return v.isoformat()
    return v


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_sentiment(symbol: str, days: int = 3) -> dict:
    """Computed sentiment score for a symbol over the trailing N days.

    Score = mean VADER+FinBERT compound (-1..1) across mapped news articles.
    Returns the deterministic score the pipeline computed — not raw news.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(AVG(sentiment_compound), 0), COUNT(*) FROM news_sentiment "
            "WHERE symbol=%s AND published_at >= now() - make_interval(days => %s)",
            (symbol, days),
        )
        avg, n = cur.fetchone()
        cur.execute(
            "SELECT title, published_at, sentiment_compound FROM news_sentiment "
            "WHERE symbol=%s AND published_at >= now() - make_interval(days => %s) "
            "ORDER BY published_at DESC LIMIT 3", (symbol, days),
        )
        recent = [{"title": t, "published": _iso(p), "compound": round(float(c), 3) if c else None}
                  for t, p, c in cur.fetchall()]
    return {"symbol": symbol, "days": days, "compound": round(float(avg), 3),
            "n_articles": int(n), "label": "POSITIVE" if avg >= 0.1 else ("NEGATIVE" if avg <= -0.1 else "NEUTRAL"),
            "recent_headlines": recent}


@mcp.tool()
def get_news(symbol: str, limit: int = 10) -> dict:
    """Raw recent news headlines for a symbol (for depth / LLM judgement)."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT title, url, source, published_at, sentiment_compound FROM news_sentiment "
            "WHERE symbol=%s ORDER BY published_at DESC LIMIT %s", (symbol, limit),
        )
        rows = [{"title": t, "url": u, "source": s, "published": _iso(p),
                 "compound": round(float(c), 3) if c else None} for t, u, s, p, c in cur.fetchall()]
    return {"symbol": symbol, "articles": rows}


@mcp.tool()
def get_candles(symbol: str, timeframe: str = "1d", limit: int = 100) -> dict:
    """Recent OHLCV candles for a symbol. timeframes: 1d, 15m, 5m."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ts, open, high, low, close, volume FROM candles "
            "WHERE symbol=%s AND timeframe=%s ORDER BY ts DESC LIMIT %s",
            (symbol, timeframe, limit),
        )
        rows = [{"ts": _iso(t), "open": float(o), "high": float(h), "low": float(l),
                 "close": float(c), "volume": int(v or 0)} for t, o, h, l, c, v in cur.fetchall()]
    rows.reverse()
    return {"symbol": symbol, "timeframe": timeframe, "candles": rows}


@mcp.tool()
def get_indicators(symbol: str, timeframe: str = "1d", limit: int = 50) -> dict:
    """Latest technical indicators: SMA20/50, EMA12/26, RSI14, MACD, Bollinger.

    Note: indicators are computed on DAILY data only — the timeframe argument
    is accepted for API compatibility but anything other than '1d' is
    rejected (it was previously silently ignored).
    """
    if timeframe != "1d":
        return {"symbol": symbol,
                "error": f"indicators are computed on daily data only (got timeframe={timeframe!r}); use '1d'"}
    from analysis.data import load_daily
    from analysis.features import compute_features
    frames = load_daily()
    df = frames.get(symbol)
    if df is None:
        return {"symbol": symbol, "error": "no daily data for symbol"}
    feats = compute_features(df).tail(limit)
    out = []
    for ts, row in feats.iterrows():
        out.append({
            "date": _iso(ts), "close": round(float(row.get("close", 0)), 2),
            "rsi14": _r(row.get("rsi14")), "macd": _r(row.get("macd")),
            "macd_signal": _r(row.get("macd_signal")), "bb_pos": _r(row.get("bb_pos")),
            "atr14": _r(row.get("atr14")), "mom_rank": _r(row.get("mom_rank")),
        })
    return {"symbol": symbol, "rows": out}


def _r(v):
    return None if v is None or v != v else round(float(v), 3)  # NaN check


@mcp.tool()
def get_market_pulse() -> dict:
    """Market regime inputs: market-wide sentiment, global cues (incl. Asia),
    and the latest paper equity vs benchmark."""
    from bot.paper import market_regime
    regime = market_regime()
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT date, direction, avg_compound, n_articles FROM market_sentiment "
                    "ORDER BY date DESC LIMIT 1")
        ms = cur.fetchone()
        cur.execute("SELECT date, direction, avg_compound, themes FROM global_cues "
                    "ORDER BY date DESC LIMIT 1")
        gc = cur.fetchone()
        cur.execute("SELECT date, equity, benchmark FROM equity_curve ORDER BY date DESC LIMIT 1")
        eq = cur.fetchone()
    themes = {}
    if gc and gc[3]:
        themes = json.loads(gc[3]) if isinstance(gc[3], str) else gc[3]
    return {
        "regime": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in regime.items()},
        "market_sentiment": {"date": _iso(ms[0]), "direction": ms[1], "compound": round(float(ms[2]), 3),
                             "articles": int(ms[3])} if ms else None,
        "global_cues": {"date": _iso(gc[0]), "direction": gc[1], "compound": round(float(gc[2]), 3),
                        "themes": themes} if gc else None,
        "equity": {"date": _iso(eq[0]), "equity": round(float(eq[1]), 2),
                   "benchmark": round(float(eq[2]), 2)} if eq else None,
    }


@mcp.tool()
def get_paper_portfolio() -> dict:
    """Current paper-trading positions, cash and total equity (₹)."""
    from bot.paper import _portfolio_equity, open_positions
    positions = open_positions()
    out = []
    for p in positions:
        from bot.paper import _last_close
        close = _last_close(p["symbol"])
        out.append({"symbol": p["symbol"], "qty": int(p["qty"]),
                    "entry_price": round(float(p["price"]), 2), "last_close": round(close, 2),
                    "unrealized_pnl": round((close - p["price"]) * p["qty"], 2)})
    return {"positions": out, "total_equity": round(_portfolio_equity(), 2)}


@mcp.tool()
def get_today_picks() -> dict:
    """Today's model picks (composite = 0.4*momentum + 0.35*ML + 0.25*sentiment),
    from the latest signal run. Includes sentiment guardrail status."""
    if not PICKS_PATH.exists():
        return {"error": "no picks file yet — run analysis.ml.score"}
    data = json.loads(PICKS_PATH.read_text())
    return {"as_of": data.get("as_of"), "next_trade_date": data.get("next_trade_date"),
            "top5": data.get("top5"), "top10": data.get("top10")}


@mcp.tool()
def get_decision_log(limit: int = 10) -> dict:
    """Recent trade decisions (the training log): what the bot considered,
    the LLM rating, and whether it executed."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, action, decision_ts, composite_score, llm_rating, llm_reason, "
            "executed, trade_id FROM trade_decisions ORDER BY decision_ts DESC LIMIT %s", (limit,),
        )
        rows = [{"symbol": s, "action": a, "at": _iso(t),
                 "composite": round(float(c), 3) if c else None,
                 "llm_rating": round(float(r), 3) if r else None,
                 "llm_reason": rs, "executed": bool(e), "trade_id": ti}
                for s, a, t, c, r, rs, e, ti in cur.fetchall()]
    return {"decisions": rows}


@mcp.tool()
def run_backtest(strategy: str = "rsi2_reversion", symbol: str = "NIFTY 50") -> dict:
    """Run one classic strategy backtest vs buy-and-hold (real Indian costs).

    strategies: rsi2_reversion, macd_cross, golden_cross, bollinger_reversion,
    momentum_12_1, donchian_breakout, buyhold.
    """
    from analysis.data import load_daily
    from analysis.engine import BacktestConfig, BacktestEngine
    from analysis.metrics import compute_metrics
    from analysis.strategies import STRATEGIES
    frames = load_daily()
    df = frames.get(symbol)
    if df is None:
        return {"error": f"no data for {symbol}"}
    if strategy not in STRATEGIES:
        return {"error": f"unknown strategy {strategy}; pick from {list(STRATEGIES)}"}
    sig = STRATEGIES[strategy]().run(df)
    res = BacktestEngine().run(df, sig, BacktestConfig(), symbol=symbol)
    m = compute_metrics(res)
    return {"strategy": strategy, "symbol": symbol, "metrics": m,
            "n_trades": len([t for t in res.trades if t.get("closed")]),
            "total_fees": round(res.total_fees, 2)}


if __name__ == "__main__":
    mcp.run()
