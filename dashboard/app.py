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

import logging
import math
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
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


def _iso(dt: datetime) -> str:
    """Serialize a tz-aware datetime as ISO8601 in Asia/Kolkata."""
    return dt.astimezone(IST).isoformat()


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


def _fetch_candle_rows(
    symbol: str, timeframe: str, days: int | None, limit: int
) -> list[dict[str, Any]]:
    """Validate symbol/timeframe and fetch OHLCV rows ordered ts ASC.

    ``days`` (when given, wins over ``limit``) fetches every candle with
    ts >= now_IST - days; otherwise the newest ``limit`` candles are returned
    (existing behaviour). Raises 404 for an unknown symbol and 400 for an
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
            if days is not None:
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
    """Reference list of all instruments in the `symbols` table."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT symbol, name, instrument_type FROM symbols ORDER BY symbol")
            rows = [dict(r) for r in cur.fetchall()]
    return {"symbols": rows}


@app.get("/api/candles")
def get_candles(
    symbol: str = Query(..., min_length=1, description="Ticker symbol, e.g. TCS"),
    timeframe: str = Query("1d", description="Candle timeframe: 1d, 5m or 15m"),
    days: int | None = Query(None, ge=1, le=365, description="Return candles with ts >= now_IST - days (wins over limit)"),
    limit: int = Query(250, ge=1, le=5000, description="Newest N candles to return when `days` is not given"),
) -> dict[str, Any]:
    """OHLCV candles for one symbol+timeframe, oldest-first.

    Either the newest ``limit`` candles or — when ``days`` is provided — every
    candle within the trailing window (``days`` takes precedence).
    """
    rows = _fetch_candle_rows(symbol, timeframe, days, limit)
    for c in rows:
        c["ts"] = _iso(c["ts"])
    return {"symbol": symbol, "timeframe": timeframe, "candles": rows}


@app.get("/api/indicators")
def get_indicators(
    symbol: str = Query(..., min_length=1, description="Ticker symbol, e.g. TCS"),
    timeframe: str = Query("1d", description="Candle timeframe: 1d, 5m or 15m"),
    days: int | None = Query(None, ge=1, le=365, description="Return candles with ts >= now_IST - days (wins over limit)"),
    limit: int = Query(250, ge=1, le=5000, description="Newest N candles to use when `days` is not given"),
) -> dict[str, Any]:
    """OHLCV candles augmented with SMA/EMA/RSI/MACD/Bollinger/volume-SMA.

    Same row-selection semantics as /api/candles (days wins over limit),
    oldest-first. Warmup rows keep null indicator values — they are not
    dropped so the frontend can render partial curves. Floats rounded to 2dp.
    """
    rows = _fetch_candle_rows(symbol, timeframe, days, limit)
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


# Everything under /static/ (plus the API routes registered above, which take
# precedence) is served from the static directory.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
