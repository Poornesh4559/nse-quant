"""Phase 2 dashboard backend — read-only JSON API over the TimescaleDB data.

Serves OHLCV candles and per-symbol quotes (latest close, prev close, change %)
from the `nse_quant` database for the Phase 2 web dashboard, plus the static
frontend files from ``dashboard/static/``.

All DB access is read-only, parameterized SQL only. Config comes from the
repo-root ``.env`` (see collector/config.py for the same convention).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

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

app = FastAPI(title="NSE Quant Dashboard API", version="0.2.0")


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
    limit: int = Query(250, ge=1, le=2000, description="Newest N candles to return"),
) -> dict[str, Any]:
    """OHLCV candles for one symbol+timeframe, oldest-first, capped to newest N."""
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
            cur.execute(
                "SELECT ts, open, high, low, close, volume FROM candles "
                "WHERE symbol = %s AND timeframe = %s ORDER BY ts DESC LIMIT %s",
                (symbol, timeframe, limit),
            )
            rows = [dict(r) for r in reversed(cur.fetchall())]  # back to oldest-first
    for c in rows:
        c["ts"] = _iso(c["ts"])
    return {"symbol": symbol, "timeframe": timeframe, "candles": rows}


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
