"""Data loading for backtesting — daily candles, sentiment, benchmark index.

All queries are parameterized. Loaded frames are cached in a module-level dict
keyed by the requested symbol tuple so repeated research runs are fast; use
:func:`clear_cache` to invalidate (e.g. after a backfill lands).

Conventions
-----------
* Daily candle timestamps are stored UTC midnight. The trading *date* is the
  IST calendar day (UTC midnight == 05:30 IST), so we convert ts -> IST date
  and use a tz-naive normalized ``DatetimeIndex`` of dates (ascending).
* ``load_daily`` returns one DataFrame per symbol: index = date, columns =
  open/high/low/close/volume (sorted, deduped — the PK already enforces this).
* Benchmark index candles live in the same `candles` table under symbol
  'NIFTY 50' (instrument_type INDEX).
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd
import psycopg2
import psycopg2.extras

from collector.config import settings

logger = logging.getLogger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
BENCHMARK_SYMBOL = "NIFTY 50"

# module-level cache: {(tuple(symbols),) -> {symbol: DataFrame}}
_CACHE: dict[tuple, dict[str, pd.DataFrame]] = {}
_SENTIMENT_CACHE: pd.DataFrame | None = None


def _conn():
    return psycopg2.connect(settings.db_url)


def _ist_date(ts: pd.Series) -> pd.Series:
    """Convert a UTC timestamptz series to tz-naive IST calendar dates."""
    tz = "Asia/Kolkata"
    return (
        pd.to_datetime(ts, utc=True)
        .dt.tz_convert(tz)
        .dt.normalize()
        .dt.tz_localize(None)
    )


def load_daily(symbols: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load daily OHLCV candles for the given symbols (or all with 1d data).

    Returns ``{symbol: DataFrame}`` where each frame is indexed by IST date
    (ascending) with columns open/high/low/close/volume.
    """
    key = tuple(sorted(symbols)) if symbols is not None else ("__all__",)
    if key in _CACHE:
        return _CACHE[key]

    sql = """
        SELECT symbol, ts, open, high, low, close, volume
        FROM candles
        WHERE timeframe = '1d'
    """
    params: tuple = ()
    if symbols is not None:
        placeholders = ", ".join(["%s"] * len(symbols))
        sql += f" AND symbol IN ({placeholders})"
        params = tuple(symbols)
    sql += " ORDER BY symbol, ts"

    frames: dict[str, pd.DataFrame] = {}
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        logger.warning("load_daily: no 1d candles found for %s", symbols or "any symbol")
        return frames

    by_symbol: dict[str, list[dict]] = {}
    for r in rows:
        by_symbol.setdefault(r["symbol"], []).append(r)

    for symbol, sym_rows in by_symbol.items():
        df = pd.DataFrame(sym_rows)
        df["date"] = _ist_date(df["ts"])
        df = (
            df[["date", *OHLCV_COLS]]
            .drop_duplicates("date")
            .set_index("date")
            .sort_index()
        )
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        frames[symbol] = df

    _CACHE[key] = frames
    return frames


def load_sentiment() -> pd.DataFrame:
    """Per-symbol daily average sentiment from `news_sentiment`.

    Returns DataFrame(symbol, date, avg_compound): mean of sentiment_compound
    grouped by symbol and IST published date. Null-symbol rows are dropped.
    """
    global _SENTIMENT_CACHE
    if _SENTIMENT_CACHE is not None:
        return _SENTIMENT_CACHE

    sql = """
        SELECT symbol, published_at, sentiment_compound
        FROM news_sentiment
        WHERE symbol IS NOT NULL AND sentiment_compound IS NOT NULL
    """
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    if not rows:
        _SENTIMENT_CACHE = pd.DataFrame(columns=["symbol", "date", "avg_compound"])
        return _SENTIMENT_CACHE

    df = pd.DataFrame(rows)
    df["date"] = _ist_date(df["published_at"])
    grouped = (
        df.groupby(["symbol", "date"])["sentiment_compound"]
        .mean()
        .rename("avg_compound")
        .reset_index()
    )
    _SENTIMENT_CACHE = grouped
    return grouped


def load_benchmark() -> pd.DataFrame:
    """Daily candles for the NIFTY 50 index (benchmark buy-and-hold)."""
    return load_daily([BENCHMARK_SYMBOL])[BENCHMARK_SYMBOL]


def clear_cache() -> None:
    """Drop cached frames (call after a backfill lands)."""
    global _SENTIMENT_CACHE
    _CACHE.clear()
    _SENTIMENT_CACHE = None
