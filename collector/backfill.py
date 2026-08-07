"""Historical backfill: pull EOD + intraday candles from Fyers into TimescaleDB.

Fyers REST history returns limited windows per request, so the target range is
split into chunks (1 day for intraday resolutions, ~180 days for daily).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, time

from collector.config import settings
from collector.db import list_symbols, upsert_candles
from collector.fyers_client import FyersClient
from collector.symbols import fyers_symbol

logger = logging.getLogger(__name__)

# Fyers history returns large ranges in a single call (verified: 60d of 5m in
# one request). Chunk sizes below are generous safe caps; weekends/holidays come
# back as 'no_data' and are simply skipped.
CHUNK_DAYS = {"1d": 365, "1m": 15, "5m": 60, "15m": 60, "30m": 60, "60m": 120}


def _chunk_windows(from_dt: datetime, to_dt: datetime, chunk_days: int):
    """Yield (start, end) inclusive-ish windows covering [from_dt, to_dt]."""
    cursor = from_dt
    while cursor < to_dt:
        end = min(cursor + timedelta(days=chunk_days), to_dt)
        yield cursor, end
        cursor = end


def backfill(
    client: FyersClient,
    timeframes: list[str],
    from_dt: datetime,
    to_dt: datetime,
    limit: int | None = None,
) -> int:
    """Backfill candles for every tracked symbol. Returns total rows written.

    Args:
        timeframes: e.g. ['1d', '5m', '15m'].
        from_dt/to_dt: inclusive history window (IST).
        limit: optional max symbols to process (handy for a first test run).
    """
    symbols = list_symbols()
    if limit:
        symbols = symbols[:limit]

    total = 0
    for tf in timeframes:
        chunk_days = CHUNK_DAYS.get(tf, 1)
        for sym in symbols:
            fyers_sym = fyers_symbol(sym["symbol"], sym["instrument_type"])
            rows = 0
            for start, end in _chunk_windows(from_dt, to_dt, chunk_days):
                try:
                    candles = client.history(fyers_sym, tf, start, end)
                except Exception as exc:
                    logger.error("backfill %s %s %s..%s failed: %s", fyers_sym, tf, start, end, exc)
                    continue
                if candles:
                    rows += upsert_candles(sym["symbol"], tf, candles)
            total += rows
            logger.info("backfilled %d rows for %s %s", rows, sym["symbol"], tf)
    return total
