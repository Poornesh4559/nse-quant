"""Intraday collection: poll the last N hours of 5m/15m candles periodically.

Designed to run from cron every ~5 minutes during market hours. Because the
current (forming) candle is not returned until closed, we fetch a window that
starts a little before the latest stored candle to fill any gaps.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from collector.db import list_symbols, latest_candle_ts, upsert_candles
from collector.fyers_client import FyersClient
from collector.symbols import fyers_symbol

logger = logging.getLogger(__name__)

WINDOW_HOURS = 3  # how far back each poll reaches to close gaps


def collect_intraday(client: FyersClient, timeframes: list[str] | None = None) -> int:
    """Fetch recent intraday candles for all symbols. Returns rows written."""
    timeframes = timeframes or ["5m", "15m"]
    symbols = list_symbols()
    total = 0
    now = datetime.now().astimezone()

    for tf in timeframes:
        for sym in symbols:
            fyers_sym = fyers_symbol(sym["symbol"], sym["instrument_type"])
            latest = latest_candle_ts(sym["symbol"], tf)
            if latest is not None:
                start = latest - timedelta(minutes=30)
            else:
                start = now - timedelta(hours=WINDOW_HOURS)
            if start >= now:
                continue
            try:
                candles = client.history(fyers_sym, tf, start, now)
            except Exception as exc:
                logger.error("intraday %s %s failed: %s", fyers_sym, tf, exc)
                continue
            if candles:
                total += upsert_candles(sym["symbol"], tf, candles)
    return total
