"""End-of-day collection: fetch today's daily candle for every symbol.

Designed to run from cron after market close (~18:00 IST) so the day's final
OHLCV is committed to TimescaleDB.
"""

from __future__ import annotations

import logging
from datetime import datetime

from collector.db import list_symbols, upsert_candles
from collector.fyers_client import FyersClient
from collector.symbols import fyers_symbol

logger = logging.getLogger(__name__)


def collect_eod(client: FyersClient) -> int:
    """Fetch today's daily candle for all symbols. Returns rows written."""
    symbols = list_symbols()
    today = datetime.now().astimezone().date()
    start = datetime.combine(today, datetime.min.time()).astimezone()
    end = datetime.combine(today, datetime.max.time()).astimezone()
    total = 0
    for sym in symbols:
        fyers_sym = fyers_symbol(sym["symbol"], sym["instrument_type"])
        try:
            candles = client.history(fyers_sym, "1d", start, end)
        except Exception as exc:
            logger.error("eod %s failed: %s", fyers_sym, exc)
            continue
        if candles:
            total += upsert_candles(sym["symbol"], "1d", candles)
    return total
