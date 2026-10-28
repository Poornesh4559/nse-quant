"""End-of-day collection: fetch today's daily candle for every symbol.

Designed to run from cron after market close (~18:00 IST) so the day's final
OHLCV is committed to TimescaleDB. All time windows are pinned to
Asia/Kolkata: on a UTC host the old local-midnight window excluded the
IST-midnight daily candle epoch and silently collected nothing.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from collector.db import list_symbols, upsert_candles
from collector.fyers_client import FyersClient
from collector.symbols import fyers_symbol

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


def is_fyers_request_limit_error(exc: Exception) -> bool:
    """Identify Fyers' daily plan cap so EOD can fall back to Yahoo."""
    text = str(exc).lower()
    return "-360" in text or "5,000 requests/day".lower() in text or "5000 requests/day" in text


def _yahoo_ticker(symbol: dict) -> str:
    if symbol["instrument_type"] == "INDEX":
        return "^NSEI" if symbol["symbol"] == "NIFTY 50" else "^NSEBANK"
    return f"{symbol['symbol']}.NS"


def _collect_yahoo_daily(symbols: list[dict], start: date, end: date) -> int:
    """Batch-fetch recent daily bars from Yahoo when Fyers exhausts its quota."""
    import yfinance as yf

    ticker_for = {row["symbol"]: _yahoo_ticker(row) for row in symbols}
    tickers = list(dict.fromkeys(ticker_for.values()))
    data = yf.download(
        tickers, start=start.isoformat(), end=end.isoformat(), auto_adjust=False,
        progress=False, group_by="ticker", threads=True, actions=False,
    )
    if data.empty:
        raise RuntimeError("Yahoo Finance returned no daily candles")

    total = 0
    for row in symbols:
        ticker = ticker_for[row["symbol"]]
        try:
            frame = data[ticker]
        except (KeyError, TypeError):
            logger.warning("Yahoo daily fallback missing %s (%s)", row["symbol"], ticker)
            continue
        candles = []
        for idx, bar in frame.iterrows():
            close = bar.get("Close")
            if pd.isna(close):
                continue
            volume = bar.get("Volume", 0)
            candles.append({
                # Match the DB convention used by Fyers daily candles: UTC midnight.
                "ts": datetime.combine(pd.Timestamp(idx).date(), time.min, tzinfo=timezone.utc),
                "open": float(bar["Open"]),
                "high": float(bar["High"]),
                "low": float(bar["Low"]),
                "close": float(close),
                "volume": 0 if pd.isna(volume) else int(volume),
            })
        total += upsert_candles(row["symbol"], "1d", candles)
    return total



def collect_eod(client: FyersClient) -> int:
    """Fetch today's daily candle for all symbols. Returns rows written."""
    symbols = list_symbols()
    today = datetime.now(IST).date()
    start = datetime.combine(today, datetime.min.time(), tzinfo=IST)
    end = datetime.combine(today, datetime.max.time(), tzinfo=IST)
    total = 0
    quota_hit = False
    for sym in symbols:
        fyers_sym = fyers_symbol(sym["symbol"], sym["instrument_type"])
        try:
            candles = client.history(fyers_sym, "1d", start, end)
        except Exception as exc:
            logger.error("eod %s failed: %s", fyers_sym, exc)
            if is_fyers_request_limit_error(exc):
                quota_hit = True
                break
            continue
        if candles:
            total += upsert_candles(sym["symbol"], "1d", candles)

    if quota_hit:
        # Recover the interrupted symbol set and recent missing sessions in one
        # batch call, rather than hammering Fyers after its daily cap is reached.
        yahoo_start = today - timedelta(days=14)
        try:
            recovered = _collect_yahoo_daily(symbols, yahoo_start, today + timedelta(days=1))
            logger.warning("Fyers daily quota reached; Yahoo backfill upserted %d rows", recovered)
            total += recovered
        except Exception:
            logger.exception("Fyers daily quota reached and Yahoo fallback failed")
    return total
