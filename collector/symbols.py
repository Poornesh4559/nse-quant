"""Symbol universe: seed the `symbols` table with the NIFTY 50 index constituents.

Primary source is the public NSE endpoint. If the VPS IP is blocked, falls
back to a bundled static list (symbols rarely change mid-year).
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from collector.db import upsert_symbols

logger = logging.getLogger(__name__)

NSE_CONSTITUENTS_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

# Indices we track separately. DB ticker -> Fyers index symbol.
INDICES = [
    {"symbol": "NIFTY 50", "fyers_symbol": "NSE:NIFTY50-INDEX", "instrument_type": "INDEX"},
    {"symbol": "BANKNIFTY", "fyers_symbol": "NSE:NIFTYBANK-INDEX", "instrument_type": "INDEX"},
]

# Fallback NIFTY 50 constituents (tickers only) in case the NSE API is blocked.
FALLBACK_NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "LICI", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "DMART", "SUNPHARMA",
    "BAJAJFINSV", "MARUTI", "TITAN", "ASIANPAINT", "HCLTECH", "WIPRO", "ULTRACEMCO", "NTPC",
    "ADANIENT", "TMPV", "ADANIPORTS", "HINDALCO", "POWERGRID", "ONGC", "JSWSTEEL",
    "TATASTEEL", "NESTLEIND", "TECHM", "GRASIM", "INDUSINDBK", "APOLLOHOSP", "DIVISLAB",
    "M&M", "BAJAJ-AUTO", "DRREDDY", "COALINDIA", "TATACONSUM", "CIPLA", "SBILIFE", "EICHERMOT",
    "HDFCLIFE", "BRITANNIA", "HEROMOTOCO", "BPCL",
]


def _fetch_from_nse() -> Optional[list[dict]]:
    """Try the NSE API; returns symbol dicts or None on any failure."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        resp = requests.get(NSE_CONSTITUENTS_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "symbol": item["symbol"],
                "name": item.get("companyName"),
                "isin": item.get("isin"),
                "sector": item.get("industry"),
                "instrument_type": "EQ",
            }
            for item in data.get("data", [])
        ]
    except Exception as exc:
        logger.warning("NSE constituents fetch failed (%s); using fallback list", exc)
        return None


def nifty50_symbols() -> list[dict]:
    """Return the NIFTY 50 symbol list, preferring the live NSE response."""
    live = _fetch_from_nse()
    if live:
        return live
    return [
        {
            "symbol": sym,
            "name": None,
            "isin": None,
            "sector": None,
            "instrument_type": "EQ",
        }
        for sym in FALLBACK_NIFTY50
    ]


def fyers_symbol(db_symbol: str, instrument_type: str = "EQ") -> str:
    """Convert a DB ticker to the Fyers exchange symbol used in API calls."""
    if instrument_type == "INDEX":
        # DB 'BANKNIFTY' -> 'NSE:NIFTYBANK-INDEX'; 'NIFTY 50' -> 'NSE:NIFTY50-INDEX'
        fyers_name = "NIFTYBANK" if db_symbol.upper() == "BANKNIFTY" else db_symbol.upper().replace(" ", "")
        return f"NSE:{fyers_name}-INDEX"
    return f"NSE:{db_symbol.upper()}-EQ"


def seed_symbols() -> int:
    """Upsert NIFTY 50 constituents + tracked indices into `symbols`."""
    rows = nifty50_symbols() + INDICES
    n = upsert_symbols(rows)
    logger.info("symbols seeded: %d (NIFTY 50 + indices)", n)
    return n
