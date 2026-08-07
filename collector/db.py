"""Thin database helpers over psycopg2 for the TimescaleDB `nse_quant` DB.

Keeps the SQL close to the schema (storage/init/01_schema.sql) and handles
the connection lifecycle + batched upserts used by the collector.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Sequence

import psycopg2
import psycopg2.extras

from collector.config import settings

logger = logging.getLogger(__name__)

# Fyers symbol -> DB symbol convention. Fyers uses 'NSE:TCS-EQ'; we store the
# bare ticker (TCS) in `symbols.symbol` and keep index names human-readable.
def fyers_to_db_symbol(fyers_symbol: str) -> str:
    """Convert an exchange symbol like 'NSE:TCS-EQ' or 'NSE:NIFTY50-INDEX' to a bare ticker."""
    return fyers_symbol.split(":", 1)[1].removesuffix("-EQ")


@contextmanager
def get_conn():
    """Context manager yielding a psycopg2 connection."""
    conn = psycopg2.connect(settings.db_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_symbols(symbols: Sequence[dict]) -> int:
    """Insert or update rows in `symbols`. Each dict: symbol, name, instrument_type, sector, isin."""
    if not symbols:
        return 0
    sql = """
        INSERT INTO symbols (symbol, name, isin, sector, instrument_type)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET
            name = EXCLUDED.name,
            isin = EXCLUDED.isin,
            sector = EXCLUDED.sector,
            instrument_type = EXCLUDED.instrument_type,
            updated_at = now()
    """
    rows = [
        (
            s["symbol"],
            s.get("name"),
            s.get("isin"),
            s.get("sector"),
            s.get("instrument_type", "EQ"),
        )
        for s in symbols
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    logger.info("upserted %d symbols", len(rows))
    return len(rows)


def list_symbols() -> list[dict]:
    """Return all symbols from the DB as dicts (symbol, name, instrument_type)."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT symbol, name, instrument_type FROM symbols ORDER BY symbol")
            return list(cur.fetchall())


def upsert_candles(symbol: str, timeframe: str, candles: Sequence[dict]) -> int:
    """Upsert OHLCV rows into the `candles` hypertable.

    Each candle dict: ts (datetime), open, high, low, close, volume.
    """
    if not candles:
        return 0
    sql = """
        INSERT INTO candles (symbol, timeframe, ts, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (symbol, timeframe, ts) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
    """
    rows = [
        (
            symbol,
            timeframe,
            c["ts"],
            c.get("open"),
            c.get("high"),
            c.get("low"),
            c.get("close"),
            c.get("volume"),
        )
        for c in candles
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    logger.info("upserted %d candles for %s %s", len(rows), symbol, timeframe)
    return len(rows)


def latest_candle_ts(symbol: str, timeframe: str):
    """Return the most recent candle timestamp for a symbol+timeframe, or None."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT max(ts) FROM candles WHERE symbol = %s AND timeframe = %s",
                (symbol, timeframe),
            )
            row = cur.fetchone()
            return row[0] if row else None
