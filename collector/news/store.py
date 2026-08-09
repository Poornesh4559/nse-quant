"""Persist scored news into `news_sentiment` with URL-based dedupe upserts.

Rows are deduped by url: a unique index is ensured on first write, then
INSERT ... ON CONFLICT (url) refreshes the sentiment score/symbol while
keeping the original published_at when the incoming value is NULL.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Sequence

import psycopg2.extras

from collector.db import get_conn

logger = logging.getLogger(__name__)

_INDEX_DDL = "CREATE UNIQUE INDEX IF NOT EXISTS idx_news_url ON news_sentiment (url)"

_UPSERT_SQL = """
    INSERT INTO news_sentiment
        (symbol, source, title, url, published_at, sentiment_compound,
         sentiment_label, fetched_at)
    VALUES %s
    ON CONFLICT (url) DO UPDATE SET
        sentiment_compound = EXCLUDED.sentiment_compound,
        sentiment_label = EXCLUDED.sentiment_label,
        symbol = EXCLUDED.symbol,
        published_at = COALESCE(EXCLUDED.published_at, news_sentiment.published_at)
"""


def _ensure_unique_index() -> None:
    """Create the url unique index if it doesn't exist yet (idempotent)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_INDEX_DDL)


def upsert_news(rows: Sequence[dict]) -> int:
    """Upsert scored news rows, deduped by url. Returns the row count.

    Each dict: {symbol: str|None, source, title, url, published_at:
    datetime|None, sentiment_compound, sentiment_label, fetched_at (optional,
    defaults to now)}. On url conflict the sentiment/symbol are refreshed;
    the original published_at is kept when the incoming value is None.
    """
    if not rows:
        return 0
    _ensure_unique_index()

    now = datetime.now(timezone.utc)
    data = [
        (
            r.get("symbol"),
            r.get("source"),
            r["title"],
            r.get("url"),
            r.get("published_at"),
            r.get("sentiment_compound"),
            r.get("sentiment_label"),
            r.get("fetched_at", now),
        )
        for r in rows
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _UPSERT_SQL, data, page_size=500)
    logger.info("upserted %d news rows", len(rows))
    return len(rows)
