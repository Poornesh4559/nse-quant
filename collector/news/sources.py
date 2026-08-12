"""News source fetchers (Phase 3 sentiment).

Every public function returns ``list[dict]`` with the uniform shape::

    {"title": str, "url": str, "published": datetime | None, "source": str}

Fetchers NEVER raise: every request is rate-limited, wrapped in try/except
and returns ``[]`` on any failure so the pipeline keeps running regardless
of a flaky feed, a blocked IP or a dead endpoint.

Sources:
- ``fetch_google_news``  — Google News RSS search (India edition, last 24h)
- ``fetch_market_feeds`` — Indian market RSS feeds (see config.MARKET_FEEDS)
- ``fetch_reddit``       — pullpush.io mirror of Reddit submissions
- ``fetch_gdelt``        — GDELT DOC 2.0 article list API
"""

from __future__ import annotations

import email.utils
import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
import xml.etree.ElementTree as ET

from collector.news import config

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, application/json;q=0.9, */*;q=0.8",
}
TIMEOUT = 20

# GDELT rate-limit responses: HTTP 429 or this body message.
_GDELT_RATE_LIMITED = "5 seconds"


def _parse_datetime(value: Any) -> datetime | None:
    """Parse RSS/JSON timestamps into tz-aware datetimes; None on failure.

    Handles RFC 822 (ET / Google News), ISO-8601 without tz (Investing.com)
    and common '%Y-%m-%d %H:%M:%S' strings.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(text)
        if parsed is not None:
            # tz-less timestamps from INDIAN feeds are IST, not UTC — treating
            # them as UTC shifted evening articles to the wrong trading day
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        except ValueError:
            continue
    return None


def _rss_items(xml_text: str) -> list[dict]:
    """Extract {title, url, published} from an RSS/Atom feed string."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("RSS parse failed (%d chars)", len(xml_text))
        return []
    items: list[dict] = []
    for node in root.findall(".//item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        if not title and not link:
            continue
        items.append(
            {
                "title": title,
                "url": link,
                "published": _parse_datetime(node.findtext("pubDate")),
            }
        )
    return items


def _get(url: str, key: str, **kwargs) -> requests.Response | None:
    """Rate-limited GET that never raises — returns None only on network errors.

    Non-2xx HTTP responses are returned as-is: RSS/JSON parsers downstream
    turn 403/503 bodies into empty results, and fetch_gdelt inspects the
    status/text itself to implement its 429 retry.
    """
    config.throttle(key)
    try:
        return requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kwargs)
    except Exception:
        logger.warning("GET failed for %s (%s): %s", key, url[:120], _exc())
        return None


def _exc() -> str:
    import traceback  # noqa: PLC0415

    return traceback.format_exc(limit=1).strip().splitlines()[-1]


# ---------------------------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------------------------
def fetch_google_news(symbol: str) -> list[dict]:
    """Search Google News (India edition, last 24h) for '<ticker> stock'.

    `symbol` is the bare DB ticker, e.g. 'TCS' -> 'TCS stock'.
    """
    # quote() so symbols with special chars (e.g. M&M) don't break the URL;
    # safe="+" keeps the space->+ encoding Google expects.
    query = quote(f"{symbol} stock".replace(" ", "+"), safe="+")
    url = config.GOOGLE_NEWS_URL.format(query=query)
    resp = _get(url, "google")
    if resp is None:
        return []
    out: list[dict] = []
    for item in _rss_items(resp.text):
        out.append(
            {
                "title": item["title"],
                "url": item["url"],
                "published": item["published"],
                "source": "google_news",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Global cues / geopolitics
# ---------------------------------------------------------------------------
# One Google News query per theme; each article carries a `theme` key so the
# pipeline can bucket aggregates (crude / fed / us_markets / usd_inr / geo).
GLOBAL_CUE_QUERIES: dict[str, str] = {
    "crude": "crude oil price markets",
    "fed": "US Federal Reserve rate decision",
    "us_markets": "US stock market overnight global markets",
    "usd_inr": "USD INR rupee dollar",
    "geo": "geopolitical risk markets war",
}


def fetch_global_cues() -> list[dict]:
    """Search Google News (last 24h) for each global-cue theme.

    Returns uniform article dicts + a ``theme`` key for bucketing. Never
    raises — any failing theme just yields nothing.
    """
    out: list[dict] = []
    for theme, q in GLOBAL_CUE_QUERIES.items():
        query = quote(q.replace(" ", "+"), safe="+")
        url = config.GOOGLE_NEWS_URL.format(query=query)
        resp = _get(url, "google")
        if resp is None:
            continue
        for item in _rss_items(resp.text):
            out.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "published": item["published"],
                    "source": "global_cues",
                    "theme": theme,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Market RSS feeds
# ---------------------------------------------------------------------------
def fetch_market_feeds() -> list[dict]:
    """Fetch every feed in config.MARKET_FEEDS; source = feed display name."""
    out: list[dict] = []
    for feed in config.MARKET_FEEDS:
        resp = _get(feed["url"], "rss")
        if resp is None:
            continue
        for item in _rss_items(resp.text):
            out.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "published": item["published"],
                    "source": feed["name"],
                }
            )
    return out


# ---------------------------------------------------------------------------
# Reddit via pullpush.io
# ---------------------------------------------------------------------------
def fetch_reddit(subreddit: str, after: float | None = None) -> list[dict]:
    """Fetch recent submissions for a subreddit via pullpush.io.

    `after`: optional epoch timestamp for incremental pulls. Title and
    selftext are concatenated into `title`; url is the canonical reddit
    permalink; source is 'reddit:<subreddit>'.
    """
    suffix = f"&after={int(after)}" if after is not None else ""
    url = config.PULLPUSH_URL.format(sub=quote(subreddit)) + suffix
    resp = _get(url, "pullpush")
    if resp is None:
        return []
    if resp.status_code == 429:
        # pullpush throttles per-IP; back off briefly and try once more.
        logger.warning("pullpush rate-limited for r/%s; retrying once in 5s", subreddit)
        time_sleep(5)
        resp = _get(url, "pullpush")
        if resp is None or resp.status_code == 429:
            logger.warning("pullpush still rate-limited for r/%s; giving up", subreddit)
            return []
    try:
        payload = resp.json()
        submissions = payload.get("data") or []
    except (ValueError, AttributeError):
        logger.warning("pullpush returned non-JSON for r/%s", subreddit)
        return []
    out: list[dict] = []
    for s in submissions:
        if not isinstance(s, dict):
            continue
        title = (s.get("title") or "").strip()
        selftext = (s.get("selftext") or "").strip()
        if not title and not selftext:
            continue
        combined = f"{title} {selftext}".strip()
        permalink = s.get("permalink")
        url = f"https://www.reddit.com{permalink}" if permalink else (s.get("url") or "")
        created = s.get("created_utc")
        published = None
        if isinstance(created, (int, float)) and created > 0:
            published = datetime.fromtimestamp(float(created), tz=timezone.utc)
        out.append(
            {
                "title": combined,
                "url": url,
                "published": published,
                "source": f"reddit:{subreddit}",
            }
        )
    return out


# ---------------------------------------------------------------------------
# GDELT DOC 2.0
# ---------------------------------------------------------------------------
def _gdelt_query(symbol: str) -> str:
    """Build the GDELT query: longest alias (quoted) OR ticker, in parens.

    GDELT rejects bare OR'd terms ('Queries containing OR'd terms must be
    surrounded by ()'), so the result is e.g. '("TATA CONSULTANCY SERVICES" OR TCS)'.
    """
    aliases = config.SYMBOL_ALIASES.get(symbol.upper(), [symbol.upper()])
    ticker = aliases[0]
    longest = max((a for a in aliases if a != ticker), key=len, default=ticker)
    return f'("{longest}" OR {ticker})'


def fetch_gdelt(symbol: str, start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Fetch articles mentioning a symbol from GDELT in [start_dt, end_dt].

    Retries once after 6 s when GDELT rate-limits us (HTTP 429 or the
    '5 seconds' body message) — the API allows 1 request per 5 s per IP and
    shared cloud egress IPs often hit the limit.
    """
    query = _gdelt_query(symbol)
    url = config.GDELT_URL.format(
        query=quote(query, safe=""),
        start=start_dt.strftime("%Y%m%d%H%M%S"),
        end=end_dt.strftime("%Y%m%d%H%M%S"),
    )
    resp = _get(url, "gdelt")
    if resp is None:
        return []
    if resp.status_code == 429 or _GDELT_RATE_LIMITED in resp.text:
        logger.warning("GDELT rate-limited for %s; retrying once in 6s", symbol)
        time_sleep(6)
        resp = _get(url, "gdelt")
        if resp is None:
            return []
        if resp.status_code == 429 or _GDELT_RATE_LIMITED in resp.text:
            logger.warning("GDELT still rate-limited for %s; giving up", symbol)
            return []
    try:
        payload = resp.json()
    except ValueError:
        logger.warning("GDELT non-JSON response for %s", symbol)
        return []
    if not isinstance(payload, dict):
        logger.warning("GDELT unexpected payload shape for %s", symbol)
        return []
    out: list[dict] = []
    for art in payload.get("articles") or []:
        if not isinstance(art, dict):
            continue
        title = (art.get("title") or "").strip()
        url = art.get("url") or ""
        if not title and not url:
            continue
        out.append(
            {
                "title": title,
                "url": url,
                "published": _parse_seendate(art.get("seendate")),
                "source": "gdelt",
            }
        )
    return out


def _parse_seendate(seendate: Any) -> datetime | None:
    """GDELT 'seendate' is YYYYMMDDHHMMSS, usually rendered as 20260802T104500Z."""
    if not seendate:
        return None
    text = str(seendate).strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return _parse_datetime(seendate)


def time_sleep(seconds: float) -> None:
    """Thin wrapper so tests can monkeypatch the retry sleep."""
    import time  # noqa: PLC0415

    time.sleep(seconds)


__all__ = [
    "fetch_google_news",
    "fetch_market_feeds",
    "fetch_reddit",
    "fetch_gdelt",
]
