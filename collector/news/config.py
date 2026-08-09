"""News pipeline configuration (Phase 3 sentiment).

Holds everything the source fetchers + mapper need:

- ``SYMBOL_ALIASES``: DB symbol -> [ticker, full company name, short names],
  used by ``mapper.map_symbols`` (whole-word match) and by ``sources.py``
  (GDELT longest-alias queries). The base map is loaded from the ``symbols``
  table at import time; curated company names / short names are merged on
  top because ``symbols.name`` is NULL for most rows.
- ``MARKET_FEEDS``: Indian market RSS feeds VERIFIED to return XML items from
  this VPS. URLs that returned 403/503 from this egress IP were dropped.
- ``REDDIT_SUBREDDITS``: Indian stock subs polled via pullpush.io.
- ``RATE_LIMITS`` + ``throttle()``: naive per-source global throttling.
- URL templates for Google News RSS, GDELT DOC 2.0 and pullpush.io.

All aliases are UPPERCASE; the mapper uppercases headlines before matching.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated company names — the `symbols` table has name=NULL for all 52 rows,
# so these fill in the "full company name" slot of each alias list.
# ---------------------------------------------------------------------------
COMPANY_NAMES: dict[str, str] = {
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports and Special Economic Zone",
    "APOLLOHOSP": "Apollo Hospitals Enterprise",
    "ASIANPAINT": "Asian Paints",
    "AXISBANK": "Axis Bank",
    "BAJAJ-AUTO": "Bajaj Auto",
    "BAJAJFINSV": "Bajaj Finserv",
    "BAJFINANCE": "Bajaj Finance",
    "BANKNIFTY": "Nifty Bank",
    "BHARTIARTL": "Bharti Airtel",
    "BPCL": "Bharat Petroleum Corporation",
    "BRITANNIA": "Britannia Industries",
    "CIPLA": "Cipla",
    "COALINDIA": "Coal India",
    "DIVISLAB": "Divi's Laboratories",
    "DMART": "Avenue Supermarts",
    "DRREDDY": "Dr. Reddy's Laboratories",
    "EICHERMOT": "Eicher Motors",
    "GRASIM": "Grasim Industries",
    "HCLTECH": "HCL Technologies",
    "HDFCBANK": "HDFC Bank",
    "HDFCLIFE": "HDFC Life Insurance",
    "HEROMOTOCO": "Hero MotoCorp",
    "HINDALCO": "Hindalco Industries",
    "HINDUNILVR": "Hindustan Unilever",
    "ICICIBANK": "ICICI Bank",
    "INDUSINDBK": "IndusInd Bank",
    "INFY": "Infosys",
    "ITC": "ITC",
    "JSWSTEEL": "JSW Steel",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LICI": "Life Insurance Corporation of India",
    "LT": "Larsen & Toubro",
    "M&M": "Mahindra & Mahindra",
    "MARUTI": "Maruti Suzuki India",
    "NESTLEIND": "Nestle India",
    "NIFTY 50": "Nifty 50",
    "NTPC": "NTPC",
    "ONGC": "Oil and Natural Gas Corporation",
    "POWERGRID": "Power Grid Corporation of India",
    "RELIANCE": "Reliance Industries",
    "SBILIFE": "SBI Life Insurance",
    "SBIN": "State Bank of India",
    "SUNPHARMA": "Sun Pharmaceutical Industries",
    "TATACONSUM": "Tata Consumer Products",
    "TATASTEEL": "Tata Steel",
    "TCS": "Tata Consultancy Services",
    "TECHM": "Tech Mahindra",
    "TITAN": "Titan Company",
    "TMPV": "Tata Motors Passenger Vehicles",
    "ULTRACEMCO": "UltraTech Cement",
    "WIPRO": "Wipro",
}

# ---------------------------------------------------------------------------
# Curated short names / common aliases (beyond ticker + company name).
# ---------------------------------------------------------------------------
EXTRA_ALIASES: dict[str, list[str]] = {
    "ADANIENT": ["ADANI ENTERPRISES", "ADANI"],
    "ADANIPORTS": ["ADANI PORTS", "ADANI PORT", "APSEZ"],
    "APOLLOHOSP": ["APOLLO HOSPITALS", "APOLLO HOSPITAL", "APOLLO"],
    "ASIANPAINT": ["ASIAN PAINTS", "ASIAN PAINT"],
    "AXISBANK": ["AXIS BANK", "AXIS"],
    "BAJAJ-AUTO": ["BAJAJ AUTO", "BAJAJ AUTOMOBILES"],
    "BAJAJFINSV": ["BAJAJ FINSERV"],
    "BAJFINANCE": ["BAJAJ FINANCE"],
    "BANKNIFTY": ["NIFTY BANK", "BANK NIFTY"],
    "BHARTIARTL": ["BHARTI AIRTEL", "AIRTEL"],
    "BPCL": ["BHARAT PETROLEUM", "BPCL"],
    "BRITANNIA": ["BRITANNIA INDUSTRIES", "BRITANNIA"],
    "CIPLA": ["CIPLA"],
    "COALINDIA": ["COAL INDIA"],
    "DIVISLAB": ["DIVI'S LABORATORIES", "DIVIS LABORATORIES", "DIVI'S LAB", "DIVIS"],
    "DMART": ["AVENUE SUPERMARTS", "DMART"],
    "DRREDDY": ["DR REDDY'S", "DR. REDDY'S", "DR REDDYS"],
    "EICHERMOT": ["EICHER MOTORS", "ROYAL ENFIELD", "EICHER"],
    "GRASIM": ["GRASIM INDUSTRIES", "GRASIM"],
    "HCLTECH": ["HCL TECHNOLOGIES", "HCL TECH", "HCL"],
    "HDFCBANK": ["HDFC BANK"],
    "HDFCLIFE": ["HDFC LIFE"],
    "HEROMOTOCO": ["HERO MOTOCORP", "HERO MOTOR", "HERO"],
    "HINDALCO": ["HINDALCO INDUSTRIES", "HINDALCO"],
    "HINDUNILVR": ["HINDUSTAN UNILEVER", "HUL", "HINDUSTAN LEVER"],
    "ICICIBANK": ["ICICI BANK"],
    "INDUSINDBK": ["INDUSIND BANK"],
    "INFY": ["INFOSYS"],
    "ITC": ["ITC LIMITED"],
    "JSWSTEEL": ["JSW STEEL"],
    "KOTAKBANK": ["KOTAK MAHINDRA BANK", "KOTAK BANK", "KOTAK"],
    "LICI": ["LIC INDIA", "LIFE INSURANCE CORPORATION", "LIC"],
    "LT": ["LARSEN & TOUBRO", "LARSEN AND TOUBRO", "L&T", "LARSEN TOUBRO"],
    "M&M": ["MAHINDRA AND MAHINDRA", "MAHINDRA & MAHINDRA", "MAHINDRA"],
    "MARUTI": ["MARUTI SUZUKI", "MARUTI"],
    "NESTLEIND": ["NESTLE INDIA", "NESTLÉ INDIA", "NESTLE"],
    "NIFTY 50": ["NIFTY", "NIFTY50"],
    "NTPC": ["NTPC LIMITED", "NTPC"],
    "ONGC": ["OIL AND NATURAL GAS", "ONGC"],
    "POWERGRID": ["POWER GRID", "POWERGRID"],
    "RELIANCE": ["RELIANCE INDUSTRIES", "RIL", "RELIANCE"],
    "SBILIFE": ["SBI LIFE"],
    "SBIN": ["STATE BANK OF INDIA", "SBI", "STATE BANK"],
    "SUNPHARMA": ["SUN PHARMA", "SUN PHARMACEUTICAL"],
    "TATACONSUM": ["TATA CONSUMER", "TATA CONSUMER PRODUCTS"],
    "TATASTEEL": ["TATA STEEL"],
    "TCS": ["TATA CONSULTANCY SERVICES", "TATA CONSULTANCY"],
    "TECHM": ["TECH MAHINDRA"],
    "TITAN": ["TITAN COMPANY", "TITAN"],
    "TMPV": ["TATA MOTORS", "TATAMOTORS", "TATA MOTORS PASSENGER"],
    "ULTRACEMCO": ["ULTRATECH CEMENT", "ULTRATECH"],
    "WIPRO": ["WIPRO"],
}


def _load_db_symbols() -> list[tuple[str, str | None]]:
    """Load (symbol, name) rows from the `symbols` table; [] if DB is down."""
    try:
        import psycopg2  # noqa: PLC0415

        from collector.config import settings  # noqa: PLC0415

        conn = psycopg2.connect(settings.db_url)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol, name FROM symbols")
                return [(str(r[0]), r[1]) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception:  # pragma: no cover - DB may be down at import time
        logger.warning("symbols table unreachable; using curated-only aliases", exc_info=True)
        return []


def _build_aliases() -> dict[str, list[str]]:
    """Merge DB symbols + curated names into the final alias map.

    Every list is [TICKER, COMPANY NAME, ...short names], all uppercase,
    deduped in order. DB rows win over the curated symbol set (in case the
    universe changes), curated names/extras fill in the gaps.
    """
    aliases: dict[str, list[str]] = {}
    for sym, name in _load_db_symbols():
        key = sym.upper()
        lst = [key]
        full = (name or COMPANY_NAMES.get(key) or "").upper()
        if full and full not in lst:
            lst.append(full)
        aliases[key] = lst
    # Ensure every curated symbol is present even if the DB query returned [].
    for key in COMPANY_NAMES:
        if key not in aliases:
            aliases[key] = [key]
    for key, extras in EXTRA_ALIASES.items():
        key = key.upper()
        lst = aliases.setdefault(key, [key])
        full = (COMPANY_NAMES.get(key) or "").upper()
        if full and full not in lst:
            lst.insert(1, full)
        for alias in extras:
            alias = str(alias).upper()
            if alias not in lst:
                lst.append(alias)
    return aliases


SYMBOL_ALIASES: dict[str, list[str]] = _build_aliases()

# ---------------------------------------------------------------------------
# Indian market RSS feeds — VERIFIED live from this VPS (2026-08-09).
# Dropped candidates that returned 403/503 from this egress IP:
#   - Moneycontrol        https://www.moneycontrol.com/rss/business-market.xml   (503)
#   - NDTV Profit         https://www.ndtv.com/business/rss/feeds                (403)
#   - Business Standard   https://www.business-standard.com/rss/markets-106.rss  (403)
#   - Financial Express   https://www.financialexpress.com/market/feed/          (403)
# ---------------------------------------------------------------------------
MARKET_FEEDS: list[dict] = [
    {"name": "Economic Times Markets", "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"},
    {"name": "Mint Markets", "url": "https://www.livemint.com/rss/markets"},
    {"name": "Investing.com India", "url": "https://in.investing.com/rss/news.rss"},
]

REDDIT_SUBREDDITS: list[str] = [
    "IndianStockMarket",
    "StockMarketIndia",
    "IndiaInvestments",
    "IndianStreetBets",
    "NSEbets",
]

# ---------------------------------------------------------------------------
# Per-source minimum seconds between requests (enforced by throttle()).
# ---------------------------------------------------------------------------
RATE_LIMITS: dict[str, float] = {
    "gdelt": 5.5,      # GDELT hard-limits 1 req / 5 s per IP
    "google": 2.0,
    "pullpush": 2.0,
    "rss": 1.0,
}

# ---------------------------------------------------------------------------
# URL templates.
# ---------------------------------------------------------------------------
GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={query}+when:1d&hl=en-IN&gl=IN&ceid=IN:en"
GDELT_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query={query}&mode=artlist&format=json"
    "&startdatetime={start}&enddatetime={end}&maxrecords=25"
)
# `after` is optional; append "&after=<epoch>" for incremental pulls.
PULLPUSH_URL = "https://api.pullpush.io/reddit/search/submission/?subreddit={sub}&size=25"


# ---------------------------------------------------------------------------
# Global throttle — simple per-key min-interval gate shared by all fetchers.
# ---------------------------------------------------------------------------
_last_call: dict[str, float] = {}


def throttle(key: str) -> None:
    """Sleep so consecutive calls with the same key are >= RATE_LIMITS[key] apart."""
    wait = RATE_LIMITS.get(key, 0.0)
    if wait <= 0:
        return
    now = time.monotonic()
    elapsed = now - _last_call.get(key, 0.0)
    if elapsed < wait:
        time.sleep(wait - elapsed)
    _last_call[key] = time.monotonic()


def reset_throttle() -> None:
    """Clear throttle state (used by tests)."""
    _last_call.clear()
