"""NSE trading calendar — trading days, holidays, IST time helpers.

The paper bot and score pipeline previously estimated the next trading day
with a plain weekday rule, so it would happily "trade" on NSE holidays at
stale closes, or skip the real next trading day when the estimate landed on
a holiday. This module is the single source of truth for "is this an NSE
trading day": weekend OR holiday -> not a trading day.

Holiday list: NSE publishes its official annual calendar (usually late in
the previous year). The fixed-date entries below are confident; the
lunar/Islamic dates are moon-sighting dependent and marked TENTATIVE.
UPDATE this set each year from NSE's official holiday list.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

#: NSE holidays. Best-effort; review yearly against the official NSE calendar.
#: Diwali 2026 (Nov 8) falls on a Sunday and is covered by the weekend rule;
#: Guru Nanak Jayanti (late Nov) should be added from the official list.
NSE_HOLIDAYS: set[date] = {
    # fixed-date national holidays (confident)
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 12, 25),  # Christmas
    # lunar/Islamic holidays (TENTATIVE — moon-sighting dependent)
    date(2026, 3, 20),   # Id-Ul-Fitr (estimate)
    date(2026, 5, 27),   # Id-Ul-Adha (estimate)
    date(2026, 6, 26),   # Muharram (estimate)
    date(2026, 9, 4),    # Id-e-Milad (estimate)
}


def is_trading_day(d: date) -> bool:
    """True on weekdays that are not NSE holidays."""
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def next_trading_day(d: date) -> date:
    """First trading day strictly after ``d`` (skips weekends + holidays)."""
    n = d + timedelta(days=1)
    while not is_trading_day(n):
        n += timedelta(days=1)
    return n


def ist_now() -> datetime:
    """Current time in Asia/Kolkata, regardless of the host's timezone."""
    return datetime.now(IST)


def ist_today() -> date:
    """Today's calendar date in India (the market's timezone)."""
    return ist_now().date()
