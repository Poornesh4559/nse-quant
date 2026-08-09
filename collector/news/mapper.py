"""Map free-text news headlines to DB symbols via whole-word alias matching.

A headline like "Tata Consultancy Services shares crash" is matched to the
`TCS` ticker by looking up every alias (and bare ticker) from
`collector.news.config.SYMBOL_ALIASES` as a whole word in the uppercased text.
"""

from __future__ import annotations

import re

from collector.news import config

# Maximum number of symbols to return for a single headline — news about a
# stock usually names 1-3 companies.
MAX_SYMBOLS = 3

# Whole-word boundary: the alias must not be flanked by another word char.
# Hyphens/spaces/& count as boundaries, so 'BAJAJ-AUTO' matches 'Bajaj-Auto'
# but not 'BAJAJ-AUTOMOTIVE'.
_BOUNDARY = r"(?<![A-Z0-9]){alias}(?![A-Z0-9])"

# Cache: uppercase alias -> DB symbol, built once from config.SYMBOL_ALIASES.
_ALIAS_TO_SYMBOL: dict[str, str] | None = None


def _alias_index() -> dict[str, str]:
    """Build (and cache) the uppercase alias -> DB symbol lookup.

    Each symbol's bare ticker is always included, plus every configured alias.
    """
    global _ALIAS_TO_SYMBOL
    if _ALIAS_TO_SYMBOL is None:
        index: dict[str, str] = {}
        for symbol, aliases in config.SYMBOL_ALIASES.items():
            for alias in [symbol, *aliases]:
                alias = str(alias).strip().upper()
                if alias:
                    index[alias] = symbol
        _ALIAS_TO_SYMBOL = index
    return _ALIAS_TO_SYMBOL


def map_symbols(text: str) -> list[str]:
    """Return DB symbols mentioned in `text`, deduped, max MAX_SYMBOLS.

    Matching is case-insensitive whole-word. Returns [] when nothing matches —
    the caller then stores symbol=NULL (market-wide news).
    """
    if not text:
        return []
    upper = text.upper()
    found: list[str] = []
    for alias, symbol in _alias_index().items():
        if re.search(_BOUNDARY.format(alias=re.escape(alias)), upper):
            if symbol not in found:
                found.append(symbol)
                if len(found) >= MAX_SYMBOLS:
                    break
    return found
