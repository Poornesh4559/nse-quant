"""News sentiment pipeline package (Phase 3): fetch -> map -> score -> store."""

from collector.news import config, mapper, scorer, sources, store  # noqa: F401

__all__ = ["config", "mapper", "scorer", "sources", "store"]
