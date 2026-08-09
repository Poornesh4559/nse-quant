"""Phase 4 — Backtesting foundation for nse-quant.

Package layout
--------------
data.py        DB loading (daily candles, sentiment, benchmark index) + caching
features.py    Future-safe indicator/feature engineering (no lookahead)
strategies.py  Pluggable classic strategies (frozen-dataclass params, long-only
               + long/short variants)
engine.py      No-lookahead backtest engine with Indian-market costs
metrics.py     Performance metrics from equity curve + trades
research.py    CLI: ``python -m analysis.research run|bench``
test_engine.py pytest suite proving no-lookahead + fee math (the bulletproof bit)
"""

from analysis.data import load_daily, load_sentiment, clear_cache  # noqa: F401
from analysis.engine import BacktestConfig, BacktestResult, run_backtest  # noqa: F401
from analysis.metrics import compute_metrics  # noqa: F401

__version__ = "0.1.0"
