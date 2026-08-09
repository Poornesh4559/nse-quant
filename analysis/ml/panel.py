"""Symbol-day panel for the ML layer — features + sentiment + target.

``build_panel`` is the single source of truth for what the model sees. Every
row is one (symbol, date); every feature at date ``t`` is computed exclusively
from data available at the close of ``t`` (trailing windows, as-of sentiment,
``shift`` lags — see :func:`_sentiment_block`). The leakage test in
``test_ml.py`` pins this invariant down.

Panel composition
-----------------
* price/tech features from ``analysis.features.compute_features``: ret_1/5/10/21,
  rsi14, macd(+signal/hist), bb_pos, atr14, vol_z, dow (all trailing-only).
* ``mom_rank``: cross-sectional 21d-return percentile rank across the whole
  symbol universe per date (``groupby(date).rank(pct=True)``, 1 = strongest
  momentum). NaN ret_21 -> NaN rank.
* sentiment features (the differentiator): ``sent_1d`` (as-of daily avg
  compound), ``sent_3d`` / ``sent_7d`` (trailing mean of the as-of series over
  3/7 trading days), ``sent_count_7d`` (# of news days in trailing 7 trading
  days), ``sent_lag1`` (previous trading day's as-of value), ``sent_missing``
  (1 = no news published at-or-before t). Missing sentiment -> 0 + flag. All
  windows end at ``t``; nothing from ``t+1`` can enter. News published on a
  non-trading day (the RSS pipeline publishes heavily on weekends) is carried
  forward as-of to the next trading date — never backward.
* target ``y``: 1 if next-day close-to-close return > 0, i.e.
  ``y_t = sign(close[t+1] / close[t] - 1)`` — known only after the close of
  ``t+1``. Rows with no ``t+1`` price (last row of a symbol) get NaN and are
  excluded from training (they are exactly the rows ``score.py`` predicts).

The panel includes all loaded symbols (index rows train fine); the portfolio
simulator separately excludes INDEX instruments from the tradable universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import load_daily, load_sentiment
from analysis.features import compute_features

#: every model feature, in a stable order. `mom_rank` is filled in after the
#: per-symbol blocks are concatenated; sentiment cols come from _sentiment_block.
FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_21",
    "rsi14", "macd", "macd_signal", "macd_hist", "bb_pos", "atr14", "vol_z",
    "mom_rank", "dow",
    "sent_1d", "sent_3d", "sent_7d", "sent_count_7d", "sent_lag1", "sent_missing",
]
SENTIMENT_COLS = ["sent_1d", "sent_3d", "sent_7d", "sent_count_7d", "sent_lag1", "sent_missing"]


def _sentiment_by_symbol(sentiment: pd.DataFrame) -> dict[str, pd.Series]:
    """{symbol: date-indexed daily avg_compound series} (ascending, deduped)."""
    out: dict[str, pd.Series] = {}
    if sentiment is None or sentiment.empty:
        return out
    s = sentiment.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    for symbol, g in s.groupby("symbol"):
        series = g.set_index("date")["avg_compound"].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        out[symbol] = series
    return out


def _sentiment_block(symbol: str, trading_dates: pd.DatetimeIndex,
                     sent_series: pd.Series | None) -> pd.DataFrame:
    """Causal sentiment features aligned to a symbol's trading dates.

    ``raw``  = daily avg compound reindexed to trading dates (NaN = no news).
    ``asof`` = raw.ffill() — last known value carried forward, so weekend news
               lands on the next trading date and stays until newer news.
    Rows before the first news for a symbol: all 0 with ``sent_missing`` = 1.
    """
    idx = pd.DatetimeIndex(trading_dates)
    out = pd.DataFrame(index=idx)
    if sent_series is None:
        out["sent_1d"] = 0.0
        out["sent_missing"] = 1.0
        out["sent_3d"] = 0.0
        out["sent_7d"] = 0.0
        out["sent_count_7d"] = 0.0
        out["sent_lag1"] = 0.0
        return out

    raw = sent_series.reindex(idx)
    asof = raw.ffill()
    out["sent_1d"] = asof.fillna(0.0)
    out["sent_missing"] = asof.isna().astype(float)
    out["sent_3d"] = asof.rolling(3, min_periods=1).mean().fillna(0.0)
    out["sent_7d"] = asof.rolling(7, min_periods=1).mean().fillna(0.0)
    out["sent_count_7d"] = raw.notna().astype(float).rolling(7, min_periods=1).sum().fillna(0.0)
    out["sent_lag1"] = out["sent_1d"].shift(1).fillna(0.0)
    return out


def build_panel(symbols: list[str] | None = None,
                frames: dict[str, pd.DataFrame] | None = None,
                sentiment: pd.DataFrame | None = None,
                as_of: str | pd.Timestamp | None = None,
                ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the training/prediction panel.

    Args:
        symbols:   symbols to include (None = all with 1d data).
        frames:    optional preloaded {symbol: OHLCV df} (tests inject these).
        sentiment: optional preloaded sentiment frame (tests inject these).
        as_of:     cap all data at this date (score.py uses it so it can never
                   peek at today's not-yet-complete session).

    Returns:
        (X, y, meta): X = float feature matrix (rows = RangeIndex),
        y = target 0/1 with NaN where the next-day return is not yet realized,
        meta = DataFrame[symbol, date, ret_next] aligned with X/y.
    """
    frames = frames if frames is not None else load_daily(symbols)
    sentiment = sentiment if sentiment is not None else load_sentiment()
    if as_of is not None:
        as_of = pd.Timestamp(as_of).normalize()
        frames = {s: df[df.index <= as_of] for s, df in frames.items()}
        if sentiment is not None and not sentiment.empty:
            sentiment = sentiment[pd.to_datetime(sentiment["date"]).dt.normalize() <= as_of]

    sent_by_sym = _sentiment_by_symbol(sentiment)
    base_cols = [c for c in FEATURE_COLS if c not in SENTIMENT_COLS and c != "mom_rank"]

    blocks: list[pd.DataFrame] = []
    for symbol, df in frames.items():
        if df.empty:
            continue
        f = compute_features(df)
        block = f[base_cols].copy()
        block["mom_rank"] = np.nan  # filled cross-sectionally after concat
        block["symbol"] = symbol
        block["date"] = f.index
        block = block.join(_sentiment_block(symbol, f.index, sent_by_sym.get(symbol)))

        close = df["close"]
        block["ret_next"] = close.shift(-1) / close - 1.0          # close[t+1]/close[t] - 1
        # y must stay NaN where ret_next is NaN: (NaN > 0) would be False -> 0
        block["y"] = np.where(block["ret_next"].notna(),
                              (block["ret_next"] > 0.0).astype(float), np.nan)
        blocks.append(block)

    if not blocks:
        empty = pd.DataFrame(columns=FEATURE_COLS, dtype=float)
        return empty, pd.Series(dtype=float), pd.DataFrame(columns=["symbol", "date", "ret_next"])

    panel = pd.concat(blocks, ignore_index=True)
    panel["mom_rank"] = panel.groupby("date")["ret_21"].rank(pct=True)

    meta = panel[["symbol", "date", "ret_next"]].copy()
    y = panel["y"].astype(float)
    X = panel[FEATURE_COLS].astype(float)
    return X, y, meta
