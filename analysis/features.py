"""Future-safe feature engineering for backtesting.

Every feature at row ``t`` is computed exclusively from data at rows ``<= t``:
all windows use ``rolling``/``ewm`` (trailing) and never ``shift(-n)``. This is
the layer where lookahead would sneak in, so the rule is enforced by
construction — a test in ``test_engine.py`` asserts the truncation invariant:
``compute_features(df).iloc[:-1]`` must equal ``compute_features(df.iloc[:-1])``.

Feature columns (all tz-naive date-indexed OHLCV frames):

  ret_1/ret_5/ret_10/ret_21  trailing returns (close.pct_change)
  rsi14                      Wilder RSI(14), ewm alpha=1/14 (dashboard style)
  macd/macd_signal/macd_hist EMA 12/26 with 9-period signal
  bb_pos                     (close - bb_lower) / (bb_upper - bb_lower), 20,2σ
  atr14                      Wilder ATR(14)
  vol_z                      (volume - vol_ma20) / vol_std20
  mom_rank                   cross-sectional 21d-return percentile rank —
                             NaN here, filled by add_cross_sectional()
  dow                        day of week (Monday=0..Sunday=6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_PERIOD, BB_STD = 20, 2.0
ATR_PERIOD = 14
VOL_PERIOD = 20
MOM_WINDOW = 21


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all single-symbol features to an OHLCV frame (idempotent).

    ``df`` must have columns open/high/low/close/volume and a DatetimeIndex.
    Warmup rows carry NaN — strategies treat NaN as flat/no-signal.
    """
    if {"open", "high", "low", "close", "volume"} - set(df.columns):
        raise ValueError(f"df needs OHLCV columns, got {list(df.columns)}")

    close = df["close"]
    out = df.copy()

    # Trailing returns — pct_change(n) uses rows [t-n, t-1] only.
    for n in (1, 5, 10, 21):
        out[f"ret_{n}"] = close.pct_change(n)

    # Wilder RSI (matches dashboard/app.py: ewm alpha=1/period, adjust=False).
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out["rsi14"] = 100 - 100 / (1 + rs)

    # MACD 12/26/9.
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Bollinger position: 0 = at lower band, 1 = at upper band.
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD).std()
    bb_lower = sma - BB_STD * std
    bb_upper = sma + BB_STD * std
    with np.errstate(divide="ignore", invalid="ignore"):
        out["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower)

    # Wilder ATR(14): TR then ewm alpha=1/period.
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    # Volume z-score over a trailing 20d window.
    vol_ma = df["volume"].rolling(VOL_PERIOD).mean()
    vol_std = df["volume"].rolling(VOL_PERIOD).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vol_z"] = (df["volume"] - vol_ma) / vol_std

    # Cross-sectional momentum rank — needs the whole universe; NaN until
    # add_cross_sectional() merges it in.
    out["mom_rank"] = np.nan

    # Day of week.
    out["dow"] = df.index.dayofweek if isinstance(df.index, pd.DatetimeIndex) else np.nan

    return out


def add_cross_sectional(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Merge cross-sectional 21d-return percentile rank into each frame.

    ``frames`` is {symbol: OHLCV df}. Returns new frames (originals untouched)
    with the ``mom_rank`` column filled: on each date, symbols are ranked by
    their trailing 21d return (pct=True -> 0..1, 1 = strongest momentum).
    Dates where a symbol has no row get NaN (not fabricated).
    """
    if not frames:
        return frames

    ret21 = {}
    for symbol, df in frames.items():
        f = compute_features(df) if "ret_21" not in df.columns else df
        ret21[symbol] = f["ret_21"].rename(symbol)
    wide = pd.concat(ret21, axis=1)
    ranks = wide.rank(axis=1, pct=True)

    out = {}
    for symbol, df in frames.items():
        f = compute_features(df) if "ret_21" not in df.columns else df
        f = f.copy()
        f["mom_rank"] = ranks[symbol]
        out[symbol] = f
    return out
