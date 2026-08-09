"""Pluggable classic strategies.

Contract (the no-lookahead half of the backtest):
  * ``generate_signals(df) -> pd.Series`` of ints/floats, indexed like ``df``,
    value = target position from close of that row: 1 = long, -1 = short,
    0 = flat, fractions allowed (scale position size).
  * A signal at row ``t`` may ONLY use data from rows ``<= t``. All strategy
    logic here uses trailing windows (see analysis.features), so this holds by
    construction. The engine independently enforces next-open execution.
  * Warmup rows (NaN features) yield 0 (flat).

Params are frozen dataclasses so configs are hashable/comparable and typos in
parameter names fail fast. ``long_short`` flips mean-reversion/momentum
strategies from long-only (0/1) to long/short (-1/0/1) where it makes sense
for cash markets; the engine's ``allow_short`` flag gates actual shorting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from analysis.features import compute_features


class Strategy(ABC):
    """Base class: name, params, and a pure signal generator."""

    name: str = "strategy"

    def __init__(self, **kwargs: Any) -> None:
        self.params = self.Params(**kwargs)  # type: ignore[attr-defined]

    @property
    def display_name(self) -> str:
        return self.name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return target position per row, using only data <= row t."""

    def run(self, df: pd.DataFrame) -> pd.Series:
        """Ensure features exist, then generate signals."""
        if not {"rsi14", "macd"}.issubset(df.columns):
            df = compute_features(df)
        return self.generate_signals(df).reindex(df.index).fillna(0.0).astype(float)


def _hold_previous(signals: pd.Series, exit_mask: pd.Series | None = None) -> pd.Series:
    """Mean-reversion state machine: 1/-1 = explicit position change, 0 = hold.

    Non-zero signals always apply. Rows in ``exit_mask`` (explicit go-flat,
    e.g. RSI crossing overbought) override the hold and drop to 0. Rows with
    0 and not in the exit mask carry the previous position forward; before the
    first entry the position is flat.
    """
    pos = signals.where(signals != 0, pd.NA).ffill().fillna(0.0).astype(float)
    if exit_mask is not None:
        pos = pos.mask(exit_mask.fillna(False), 0.0)
    return pos


# ---------------------------------------------------------------------------
# Strategy zoo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuyHoldParams:
    long_short: bool = False  # no-op: buy-hold is long-only


class BuyHold(Strategy):
    """Always fully long from the first tradable day."""

    name = "buyhold"
    Params = BuyHoldParams

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


@dataclass(frozen=True)
class Rsi2ReversionParams:
    oversold: float = 30.0
    overbought: float = 70.0
    period: int = 14
    long_short: bool = False


class Rsi2Reversion(Strategy):
    """Mean reversion on RSI: buy oversold, exit/sell overbought.

    Long-only: rsi < oversold -> 1, rsi > overbought -> 0, else hold.
    Long/short: rsi > overbought -> -1 instead of 0.
    """

    name = "rsi2_reversion"
    Params = Rsi2ReversionParams

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        rsi = df[f"rsi{p.period}"] if f"rsi{p.period}" in df.columns else df["rsi14"]
        sig = pd.Series(0.0, index=df.index)
        sig[rsi < p.oversold] = 1.0
        if p.long_short:
            sig[rsi > p.overbought] = -1.0
        return _hold_previous(sig, exit_mask=(rsi > p.overbought) if not p.long_short else None)


@dataclass(frozen=True)
class MacdCrossParams:
    long_short: bool = False


class MacdCross(Strategy):
    """Trend following on MACD vs its signal line (12/26/9).

    Long-only (India cash): macd > signal -> 1 else 0.
    Long/short: macd < signal -> -1.
    """

    name = "macd_cross"
    Params = MacdCrossParams

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        hist = df["macd_hist"]
        sig = (hist > 0).astype(float)
        if p.long_short:
            sig = sig.where(hist > 0, -1.0)
        return sig


@dataclass(frozen=True)
class GoldenCrossParams:
    fast: int = 20
    slow: int = 50
    long_short: bool = False


class GoldenCross(Strategy):
    """Classic SMA cross: fast SMA > slow SMA -> long, else flat/short."""

    name = "golden_cross"
    Params = GoldenCrossParams

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        fast = df["close"].rolling(p.fast).mean()
        slow = df["close"].rolling(p.slow).mean()
        sig = (fast > slow).astype(float)
        if p.long_short:
            sig = sig.where(fast > slow, -1.0)
        return sig


@dataclass(frozen=True)
class BollingerReversionParams:
    period: int = 20
    n_std: float = 2.0
    long_short: bool = False


class BollingerReversion(Strategy):
    """Mean reversion at the Bollinger bands (own bands, trailing window).

    close < lower band -> 1, close > upper band -> 0 (or -1), else hold.
    """

    name = "bollinger_reversion"
    Params = BollingerReversionParams

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        sma = close.rolling(p.period).mean()
        std = close.rolling(p.period).std()
        lower = sma - p.n_std * std
        upper = sma + p.n_std * std
        sig = pd.Series(0.0, index=df.index)
        sig[close < lower] = 1.0
        if p.long_short:
            sig[close > upper] = -1.0
        return _hold_previous(sig, exit_mask=(close > upper) if not p.long_short else None)


@dataclass(frozen=True)
class Momentum121Params:
    hold_months: int = 12
    skip_months: int = 1
    long_short: bool = False


class Momentum121(Strategy):
    """12-1 momentum: 12-month return skipping the most recent month.

    Ret = close[t-21] / close[t-252] - 1 (1 month ~= 21 trading days).
    Positive -> 1, negative -> 0 (or -1 in long/short mode).
    """

    name = "momentum_12_1"
    Params = Momentum121Params

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        skip = p.skip_months * 21
        hold = p.hold_months * 21
        mom = df["close"].shift(skip) / df["close"].shift(hold) - 1.0
        sig = (mom > 0).astype(float)
        if p.long_short:
            sig = sig.where(mom > 0, -1.0)
        return sig


@dataclass(frozen=True)
class DonchianBreakoutParams:
    lookback: int = 20
    long_short: bool = False


class DonchianBreakout(Strategy):
    """Donchian channel breakout on *prior* channel (no self-inclusion).

    close > max(high, prior `lookback` days) -> 1; close < min(low, prior
    days) -> 0 (or -1). The channel uses shifted rolling extrema so today's
    own candle cannot signal its own breakout.
    """

    name = "donchian_breakout"
    Params = DonchianBreakoutParams

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        prior_high = df["high"].rolling(p.lookback).max().shift(1)
        prior_low = df["low"].rolling(p.lookback).min().shift(1)
        sig = pd.Series(0.0, index=df.index)
        sig[close > prior_high] = 1.0
        sig[close < prior_low] = -1.0 if p.long_short else 0.0
        return sig


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (
        BuyHold,
        Rsi2Reversion,
        MacdCross,
        GoldenCross,
        BollingerReversion,
        Momentum121,
        DonchianBreakout,
    )
}


def all_strategies(**overrides: dict[str, dict]) -> list[Strategy]:
    """Instantiate every strategy; ``overrides`` = {strategy_name: param_kwargs}."""
    out = []
    for name, cls in STRATEGIES.items():
        out.append(cls(**(overrides.get(name, {}) or {})))
    return out
