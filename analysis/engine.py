"""No-lookahead backtest engine with realistic Indian-market costs.

Execution model (sacred — tested in ``test_engine.py``):
  * A signal generated at the **close** of day ``t`` is executed at the
    **open** of day ``t+1``. The engine applies ``signals.shift(1)`` itself,
    so *any* signal series fed in gets next-open execution — callers cannot
    accidentally get close-of-signal fills.
  * A position is held until the signal flips; exits also fill at the next
    day's open. Daily resolution: the only intraday price used is the open.
  * Costs per side: ``max(cost_rate * notional, min_fee)`` — 0.1% notional or
    ₹20, whichever is larger, mirroring Indian broker brokerage+STT+slippage.
    Slippage is applied on top as a price adjustment (``slippage_bps``).
  * Equity is marked-to-market daily at the close: ``cash + qty * close``.
  * Missing days: the frame is reindexed to the union of df/signal dates.
    Prices are **never** forward-filled (no fabricated quotes); only the
    position state persists (a missing trading day simply carries the
    position, and any pending signal change fills at the next valid open).

Position sizing: each (re)entry invests ``position_pct`` of current equity
scaled by ``|signal|``, rounded down to whole shares. ``allow_short=False``
clamps negative targets to flat (India cash markets). A position still open
at the end of the series is force-valued at the last close and reported with
``closed=False`` so metrics can treat it as unrealized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeeSchedule:
    """Realistic Indian equity DELIVERY cost schedule (defaults: Fyers + NSE).

    Every knob is tweakable. Defaults reflect the real break-down for a
    delivery (swing) trade — this is the "close to real life" simulator:

      * brokerage:  min(₹20, 0.3% * notional) per order  (Fyers delivery pricing)
      * STT:        0.1% of sell-side notional only      (delivery)
      * exchange:   0.00297% of notional (NSE)           (both sides)
      * SEBI fee:   ₹10 per crore = 0.0001%              (both sides)
      * stamp:      0.015% of buy-side notional (NSE delivery)
      * GST:        18% on (brokerage + exchange + SEBI)
      * DP charge:  ₹13.5 + 18% GST per scrip per side (delivery settlement)
      * slippage:   5 bps price adjustment per fill

    Example: a ₹6,000 position round-trips at ~1.36% all-in — which is why
    naive 0.1%-cost backtests overstate returns for small accounts.
    """

    brokerage_fixed: float = 20.0      # ₹ per order
    brokerage_pct: float = 0.003       # 0.3% delivery (whichever lower)
    stt_sell_pct: float = 0.001        # 0.1% delivery sell
    exchange_pct: float = 0.0000297    # NSE 0.00297%
    sebi_pct: float = 0.000001         # ₹10/crore
    stamp_buy_pct: float = 0.00015     # 0.015% delivery buy
    gst_pct: float = 0.18
    dp_charge: float = 13.5            # ₹/scrip/side (+ GST applied)
    slippage_bps: float = 5.0          # 5 bps = 0.05% price adjust per fill

    @property
    def dp_charge_gst(self) -> float:
        return self.dp_charge * (1 + self.gst_pct)

    def _base(self, notional: float) -> tuple[float, float, float, float]:
        brk = min(self.brokerage_fixed, self.brokerage_pct * notional)
        exch = self.exchange_pct * notional
        sebi = self.sebi_pct * notional
        gst = self.gst_pct * (brk + exch + sebi)
        return brk, exch, sebi, gst

    def entry_fee(self, notional: float) -> float:
        brk, exch, sebi, gst = self._base(notional)
        stamp = self.stamp_buy_pct * notional
        return brk + exch + sebi + stamp + gst + self.dp_charge_gst

    def exit_fee(self, notional: float) -> float:
        brk, exch, sebi, gst = self._base(notional)
        stt = self.stt_sell_pct * notional
        return brk + exch + sebi + stt + gst + self.dp_charge_gst


@dataclass(frozen=True)
class BacktestConfig:
    """Execution knobs. All tweakable — defaults model Indian cash delivery."""

    initial_capital: float = 1_000_000.0
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    position_pct: float = 0.95
    allow_short: bool = False


@dataclass
class BacktestResult:
    """Outcome of one strategy x symbol backtest."""

    symbol: str
    equity_curve: pd.Series  # indexed by date, daily MTM
    position: pd.Series  # signed share count per day
    trades: list[dict]  # round trips (see _close_leg)
    config: BacktestConfig
    total_fees: float = 0.0
    metrics: dict = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])


class BacktestEngine:
    """Runs a precomputed signal Series through the execution model."""

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        config: BacktestConfig,
        symbol: str = "",
    ) -> BacktestResult:
        """Backtest ``signals`` against ``df`` (OHLCV, date-indexed)."""
        if not {"open", "close"}.issubset(df.columns):
            raise ValueError(f"df needs at least open/close, got {list(df.columns)}")

        df = df.sort_index()
        signals = (
            pd.Series(signals, index=pd.DatetimeIndex(signals.index))
            .astype(float)
            .sort_index()
        )

        # ---- align to union calendar; ffill ONLY the position state ----
        union_idx = df.index.union(signals.index).sort_values()
        prices = df["open"].reindex(union_idx)  # NaN on missing days
        closes = df["close"].reindex(union_idx)
        # signal known at close of t-1 -> effective from open of t
        planned = signals.reindex(union_idx).shift(1)
        planned = planned.ffill().fillna(0.0)  # state persists; prices don't

        cash = float(config.initial_capital)
        qty = 0.0  # signed shares; >0 long, <0 short
        cur_target = 0.0  # last executed position target
        slip = config.fees.slippage_bps / 1e4
        total_fees = 0.0
        equities: list[float] = []
        positions: list[float] = []
        trades: list[dict] = []
        leg: Optional[dict] = None  # open round trip
        last_close: Optional[float] = None

        for t, target in planned.items():
            op, cl = prices[t], closes[t]

            # ---- 1) execute any signal change at today's open ----
            if not pd.isna(op):
                if not config.allow_short and target < 0:
                    target = 0.0  # shorts clamped to flat for India cash
                if target != cur_target:
                    px = float(op)
                    buy_px, sell_px = px * (1 + slip), px * (1 - slip)
                    # desired shares for the new target (buy-side sizing)
                    if target == 0.0:
                        desired = 0.0
                    else:
                        equity_now = cash + qty * px
                        shares = math.floor(
                            equity_now * config.position_pct * abs(target) / buy_px
                        )
                        desired = shares * (1 if target > 0 else -1)

                    acted = False
                    # (a) close existing position if target flips to flat/opposite
                    if qty != 0.0 and (desired == 0.0 or (qty > 0) != (desired > 0)):
                        assert leg is not None, "open position must have a leg"
                        leg_px = sell_px if qty > 0 else buy_px
                        notional = abs(qty) * leg_px
                        fee = config.fees.exit_fee(notional)
                        cash += notional - fee if qty > 0 else -notional - fee
                        total_fees += fee
                        leg = self._close_leg(leg, trades, t, leg_px, exit_fee=fee, closed=True)
                        qty = 0.0
                        acted = True
                    # (b) open fresh position (from flat, or after the close above)
                    if desired != 0.0 and qty == 0.0:
                        side = "LONG" if desired > 0 else "SHORT"
                        fill_px = buy_px if side == "LONG" else sell_px
                        notional = abs(desired) * fill_px
                        fee = config.fees.entry_fee(notional)
                        cash += -notional - fee if side == "LONG" else notional - fee
                        total_fees += fee
                        qty = desired
                        leg = {
                            "symbol": symbol,
                            "side": side,
                            "entry_date": t,
                            "entry_price": fill_px,
                            "qty": abs(qty),
                            "entry_fee": fee,
                        }
                        acted = True
                    # (c) same-side resize (only reachable with fractional signals)
                    if not acted and desired != qty:
                        assert leg is not None, "resize requires an open leg"
                        dq = desired - qty
                        if dq > 0:
                            notional = dq * buy_px
                            fee = config.fees.entry_fee(notional)
                            cash -= notional + fee
                            leg["entry_price"] = (
                                leg["entry_price"] * leg["qty"] + buy_px * dq
                            ) / (leg["qty"] + dq)
                        else:
                            notional = (-dq) * sell_px
                            fee = config.fees.exit_fee(notional)
                            cash += notional - fee
                        total_fees += fee
                        leg["qty"] = abs(desired)
                        leg["entry_fee"] += fee
                        qty = desired
                        acted = True
                    if acted:
                        cur_target = target

            # (no valid open today: nothing happens, position state persists)

            # ---- 2) mark to market at today's close (last close if missing) ----
            if not pd.isna(cl):
                last_close = float(cl)
            px_mtm = last_close if last_close is not None else 0.0
            equities.append(cash + qty * px_mtm)
            positions.append(qty)

        # ---- force-value any still-open position at the last close ----
        if leg is not None:
            assert last_close is not None, "open leg but no close ever seen"
            leg = self._close_leg(
                leg, trades, union_idx[-1], last_close, exit_fee=0.0, closed=False
            )

        equity_curve = pd.Series(equities, index=union_idx, name="equity")
        position = pd.Series(positions, index=union_idx, name="position")

        return BacktestResult(
            symbol=symbol,
            equity_curve=equity_curve,
            position=position,
            trades=trades,
            config=config,
            total_fees=total_fees,
        )

    # ------------------------------------------------------------------
    # round-trip bookkeeping
    # ------------------------------------------------------------------

    @staticmethod
    def _close_leg(
        leg: dict, trades: list[dict], exit_date, exit_px: float,
        exit_fee: float, closed: bool,
    ) -> None:
        """Emit a round-trip trade dict; PnL reconciles with the cash curve.

        Cash effect of the round trip = gross P&L - entry_fee - exit_fee, so
        both fees are subtracted here and reported in the ``fees`` field.
        """
        entry_px, qty, entry_fee = leg["entry_price"], leg["qty"], leg["entry_fee"]
        gross = (exit_px - entry_px) * qty if leg["side"] == "LONG" else (entry_px - exit_px) * qty
        pnl = gross - entry_fee - exit_fee
        cost_basis = entry_px * qty
        trades.append(
            {
                "symbol": leg.get("symbol", ""),
                "side": leg["side"],
                "entry_date": leg["entry_date"],
                "entry_price": round(entry_px, 4),
                "exit_date": exit_date,
                "exit_price": round(exit_px, 4),
                "qty": qty,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / cost_basis, 6) if cost_basis else 0.0,
                "fees": round(entry_fee + exit_fee, 2),
                "closed": closed,
            }
        )
        return None


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
    symbol: str = "",
) -> BacktestResult:
    """One-shot convenience wrapper over :class:`BacktestEngine`.

    Computes metrics too, so research code can call this directly.
    """
    from analysis.metrics import compute_metrics  # local import avoids cycle

    result = BacktestEngine().run(df, signals, config or BacktestConfig(), symbol)
    result.metrics = compute_metrics(result)
    return result
