"""Cross-sectional top-N OOS portfolio simulation vs buy-hold NIFTY 50.

Execution model (mirrors analysis.engine conventions for Indian cash):
  * A signal is generated at the CLOSE of day ``t`` (from the model's OOS
    P_up at row ``t``) and executed at the OPEN of day ``t+1``.
  * Each day the model ranks every tradable symbol by P_up, applies the
    sentiment guardrail, and the top ``top_n`` names are the target portfolio.
  * Costs 0.1%/side (``max(cost_rate * notional, min_fee)``) apply ONLY on
    CHANGES: buys and sells. A position that stays in the top-N is held
    untouched (weights are free to drift — no rebalancing tax).
  * -5% STOP-LOSS: if a held symbol's close <= entry_price * 0.95, the whole
    position is exited at the NEXT day's open, regardless of the target list.
    A stopped symbol is never re-entered on the same open (that would be an
    immediate round trip at double cost); it may be re-picked later.
  * Sentiment GUARDRAIL: a symbol with sent_3d <= -0.1 can never be entered.
  * Equity is marked to market daily at the close. Equal-weight entries at
    ``weight`` (= 1/top_n) of pre-trade equity; fewer than top_n names passing
    the guardrail leaves cash idle.

The benchmark is buy-and-hold NIFTY 50 over the SAME trade window, with one
~0.1% round trip of index costs (0.05%/side). Honest comparison: same dates,
same capital, both marked daily at close.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.data import BENCHMARK_SYMBOL, load_daily
from analysis.engine import FeeSchedule
from analysis.ml.train import OOS_PATH, MODELS_DIR, walk_forward_train
from analysis.ml.panel import build_panel

TRADING_DAYS = 252
DEFAULT_RF = 0.065


@dataclass(frozen=True)
class PortfolioConfig:
    top_n: int = 5
    weight: float = 0.2            # 1/top_n — equal weight per entry
    stop_loss: float = 0.05        # -5% intraday close stop -> exit next open
    sent_guardrail: float = -0.1   # never enter symbols with sent_3d <= this
    fees: FeeSchedule = field(default_factory=FeeSchedule)  # REAL Indian costs
    turnover_buffer: int = 2       # hold until rank > top_n + buffer (cut churn)
    regime_sma: int = 50           # price regime proxy: NIFTY 50 above SMA50
    regime_min_above_pct: float = 0.0  # require close > sma (0 = just above)
    initial_capital: float = 1_000_000.0
    index_cost_round_trip: float = 0.001  # 0.1% total for the buy-hold benchmark
    exclude_symbols: tuple[str, ...] = (BENCHMARK_SYMBOL, "BANKNIFTY")  # INDEX instruments


def _metrics(equity: pd.Series, capital: float) -> dict:
    """total_return / cagr / max_dd / sharpe / ann_vol from a daily MTM curve."""
    if equity is None or len(equity) < 2:
        return {k: float("nan") for k in
                ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown")}
    final = float(equity.iloc[-1])
    total_return = final / capital - 1.0
    days = (equity.index[-1] - equity.index[0]).days
    cagr = (final / capital) ** (365.25 / days) - 1.0 if days > 0 else 0.0
    rets = equity.pct_change().dropna()
    ann_vol = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    sharpe = (float(rets.mean()) * TRADING_DAYS - DEFAULT_RF) / ann_vol if ann_vol > 0 else 0.0
    dd = equity / equity.cummax() - 1.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
    }


def simulate_portfolio(frames: dict[str, pd.DataFrame], preds: pd.DataFrame,
                       cfg: PortfolioConfig | None = None,
                       regime: pd.Series | None = None) -> dict:
    """Run the top-N daily-rebalance sim over the OOS window.

    ``preds`` must have columns symbol, date (signal date t), p_up, sent_3d.
    Trade date = next calendar day after the signal date. ``regime`` (optional)
    is a Series indexed by SIGNAL date with True = risk-on: on risk-off days
    everything is sold at the next open and no buys happen (market gate).
    Returns a dict with the equity curve, trade log, stop/fee counters/metrics.
    """
    cfg = cfg or PortfolioConfig()
    calendar = pd.DatetimeIndex(sorted({d for df in frames.values() for d in df.index}))
    tradable = [s for s in frames if s not in cfg.exclude_symbols]

    pred_map: dict[pd.Timestamp, pd.DataFrame] = {}
    for sig_date, g in preds.groupby("date"):
        pred_map[pd.Timestamp(sig_date)] = g
    signal_dates = set(pred_map)
    risk_on = regime if regime is not None else None

    cash = cfg.initial_capital
    qty: dict[str, float] = {}
    entry: dict[str, float] = {}
    last_close: dict[str, float] = {}
    trades: list[dict] = []
    fees_total = 0.0
    n_stops = 0
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    traded_days = 0

    def px(symbol: str, d: pd.Timestamp, col: str) -> float | None:
        df = frames[symbol]
        if d in df.index:
            v = df.at[d, col]
            return float(v) if not pd.isna(v) else None
        return None

    # first trade date = the calendar day after the first OOS signal date
    first_trade_idx = next((i for i in range(1, len(calendar)) if calendar[i - 1] in signal_dates), None)
    if first_trade_idx is None:
        return {"equity": pd.Series(dtype=float), "trades": [], "n_stops": 0,
                "n_trades": 0, "fees": 0.0, "metrics": _metrics(None, cfg.initial_capital),
                "trade_window": None}

    for i in range(first_trade_idx, len(calendar)):
        d = calendar[i]
        sig_date = calendar[i - 1]
        has_signal = sig_date in signal_dates

        stopped_today: set[str] = set()
        # ---- 0) market regime gate: risk-off sells everything at next open ----
        regime_off = bool(risk_on is not None and sig_date in risk_on.index and not bool(risk_on.loc[sig_date]))
        if regime_off:
            for s in list(qty):
                op = px(s, d, "open")
                if op is None:
                    continue
                notional = qty[s] * op
                fee = cfg.fees.exit_fee(notional)
                cash += notional - fee
                fees_total += fee
                trades.append({"symbol": s, "side": "REGIME-OFF", "date": d, "price": op,
                               "qty": qty[s], "notional": notional, "fee": fee})
                del qty[s], entry[s]

        # ---- 1) stop-loss exits at today's open (checked on yesterday's close) ----
        for s in list(qty):
            lc = last_close.get(s)
            if lc is not None and lc <= entry[s] * (1.0 - cfg.stop_loss):
                op = px(s, d, "open")
                if op is None:
                    continue  # no quote today — defer the stop to the next valid open
                notional = qty[s] * op
                fee = cfg.fees.exit_fee(notional)
                cash += notional - fee
                fees_total += fee
                n_stops += 1
                trades.append({"symbol": s, "side": "STOP", "date": d, "price": op,
                               "qty": qty[s], "notional": notional, "fee": fee})
                del qty[s], entry[s]
                stopped_today.add(s)

        # ---- 2) rebalance to the top-N target (only when we have a signal) ----
        if has_signal and not regime_off:
            sub = pred_map[sig_date]
            sub = sub[sub["symbol"].isin(tradable)]
            sub = sub[sub["sent_3d"] > cfg.sent_guardrail]           # guardrail
            sub = sub.sort_values("p_up", ascending=False)
            target = set(sub["symbol"].iloc[: cfg.top_n])
            ranked = list(sub["symbol"].iloc[: cfg.top_n + cfg.turnover_buffer])
            rank_pos = {s: i for i, s in enumerate(ranked)}

            equity_open = cash
            for s in qty:
                op = px(s, d, "open")
                equity_open += qty[s] * (op if op is not None else last_close.get(s, 0.0))
            # sells: held names that fell OUT of the buffer zone
            # (a name ranked top_n..top_n+buffer is held — no churn tax)
            for s in list(qty):
                if s in stopped_today:
                    continue
                hold_rank = rank_pos.get(s)
                if hold_rank is not None and hold_rank < cfg.top_n + cfg.turnover_buffer:
                    continue
                op = px(s, d, "open")
                if op is None:
                    continue
                notional = qty[s] * op
                fee = cfg.fees.exit_fee(notional)
                cash += notional - fee
                fees_total += fee
                trades.append({"symbol": s, "side": "SELL", "date": d, "price": op,
                               "qty": qty[s], "notional": notional, "fee": fee})
                del qty[s], entry[s]
            # buys: fill free top-N slots with target names not held
            # (stopped symbols wait until next day)
            for s in sorted(target - set(qty) - stopped_today):
                if len(qty) >= cfg.top_n:
                    break
                op = px(s, d, "open")
                if op is None or op <= 0:
                    continue
                notional = cfg.weight * equity_open
                q = notional / op
                if q <= 0:
                    continue
                fee = cfg.fees.entry_fee(notional)
                cash -= notional + fee
                fees_total += fee
                qty[s] = q
                entry[s] = op
                trades.append({"symbol": s, "side": "BUY", "date": d, "price": op,
                               "qty": q, "notional": notional, "fee": fee})

        # ---- 3) mark to market at today's close ----
        for s in list(qty):
            cl = px(s, d, "close")
            if cl is not None:
                last_close[s] = cl
        equity = cash + sum(qty[s] * last_close.get(s, 0.0) for s in qty)
        equity_curve.append((d, equity))
        traded_days += 1

    equity_series = pd.Series([e for _, e in equity_curve],
                              index=pd.DatetimeIndex([d for d, _ in equity_curve]),
                              name="equity")
    return {
        "equity": equity_series,
        "trades": trades,
        "n_stops": n_stops,
        "n_trades": len(trades),
        "fees": fees_total,
        "metrics": _metrics(equity_series, cfg.initial_capital),
        "trade_window": (equity_series.index[0], equity_series.index[-1]) if len(equity_series) else None,
    }


def simulate_buyhold(frames: dict[str, pd.DataFrame], trade_window: tuple[pd.Timestamp, pd.Timestamp],
                     cfg: PortfolioConfig | None = None) -> dict:
    """Buy-and-hold NIFTY 50 over the same trade window, ~0.1% round trip of costs."""
    cfg = cfg or PortfolioConfig()
    bm = frames[BENCHMARK_SYMBOL]
    d0, d1 = trade_window
    dates = [d for d in bm.index if d0 <= d <= d1]
    if not dates:
        return {"equity": pd.Series(dtype=float), "metrics": _metrics(None, cfg.initial_capital)}
    half = cfg.index_cost_round_trip / 2.0
    open0 = float(bm.at[dates[0], "open"])
    qty_bh = cfg.initial_capital * (1.0 - half) / open0
    closes = bm["close"].reindex(dates).ffill()
    curve = qty_bh * closes
    curve.iloc[-1] *= (1.0 - half)
    return {"equity": curve, "metrics": _metrics(curve, cfg.initial_capital)}


def run_comparison(frames: dict[str, pd.DataFrame], preds: pd.DataFrame,
                   cfg: PortfolioConfig | None = None,
                   regime: pd.Series | None = None) -> dict:
    """Full OOS comparison: top-N portfolio vs buy-hold NIFTY 50."""
    cfg = cfg or PortfolioConfig()
    ml = simulate_portfolio(frames, preds, cfg, regime=regime)
    bh = simulate_buyhold(frames, ml["trade_window"], cfg)
    ml_m, bh_m = ml["metrics"], bh["metrics"]
    return {
        "ml": ml, "bench": bh,
        "excess_return": ml_m["total_return"] - bh_m["total_return"],
        "excess_cagr": ml_m["cagr"] - bh_m["cagr"],
    }


def print_table(comp: dict) -> None:
    ml_m, bh_m = comp["ml"]["metrics"], comp["bench"]["metrics"]
    ml, bh = comp["ml"], comp["bench"]
    rows = [
        ("Total return", f"{ml_m['total_return'] * 100:8.2f}%", f"{bh_m['total_return'] * 100:8.2f}%"),
        ("CAGR", f"{ml_m['cagr'] * 100:8.2f}%", f"{bh_m['cagr'] * 100:8.2f}%"),
        ("Ann. vol", f"{ml_m['ann_vol'] * 100:8.2f}%", f"{bh_m['ann_vol'] * 100:8.2f}%"),
        ("Sharpe", f"{ml_m['sharpe']:8.2f}", f"{bh_m['sharpe']:8.2f}"),
        ("Max drawdown", f"{ml_m['max_drawdown'] * 100:8.2f}%", f"{bh_m['max_drawdown'] * 100:8.2f}%"),
        ("Fees paid", f"₹{ml['fees']:,.0f}", f"₹{0:,.0f}"),
        ("Trades / stops", f"{ml['n_trades']} / {ml['n_stops']}", "1 / 0"),
    ]
    w0 = max(len(r[0]) for r in rows) + 1
    print(f"\n{'metric':<{w0}}{'ML top-5':>14}{'BuyHold NIFTY 50':>18}")
    print("-" * (w0 + 32))
    for name, v1, v2 in rows:
        print(f"{name:<{w0}}{v1:>14}{v2:>18}")
    print("-" * (w0 + 32))
    print(f"{'Excess return (ML - BH)':<{w0}}{comp['excess_return'] * 100:>+13.2f}%{'':>18}")
    print(f"{'Excess CAGR':<{w0}}{comp['excess_cagr'] * 100:>+13.2f}%{'':>18}")


def main(argv: list[str] | None = None) -> int:
    t0 = dt.datetime.now()
    cfg = PortfolioConfig()
    oos_path = MODELS_DIR / "oos_predictions.csv"
    if oos_path.exists():
        preds = pd.read_csv(oos_path, parse_dates=["date"])
        print(f"Loaded OOS predictions: {oos_path} ({len(preds):,} rows)")
    else:
        print("No saved OOS predictions — running walk-forward first...")
        X, y, meta = build_panel()
        result = walk_forward_train(X, y, meta)
        preds = result["oos"]
        preds.to_csv(oos_path, index=False)

    frames = load_daily()
    comp = run_comparison(frames, preds, cfg)
    print(f"\nOOS window: {comp['ml']['trade_window'][0].date()} -> {comp['ml']['trade_window'][1].date()} "
          f"({len(comp['ml']['equity'])} trading days) | capital ₹{cfg.initial_capital:,.0f} | "
          f"fees: real Fyers/NSE | stop -{cfg.stop_loss * 100:.0f}% | "
          f"guardrail sent_3d <= {cfg.sent_guardrail} | buffer {cfg.turnover_buffer}")
    print_table(comp)

    # honest verdict
    ex = comp["excess_return"]
    if ex > 0:
        print(f"\nVerdict: ML top-5 beat buy-hold NIFTY 50 by {ex * 100:.2f}pp over the OOS window.")
    else:
        print(f"\nVerdict: ML top-5 LOST to buy-hold NIFTY 50 by {-ex * 100:.2f}pp over the OOS window — "
              "the ranking signal does not yet cover its costs. Research result, not marketing.")
    print(f"(took {dt.datetime.now() - t0})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
