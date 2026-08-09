"""Performance metrics from an equity curve + trade list.

Annualization uses 252 trading days/year; the Sharpe risk-free rate is 6.5%
annual (India 10Y G-sec ballpark, tweakable). ``benchmark_compare`` compares a
strategy result against a *same-execution-model* buy-and-hold result (i.e. the
BuyHold strategy run through the same engine), so the comparison is apples to
apples — both pay the same costs and fill at next open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_RF = 0.065  # annual risk-free rate


def compute_metrics(result, rf: float = DEFAULT_RF) -> dict:
    """Metrics from a :class:`~analysis.engine.BacktestResult`."""
    eq = result.equity_curve
    initial = float(result.config.initial_capital)
    final = float(eq.iloc[-1])
    n = len(eq)

    total_return = final / initial - 1.0

    days = (eq.index[-1] - eq.index[0]).days if n > 1 else 0
    cagr = (final / initial) ** (365.25 / days) - 1.0 if days > 0 else 0.0

    rets = eq.pct_change().dropna()
    ann_vol = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    if ann_vol > 0:
        sharpe = (float(rets.mean()) * TRADING_DAYS - rf) / ann_vol
    else:
        sharpe = 0.0

    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())

    closed = [t for t in result.trades if t["closed"]]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] < 0]
    win_rate = len(wins) / len(closed) if closed else 0.0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)

    position = result.position
    exposure = float((position != 0).mean()) if len(position) else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": len(result.trades),
        "n_closed": len(closed),
        "exposure_pct": exposure,
        "total_fees": result.total_fees,
        "rf": rf,
    }


def benchmark_compare(result, buyhold_result) -> dict:
    """Excess return + information ratio vs a same-engine buy-and-hold run."""
    strat_ret = result.equity_curve.pct_change().dropna()
    bh_ret = buyhold_result.equity_curve.pct_change().dropna()
    idx = strat_ret.index.intersection(bh_ret.index)
    strat_ret, bh_ret = strat_ret.loc[idx], bh_ret.loc[idx]
    excess = strat_ret - bh_ret

    info_ratio = (
        float(excess.mean()) / float(excess.std(ddof=0)) * np.sqrt(TRADING_DAYS)
        if len(excess) > 1 and excess.std(ddof=0) > 0
        else 0.0
    )
    return {
        "excess_return": result.metrics["total_return"] - buyhold_result.metrics["total_return"],
        "excess_cagr": result.metrics["cagr"] - buyhold_result.metrics["cagr"],
        "info_ratio": info_ratio,
    }
