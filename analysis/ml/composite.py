"""Composite v1 strategy — the rules the ₹30k paper bot actually runs.

Ranking score per symbol-day (0..1):
    score = 0.40 * mom_rank          (21d cross-sectional momentum percentile)
          + 0.35 * ((P_up - 0.5) * 2)  (ML next-day prob, centered/scaled)
          + 0.25 * sent_score          (sentiment: sent_3d clipped [-1,1] -> [0,1])

Sentiment is a GATE (sent_3d <= -0.1 blocks entry), a TILT (score term) and a
REGIME input (market_sentiment + global_cues -> risk-on/off in the live bot).
Here, for OOS validation, the regime uses a price proxy (NIFTY 50 above its
SMA) since the sentiment tables only start today.

Validated end-to-end against buy-hold NIFTY 50 with REAL Fyers/NSE costs
(FeeSchedule) and a turnover buffer to keep churn low.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

from analysis.data import BENCHMARK_SYMBOL, load_daily
from analysis.ml.panel import build_panel
from analysis.ml.portfolio import (
    PortfolioConfig,
    run_comparison,
    simulate_buyhold,
    simulate_portfolio,
)
from analysis.ml.train import MODELS_DIR, walk_forward_train

MOM_W = 0.40   # momentum percentile weight
ML_W = 0.35    # ML probability weight
SENT_W = 0.25  # sentiment weight


def composite_score(panel_meta: pd.DataFrame) -> pd.DataFrame:
    """Build the composite ranking from panel features + OOS ML predictions.

    panel_meta: (symbol, date, mom_rank, sent_3d, ...) per symbol-day.
    Returns a DataFrame with symbol, date, p_up (=composite score), sent_3d,
    mom_rank — the exact shape simulate_portfolio consumes.
    """
    df = panel_meta.copy()
    df["mom_rank"] = df["mom_rank"].fillna(0.5)
    df["sent_3d"] = df["sent_3d"].fillna(0.0)
    # sent score: [-1, 1] -> [0, 1]
    sent_score = (df["sent_3d"].clip(-1.0, 1.0) + 1.0) / 2.0
    # ML probability: start neutral (0.5) where no model output exists
    ml_p = df.get("p_up", pd.Series(0.5, index=df.index)).fillna(0.5)
    df["p_up"] = (MOM_W * df["mom_rank"] + ML_W * ((ml_p - 0.5) * 2.0) + SENT_W * sent_score).clip(0.0, 1.0)
    return df[["symbol", "date", "p_up", "sent_3d", "mom_rank"]]


def price_regime(frames: dict[str, pd.DataFrame], sma: int = 50) -> pd.Series:
    """Risk-on series indexed by date: NIFTY 50 close above its SMA(sma)."""
    bm = frames[BENCHMARK_SYMBOL]
    close = bm["close"]
    sma_s = close.rolling(sma).mean()
    return (close > sma_s).reindex(close.index).fillna(False)


def run_validation(top_n: int = 5, buffer: int = 2) -> dict:
    """Walk-forward ML + composite ranking, OOS portfolio vs buy-hold."""
    print("Building panel (5y daily x 52 symbols)...")
    X, y, meta = build_panel()

    print("Running walk-forward LightGBM (this is the slow step)...")
    result = walk_forward_train(X, y, meta)
    oos = result["oos"].copy()  # columns: symbol, date, p_up, ... (OOS only)

    # join the composite features onto OOS predictions
    feat = pd.DataFrame({
        "symbol": meta["symbol"].values,
        "date": meta["date"].values,
        "mom_rank": X["mom_rank"].values,
        "sent_3d": X["sent_3d"].values,
    })
    oos_f = oos[["symbol", "date", "p_up"]].merge(feat, on=["symbol", "date"], how="left")

    comp = composite_score(oos_f)
    cfg = PortfolioConfig(top_n=top_n, turnover_buffer=buffer)
    frames = load_daily()
    regime = price_regime(frames, cfg.regime_sma)

    ml_only = run_comparison(frames, oos_f, cfg)                 # pure ML ranking
    ml_only["regime"] = None
    blended = run_comparison(frames, comp, cfg, regime=regime)   # composite + gate
    blended["regime"] = regime

    print(f"\n{'metric':<16}{'ML top-5':>14}{'Composite+gate':>18}{'BuyHold NIFTY':>16}")
    print("-" * 64)
    for m in ("total_return", "cagr", "sharpe", "max_drawdown"):
        a = ml_only["ml"]["metrics"][m]
        b = blended["ml"]["metrics"][m]
        c = ml_only["bench"]["metrics"][m]
        print(f"{m:<16}{a * 100:>+13.2f}%{b * 100:>+17.2f}%{c * 100:>+15.2f}%")
    print(f"{'Fees':<16}₹{ml_only['ml']['fees']:>12,.0f}₹{blended['ml']['fees']:>16,.0f}{'':>16}")
    print(f"{'Trades/stops':<16}{ml_only['ml']['n_trades']:>8}/{ml_only['ml']['n_stops']:<5}"
          f"{blended['ml']['n_trades']:>10}/{blended['ml']['n_stops']:<5}{'1/0':>14}")
    ex_ml = ml_only["excess_return"]
    ex_c = blended["excess_return"]
    print(f"\nExcess vs buy-hold:  ML-only {ex_ml * 100:+.2f}pp | composite+gate {ex_c * 100:+.2f}pp")
    verdict = "BEATS" if ex_c > 0 else "STILL LOSES TO"
    print(f"Verdict: composite v1 {verdict} buy-hold NIFTY 50 by {abs(ex_c) * 100:.2f}pp (OOS, real costs)")
    return {"ml_only": ml_only, "blended": blended}


if __name__ == "__main__":
    sys.exit(0 if run_validation() else 1)
