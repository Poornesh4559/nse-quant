"""Tests for the Phase 4 ML layer — leakage, splits, portfolio math.

Run from repo root:  .venv/bin/python -m pytest analysis/ml/test_ml.py -v

(a) Leakage: panel features at date t never contain values from t+1
    (sentiment shift + truncation invariant on a known synthetic frame).
(b) Walk-forward split sanity: every training window strictly precedes its
    test window; windows are consecutive 30-day blocks; 60% fallback works.
(c) Portfolio hand-checks: exact entry/exit math with costs, hold-unchanged
    (no cost), stop-loss exit at next open overriding the target list,
    no same-day re-entry after a stop, sentiment guardrail blocking entry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.ml.panel import build_panel
from analysis.engine import FeeSchedule
from analysis.ml.portfolio import PortfolioConfig, simulate_portfolio
from analysis.ml.train import walk_forward_splits


def make_frame(opens: list[float], closes: list[float],
               start: str = "2024-01-01", n: int | None = None) -> pd.DataFrame:
    n = n or len(opens)
    idx = pd.date_range(start, periods=n, freq="B")
    opens = opens + [opens[-1]] * (n - len(opens))
    closes = closes + [closes[-1]] * (n - len(closes))
    return pd.DataFrame({
        "open": opens,
        "high": [max(o, c) for o, c in zip(opens, closes)],
        "low": [min(o, c) for o, c in zip(opens, closes)],
        "close": closes,
        "volume": [1_000_000] * n,
    }, index=idx)


def make_preds(rows: list[tuple[str, pd.Timestamp, float, float]]) -> pd.DataFrame:
    """(symbol, signal_date, p_up, sent_3d) -> preds frame."""
    return pd.DataFrame(rows, columns=["symbol", "date", "p_up", "sent_3d"])


# ---------------------------------------------------------------------------
# (a) leakage
# ---------------------------------------------------------------------------

def test_panel_has_no_future_leakage():
    """sent shift + truncation invariant on a tiny known frame."""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    opens = [100.0, 101.0, 102.0, 103.0, 104.0]
    frames = {"A": make_frame(opens, opens), "B": make_frame([100.0] * 5, [100.0] * 5)}
    sent = pd.DataFrame({
        "symbol": ["A", "A"],
        "date": [idx[3], idx[4]],           # news arrives on days 3 and 4
        "avg_compound": [0.5, 0.9],
    })

    X, y, meta = build_panel(frames=frames, sentiment=sent)
    A = meta["symbol"] == "A"

    # --- sentiment: day-3 features must not know day-4's news (0.9) ---
    r3 = meta.index[A & (meta["date"] == idx[3])][0]
    r4 = meta.index[A & (meta["date"] == idx[4])][0]
    assert X.at[r3, "sent_1d"] == pytest.approx(0.5)       # same-day only
    assert X.at[r3, "sent_3d"] == pytest.approx(0.5)
    assert X.at[r3, "sent_lag1"] == pytest.approx(0.0)     # nothing before day 3
    assert X.at[r3, "sent_count_7d"] == pytest.approx(1.0)
    assert X.at[r4, "sent_lag1"] == pytest.approx(0.5)     # shift: prev day's value
    assert X.at[r4, "sent_count_7d"] == pytest.approx(2.0)
    # no-news rows are 0 + missing flag
    r0 = meta.index[A & (meta["date"] == idx[0])][0]
    assert X.at[r0, "sent_1d"] == pytest.approx(0.0)
    assert X.at[r0, "sent_missing"] == pytest.approx(1.0)
    assert X.at[r3, "sent_missing"] == pytest.approx(0.0)

    # --- price features: day-3 ret_1 uses close[2], never close[4] ---
    assert X.at[r3, "ret_1"] == pytest.approx(103.0 / 102.0 - 1.0)
    assert X.at[r3, "ret_1"] != pytest.approx(104.0 / 103.0 - 1.0)

    # --- truncation invariant: features at t identical whether or not t+1 exists ---
    X2, y2, meta2 = build_panel(
        frames={s: df.iloc[:4] for s, df in frames.items()}, sentiment=sent)
    for r_full, r_trunc in zip(
            meta.index[A & (meta["date"] <= idx[3])],
            meta2.index[meta2["symbol"] == "A"]):
        pd.testing.assert_series_equal(X.loc[r_full], X2.loc[r_trunc],
                                       check_names=False, rtol=1e-12)

    # --- target: y at t uses close[t+1]; last row unlabeled ---
    assert y.at[r3] == 1.0                     # 104 > 103
    assert np.isnan(y.at[r4])                  # no t+1 price yet
    assert y.at[r0] == 1.0                     # 101 > 100


def test_mom_rank_is_cross_sectional():
    """21d-return rank is computed per date across symbols (groupby rank)."""
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    trend = [100 + i for i in range(n)]              # strong uptrend
    flat = [100.0] * n                               # no momentum
    frames = {"UP": make_frame(trend, trend), "FLAT": make_frame(flat, flat)}
    X, y, meta = build_panel(frames=frames)
    last = meta["date"].max()
    sel = meta["date"] == last
    rank = X.loc[sel].set_index(meta.loc[sel, "symbol"])["mom_rank"]
    assert rank["UP"] == pytest.approx(1.0)          # strongest momentum -> 1.0
    assert rank["FLAT"] == pytest.approx(0.5)        # pandas pct rank: 1/n .. 1.0 (2 symbols)
    # NaN only where ret_21 is NaN (21-day warmup per symbol), filled elsewhere
    warmup = X["ret_21"].isna()
    assert X.loc[warmup, "mom_rank"].isna().all()
    assert not X.loc[~warmup, "mom_rank"].isna().any()


# ---------------------------------------------------------------------------
# (b) walk-forward splits
# ---------------------------------------------------------------------------

def test_walk_forward_split_sanity():
    dates = pd.date_range("2021-01-01", periods=1300, freq="B")  # ~5.2 years
    splits = walk_forward_splits(dates, initial_years=3.0, fold_days=30)
    assert len(splits) > 10
    # strict date separation + block structure
    for tr, te in splits:
        assert tr[-1] < te[0]
        assert 1 <= len(te) <= 30
    # expanding window: next train == previous train + previous test
    assert np.array_equal(splits[1][0], np.concatenate([splits[0][0], splits[0][1]]))
    # initial train ~= first 3 years of business days
    assert 740 <= len(splits[0][0]) <= 800


def test_walk_forward_60pct_fallback():
    dates = pd.date_range("2024-01-01", periods=400, freq="B")  # ~1.6y < 3y
    splits = walk_forward_splits(dates, initial_years=3.0, fold_days=30)
    assert len(splits[0][0]) == int(400 * 0.6)      # 60% fallback kicks in
    assert len(splits[0][1]) == 30


# ---------------------------------------------------------------------------
# (c) portfolio math
# ---------------------------------------------------------------------------

def _port_cfg(**kw) -> PortfolioConfig:
    kw.setdefault("initial_capital", 100_000.0)
    kw.setdefault("fees", FeeSchedule(
        brokerage_fixed=1e9, brokerage_pct=0.001,  # uncapped 0.1% — legacy flat model for hand-checks
        stt_sell_pct=0.0, exchange_pct=0.0, sebi_pct=0.0,
        stamp_buy_pct=0.0, gst_pct=0.0, dp_charge=0.0, slippage_bps=0.0,
    ))
    kw.setdefault("weight", 0.5)
    kw.setdefault("top_n", 2)
    kw.setdefault("turnover_buffer", 0)  # strict top-N for hand-check tests
    kw.setdefault("exclude_symbols", ())
    return PortfolioConfig(**kw)


def test_portfolio_rebalance_math_and_guardrail():
    """Buy at open, hold unchanged (no cost), sell when dropped, guardrail."""
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    frames = {
        "A": make_frame([100, 100, 100, 100], [100, 101, 102, 103]),
        "B": make_frame([100, 100, 100, 100], [100, 100, 100, 100]),
        "C": make_frame([100, 100, 100, 100], [100, 99, 99, 99]),
        "D": make_frame([100, 100, 100, 100], [100, 100, 100, 100]),
    }
    preds = make_preds([
        ("A", idx[0], 0.90, 0.0), ("B", idx[0], 0.80, 0.0), ("C", idx[0], 0.70, 0.0),
        ("A", idx[1], 0.90, 0.0), ("B", idx[1], 0.20, 0.0), ("C", idx[1], 0.95, 0.0),
        ("A", idx[2], 0.95, 0.0), ("B", idx[2], 0.10, 0.0), ("C", idx[2], 0.90, 0.0),
        # D has the highest P_up but sent_3d = -0.5 -> guardrail must block it
        ("D", idx[0], 0.99, -0.5), ("D", idx[1], 0.99, -0.5), ("D", idx[2], 0.99, -0.5),
    ])
    res = simulate_portfolio(frames, preds, _port_cfg())

    sides = [(t["symbol"], t["side"], t["date"]) for t in res["trades"]]
    assert ("A", "BUY", idx[1]) in sides
    assert ("B", "BUY", idx[1]) in sides
    assert ("B", "SELL", idx[2]) in sides              # B dropped out of top-2
    assert ("C", "BUY", idx[2]) in sides
    assert not any(t["symbol"] == "D" for t in res["trades"])   # guardrail blocked D
    assert not any(t["date"] == idx[3] for t in res["trades"])  # A/C held: no cost day 3
    assert res["n_stops"] == 0

    # hand-computed: buys sized at weight * pre-trade equity (100k), 0.1% fees
    # d1: BUY A 500@100 (fee 50), BUY B 500@100 (fee 50) -> cash -100
    # d2: SELL B @100 (fee 50), BUY C 499.5@100 (fee 49.95) -> cash -149.95
    assert res["equity"].iloc[0] == pytest.approx(100_400.0)           # close A=101,B=100
    assert res["equity"].iloc[1] == pytest.approx(100_300.55)          # close A=102,C=99
    assert res["equity"].iloc[-1] == pytest.approx(100_800.55)         # close A=103,C=99
    assert res["fees"] == pytest.approx(50 + 50 + 50 + 49.95, rel=1e-9)
    assert res["n_trades"] == 4


def test_portfolio_stop_loss_exit():
    """Stop-loss: close <= entry*0.95 -> exit at next open even if still top-ranked."""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    frames = {
        "A": make_frame([100, 100, 100, 100, 100], [100, 101, 102, 100, 100]),
        # B crashes on day 3 (close 90 <= 95) but stays top-2 in the target
        "B": make_frame([100, 100, 100, 100, 100], [100, 101, 90, 100, 101]),
        "C": make_frame([100, 100, 100, 100, 100], [100, 99, 99, 99, 99]),
    }
    preds = make_preds([
        ("A", idx[0], 0.90, 0.0), ("B", idx[0], 0.80, 0.0), ("C", idx[0], 0.10, 0.0),
        ("A", idx[1], 0.90, 0.0), ("B", idx[1], 0.80, 0.0), ("C", idx[1], 0.10, 0.0),
        ("A", idx[2], 0.90, 0.0), ("B", idx[2], 0.85, 0.0), ("C", idx[2], 0.80, 0.0),  # B still #2
        ("A", idx[3], 0.90, 0.0), ("B", idx[3], 0.85, 0.0), ("C", idx[3], 0.80, 0.0),
    ])
    res = simulate_portfolio(frames, preds, _port_cfg())

    stop = [t for t in res["trades"] if t["side"] == "STOP"]
    assert len(stop) == 1
    assert stop[0]["symbol"] == "B"
    assert stop[0]["date"] == idx[3]                 # exited at the next open after close<=95
    assert stop[0]["price"] == pytest.approx(100.0)  # open of day 4
    assert stop[0]["qty"] == pytest.approx(500.0)    # the whole position, no same-day re-entry

    # B was #2 in the target on day 3, but was stopped — no re-buy that open
    assert not any(t["symbol"] == "B" and t["side"] == "BUY" and t["date"] == idx[3]
                   for t in res["trades"])
    # B re-enters the next day (allowed after the stop day)
    assert any(t["symbol"] == "B" and t["side"] == "BUY" and t["date"] == idx[4]
               for t in res["trades"])
    # A never sold: exactly one BUY
    assert sum(1 for t in res["trades"] if t["symbol"] == "A") == 1

    assert res["n_stops"] == 1
    assert res["n_trades"] == 4                      # BUY A, BUY B, STOP B, BUY B
    assert res["fees"] == pytest.approx(50 + 50 + 50 + 49.925, rel=1e-9)
    assert res["equity"].iloc[-1] == pytest.approx(100_299.325, rel=1e-9)
    assert len(res["equity"]) == 4                   # trade dates idx[1..4] only
