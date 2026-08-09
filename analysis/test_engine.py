"""Bulletproof tests for the backtest engine — no-lookahead + fee math.

Run from repo root:  .venv/bin/python -m pytest analysis/test_engine.py -v

What is proven here
-------------------
(a) Next-open execution: a signal on row t fills at row t+1's OPEN, never at
    row t's close (hand-computed, zero-cost config so prices are exact).
(b) Future-peek detection: a signal Series built with ``.iloc[-1]``-style
    lookahead produces a *different* result than honest execution, and the
    engine still fills at next open — the no-lookahead guarantee lives in
    (1) the engine's mandatory ``shift(1)`` execution and (2) features that
    never peek (verified by the truncation invariant below).
(c) Fee math: one round trip with 0.1% cost_rate yields the exact expected
    PnL and fees; the ₹20-per-side minimum is exercised separately.
(d) Buy-and-hold on a monotonic frame returns exactly the expected equity.
(e) Features are future-safe: ``compute_features(df).iloc[:-1]`` equals
    ``compute_features(df.iloc[:-1])`` — no row t uses row t+1.
(f) Missing days: no fabricated prices; pending fills happen at the next
    valid open and positions persist across gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.engine import BacktestConfig, BacktestEngine, FeeSchedule, run_backtest
from analysis.features import add_cross_sectional, compute_features
from analysis.metrics import compute_metrics
from analysis.strategies import STRATEGIES, Strategy

ENGINE = BacktestEngine()

# A zero-cost config makes hand-computed prices exact.
ZERO_COST = BacktestConfig(
    initial_capital=1_000_000.0,
    fees=FeeSchedule(
        brokerage_fixed=0.0, brokerage_pct=0.0, stt_sell_pct=0.0,
        exchange_pct=0.0, sebi_pct=0.0, stamp_buy_pct=0.0,
        gst_pct=0.0, dp_charge=0.0, slippage_bps=0.0,
    ),
    position_pct=1.0,
)
# The real Fyers/NSE delivery schedule (defaults).
REAL_COST = BacktestConfig(initial_capital=1_000_000.0)


def make_frame(
    opens: list[float], closes: list[float],
    start: str = "2024-01-01", step_days: int = 1,
) -> pd.DataFrame:
    """Build a date-indexed OHLCV frame (high/low = max/min of o/c, vol=1e6)."""
    idx = pd.date_range(start, periods=len(opens), freq=f"{step_days}D")
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": [min(o, c) for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1_000_000] * len(opens),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# (a) No-lookahead: signal at close of row t -> fill at open of row t+1
# ---------------------------------------------------------------------------

def test_signal_fills_at_next_day_open_not_signal_close():
    """Hand-computed: BUY signal on row 4 must fill at row 5 OPEN, never row-4 close."""
    opens = [100.0, 101.0, 102.0, 103.0, 104.0, 200.0]  # row-5 open is the tell
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]  # row-4 close = 104
    df = make_frame(opens, closes)
    sig = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], index=df.index)  # BUY at close of row 4

    res = ENGINE.run(df, sig, ZERO_COST)

    # exactly one round trip (still open -> force-valued at the end)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t["entry_date"] == df.index[5], "must fill at the open AFTER the signal day"
    assert t["entry_price"] == 200.0, "fill price is row-5 OPEN, not row-4 close (104)"
    assert t["entry_price"] != df["close"].iloc[4]

    # flat on the signal day, long from the next day
    assert res.position.iloc[4] == 0.0
    assert res.position.iloc[5] > 0.0

    # equity on signal day (row 4) is still 100% cash
    assert res.equity_curve.iloc[4] == pytest.approx(1_000_000.0)


def test_exit_also_fills_at_next_open():
    """Sell signal on row 6 -> exit at row 7 OPEN (not row-6 close)."""
    opens = [100.0] * 7 + [250.0]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    df = make_frame(opens, closes)
    # long from row 3 signal, flat from row 6 signal
    sig = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], index=df.index)

    res = ENGINE.run(df, sig, ZERO_COST)
    closed = [t for t in res.trades if t["closed"]]
    assert len(closed) == 1
    assert closed[0]["exit_date"] == df.index[7]
    assert closed[0]["exit_price"] == 250.0  # row-7 open


# ---------------------------------------------------------------------------
# (b) Future-peek: lookahead in the signal series changes results
# ---------------------------------------------------------------------------

def _walk_frame(n: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1.5, n))
    opens = np.concatenate([[100.0], closes[:-1]])  # open == prev close
    return make_frame(list(opens), list(closes))


def test_future_peek_signal_changes_results():
    """A signal Series built with tomorrow's data yields different equity —
    the engine does NOT mask lookahead, so strategies must never peek."""
    df = _walk_frame()
    honest = (df["close"] > df["close"].shift(1)).astype(float)  # known at close of t
    peeked = (df["close"].shift(-1) > df["close"]).astype(float)  # uses close[t+1]
    assert not peeked.equals(honest)

    r_honest = ENGINE.run(df, honest, ZERO_COST)
    r_peeked = ENGINE.run(df, peeked, ZERO_COST)

    # lookahead changes the outcome
    assert not np.isclose(r_honest.final_equity, r_peeked.final_equity)

    # but the engine's next-open rule still binds: first fill is the open
    # AFTER the first 1-signal, never the signal day's close
    first_sig_pos = int(np.argmax(peeked.values == 1))
    assert r_peeked.trades[0]["entry_date"] == df.index[first_sig_pos + 1]
    # entry_price is rounded to 4dp in the trade dict
    assert r_peeked.trades[0]["entry_price"] == pytest.approx(
        df["open"].iloc[first_sig_pos + 1], abs=1e-3
    )


def test_peeking_strategy_is_detectable_and_still_next_open():
    """A strategy that peeks via shift(-1) produces signals that differ from
    the honest strategy's — and the engine still fills them at next open.
    (Engine design: signals are precomputed Series; the no-lookahead
    guarantee = engine shift(1) execution + feature shift discipline.)"""

    class PeekStrategy(Strategy):
        name = "peek"
        class Params:  # noqa: N801 - test stub
            long_short: bool = False

        def generate_signals(self, df: pd.DataFrame) -> pd.Series:
            return (df["close"].shift(-1) > df["close"]).astype(float)

    class HonestStrategy(Strategy):
        name = "honest"
        class Params:  # noqa: N801 - test stub
            long_short: bool = False

        def generate_signals(self, df: pd.DataFrame) -> pd.Series:
            return (df["close"] > df["close"].shift(1)).astype(float)

    df = _walk_frame(15, seed=3)
    peek_sig = PeekStrategy().run(df)
    honest_sig = HonestStrategy().run(df)
    assert not peek_sig.equals(honest_sig)  # the peek is visible in the signals

    r_peek = ENGINE.run(df, peek_sig, ZERO_COST)
    r_honest = ENGINE.run(df, honest_sig, ZERO_COST)
    assert not np.isclose(r_peek.final_equity, r_honest.final_equity)

    first = int(np.argmax(peek_sig.values == 1))
    assert r_peek.trades[0]["entry_date"] == df.index[first + 1]


# ---------------------------------------------------------------------------
# (c) Fee math — exact, hand-computed
# ---------------------------------------------------------------------------

def test_fee_schedule_hand_computed_round_trip():
    """Real Fyers/NSE delivery schedule, exact hand-computed numbers.

    Notional ₹6,000 (typical small paper-trade position):
      entry = brk min(20, 0.3%*6000=18) 18.0 + exch 0.1782 + sebi 0.006
            + stamp 0.9 + gst 18%*(18+0.1782+0.006)=3.27316 + dp 13.5*1.18=15.93
            = 38.28736
      exit  = 18.0 + 0.1782 + 0.006 + stt 6.0 + 3.27316 + 15.93 = 43.38736
    """
    fees = FeeSchedule()
    entry = fees.entry_fee(6_000)
    exit_ = fees.exit_fee(6_000)
    assert entry == pytest.approx(38.28736, abs=1e-3)
    assert exit_ == pytest.approx(43.38736, abs=1e-3)
    # ~1.36% round trip on a ₹6k position — the small-account killer
    assert (entry + exit_) / 6_000 == pytest.approx(0.0136, abs=1e-3)

    # big notional: brokerage caps at ₹20 (min(20, 0.3%*N) = 20 when N > 6,667)
    big = fees.entry_fee(1_000_000)
    assert big == pytest.approx(20 + 29.7 + 1.0 + 150 + 0.18 * (20 + 29.7 + 1.0) + 15.93, abs=0.5)


def test_engine_round_trip_with_real_fees_reconciles():
    """Full engine round trip under REAL_COST: equity curve == hand-computed."""
    opens = [100.0] * 7 + [110.0]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 105.0, 110.0]
    df = make_frame(opens, closes)
    sig = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], index=df.index)

    res = ENGINE.run(df, sig, REAL_COST)
    closed = [t for t in res.trades if t["closed"]]
    assert len(closed) == 1
    t = closed[0]
    # slippage 5bps: buy at 100.05, sell at 109.945
    buy_px, sell_px = 100.0 * 1.0005, 110.0 * 0.9995
    assert t["qty"] == 9_495  # floor(1e6 * 0.95 / 100.05)
    assert t["entry_price"] == pytest.approx(buy_px)
    assert t["exit_price"] == pytest.approx(sell_px)
    # fees from the REAL schedule on the actual notionals
    entry_fee = FeeSchedule().entry_fee(9_495 * buy_px)
    exit_fee = FeeSchedule().exit_fee(9_495 * sell_px)
    assert t["fees"] == pytest.approx(entry_fee + exit_fee, abs=0.01)  # engine rounds to paise
    assert t["pnl"] == pytest.approx((sell_px - buy_px) * 9_495 - (entry_fee + exit_fee), abs=0.02)
    assert res.total_fees == pytest.approx(entry_fee + exit_fee, abs=0.01)
    assert res.final_equity == pytest.approx(1_000_000 + t["pnl"], abs=0.05)

    # and the trade PnL reconciles with the equity curve
    assert res.equity_curve.iloc[-1] - res.equity_curve.iloc[0] == pytest.approx(t["pnl"])


def test_small_position_fees_are_dp_dominated():
    """₹900 notional: Fyers charges the LOWER of ₹20 / 0.3% (=₹2.70) but the
    fixed DP charge + GST still make a small round trip cost ~₹38 — the real
    reason tiny positions are uneconomical."""
    opens = [100.0] * 7 + [110.0]
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 105.0, 110.0]
    df = make_frame(opens, closes)
    sig = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], index=df.index)
    cfg = BacktestConfig(initial_capital=1_000.0, position_pct=0.95)

    res = ENGINE.run(df, sig, cfg)
    t = [x for x in res.trades if x["closed"]][0]
    assert t["qty"] == 9  # floor(1000 * 0.95 / (100*1.0005))
    # slippage-adjusted notionals: 9*100.05 entry, 9*109.945 exit
    expected = FeeSchedule().entry_fee(9 * 100 * 1.0005) + FeeSchedule().exit_fee(9 * 110 * 0.9995)
    assert t["fees"] == pytest.approx(expected, abs=0.01)  # engine rounds to paise
    assert t["fees"] > 30.0  # DP+GST dominates; no such thing as a ₹0 trade
    assert t["pnl"] == pytest.approx((110 * 0.9995 - 100 * 1.0005) * 9 - expected, abs=0.02)
    assert res.final_equity == pytest.approx(1_000 + t["pnl"], abs=0.05)


def test_slippage_adjusts_fills():
    """5 bps slippage: buy at open*1.0005, sell at open*0.9995."""
    opens = [100.0] * 7 + [110.0]
    closes = [100.0] * 8
    df = make_frame(opens, closes)
    sig = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], index=df.index)
    cfg = BacktestConfig(
        initial_capital=1_000_000.0,
        fees=FeeSchedule(
            brokerage_fixed=0.0, brokerage_pct=0.0, stt_sell_pct=0.0,
            exchange_pct=0.0, sebi_pct=0.0, stamp_buy_pct=0.0,
            gst_pct=0.0, dp_charge=0.0, slippage_bps=5.0,
        ),
        position_pct=1.0,
    )

    res = ENGINE.run(df, sig, cfg)
    t = [x for x in res.trades if x["closed"]][0]
    assert t["entry_price"] == pytest.approx(100.0 * 1.0005)
    assert t["exit_price"] == pytest.approx(110.0 * 0.9995)


# ---------------------------------------------------------------------------
# (d) Buy-and-hold on a monotonic frame
# ---------------------------------------------------------------------------

def test_buyhold_monotonic_expected_equity():
    """All-1 signal on a monotonically rising frame: enter at row-1 open,
    hold to the end -> exactly 10,000 shares * 109 close = 1,090,000."""
    closes = [100.0 + i for i in range(10)]  # 100..109
    opens = [100.0] * 10
    df = make_frame(opens, closes)
    sig = pd.Series(1.0, index=df.index)

    res = run_backtest(df, sig, ZERO_COST, symbol="TEST")

    assert res.position.iloc[0] == 0.0, "no position on day 0 (no prior signal)"
    assert res.position.iloc[1] == 10_000.0  # floor(1e6 * 1.0 / 100)
    assert res.final_equity == pytest.approx(1_090_000.0)
    assert res.metrics["total_return"] == pytest.approx(0.09)
    assert res.metrics["cagr"] == pytest.approx(
        (1.09) ** (365.25 / 9) - 1.0, rel=1e-9
    )
    # equity is monotonically non-decreasing
    assert (res.equity_curve.diff().dropna() >= 0).all()
    assert res.metrics["n_trades"] == 1  # the still-open round trip
    assert res.metrics["exposure_pct"] == pytest.approx(9 / 10)


# ---------------------------------------------------------------------------
# (e) Features are future-safe (truncation invariant)
# ---------------------------------------------------------------------------

def test_features_no_future_leakage():
    """compute_features on the full frame equals compute_features on the
    truncated frame for every overlapping row — no row uses future data."""
    rng = np.random.default_rng(42)
    n = 80
    closes = 100 + np.cumsum(rng.normal(0, 1.0, n))
    df = make_frame(
        list(np.concatenate([[100.0], closes[:-1]])),
        list(closes),
    )
    df["volume"] = rng.integers(100_000, 500_000, n)

    full = compute_features(df)
    truncated = compute_features(df.iloc[:-1])
    pd.testing.assert_frame_equal(full.iloc[:-1], truncated)


def test_mom_rank_cross_sectional():
    """mom_rank: on each date the strongest 21d mover ranks highest."""
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    slow = 100 + np.arange(60)  # +1/day  (~21% / 21d)
    fast = 100 + 2.0 * np.arange(60)  # +2/day (~29% / 21d)
    frames = {
        "SLOW": make_frame(list(slow), list(slow)).reindex(idx),
        "FAST": make_frame(list(fast), list(fast)).reindex(idx),
    }
    out = add_cross_sectional(frames)
    ranks = out["FAST"]["mom_rank"].dropna()
    assert (ranks > 0.5).all()  # FAST outranks SLOW on every common date
    assert out["SLOW"]["mom_rank"].dropna().between(0, 0.5).all()


# ---------------------------------------------------------------------------
# (f) Missing days: no fabricated prices, position state persists
# ---------------------------------------------------------------------------

def test_missing_day_pending_fill_executes_at_next_valid_open():
    """Signal flips on d1 but d2 has no candle -> fill at d3 OPEN (200),
    never at d2 (missing) and never at d1's close (100)."""
    full = make_frame(
        opens=[100.0, 100.0, 100.0, 200.0, 200.0],
        closes=[100.0, 100.0, 100.0, 200.0, 200.0],
        step_days=1,
    )
    df = full.drop(full.index[2])  # d2 missing (exchange holiday / no candle)

    sig = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=full.index)  # BUY from close of d1
    res = ENGINE.run(df, sig, ZERO_COST)

    assert res.trades[0]["entry_date"] == full.index[3]  # first valid open after the flip
    assert res.trades[0]["entry_price"] == 200.0  # actual d3 open, not a ffill'd 100


def test_missing_day_position_persists():
    """Already long when a day goes missing: position carries through the gap
    (state ffill) and MTM uses the last known close, not a fabricated price."""
    full = make_frame(
        opens=[100.0, 100.0, 100.0, 100.0, 100.0],
        closes=[100.0, 105.0, 110.0, 115.0, 120.0],
        step_days=1,
    )
    df = full.drop(full.index[2])
    sig = pd.Series([1.0] * 5, index=full.index)

    res = ENGINE.run(df, sig, ZERO_COST)
    # position is nonzero on every valid day and across the gap
    assert (res.position.iloc[1:] > 0).all()
    # equity on the missing day equals last known close valuation
    # (d1 close 105 -> same 105 carried to d2, then d3 close 115)
    assert res.equity_curve.iloc[2] == pytest.approx(res.equity_curve.iloc[1])
    assert res.equity_curve.iloc[3] > res.equity_curve.iloc[2]


# ---------------------------------------------------------------------------
# Smoke: every shipped strategy runs on realistic data without error
# ---------------------------------------------------------------------------

def test_all_strategies_smoke():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0.05, 1.2, 300))
    df = make_frame(
        list(np.concatenate([[100.0], closes[:-1]])),
        list(closes),
    )
    df["volume"] = rng.integers(100_000, 500_000, 300)
    df = compute_features(df)

    for name, cls in STRATEGIES.items():
        strat = cls()
        sig = strat.run(df)
        assert sig.index.equals(df.index)
        assert set(sig.dropna().unique()) <= {-1.0, 0.0, 1.0}
        res = ENGINE.run(df, sig, REAL_COST, symbol="SMOKE")
        assert res.final_equity > 0
        # mean-reversion strategies may never trigger on this particular walk —
        # but if a signal ever fires, it must produce at least one round trip
        if sig.any():
            assert len(res.trades) >= 1
        metrics = compute_metrics(res)
        assert -1.0 < metrics["total_return"] < 5.0
        assert metrics["max_drawdown"] <= 0.0
