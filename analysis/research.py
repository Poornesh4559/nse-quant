"""Research CLI — compare every strategy against buy-and-hold on real data.

Usage (from repo root):

    .venv/bin/python -m analysis.research run [--symbols "NIFTY 50,TCS,RELIANCE"] [--min-years 1]
    .venv/bin/python -m analysis.research bench

``run``  — every strategy x every symbol; prints a comparison table and writes
           ``data/research/report_<timestamp>.md``.
``bench`` — buy-and-hold stats per symbol (same engine, same costs).

Default symbol set: NIFTY 50 (always included) + the 5 symbols with the most
1d candle rows that also span >= ``--min-years`` (default 1) of calendar time.
Explicit ``--symbols`` are always included regardless of history length.

All strategies run through the same no-lookahead engine (next-open fills,
₹20-min/0.1% fees), so the buy-hold column is directly comparable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from analysis.data import BENCHMARK_SYMBOL, clear_cache, load_daily
from analysis.engine import BacktestConfig, run_backtest
from analysis.features import add_cross_sectional
from analysis.metrics import benchmark_compare
from analysis.strategies import STRATEGIES, BuyHold, Strategy

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "data" / "research"

DEFAULT_CAPITAL = 1_000_000.0


def _table_row(strat: str, symbol: str, m: dict, bh: dict) -> list[str]:
    return [
        strat,
        symbol,
        f"{m['total_return'] * 100:7.2f}%",
        f"{m['cagr'] * 100:7.2f}%",
        f"{m['sharpe']:6.2f}",
        f"{m['max_drawdown'] * 100:7.2f}%",
        f"{m['win_rate'] * 100:5.1f}%",
        str(m["n_trades"]),
        f"{(m['total_return'] - bh['total_return']) * 100:+7.2f}%",
    ]


def print_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


def _pick_symbols(frames: dict[str, pd.DataFrame], min_years: float, explicit: list[str]) -> list[str]:
    """Default: NIFTY 50 + top-5 symbols by row count that span >= min_years.

    Explicit ``--symbols`` replace the default pick entirely.
    """
    if explicit:
        return [s for s in explicit if s in frames]
    pool = []
    for sym, df in frames.items():
        if sym == BENCHMARK_SYMBOL:
            continue
        span_days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 0
        if span_days >= 365 * min_years:
            pool.append((len(df), sym))
    pool.sort(reverse=True)
    return [BENCHMARK_SYMBOL, *[s for _, s in pool[:5]]]


def _run_symbol(symbol: str, df: pd.DataFrame, cfg: BacktestConfig,
                strategies: list[Strategy]) -> dict[str, dict]:
    """Run every strategy on one symbol. Returns {strategy_name: result_dict}."""
    out: dict[str, dict] = {}
    for strat in strategies:
        signals = strat.run(df)
        result = run_backtest(df, signals, cfg, symbol=symbol)
        entry = {"result": result, "metrics": result.metrics}
        out[strat.name] = entry
    # benchmark = BuyHold through the same engine (apples-to-apples)
    bh = out[BuyHold.name]
    for name, entry in out.items():
        entry["bench"] = benchmark_compare(entry["result"], bh["result"])
    return out


def cmd_run(args: argparse.Namespace) -> int:
    explicit = [s.strip() for s in ",".join(args.symbols or []).split(",") if s.strip()]
    frames = load_daily()
    if explicit:
        missing = [s for s in explicit if s not in frames]
        if missing:
            print(f"WARNING: no 1d candles for {missing}; skipping them", file=sys.stderr)
        explicit = [s for s in explicit if s in frames]
    if not frames:
        print("No 1d candle data in DB — is the backfill running?", file=sys.stderr)
        return 1

    chosen = _pick_symbols(frames, args.min_years, explicit)
    frames = {s: frames[s] for s in chosen}
    frames = add_cross_sectional(frames)  # fills mom_rank (cross-sectional feature)

    cfg = BacktestConfig(initial_capital=DEFAULT_CAPITAL)
    strategies = [cls() for cls in STRATEGIES.values()]

    print(f"Universe: {len(chosen)} symbols | capital ₹{DEFAULT_CAPITAL:,.0f} | "
          f"fees: real Fyers/NSE delivery | slip {cfg.fees.slippage_bps:.0f}bps | "
          f"short: {'on' if cfg.allow_short else 'off'}\n")

    rows: list[list[str]] = []
    meta: dict[str, dict] = {}  # per-symbol data-window info
    strat_meta: dict[str, dict] = {}  # per-strategy per-symbol results
    for symbol in chosen:
        df = frames[symbol]
        meta[symbol] = {
            "start": df.index[0].date(),
            "end": df.index[-1].date(),
            "n_days": len(df),
            "total_fees": 0.0,
        }
        results = _run_symbol(symbol, df, cfg, strategies)
        bh_m = results[BuyHold.name]["metrics"]
        meta[symbol]["total_fees"] = bh_m["total_fees"]
        for name, entry in results.items():
            m = entry["metrics"]
            b = entry["bench"]
            rows.append(_table_row(name, symbol, m, bh_m) + [f"{b['info_ratio']:6.2f}"])
            strat_meta.setdefault(name, {})[symbol] = {"metrics": m, "bench": b}

    header = ["strategy", "symbol", "total_ret", "cagr", "sharpe", "max_dd",
              "win_rate", "n_trades", "excess_vs_bh", "info_ratio"]
    print_table(header, rows)

    # ---- markdown report ----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"report_{stamp}.md"
    lines = [
        f"# Research report {stamp}",
        "",
        f"- Capital: ₹{DEFAULT_CAPITAL:,.0f} — fees: real Fyers/NSE delivery schedule "
        f"(brokerage min(₹20,0.3%), STT 0.1% sell, exchange 0.00297%, stamp 0.015%, "
        f"GST 18%, DP ₹13.5+GST/side) — slippage: {cfg.fees.slippage_bps:.0f} bps — "
        f"shorting: {'on' if cfg.allow_short else 'off'}",
        f"- Execution: signals at close → fills at next open; daily MTM at close.",
        "",
        "## Data window",
        "",
        "| symbol | start | end | days |",
        "|---|---|---|---|",
    ]
    for symbol, m in meta.items():
        lines.append(f"| {symbol} | {m['start']} | {m['end']} | {m['n_days']} |")
    lines += ["", "## Strategy x symbol comparison", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    lines.append("_excess_vs_bh = strategy total return minus BuyHold total return "
                 "(both through the same engine). win_rate over closed trades; "
                 "n_trades counts round trips incl. any still-open position._")
    report_path.write_text("\n".join(lines))
    print(f"\nReport written: {report_path}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    frames = load_daily()
    if not frames:
        print("No 1d candle data in DB — is the backfill running?", file=sys.stderr)
        return 1
    cfg = BacktestConfig(initial_capital=DEFAULT_CAPITAL)
    buyhold = BuyHold()
    rows = []
    for symbol in sorted(frames):
        df = frames[symbol]
        result = run_backtest(df, buyhold.run(df), cfg, symbol=symbol)
        m = result.metrics
        rows.append([
            symbol,
            str(df.index[0].date()),
            str(df.index[-1].date()),
            str(len(df)),
            f"{m['total_return'] * 100:7.2f}%",
            f"{m['cagr'] * 100:7.2f}%",
            f"{m['sharpe']:6.2f}",
            f"{m['max_drawdown'] * 100:7.2f}%",
        ])
    print_table(["symbol", "start", "end", "days", "total_ret", "cagr", "sharpe", "max_dd"], rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysis.research", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run all strategies vs buy-and-hold")
    p_run.add_argument("--symbols", nargs="+", default=[],
                       help='comma-separated symbols, e.g. "NIFTY 50,TCS,RELIANCE"')
    p_run.add_argument("--min-years", type=float, default=1.0,
                       help="min history (years) for symbols in the default top-5 pick (default 1)")
    p_run.add_argument("--no-cache", action="store_true", help="clear the load cache first")
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser("bench", help="buy-and-hold stats per symbol")
    p_bench.add_argument("--no-cache", action="store_true", help="clear the load cache first")
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    if getattr(args, "no_cache", False):
        clear_cache()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
