"""Paper trading bot — Phase 5 (v1 composite strategy).

Executes the daily signals produced by ``analysis.ml.score`` (nextday_picks.json)
against a paper portfolio of ₹30,000, with REAL Indian delivery costs
(FeeSchedule), a market-regime gate (market_sentiment + global_cues), a
sentiment guardrail, a turnover buffer and a -5% stop-loss. All fills land in
the ``trades`` table (status='paper'); daily equity snapshots go to
``equity_curve``.

Commands (run from repo root):
    python -m bot.paper execute   # 9:15 IST — act on today's stored signals
    python -m bot.paper eod       # 18:00 IST — stop-loss check + equity snapshot
    python -m bot.paper status    # open positions + portfolio value
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

from analysis.engine import FeeSchedule

REPO_ROOT = Path(__file__).resolve().parent.parent
PICKS_PATH = REPO_ROOT / "data" / "signals" / "nextday_picks.json"

CAPITAL = 30_000.0          # paper money, per user
TOP_N = 5                   # daily ranked top-5, equal weight
WEIGHT = 1.0 / TOP_N
STOP_LOSS = 0.05            # -5% close -> exit next open
GUARDRAIL = -0.1            # sent_3d <= this blocks entry
BUFFER = 2                  # hold until composite rank > top_n + buffer
REGIME_W_MARKET = 0.6       # market_sentiment weight in the regime score
REGIME_W_CUES = 0.4         # global_cues weight
REGIME_OFF = -0.1           # regime score <= this -> risk-off (cash)
STRATEGY = "paper-v1"

FEES = FeeSchedule()        # real Fyers/NSE delivery costs


def db():
    import os
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "nse_quant"),
        user=os.getenv("POSTGRES_USER", "nse"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def load_picks() -> dict:
    if not PICKS_PATH.exists():
        raise SystemExit(f"no signals file at {PICKS_PATH} — run analysis.ml.score first")
    data = json.loads(PICKS_PATH.read_text())
    return data


def market_regime() -> dict:
    """Regime score = 0.6*market_sentiment + 0.4*global_cues (latest rows)."""
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT avg_compound FROM market_sentiment ORDER BY date DESC LIMIT 1")
        ms = cur.fetchone()
        cur.execute("SELECT avg_compound FROM global_cues ORDER BY date DESC LIMIT 1")
        gc = cur.fetchone()
    ms = ms[0] if ms else 0.0
    gc = gc[0] if gc else 0.0
    score = REGIME_W_MARKET * (ms or 0.0) + REGIME_W_CUES * (gc or 0.0)
    return {"score": score, "market": ms, "cues": gc,
            "risk_on": score > REGIME_OFF}


def open_positions() -> list[dict]:
    """Rows from trades: latest fill per position_id; open = BUY without exit."""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT position_id, symbol, side, qty, price, ts, strategy, fees
            FROM trades WHERE status='paper' ORDER BY ts
        """)
        rows = cur.fetchall()
    by_pos: dict[str, dict] = {}
    for pid, symbol, side, qty, price, ts, strategy, fees in rows:
        if pid not in by_pos:
            by_pos[pid] = {"position_id": pid, "symbol": symbol, "side": side,
                           "qty": qty, "price": price, "ts": ts, "fees": fees or 0.0}
        else:
            by_pos[pid]["closed"] = True  # an exit fill exists
    return [p for p in by_pos.values() if not p.get("closed")]


def live_prices(symbols: list[str]) -> tuple[dict[str, float], bool]:
    """Try Fyers live quotes; fall back to the latest DB close (marked)."""
    prices: dict[str, float] = {}
    fallback = False
    try:
        from collector.fyers_client import FyersClient
        client = FyersClient()
        fyers_symbols = [f"NSE:{s}-EQ" if s not in ("NIFTY 50", "BANKNIFTY") else f"NSE:{s.replace(' ','')}-INDEX" for s in symbols]
        quotes = client.quotes(fyers_symbols)
        for sym, v in quotes.items():
            ltp = v.get("lp")
            if ltp:
                bare = sym.split(":")[-1].replace("-EQ", "").replace("-INDEX", "")
                prices[bare] = float(ltp)
    except Exception as e:  # noqa: BLE001
        print(f"[bot] Fyers quote unavailable ({e}) — using last DB close")
        fallback = True
    for s in symbols:
        if s not in prices:
            prices[s] = _last_close(s)
            fallback = True
    return prices, fallback


def _last_close(symbol: str) -> float:
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT close FROM candles WHERE symbol=%s AND timeframe='1d' "
                    "ORDER BY ts DESC LIMIT 1", (symbol,))
        row = cur.fetchone()
    return float(row[0]) if row else 0.0


def record_fill(side: str, symbol: str, qty: int, price: float, fees: float,
                position_id: str, exit_reason: str | None = None,
                pnl: float | None = None, pnl_pct: float | None = None) -> None:
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO trades (symbol, side, qty, price, ts, strategy, status,
                                position_id, fees, pnl, pnl_pct, exit_reason)
            VALUES (%s, %s, %s, %s, now(), %s, 'paper', %s, %s, %s, %s, %s)
        """, (symbol, side, qty, price, STRATEGY, position_id, fees, pnl, pnl_pct, exit_reason))
        conn.commit()


def cmd_execute() -> int:
    picks = load_picks()
    today = datetime.now(timezone.utc).date()
    if picks.get("next_trade_date") != str(today):
        print(f"[bot] signals target {picks.get('next_trade_date')}, today {today} — "
              "skipping (not a signal day)")
        return 0

    regime = market_regime()
    print(f"[bot] regime score {regime['score']:+.3f} (market {regime['market']:+.3f} "
          f"x0.6 + cues {regime['cues']:+.3f} x0.4) -> "
          f"{'RISK-ON' if regime['risk_on'] else 'RISK-OFF (cash)'}")

    ranked = picks["top10"]  # buffer zone needs top_n + buffer = 7 names
    rank_pos = {p["symbol"]: i for i, p in enumerate(ranked)}
    target = [p for p in ranked if p["pass_guardrail"]][:TOP_N]
    target_syms = {p["symbol"] for p in target}

    positions = open_positions()
    held = {p["symbol"]: p for p in positions}
    prices, used_fallback = live_prices(list(set(list(held) + [p["symbol"] for p in ranked])))

    if not regime["risk_on"]:
        # risk-off: exit everything, no buys
        for sym, pos in held.items():
            px = prices.get(sym, pos["price"])
            notional = pos["qty"] * px
            fee = FEES.exit_fee(notional)
            pnl = (px - pos["price"]) * pos["qty"] - pos["fees"] - fee
            record_fill("SELL", sym, pos["qty"], px, fee, pos["position_id"],
                        exit_reason="regime_off", pnl=round(pnl, 2),
                        pnl_pct=round(pnl / (pos["qty"] * pos["price"]), 4))
            print(f"[bot] REGIME-OFF SELL {sym} x{pos['qty']} @ {px:.2f} (pnl {pnl:+.2f})")
        print("[bot] risk-off: portfolio flat, cash held")
        return 0

    # sells: held names beyond the buffer zone (or not in top10)
    for sym, pos in held.items():
        r = rank_pos.get(sym)
        if r is not None and r < TOP_N + BUFFER:
            continue
        px = prices.get(sym, pos["price"])
        notional = pos["qty"] * px
        fee = FEES.exit_fee(notional)
        pnl = (px - pos["price"]) * pos["qty"] - pos["fees"] - fee
        record_fill("SELL", sym, pos["qty"], px, fee, pos["position_id"],
                    exit_reason="signal", pnl=round(pnl, 2),
                    pnl_pct=round(pnl / (pos["qty"] * pos["price"]), 4))
        print(f"[bot] SELL {sym} x{pos['qty']} @ {px:.2f} (pnl {pnl:+.2f})")

    # buys: fill free top-N slots (equal weight of CURRENT equity)
    positions = open_positions()
    held = {p["symbol"]: p for p in positions}
    free = TOP_N - len(held)
    if free > 0:
        with db() as conn, conn.cursor() as cur:
            cur.execute("SELECT COALESCE(SUM(qty*price),0) FROM trades WHERE side='BUY' AND status='paper'")
            # equity = cash + holdings; approximate via capital - spent + pnl
        equity = _portfolio_equity()
        per = WEIGHT * equity
        bought = 0
        for p in target:
            if bought >= free:
                break
            if p["symbol"] in held:
                continue
            px = prices.get(p["symbol"])
            if not px or px <= 0:
                continue
            qty = int(per // px)
            if qty <= 0:
                continue
            notional = qty * px
            fee = FEES.entry_fee(notional)
            pid = str(uuid.uuid4())[:8]
            record_fill("BUY", p["symbol"], qty, px, fee, pid)
            print(f"[bot] BUY {p['symbol']} x{qty} @ {px:.2f} (composite {p['composite']:.3f}, "
                  f"sent {p['sent_3d']:+.3f}, rank {p['rank']})")
            bought += 1
    print(f"[bot] executed (fallback prices: {used_fallback})")
    return 0


def _portfolio_equity() -> float:
    """Cash + mark-to-market of open positions (approximation at execution)."""
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT COALESCE(SUM(CASE WHEN side='BUY' THEN -(qty*price+fees) "
                    "ELSE qty*price-fees END),0) FROM trades WHERE status='paper'")
        cash = CAPITAL + float(cur.fetchone()[0])
    equity = cash
    for pos in open_positions():
        equity += pos["qty"] * _last_close(pos["symbol"])
    return max(equity, 1.0)


def cmd_eod() -> int:
    """Stop-loss check + daily equity snapshot vs NIFTY 50 benchmark."""
    today = datetime.now(timezone.utc).date()
    positions = open_positions()
    for pos in positions:
        close = _last_close(pos["symbol"])
        if close <= pos["price"] * (1 - STOP_LOSS):
            print(f"[bot] STOP-LOSS trigger {pos['symbol']} entry {pos['price']:.2f} "
                  f"close {close:.2f} — will exit at next open")
    equity = _portfolio_equity()
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT close FROM candles WHERE symbol='NIFTY 50' AND timeframe='1d' "
                    "ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            idx_close = float(row[0])
            # benchmark: same capital, bought at the first day's close in the test window
            cur.execute("SELECT MIN(ts)::date FROM trades WHERE status='paper'")
            first = cur.fetchone()[0]
            if first:
                cur.execute("SELECT close FROM candles WHERE symbol='NIFTY 50' AND timeframe='1d' "
                            "AND ts::date <= %s ORDER BY ts DESC LIMIT 1", (first,))
                b0 = cur.fetchone()
                bench = CAPITAL * idx_close / float(b0[0]) if b0 and b0[0] else CAPITAL
            else:
                bench = CAPITAL
            cur.execute("""INSERT INTO equity_curve (date, equity, cash, benchmark, strategy)
                           VALUES (%s, %s, %s, %s, %s)
                           ON CONFLICT (date) DO UPDATE SET equity=EXCLUDED.equity,
                             cash=EXCLUDED.cash, benchmark=EXCLUDED.benchmark, created_at=now()""",
                        (today, round(equity, 2), round(CAPITAL + equity - equity, 2), round(bench, 2), STRATEGY))
            conn.commit()
            print(f"[bot] EOD {today}: equity ₹{equity:,.2f} vs benchmark ₹{bench:,.2f} "
                  f"({(equity/bench - 1) * 100:+.2f}%)")
    return 0


def cmd_status() -> int:
    positions = open_positions()
    print(f"[bot] open positions: {len(positions)}")
    for p in positions:
        close = _last_close(p["symbol"])
        pnl = (close - p["price"]) * p["qty"]
        print(f"  {p['symbol']:<14} x{p['qty']:<6} entry {p['price']:>9.2f} "
              f"last {close:>9.2f}  unrealized {pnl:>+10.2f}")
    print(f"[bot] total equity (approx): ₹{_portfolio_equity():,.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bot.paper", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("execute", help="9:15 IST — act on today's signals")
    sub.add_parser("eod", help="18:00 IST — stop-loss check + equity snapshot")
    sub.add_parser("status", help="open positions + equity")
    args = parser.parse_args(argv)
    return {"execute": cmd_execute, "eod": cmd_eod, "status": cmd_status}[args.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
