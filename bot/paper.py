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
from datetime import timedelta
from pathlib import Path

import psycopg2

from analysis.engine import FeeSchedule
from bot.llm_gate import LLM_MIN_RATING, build_context, llm_rate, log_decision
from analysis.calendar import is_trading_day, ist_today

REPO_ROOT = Path(__file__).resolve().parent.parent
PICKS_PATH = REPO_ROOT / "data" / "signals" / "nextday_picks.json"

CAPITAL = 100_000.0         # paper money (upgraded ₹30k → ₹1L)
TOP_N = 5                   # daily ranked top-5, equal weight
WEIGHT = 1.0 / TOP_N
STOP_LOSS = 0.05            # -5% close -> exit next open (hard floor, always)
GUARDRAIL = -0.1            # sent_3d <= this blocks entry
BUFFER = 2                  # hold until composite rank > top_n + buffer
REGIME_W_MARKET = 0.6       # market_sentiment weight in the regime score
REGIME_W_CUES = 0.25        # global-cues news weight
REGIME_W_ASIA = 0.15        # Asian-market early-cues weight (open before India)
REGIME_OFF = -0.1           # regime score <= this -> risk-off (cash)
STRATEGY = "paper-v1"

# --- smart-exit rules (v2) ---
PROFIT_TAKE_PCT = 0.05      # good profit margin: >= +5% unlocks trend-break exits
HOLD_PNL_BUFFER = 0.005     # hold when |pnl| < round-trip fees + 0.5% (no churn)
TREND_SMA = 20              # "trend good" = close above SMA20
SENT_HOLD = 0.0             # "sentiment good" = sent_3d > 0

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
    """Regime score = 0.6*market_sentiment + 0.25*cues news + 0.15*asia cues.

    Asia component: average % move of Nikkei/HSI/Shanghai/STI (they open
    BEFORE India), clipped to ±2% and scaled to [-1, 1] (/2).

    Freshness: missing OR stale rows (pipeline stalled > 4 days ago — a
    weekend + a holiday) force risk-OFF. Previously empty tables scored 0 and
    "passed" the gate, so a dead cues job silently flipped the regime to
    risk-ON with no data behind it.
    """
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT avg_compound, date FROM market_sentiment ORDER BY date DESC LIMIT 1")
        ms_row = cur.fetchone()
        cur.execute("SELECT avg_compound, themes, date FROM global_cues ORDER BY date DESC LIMIT 1")
        gc_row = cur.fetchone()
    ms = ms_row[0] if ms_row else 0.0
    gc = gc_row[0] if gc_row else 0.0
    asia_avg = 0.0
    if gc_row and gc_row[1]:
        import json as _json
        try:
            themes = _json.loads(gc_row[1]) if isinstance(gc_row[1], str) else gc_row[1]
            a = (themes or {}).get("asia", {})
            asia_avg = float(a.get("avg_chg_pct", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            asia_avg = 0.0
    asia_score = max(-1.0, min(1.0, asia_avg / 2.0))  # ±2% -> ±1
    score = REGIME_W_MARKET * (ms or 0.0) + REGIME_W_CUES * (gc or 0.0) + REGIME_W_ASIA * asia_score
    today = ist_today()
    stale = False
    for label, d in (("market_sentiment", ms_row[1] if ms_row else None),
                     ("global_cues", gc_row[2] if gc_row else None)):
        if d is None or d < today - timedelta(days=4):
            stale = True
            print(f"[bot] regime input {label} is stale (latest {d}) — forcing risk-off")
    return {"score": score, "market": ms, "cues": gc, "asia": round(asia_avg, 2),
            "risk_on": (score > REGIME_OFF) and not stale, "stale": stale}


def open_positions() -> list[dict]:
    """Open positions from trades: net of BUY/SELL fills per position_id.

    A position is open when its net quantity is > 0. Aggregating (rather than
    the old row-count heuristic) survives averaging entries and partial
    exits — the old code marked a position "closed" on ANY second fill, and a
    SELL row without its BUY row became a phantom open position that got sold
    again. Entry price = volume-weighted average of the BUY fills; fees = sum
    of entry fees (both match what a single-fill position used to record).
    """
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT position_id, symbol,
                   SUM(CASE WHEN side='BUY' THEN qty ELSE 0 END) AS bought_qty,
                   SUM(CASE WHEN side='BUY' THEN qty*price END) AS buy_notional,
                   SUM(CASE WHEN side='BUY' THEN fees ELSE 0 END) AS buy_fees,
                   SUM(CASE WHEN side='BUY' THEN qty ELSE -qty END) AS net_qty,
                   MAX(ts) AS last_ts
            FROM trades WHERE status='paper'
            GROUP BY position_id, symbol
        """)
        rows = cur.fetchall()
    out: list[dict] = []
    for pid, symbol, bought_qty, buy_notional, buy_fees, net_qty, last_ts in rows:
        net_qty = net_qty or 0
        if net_qty <= 0:
            continue
        bought_qty = bought_qty or 0
        price = (buy_notional or 0.0) / bought_qty if bought_qty else 0.0
        out.append({
            "position_id": pid, "symbol": symbol, "side": "BUY",
            "qty": int(net_qty), "price": float(price), "ts": last_ts,
            "fees": float(buy_fees or 0.0),
        })
    return out


def live_prices(symbols: list[str]) -> tuple[dict[str, float], bool]:
    """Try Fyers live quotes; fall back to the latest DB close (marked)."""
    prices: dict[str, float] = {}
    fallback = False
    try:
        from collector.fyers_client import FyersClient
        client = FyersClient()
        # Map DB tickers -> Fyers symbols (BANKNIFTY is NSE:NIFTYBANK-INDEX,
        # NOT NSE:BANKNIFTY-INDEX), and keep a reverse map so quote keys map
        # back to the DB tickers the rest of the bot uses.
        fyers_symbols, db_by_fyers = [], {}
        for s in symbols:
            if s == "BANKNIFTY":
                fy = "NSE:NIFTYBANK-INDEX"
            elif s == "NIFTY 50":
                fy = "NSE:NIFTY50-INDEX"
            else:
                fy = f"NSE:{s}-EQ"
            fyers_symbols.append(fy)
            db_by_fyers[fy] = s
        quotes = client.quotes(fyers_symbols)
        for sym, v in quotes.items():
            ltp = v.get("lp")
            if ltp:
                db_sym = db_by_fyers.get(sym)
                if db_sym:
                    prices[db_sym] = float(ltp)
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


def _sma(symbol: str, period: int = TREND_SMA) -> float | None:
    """Simple moving average of the last `period` daily closes."""
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT close FROM candles WHERE symbol=%s AND timeframe='1d' "
                    "ORDER BY ts DESC LIMIT %s", (symbol, period))
        rows = [r[0] for r in cur.fetchall()]
    if len(rows) < period:
        return None
    return sum(rows) / len(rows)


def _sent_3d(symbol: str) -> float:
    """Avg sentiment compound for a symbol over the trailing 3 days."""
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(AVG(sentiment_compound), 0) FROM news_sentiment "
            "WHERE symbol=%s AND published_at >= now() - interval '3 days'",
            (symbol,),
        )
        return float(cur.fetchone()[0])


def decide_exit(pos: dict, px: float, rank_pos: dict) -> tuple[str | None, str]:
    """Smart-exit decision for one open position.

    Priority: stop-loss (hard) > profit-take on trend/sentiment break >
    rank exit (fee-aware, trend/sentiment-aware) > hold.

    Returns (exit_reason_or_None, hold_reason_text). The decision is logged
    either way so the training set sees every HOLD too.
    """
    entry = pos["price"]
    qty = pos["qty"]
    last_close = _last_close(pos["symbol"])
    pnl_pct = (px - entry) / entry if entry else 0.0
    rt_fees = FEES.entry_fee(qty * entry) + FEES.exit_fee(qty * px)
    rt_pct = rt_fees / (qty * entry) if qty * entry else 1.0
    sma = _sma(pos["symbol"])
    trend_ok = sma is not None and last_close > sma
    sent_ok = _sent_3d(pos["symbol"]) > SENT_HOLD

    # 1) hard stop-loss — always exits, charges be damned
    if last_close <= entry * (1 - STOP_LOSS):
        return "stop_loss", ""

    # 2) good profit margin + trend/sentiment turning -> book it
    if pnl_pct >= PROFIT_TAKE_PCT and not (trend_ok and sent_ok):
        return "profit_take_trend_break", ""

    # 3) rank fell out of the buffer zone
    r = rank_pos.get(pos["symbol"])
    in_buffer = r is not None and r < TOP_N + BUFFER
    if not in_buffer:
        # fee-aware hold: |pnl| below round-trip charges (+buffer) and the
        # name still has trend or sentiment support -> hold, don't feed fees
        if abs(pnl_pct) < rt_pct + HOLD_PNL_BUFFER and (trend_ok or sent_ok):
            return None, f"small_pnl_{pnl_pct:+.2%}_vs_fees_{rt_pct:.2%}"
        return "signal", ""

    # 4) still in buffer zone -> hold
    reasons = []
    if trend_ok:
        reasons.append("trend_ok")
    if sent_ok:
        reasons.append("sentiment_ok")
    return None, ("rank_in_buffer" + (f"+{'+'.join(reasons)}" if reasons else ""))


def record_fill(side: str, symbol: str, qty: int, price: float, fees: float,
                position_id: str, exit_reason: str | None = None,
                pnl: float | None = None, pnl_pct: float | None = None) -> int:
    with db() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO trades (symbol, side, qty, price, ts, strategy, status,
                                position_id, fees, pnl, pnl_pct, exit_reason)
            VALUES (%s, %s, %s, %s, now(), %s, 'paper', %s, %s, %s, %s, %s)
            RETURNING id
        """, (symbol, side, qty, price, STRATEGY, position_id, fees, pnl, pnl_pct, exit_reason))
        trade_id = cur.fetchone()[0]
        conn.commit()
    return trade_id


# Arbitrary fixed advisory-lock key for cmd_execute: prevents two overlapping
# runs (cron + manual rerun) from double-buying the same slots.
EXECUTE_LOCK_KEY = 0x6E5E5155


def cmd_execute() -> int:
    """Single-flight wrapper around _execute_locked.

    An advisory lock (Postgres session lock) ensures only one execute runs at
    a time. Two overlapping runs used to both see zero open positions and
    fill the same top-5 slots -> 2x exposure and 2x fees.
    """
    with db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (EXECUTE_LOCK_KEY,))
        if not (cur.fetchone() or (False,))[0]:
            print("[bot] another execute run holds the lock — skipping this invocation")
            return 0
        try:
            rc = _execute_locked()
        finally:
            cur.execute("SELECT pg_advisory_unlock(%s)", (EXECUTE_LOCK_KEY,))
            conn.commit()
    return rc


def _execute_locked() -> int:
    picks = load_picks()
    today = ist_today()
    # never "trade" on a weekend or NSE holiday (would use stale closes)
    if not is_trading_day(today):
        print(f"[bot] {today} is not an NSE trading day — skipping")
        return 0
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

    # sells: smart-exit engine (stop-loss > profit-take > fee-aware rank exit)
    for sym, pos in held.items():
        px = prices.get(sym, pos["price"])
        reason, hold_text = decide_exit(pos, px, rank_pos)
        if not reason:
            # HOLD — still logged to the training set with why (never LLM-gated)
            ctx = build_context(sym, "HOLD", px, pos["qty"], None, regime)
            with db() as conn:
                log_decision(conn, ctx, (None, hold_text, "rules"), None, False)
            print(f"[bot] HOLD {sym} ({hold_text})")
            continue
        notional = pos["qty"] * px
        fee = FEES.exit_fee(notional)
        pnl = (px - pos["price"]) * pos["qty"] - pos["fees"] - fee
        trade_id = record_fill("SELL", sym, pos["qty"], px, fee, pos["position_id"],
                               exit_reason=reason, pnl=round(pnl, 2),
                               pnl_pct=round(pnl / (pos["qty"] * pos["price"]), 4))
        ctx = build_context(sym, "SELL", px, pos["qty"], None, regime)
        with db() as conn:
            log_decision(conn, ctx, (None, None, None), None, True, trade_id)
        print(f"[bot] SELL {sym} x{pos['qty']} @ {px:.2f} ({reason}, pnl {pnl:+.2f})")

    # buys: fill free top-N slots (equal weight of CURRENT equity)
    positions = open_positions()
    held = {p["symbol"]: p for p in positions}
    free = TOP_N - len(held)
    if free > 0:
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
                ctx = build_context(p["symbol"], "SKIP", px, 0, p, regime)
                with db() as conn:
                    log_decision(conn, ctx, (None, "price too high for per-position budget", "budget"),
                                 None, False)
                print(f"[bot] SKIP {p['symbol']} — ₹{px:,.0f}/share exceeds the ₹{per:,.0f} budget slot")
                continue
            notional = qty * px
            fee = FEES.entry_fee(notional)
            ctx = build_context(p["symbol"], "BUY", px, qty, p, regime)
            with db() as conn:
                dec_id = log_decision(conn, ctx, (None, None, None), False, False)
            # ---- LLM final rating gate ----
            rating, reason, model = llm_rate(ctx)
            # Fail OPEN when the LLM is unavailable (rating is None): the gate
            # must only block on an actual low rating. A dead CLI/timeout used
            # to silently freeze ALL buys while the docstring promised
            # fail-open — the gap stays visible in the log via llm_model.
            if rating is None:
                gate_pass = True
                print(f"[bot] LLM gate unavailable ({model}) — failing open on composite")
            else:
                gate_pass = rating >= LLM_MIN_RATING
            with db() as conn:
                cur = conn.cursor()
                cur.execute("""UPDATE trade_decisions SET llm_rating=%s, llm_reason=%s,
                               llm_model=%s, llm_gate_pass=%s WHERE id=%s""",
                            (rating, reason, model, gate_pass, dec_id))
                conn.commit()
            if not gate_pass:
                print(f"[bot] SKIP {p['symbol']} — LLM rating {rating} (< {LLM_MIN_RATING})"
                      + (f" — {reason}" if reason else ""))
                continue
            pid = str(uuid.uuid4())[:8]
            trade_id = record_fill("BUY", p["symbol"], qty, px, fee, pid)
            with db() as conn:
                cur = conn.cursor()
                cur.execute("UPDATE trade_decisions SET executed=TRUE, trade_id=%s WHERE id=%s",
                            (trade_id, dec_id))
                conn.commit()
            print(f"[bot] BUY {p['symbol']} x{qty} @ {px:.2f} (composite {p['composite']:.3f}, "
                  f"sent {p['sent_3d']:+.3f}, LLM {rating})")
            bought += 1
    print(f"[bot] executed (fallback prices: {used_fallback})")
    return 0


def _cash_balance() -> float:
    """Cash on hand: CAPITAL + realized cashflows of every paper fill."""
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT COALESCE(SUM(CASE WHEN side='BUY' THEN -(qty*price+fees) "
                    "ELSE qty*price-fees END),0) FROM trades WHERE status='paper'")
        return CAPITAL + float((cur.fetchone() or (0.0,))[0])


def _portfolio_equity() -> float:
    """Cash + mark-to-market of open positions (approximation at execution)."""
    cash = _cash_balance()
    equity = cash
    for pos in open_positions():
        equity += pos["qty"] * _last_close(pos["symbol"])
    return max(equity, 1.0)


def cmd_eod() -> int:
    """Stop-loss check + daily equity snapshot vs NIFTY 50 benchmark."""
    today = ist_today()
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
                        (today, round(equity, 2), round(_cash_balance(), 2), round(bench, 2), STRATEGY))
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
