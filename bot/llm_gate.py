"""LLM rating gate + decision log for the paper bot.

Every trade the bot CONSIDERS gets logged to ``trade_decisions`` with the full
context it saw (composite score, ML prob, sentiment, regime, technicals, price)
— that table becomes the training set for the next model generation.

Before executing, the bot optionally asks an LLM (deepseek-v4-flash via the
opencode CLI) for a final 0..1 rating. If the rating >= LLM_MIN_RATING the
trade passes; the rating + reason are stored with the decision. If the LLM is
unavailable the gate fails OPEN (trades on the composite alone, logged as
llm_rating=NULL) so the pipeline never stalls — paper money, and the gap is
visible in the log.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data import load_daily
from analysis.engine import FeeSchedule
from analysis.features import compute_features

LLM_MODEL = "opencode-go/deepseek-v4-flash"
LLM_MIN_RATING = 0.6          # gate threshold (tweakable)
LLM_TIMEOUT_S = 60
LLM_ENABLED = True            # flip to False to bypass the gate entirely

FEATURES_CACHE: dict[str, dict] = {}


def technical_snapshot(symbol: str) -> dict:
    """Latest indicator values for a symbol (rsi, macd, bb, atr, returns, vol)."""
    if symbol in FEATURES_CACHE:
        return FEATURES_CACHE[symbol]
    frames = load_daily()
    df = frames.get(symbol)
    out: dict = {}
    if df is not None and len(df) >= 30:
        feats = compute_features(df)
        last = feats.iloc[-1]
        for col in ("rsi14", "macd", "bb_pos", "atr14", "ret_1", "ret_5", "ret_21", "vol_z"):
            v = last.get(col)
            out[col] = None if (v is None or pd.isna(v)) else round(float(v), 4)
    FEATURES_CACHE[symbol] = out
    return out


def build_context(symbol: str, action: str, price: float, qty: int,
                  pick: dict | None, regime: dict) -> dict:
    """Full decision context: ranking inputs + regime + technicals."""
    tech = technical_snapshot(symbol)
    sent_7d = None
    if pick:
        ctx = {
            "symbol": symbol, "action": action, "price": round(price, 2), "qty": qty,
            "composite_score": pick.get("composite"),
            "mom_rank": pick.get("mom_rank"),
            "ml_p_up": pick.get("p_up"),
            "sent_3d": pick.get("sent_3d"),
        }
    else:
        ctx = {"symbol": symbol, "action": action, "price": round(price, 2), "qty": qty}
    ctx.update({
        "market_sentiment": regime.get("market"),
        "global_cues": regime.get("cues"),
        "regime_score": round(regime.get("score", 0.0), 4),
        "regime_risk_on": regime.get("risk_on"),
        **tech,
    })
    return ctx


def llm_rate(ctx: dict) -> tuple[float | None, str | None, str | None]:
    """Ask deepseek-v4-flash for a final 0..1 rating. Returns (rating, reason, model).

    Never raises: any failure returns (None, None, 'unavailable') so the caller
    can fail open. Uses the opencode CLI (proven path, no extra API keys).
    """
    if not LLM_ENABLED:
        return None, None, "disabled"
    opencode = shutil.which("opencode")
    if not opencode:
        return None, None, "opencode_cli_missing"
    prompt = (
        "You are a senior Indian equity quant trader. Rate this trade candidate "
        "0.0 to 1.0 (0 = terrible, 1 = excellent) considering momentum, sentiment, "
        f"technicals and market regime. Context JSON: {json.dumps(ctx)}. "
        'Reply with ONLY a JSON object: {"rating": <0.0-1.0>, "reason": "<under 12 words>"}'
    )
    try:
        proc = subprocess.run(
            [opencode, "run", prompt, "--model", LLM_MODEL],
            capture_output=True, text=True, timeout=LLM_TIMEOUT_S,
        )
        raw = (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:  # noqa: BLE001
        print(f"[llm-gate] call failed: {e}")
        return None, None, f"error:{type(e).__name__}"
    m = re.search(r"\{[^{}]*\"rating\"[^{}]*\}", raw, re.DOTALL)
    if not m:
        print(f"[llm-gate] no JSON in LLM output: {raw[:200]!r}")
        return None, None, "unparseable"
    try:
        obj = json.loads(m.group(0))
        rating = float(np.clip(float(obj.get("rating", 0.5)), 0.0, 1.0))
        return round(rating, 4), str(obj.get("reason", ""))[:200], LLM_MODEL
    except Exception as e:  # noqa: BLE001
        print(f"[llm-gate] bad JSON: {e}")
        return None, None, "unparseable"


def log_decision(db_conn, ctx: dict, llm: tuple, gate_pass: bool,
                 executed: bool, trade_id: int | None = None) -> int:
    """Insert/update a trade_decisions row. Returns the row id."""
    rating, reason, model = llm
    with db_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO trade_decisions
              (symbol, action, price, qty, composite_score, mom_rank, ml_p_up,
               sent_3d, market_sentiment, global_cues, regime_score, regime_risk_on,
               rsi14, macd, bb_pos, atr14, ret_1, ret_5, ret_21, vol_z,
               llm_rating, llm_reason, llm_model, llm_gate_pass, executed, trade_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, decision_ts) DO UPDATE SET
              llm_rating=EXCLUDED.llm_rating, llm_reason=EXCLUDED.llm_reason,
              llm_model=EXCLUDED.llm_model, llm_gate_pass=EXCLUDED.llm_gate_pass,
              executed=EXCLUDED.executed, trade_id=EXCLUDED.trade_id
            RETURNING id
        """, (
            ctx["symbol"], ctx["action"], ctx.get("price"), ctx.get("qty"),
            ctx.get("composite_score"), ctx.get("mom_rank"), ctx.get("ml_p_up"),
            ctx.get("sent_3d"), ctx.get("market_sentiment"), ctx.get("global_cues"),
            ctx.get("regime_score"), ctx.get("regime_risk_on"),
            ctx.get("rsi14"), ctx.get("macd"), ctx.get("bb_pos"), ctx.get("atr14"),
            ctx.get("ret_1"), ctx.get("ret_5"), ctx.get("ret_21"), ctx.get("vol_z"),
            rating, reason, model, gate_pass, executed, trade_id,
        ))
        return cur.fetchone()[0]
