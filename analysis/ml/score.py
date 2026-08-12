"""Score today's top-N picks with the saved walk-forward model.

``predict_next`` builds the panel using ONLY data available as of today
(``as_of`` cap — no today's session, nothing not yet published), loads the
saved LightGBM model, ranks every symbol by P_up on the latest completed
trading date, applies the sentiment guardrail (skip sent_3d <= -0.1), and
prints the top-10 with the top-5 picks. The picks are written to
``data/signals/nextday_picks.json`` for the Stage-C paper bot.

The market gate (index filter) is intentionally skipped for now — it lands in
a later phase; ``score.py`` only needs the guardrail.

Run:  .venv/bin/python -m analysis.ml.score [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analysis.ml.panel import build_panel
from analysis.ml.train import FEATURES_PATH, MODEL_PATH
from analysis.calendar import next_trading_day

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIGNALS_DIR = REPO_ROOT / "data" / "signals"
PICKS_PATH = SIGNALS_DIR / "nextday_picks.json"

GUARDRAIL = -0.1
TOP_N = 5
# Composite weights (the v1 bot ranking — sentiment is a TILT + GATE):
MOM_W = 0.40   # cross-sectional momentum percentile
ML_W = 0.35    # ML next-day probability
SENT_W = 0.25  # sentiment (sent_3d, clipped [-1,1] -> [0,1])


def composite(row) -> float:
    """Blend momentum + ML + sentiment into the v1 ranking score (0..1)."""
    mom = 0.5 if pd.isna(row.get("mom_rank")) else float(row["mom_rank"])
    ml = 0.5 if pd.isna(row.get("p_up")) else float(row["p_up"])
    sent = 0.0 if pd.isna(row.get("sent_3d")) else float(np.clip(row["sent_3d"], -1, 1))
    return float(np.clip(MOM_W * mom + ML_W * ((ml - 0.5) * 2.0) + SENT_W * ((sent + 1) / 2.0), 0, 1))


def predict_next(today: str | dt.date | None = None,
                 model_path: Path = MODEL_PATH,
                 features_path: Path = FEATURES_PATH,
                 top_n: int = TOP_N) -> dict:
    today = pd.Timestamp(today or dt.date.today()).normalize()

    model = joblib.load(model_path)
    feature_names = json.loads(features_path.read_text())["features"]

    # as_of cap: the model can never see today's (possibly partial) session —
    # it predicts the NEXT day from the last COMPLETED trading day.
    X, y, meta = build_panel(as_of=today)
    X = X[feature_names]

    valid = X.notna().all(axis=1)
    if not valid.any():
        raise RuntimeError("no rows with complete features in the panel — check data freshness")
    last_date = meta.loc[valid, "date"].max()
    sel = valid & (meta["date"] == last_date)

    # the paper bot trades equities only — drop INDEX instruments from the rank
    from analysis.data import BENCHMARK_SYMBOL
    tradable = ~meta.loc[sel, "symbol"].isin((BENCHMARK_SYMBOL, "BANKNIFTY"))
    sel_idx = sel[sel].index[tradable.values]

    p_up = model.predict_proba(X.loc[sel_idx])[:, 1]
    out = pd.DataFrame({
        "symbol": meta.loc[sel_idx, "symbol"].values,
        "p_up": p_up,
        "sent_3d": X.loc[sel_idx, "sent_3d"].values,
        "mom_rank": X.loc[sel_idx, "mom_rank"].values,
    })
    out["rank"] = out["p_up"].rank(ascending=False, method="first").astype(int)
    out["composite"] = out.apply(composite, axis=1)
    out = out.sort_values("composite", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out["pass_guardrail"] = out["sent_3d"] > GUARDRAIL

    picks = out[out["pass_guardrail"]].head(top_n)
    calendar = sorted(meta["date"].unique())
    nxt = next((d for d in calendar if d > last_date), None)
    nxt_estimated = False
    if nxt is None:
        # market calendar not in the DB yet (weekend/holiday) — estimate the
        # next NSE TRADING day (weekday minus holidays). The old bdate_range
        # fallback could land on an NSE holiday, so the bot would execute at
        # stale closes or skip the real next trading day.
        nxt = pd.Timestamp(next_trading_day(last_date.date()))
        nxt_estimated = True

    result = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "as_of": str(last_date.date()),
        "next_trade_date": str(nxt.date()),
        "next_trade_date_estimated": nxt_estimated,
        "model": str(model_path),
        "guardrail": GUARDRAIL,
        "top5": [{"rank": int(r["rank"]), "symbol": r["symbol"], "p_up": round(float(r["p_up"]), 4),
                  "sent_3d": round(float(r["sent_3d"]), 4),
                  "mom_rank": None if np.isnan(r["mom_rank"]) else round(float(r["mom_rank"]), 4),
                  "composite": round(float(r["composite"]), 4)}
                 for _, r in picks.iterrows()],
        "top10": [{"rank": int(r["rank"]), "symbol": r["symbol"], "p_up": round(float(r["p_up"]), 4),
                   "sent_3d": round(float(r["sent_3d"]), 4),
                   "composite": round(float(r["composite"]), 4),
                   "pass_guardrail": bool(r["pass_guardrail"])}
                  for _, r in out.head(10).iterrows()],
    }
    return result


def print_picks(result: dict) -> None:
    est = " (estimated)" if result["next_trade_date_estimated"] else ""
    print(f"\nTODAY'S TOP-{TOP_N} PICKS  (signals as of {result['as_of']}, "
          f"next trade date {result['next_trade_date']}{est})")
    print("guardrail: sent_3d <= -0.1 blocks entry | market gate: off (later phase) | INDEX symbols excluded")
    print(f"\n{'#':<3}{'symbol':<14}{'P_up':>8}{'sent_3d':>9}{'mom_rank':>10}")
    print("-" * 44)
    for p in result["top5"]:
        mr = f"{p['mom_rank']:.3f}" if p["mom_rank"] is not None else "n/a"
        print(f"{p['rank']:<3}{p['symbol']:<14}{p['p_up']:>8.4f}{p['sent_3d']:>9.3f}{mr:>10}")
    print("\nTop-10 (incl. guardrail-blocked):")
    for p in result["top10"]:
        flag = "" if p["pass_guardrail"] else "  <-- blocked by sentiment"
        print(f"  {p['rank']:>2}. {p['symbol']:<14} P_up={p['p_up']:.4f} sent_3d={p['sent_3d']:+.3f}{flag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysis.ml.score", description=__doc__)
    parser.add_argument("--date", default=None, help="as-of date override (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    result = predict_next(today=args.date)
    print_picks(result)

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {PICKS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
