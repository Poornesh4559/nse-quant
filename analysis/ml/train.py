"""Walk-forward training for the next-day direction model.

Method (standard expanding-window walk-forward, row-date based):
  * Panel rows are split by date. Initial train = the first ``initial_years``
    (default 3) of calendar time, or 60% of the calendar if the data spans less.
  * Each fold predicts the NEXT ``fold_days`` (30) trading days of rows, then
    the training window expands by those 30 days and the model is refit.
  * Because ``y_t = sign(close[t+1]/close[t] - 1)``, a row at date ``t`` can
    only be labelled after the close of ``t+1``; all training labels in a fold
    are realized before the fold's first OOS row's features are even observed.
  * Models: LGBMClassifier (early stopping on the last 10% of the training
    window by date) vs a LogisticRegression baseline (StandardScaler pipeline,
    same training rows). P_up = P(y=1) = probability the NEXT day is up.
  * The final model (most recent fold) is refit on all its training rows at the
    early-stopped iteration count and saved for ``score.py``.

Outputs (data/models/):
  * lgbm_nextday.joblib  — final LightGBM model (joblib)
  * features.json        — ordered feature names
  * oos_predictions.csv  — every OOS row: symbol, date, p_up, p_up_lr, label,
                           ret_next, sent_3d (portfolio sim consumes this)
  * walkforward_metrics.json — per-fold + overall accuracy

Run:  .venv/bin/python -m analysis.ml.train
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.ml.panel import FEATURE_COLS, build_panel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "data" / "models"
MODEL_PATH = MODELS_DIR / "lgbm_nextday.joblib"
FEATURES_PATH = MODELS_DIR / "features.json"
OOS_PATH = MODELS_DIR / "oos_predictions.csv"
METRICS_PATH = MODELS_DIR / "walkforward_metrics.json"

TRADING_DAYS = 252
DEFAULT_RF = 0.065


@dataclass(frozen=True)
class TrainConfig:
    initial_years: float = 3.0
    min_frac: float = 0.6          # fallback initial train when data < initial_years
    fold_days: int = 30
    val_frac: float = 0.10         # last fraction of train dates held out for early stopping
    seed: int = 42
    lgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 1,
        "n_jobs": -1,
        "verbosity": -1,
    })


def walk_forward_splits(dates: pd.DatetimeIndex | np.ndarray,
                        initial_years: float = 3.0,
                        fold_days: int = 30,
                        min_frac: float = 0.6) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window (train_dates, test_dates) pairs over a date calendar.

    Test windows are consecutive ``fold_days`` blocks; the last may be short.
    All dates in ``train`` strictly precede all dates in ``test``.
    """
    dates = np.asarray(pd.DatetimeIndex(dates).sort_values())
    if len(dates) == 0:
        return []
    span_years = pd.Timedelta(dates[-1] - dates[0]).days / 365.25
    if span_years < initial_years:
        init = max(int(len(dates) * min_frac), 1)
    else:
        cutoff = dates[0] + np.timedelta64(int(initial_years * 365.25), "D")
        init = int(np.searchsorted(dates, cutoff, side="right"))
    init = max(init, 1)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = init
    while start < len(dates):
        end = min(start + fold_days, len(dates))
        splits.append((dates[:start], dates[start:end]))
        start = end
    return splits


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return float("nan")
    return float(((pred[mask] > 0.5) == (y[mask] > 0.5)).mean())


def _overall_acc(pred: np.ndarray, labels: np.ndarray) -> float:
    mask = ~np.isnan(labels)
    if not mask.any():
        return float("nan")
    return float(((pred[mask] > 0.5) == (labels[mask] > 0.5)).mean())


def walk_forward_train(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                       cfg: TrainConfig | None = None) -> dict:
    """Run the walk-forward experiment. Returns a dict with per-fold results,
    overall OOS accuracy, the final model and the OOS prediction frame."""
    cfg = cfg or TrainConfig()
    cal = np.asarray(pd.DatetimeIndex(np.unique(meta["date"])).sort_values())
    splits = walk_forward_splits(cal, cfg.initial_years, cfg.fold_days, cfg.min_frac)

    feat_ok = X.notna().all(axis=1)
    label_ok = y.notna()
    idx = np.arange(len(X))

    oos_rows: list[dict] = []
    fold_metrics: list[dict] = []
    final_model = None
    final_feats = list(X.columns)

    for fold, (tr_dates, te_dates) in enumerate(splits):
        tr_mask = meta["date"].isin(tr_dates)
        te_mask = meta["date"].isin(te_dates)

        tr_idx = idx[tr_mask.values & feat_ok.values & label_ok.values]
        te_idx = idx[te_mask.values & feat_ok.values]
        if len(tr_idx) < 200 or len(te_idx) == 0:
            continue

        # validation slice = last val_frac of the training calendar
        tr_cal = np.asarray(pd.DatetimeIndex(np.unique(meta.loc[tr_idx, "date"])).sort_values())
        n_val = max(int(len(tr_cal) * cfg.val_frac), 1)
        val_cut = tr_cal[-n_val]
        is_val = meta.loc[tr_idx, "date"].values >= val_cut
        fit_idx, va_idx = tr_idx[~is_val], tr_idx[is_val]

        Xtr, ytr = X.iloc[fit_idx], y.iloc[fit_idx]
        Xva, yva = X.iloc[va_idx], y.iloc[va_idx]

        # ---- LightGBM with early stopping on the validation slice ----
        model = lgb.LGBMClassifier(**cfg.lgb_params, random_state=cfg.seed)
        if len(va_idx) >= 50:
            model.fit(Xtr, ytr, eval_X=Xva, eval_y=yva,
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        else:
            model.fit(Xtr, ytr)
        best_iter = getattr(model, "best_iteration_", None) or model.n_estimators

        # ---- LogisticRegression baseline (same training rows) ----
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, random_state=cfg.seed)),
        ])
        lr.fit(Xtr, ytr)

        Xte = X.iloc[te_idx]
        p_up = model.predict_proba(Xte)[:, 1]
        p_up_lr = lr.predict_proba(Xte)[:, 1]
        yte = y.iloc[te_idx].values
        acc_lgb, acc_lr = _accuracy(p_up, yte), _accuracy(p_up_lr, yte)

        fold_metrics.append({
            "fold": fold,
            "train_start": str(tr_dates[0])[:10], "train_end": str(tr_dates[-1])[:10],
            "test_start": str(te_dates[0])[:10], "test_end": str(te_dates[-1])[:10],
            "n_train": int(len(fit_idx) + len(va_idx)), "n_test": int(len(te_idx)),
            "acc_lgb": acc_lgb, "acc_lr": acc_lr, "best_iter": int(best_iter),
        })
        for i, r in enumerate(te_idx):
            oos_rows.append({
                "symbol": meta.at[r, "symbol"],
                "date": meta.at[r, "date"],
                "p_up": float(p_up[i]),
                "p_up_lr": float(p_up_lr[i]),
                "label": float(yte[i]) if not np.isnan(yte[i]) else None,
                "ret_next": meta.at[r, "ret_next"],
                "sent_3d": X.at[r, "sent_3d"],
            })
        final_model, final_feats = model, list(X.columns)

    if final_model is None:
        raise RuntimeError("walk-forward produced no folds — check panel data")

    # ---- refit the final model on ALL its training rows at the tuned depth ----
    best_iter = fold_metrics[-1]["best_iter"]
    last_tr_dates = splits[-1][0]
    tr_mask = meta["date"].isin(last_tr_dates)
    tr_idx = idx[tr_mask.values & feat_ok.values & label_ok.values]
    final_params = dict(cfg.lgb_params)
    final_params["n_estimators"] = best_iter
    model_final = lgb.LGBMClassifier(**final_params, random_state=cfg.seed)
    model_final.fit(X.iloc[tr_idx], y.iloc[tr_idx])

    oos = pd.DataFrame(oos_rows)
    label_arr = oos["label"].to_numpy(dtype=float)
    overall = {
        "acc_lgb": _overall_acc(oos["p_up"].to_numpy(), label_arr),
        "acc_lr": _overall_acc(oos["p_up_lr"].to_numpy(), label_arr),
        "n_oos": int(len(oos)),
        "n_labeled": int((~np.isnan(label_arr)).sum()),
        "up_rate_oos": float(label_arr[~np.isnan(label_arr)].mean()) if (~np.isnan(label_arr)).sum() else float("nan"),
    }

    return {
        "folds": fold_metrics,
        "overall": overall,
        "oos": oos,
        "model": model_final,
        "feature_names": final_feats,
    }


def save_artifacts(result: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(result["model"], MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps({"features": result["feature_names"]}, indent=2))
    result["oos"].to_csv(OOS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps({"overall": result["overall"], "folds": result["folds"]},
                                       indent=2, default=str))


def print_importances(model, feature_names: list[str], top: int = 15) -> None:
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\nTop-{top} feature importances (final model, gain):")
    for i, (name, val) in enumerate(imp.head(top).items(), 1):
        print(f"  {i:2d}. {name:<16s} {val:8.0f}")


def main(argv: list[str] | None = None) -> int:
    t0 = dt.datetime.now()
    print("Building panel (5y daily x 52 symbols + sentiment)...")
    X, y, meta = build_panel()
    print(f"  panel: {len(X):,} rows, {X.shape[1]} features, "
          f"dates {meta['date'].min().date()} -> {meta['date'].max().date()}")
    print(f"  sentiment coverage: {(X['sent_count_7d'] > 0).mean() * 100:.2f}% of rows "
          f"have news in the trailing 7d; {X['sent_missing'].mean() * 100:.1f}% never seen news")

    result = walk_forward_train(X, y, meta)
    folds, overall, oos = result["folds"], result["overall"], result["oos"]

    print(f"\nWalk-forward: {len(folds)} folds x {30}-day OOS blocks "
          f"(trained in {dt.datetime.now() - t0})")
    hdr = ["fold", "train_end", "test_start", "test_end", "n_train", "n_test", "acc_lgb", "acc_lr", "best_iter"]
    widths = {h: len(h) for h in hdr}
    rows = [[str(f["fold"]), f["train_end"], f["test_start"], f["test_end"],
             f"{f['n_train']:,}", f"{f['n_test']:,}",
             f"{f['acc_lgb']:.4f}", f"{f['acc_lr']:.4f}", str(f["best_iter"])] for f in folds]
    for r in rows:
        for h, v in zip(hdr, r):
            widths[h] = max(widths[h], len(v))
    fmt = "  ".join(f"{{:<{widths[h]}}}" for h in hdr)
    print(fmt.format(*hdr))
    print("  ".join("-" * widths[h] for h in hdr))
    for r in rows:
        print(fmt.format(*r))

    print(f"\nOVERALL OOS ({overall['n_labeled']:,} labeled rows, up-rate {overall['up_rate_oos']:.3f}):")
    print(f"  LightGBM accuracy:        {overall['acc_lgb'] * 100:.2f}%")
    print(f"  LogisticRegression acc:   {overall['acc_lr'] * 100:.2f}%")

    print_importances(result["model"], result["feature_names"])
    save_artifacts(result)
    print(f"\nSaved: {MODEL_PATH}, {FEATURES_PATH}, {OOS_PATH}, {METRICS_PATH}")
    print(f"Total: {dt.datetime.now() - t0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
