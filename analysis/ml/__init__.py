"""Phase 4 ML research layer — walk-forward next-day direction models.

Package layout
--------------
panel.py     build the symbol-day feature panel (price/tech + sentiment + cross-sectional rank)
train.py     walk-forward LightGBM training vs a LogisticRegression baseline
portfolio.py cross-sectional top-N OOS portfolio simulation (costs, stop-loss, sentiment guardrail)
score.py     today's top-N picks for the Stage-C paper bot
test_ml.py   leakage / split / portfolio-math tests

Run (from repo root):
    .venv/bin/python -m pytest analysis/ml/test_ml.py -v
    .venv/bin/python -m analysis.ml.train
    .venv/bin/python -m analysis.ml.portfolio
    .venv/bin/python -m analysis.ml.score
"""
