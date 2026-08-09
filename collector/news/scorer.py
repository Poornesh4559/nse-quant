"""Dual-model headline sentiment: VADER + FinBERT, averaged.

Model 1 (VADER): fast lexicon scorer, compound in [-1, 1].
Model 2 (FinBERT): ProsusAI/finbert transformer fine-tuned on financial
    text. Its argmax label is converted to a compound-like score via
    pos_score - neg_score: positive -> +score, negative -> -score,
    neutral -> 0.0.

The final `compound` is the mean of both model scores. FinBERT is lazy-loaded
on first use; any import/download/run failure degrades gracefully to
VADER-only with a warning — scoring never crashes the pipeline.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Truncate headlines before scoring: keeps FinBERT inside its max length and
# avoids long-tail noise dominating either model.
MAX_CHARS = 200

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - only if a dep is missing
    SentimentIntensityAnalyzer = None  # type: ignore[assignment]

_vader: SentimentIntensityAnalyzer | None = None
_finbert_pipe = None
_finbert_failed = False


def _get_vader() -> SentimentIntensityAnalyzer | None:
    """Lazily build the (thread-safe-enough) VADER singleton."""
    global _vader
    if _vader is None and SentimentIntensityAnalyzer is not None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def _get_finbert():
    """Lazily load the FinBERT pipeline singleton.

    Returns the pipeline, or None if transformers is missing / the model fails
    to download or load. Failures are remembered so we don't retry every call.
    """
    global _finbert_pipe, _finbert_failed
    if _finbert_pipe is None and not _finbert_failed:
        try:
            from transformers import pipeline

            _finbert_pipe = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=MAX_CHARS,
            )
            logger.info("FinBERT pipeline loaded")
        except Exception as exc:  # noqa: BLE001 - fall back, never crash
            _finbert_failed = True
            logger.warning("FinBERT unavailable (%s); falling back to VADER-only", exc)
    return _finbert_pipe


def _finbert_compound(text: str) -> float | None:
    """Run FinBERT and convert its argmax label to a compound-like score."""
    pipe = _get_finbert()
    if pipe is None:
        return None
    try:
        result = pipe(text[:MAX_CHARS])[0]
        label, score = result["label"].lower(), float(result["score"])
        if label == "positive":
            return score
        if label == "negative":
            return -score
        return 0.0  # neutral
    except Exception as exc:  # noqa: BLE001 - a bad run shouldn't kill the run
        logger.warning("FinBERT scoring failed (%s); falling back to VADER-only", exc)
        return None


def score_text(text: str) -> dict:
    """Score a headline with both models and return the averaged sentiment.

    Returns a dict with:
      compound         averaged VADER+FinBERT score (-1..1)
      pos/neg/neu      VADER valence proportions
      label            POSITIVE (>= 0.05) / NEGATIVE (<= -0.05) / NEUTRAL
      vader_compound   raw VADER compound (debug)
      finbert_compound raw FinBERT compound, or None if FinBERT unavailable
    """
    text = (text or "").strip()[:MAX_CHARS]
    vader = _get_vader()
    if vader is not None:
        vs = vader.polarity_scores(text)
    else:
        vs = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    vader_compound = float(vs["compound"])
    finbert_compound = _finbert_compound(text)

    if finbert_compound is not None:
        compound = (vader_compound + finbert_compound) / 2.0
    else:
        compound = vader_compound

    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "compound": compound,
        "pos": float(vs["pos"]),
        "neg": float(vs["neg"]),
        "neu": float(vs["neu"]),
        "label": label,
        "vader_compound": vader_compound,
        "finbert_compound": finbert_compound,
    }
