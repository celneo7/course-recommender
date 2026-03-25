"""
method_zeroshot.py — Zero-shot classification via HuggingFace transformers
===========================================================================
Uses facebook/bart-large-mnli to classify reviews into difficulty categories
WITHOUT any training data. The model treats it as natural language inference:
  "Does this review entail that the course is difficult?"

Prerequisites (run locally):
  pip install transformers torch

First run will download the model (~1.6GB). Subsequent runs use cache.

Pros:  No training needed, understands context/semantics, catches implicit signals
Cons:  Requires torch (~900MB+), slower than keyword/VADER, less transparent
"""

from transformers import pipeline
from shared import make_result

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Candidate labels — the model scores how well each label describes the review
CANDIDATE_LABELS = [
    "This course is very difficult and has heavy workload",
    "This course is easy and manageable",
    "This review does not discuss course difficulty",
]

# Short aliases for output
LABEL_MAP = {
    "This course is very difficult and has heavy workload": "hard",
    "This course is easy and manageable": "easy",
    "This review does not discuss course difficulty": "neutral",
}

# Model — bart-large-mnli is the standard for zero-shot classification
MODEL_NAME = "facebook/bart-large-mnli"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_classifier = None

def _get_classifier():
    """Lazy-load the model (heavy, only load once)."""
    global _classifier
    if _classifier is None:
        print("  Loading zero-shot model (first run downloads ~1.6GB)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,  # CPU; change to 0 for GPU
        )
    return _classifier


def run(reviews):
    """Run zero-shot classification on all reviews. Returns list of results."""
    classifier = _get_classifier()
    results = []

    for i, rev in enumerate(reviews):
        msg = rev.get("message", "")

        if rev.get("_too_short", False):
            results.append(make_result(
                rev.get("post_id", ""), rev.get("author", ""),
                msg, "neutral", 0.0, "zeroshot", "Too short to score"
            ))
            continue

        # Truncate long reviews (model has token limit)
        text = msg[:512]

        print(f"  Zero-shot [{i+1}/{len(reviews)}]...", end=" ", flush=True)
        output = classifier(text, CANDIDATE_LABELS, multi_label=False)

        # output: {"labels": [...], "scores": [...]}
        top_label = output["labels"][0]
        top_score = output["scores"][0]
        label = LABEL_MAP.get(top_label, "neutral")

        # Build explanation showing all scores
        score_parts = []
        for lbl, sc in zip(output["labels"], output["scores"]):
            short = LABEL_MAP.get(lbl, lbl)
            score_parts.append(f"{short}={sc:.3f}")
        explanation = "Scores: " + ", ".join(score_parts)

        # Confidence: use the margin between top and second-best
        if len(output["scores"]) >= 2:
            margin = output["scores"][0] - output["scores"][1]
            confidence = min(margin * 2, 1.0)  # Scale margin to 0-1
        else:
            confidence = top_score

        print(f"{label} ({confidence:.2f})")

        results.append(make_result(
            rev.get("post_id", ""), rev.get("author", ""),
            msg, label, round(confidence, 2), "zeroshot", explanation
        ))

    return results
