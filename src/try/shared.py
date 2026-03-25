"""
shared.py — Common utilities for all difficulty sentiment methods
================================================================
Standard output format, summary aggregation, review loading/filtering.
"""

import json
import re
from collections import Counter


# ============================================================================
# Standard result format (every method must return a list of these)
# ============================================================================

def make_result(post_id, author, message, label, confidence, method, explanation):
    """
    label:       "hard" | "neutral" | "easy"
    confidence:  0.0 - 1.0
    method:      string identifying which method produced this
    explanation: human-readable reason for the label
    """
    return {
        "post_id": str(post_id),
        "author": author,
        "message_preview": message[:150] + ("..." if len(message) > 150 else ""),
        "message_full": message,
        "label": label,
        "confidence": round(confidence, 2),
        "method": method,
        "explanation": explanation,
    }


# ============================================================================
# Aggregation
# ============================================================================

def summarize(results, course_code="UNKNOWN"):
    """Aggregate scored results into a course-level summary."""
    scored = [r for r in results if r["label"] in ("hard", "neutral", "easy")]
    dist = Counter(r["label"] for r in scored)
    total = len(scored)

    pcts = {}
    for label in ["hard", "neutral", "easy"]:
        pcts[label] = round(dist.get(label, 0) / total * 100, 1) if total else 0.0

    # Weighted score: hard=5, neutral=3, easy=1
    weight_map = {"hard": 5.0, "neutral": 3.0, "easy": 1.0}
    if total > 0:
        score = round(sum(weight_map[r["label"]] for r in scored) / total, 2)
    else:
        score = 3.0

    return {
        "course_code": course_code,
        "total_scored": total,
        "distribution": dict(dist),
        "percentages": pcts,
        "difficulty_score": score,
    }


def print_summary(results, course_code="UNKNOWN", method_name=""):
    """Pretty-print a method's results."""
    s = summarize(results, course_code)
    bar_width = 30

    header = f"  {method_name}" if method_name else ""
    print(f"\n  Course: {s['course_code']}{header}  |  Scored: {s['total_scored']} reviews")
    print(f"  Difficulty Score: {s['difficulty_score']} / 5.0\n")

    for label, emoji in [("hard", "🔴"), ("neutral", "🟡"), ("easy", "🟢")]:
        count = s["distribution"].get(label, 0)
        pct = s["percentages"][label]
        filled = int(pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  {emoji} {label.capitalize():8s} {bar} {count:3d} ({pct:.1f}%)")


# ============================================================================
# Loading & filtering
# ============================================================================

def load_reviews(path, top_level_only=True, min_length=15):
    """Load reviews JSON; optionally filter to top-level and non-trivial."""
    with open(path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    total = len(reviews)

    if top_level_only:
        reviews = [r for r in reviews if not r.get("reply_to_post_id")]

    # Tag short reviews but keep them (methods can skip)
    for r in reviews:
        r["_too_short"] = len((r.get("message") or "").strip()) < min_length

    filtered = len(reviews)
    short = sum(1 for r in reviews if r["_too_short"])
    print(f"Loaded {total} reviews → {filtered} top-level ({short} too short, will be neutral)")
    return reviews


def preprocess(text):
    """Lowercase, normalize whitespace, strip non-alpha."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
