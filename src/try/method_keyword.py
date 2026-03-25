"""
method_keyword.py — Weighted keyword/phrase lexicon
====================================================
Matches curated difficulty phrases against review text.
Longer phrases (bigrams, trigrams) take priority over unigrams
to prevent double-counting (e.g. "not hard" won't also match "hard").

Pros:  Fast, transparent, zero cost, fully explainable
Cons:  Misses implicit difficulty, no context/sarcasm handling
"""

import re
from shared import make_result, preprocess


# ---------------------------------------------------------------------------
# Lexicons — positive weight = hard signal, scored by confidence
# ---------------------------------------------------------------------------

HARD_PHRASES = {
    # Trigrams+ (highest confidence)
    "steep learning curve": 3.0, "extremely difficult": 3.0, "incredibly hard": 3.0,
    "most difficult module": 3.0, "hardest module": 3.0, "insanely hard": 3.0,
    "prepare to suffer": 3.0, "very heavy workload": 3.0, "extremely heavy": 3.0,
    # Strong bigrams
    "very time consuming": 2.5, "super hard": 2.5, "really hard": 2.5,
    "really difficult": 2.5, "very hard": 2.5, "very difficult": 2.5,
    "too hard": 2.5, "too difficult": 2.5, "killer module": 2.5, "killer mod": 2.5,
    "heavy workload": 2.5, "rip your grades": 2.5,
    # Moderate bigrams
    "quite challenging": 2.0, "pretty hard": 2.0, "pretty difficult": 2.0,
    "quite hard": 2.0, "quite difficult": 2.0, "so hard": 2.0, "so difficult": 2.0,
    "content heavy": 2.0, "workload is heavy": 2.0, "no life": 2.0,
    # Unigrams (lower weight, more ambiguous)
    "time consuming": 1.5, "nightmare": 2.5, "brutal": 2.0, "hell": 2.0,
    "overwhelming": 2.0, "impossible": 2.5,
    "difficult": 1.5, "hard": 1.0, "challenging": 1.0, "tough": 1.2,
    "struggle": 1.5, "struggled": 1.5, "struggling": 1.5,
    "stressful": 1.5, "demanding": 1.5, "confusing": 1.2, "complicated": 1.2,
    "tricky": 1.0, "painful": 1.5, "intense": 1.0, "tedious": 1.5,
    "suffer": 1.5, "suffered": 1.5, "cry": 1.5, "cried": 1.5,
}

EASY_PHRASES = {
    # Negation phrases (must match before unigrams)
    "not that hard": 2.5, "not that difficult": 2.5,
    "not too hard": 2.0, "not too difficult": 2.0,
    "not very hard": 2.0, "not very difficult": 2.0,
    # Strong easy signals
    "pretty easy": 2.0, "quite easy": 2.0, "very easy": 2.5,
    "really easy": 2.5, "super easy": 2.5, "fairly easy": 2.0,
    "relatively easy": 2.0, "quite straightforward": 2.0,
    "very straightforward": 2.5, "free grade": 2.5, "easy a": 2.5,
    "easy module": 2.0, "easy mod": 2.0, "chill module": 2.0, "chill mod": 2.0,
    "light workload": 2.0, "very manageable": 2.0, "manageable workload": 2.0,
    "not hard": 1.5, "not difficult": 1.5,
    # Unigrams
    "easy": 1.0, "straightforward": 1.2, "manageable": 1.0, "simple": 0.8,
    "chill": 1.2, "breeze": 1.5, "effortless": 2.0, "doable": 0.8,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _keyword_score(message):
    """
    Score a message against both lexicons.
    Returns (hard_score, easy_score, hard_matches, easy_matches).
    Longer phrases matched first; consumed spans prevent double-counting.
    """
    clean = preprocess(message)
    if not clean:
        return 0, 0, [], []

    consumed = []

    def is_consumed(s, e):
        return any(s < ce and e > cs for cs, ce in consumed)

    hard_score, easy_score = 0.0, 0.0
    hard_matches, easy_matches = [], []

    # Easy first (handles negation before "hard" unigram fires)
    for phrase, w in sorted(EASY_PHRASES.items(), key=lambda x: len(x[0]), reverse=True):
        for m in re.finditer(r'\b' + re.escape(phrase) + r'\b', clean):
            if not is_consumed(m.start(), m.end()):
                easy_score += w
                easy_matches.append(phrase)
                consumed.append((m.start(), m.end()))

    for phrase, w in sorted(HARD_PHRASES.items(), key=lambda x: len(x[0]), reverse=True):
        for m in re.finditer(r'\b' + re.escape(phrase) + r'\b', clean):
            if not is_consumed(m.start(), m.end()):
                hard_score += w
                hard_matches.append(phrase)
                consumed.append((m.start(), m.end()))

    return hard_score, easy_score, hard_matches, easy_matches


def run(reviews):
    """Run keyword method on a list of review dicts. Returns list of results."""
    results = []
    for rev in reviews:
        msg = rev.get("message", "")

        if rev.get("_too_short", False):
            results.append(make_result(
                rev.get("post_id", ""), rev.get("author", ""),
                msg, "neutral", 0.0, "keyword", "Too short to score"
            ))
            continue

        hs, es, hm, em = _keyword_score(msg)
        net = hs - es

        if net >= 1.0:
            label = "hard"
        elif net <= -1.0:
            label = "easy"
        else:
            label = "neutral"

        confidence = min(abs(net) / 5.0, 1.0) if abs(net) >= 1.0 else abs(net) / 2.0

        parts = []
        if hm:
            parts.append(f"hard signals: {', '.join(hm)}")
        if em:
            parts.append(f"easy signals: {', '.join(em)}")
        explanation = "; ".join(parts) if parts else "No difficulty keywords found"

        results.append(make_result(
            rev.get("post_id", ""), rev.get("author", ""),
            msg, label, confidence, "keyword", explanation
        ))

    return results
