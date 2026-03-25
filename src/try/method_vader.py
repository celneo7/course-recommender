"""
method_vader.py — VADER sentiment with academic context filtering
=================================================================
Instead of running VADER on the whole review (which conflates
"great professor" with "hard exam"), we:
  1. Split the review into sentences
  2. Keep only sentences containing academic keywords
  3. Run VADER on those filtered sentences
  4. Map negative sentiment in academic context → "hard"

Pros:  No training needed, fast, handles some context
Cons:  VADER is general-purpose; negative ≠ difficult always
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from shared import make_result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ACADEMIC_KEYWORDS = {
    "exam", "exams", "midterm", "midterms", "final", "finals", "quiz", "quizzes",
    "assignment", "assignments", "homework", "hw", "problem set", "pset", "lab", "labs",
    "project", "projects", "workload", "content", "material", "lecture", "lectures",
    "tutorial", "tutorials", "module", "mod", "course", "grading", "grade", "grades",
    "bell curve", "bellcurve", "curriculum", "syllabus", "topic", "topics",
    "concept", "concepts", "coding", "programming", "paper", "essay",
    # Expanded: difficulty-adjacent terms often used without explicit academic nouns
    "hard", "difficult", "tough", "easy", "challenging", "struggle", "manageable",
    "workload", "hours", "week", "time", "effort", "study", "revision", "revise",
    "cheatsheet", "cheat sheet", "textbook", "notes", "past year", "pyp",
}

# Sentences with these words are probably about the professor, not difficulty
PROFESSOR_KEYWORDS = {
    "professor", "prof", "lecturer", "teacher", "tutor", "ta",
    "teaching", "explains", "explained", "helpful", "friendly",
}

analyzer = SentimentIntensityAnalyzer()

# Inject difficulty-specific words that VADER doesn't know about
# VADER uses a -4 to +4 scale for its lexicon
CUSTOM_LEXICON_UPDATES = {
    "brutal": -2.5, "nightmare": -3.0, "insane": -1.5, "killer": -2.0,
    "overwhelming": -2.0, "tedious": -1.5, "grueling": -2.5, "gruelling": -2.5,
    "stressful": -2.0, "struggled": -1.5, "struggling": -1.5,
    "manageable": 1.5, "doable": 1.0, "straightforward": 1.5, "chill": 1.5,
    "breeze": 2.0, "effortless": 2.5, "lightweight": 1.0,
}
analyzer.lexicon.update(CUSTOM_LEXICON_UPDATES)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _split_sentences(text):
    """Simple sentence splitter."""
    # Split on period, exclamation, question mark, or newline
    sentences = re.split(r'[.!?\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


def _has_academic_context(sentence):
    """Check if a sentence mentions academic concepts."""
    lower = sentence.lower()
    return any(kw in lower for kw in ACADEMIC_KEYWORDS)


def _is_professor_focused(sentence):
    """Check if sentence is primarily about the teacher."""
    lower = sentence.lower()
    return any(kw in lower for kw in PROFESSOR_KEYWORDS)


def _score_review(message):
    """
    Score a review using VADER on academic-context sentences.
    Returns (compound_score, num_academic_sentences, explanation).
    """
    sentences = _split_sentences(message)

    # Filter to academic-context sentences, excluding professor-focused ones
    academic_sentences = []
    for s in sentences:
        if _has_academic_context(s) and not _is_professor_focused(s):
            academic_sentences.append(s)

    if not academic_sentences:
        # Fallback: score the whole review if no academic sentences found
        scores = analyzer.polarity_scores(message)
        return scores["compound"], 0, "No academic-context sentences; scored full review"

    # Average VADER compound across academic sentences
    compounds = []
    scored_texts = []
    for s in academic_sentences:
        vs = analyzer.polarity_scores(s)
        compounds.append(vs["compound"])
        scored_texts.append(f'"{s[:60]}" → {vs["compound"]:.2f}')

    avg_compound = sum(compounds) / len(compounds)
    explanation = f"Scored {len(academic_sentences)} academic sentences: " + "; ".join(scored_texts[:3])

    return avg_compound, len(academic_sentences), explanation


def run(reviews):
    """Run VADER+academic-filter method on reviews. Returns list of results."""
    results = []
    for rev in reviews:
        msg = rev.get("message", "")

        if rev.get("_too_short", False):
            results.append(make_result(
                rev.get("post_id", ""), rev.get("author", ""),
                msg, "neutral", 0.0, "vader", "Too short to score"
            ))
            continue

        compound, n_academic, explanation = _score_review(msg)

        # Map VADER compound to difficulty label
        # VADER compound: -1 (negative) to +1 (positive)
        # Negative academic sentiment → hard course
        # Positive academic sentiment → easy course
        if compound <= -0.15:
            label = "hard"
        elif compound >= 0.15:
            label = "easy"
        else:
            label = "neutral"

        # Confidence: how far from neutral, scaled
        confidence = min(abs(compound), 1.0)
        # Lower confidence if we had no academic sentences (used fallback)
        if n_academic == 0:
            confidence *= 0.5
            explanation = "[FALLBACK] " + explanation

        results.append(make_result(
            rev.get("post_id", ""), rev.get("author", ""),
            msg, label, confidence, "vader", explanation
        ))

    return results
