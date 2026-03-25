"""
method_rules.py — Rule-based structural pattern matching
=========================================================
Goes beyond keyword matching by looking for structural patterns:
  - "spent [N] hours" where N is high → hard
  - [difficulty adjective] near [academic noun] → hard
  - Negation-aware: "not difficult" handled correctly
  - Numeric workload signals: "3 assignments per week"

Pros:  Transparent (every rule is readable), handles some implicit signals
Cons:  Brittle to phrasing variations, requires manual rule-writing
"""

import re
from shared import make_result, preprocess


# ---------------------------------------------------------------------------
# Rule definitions — each returns (score, explanation) or (0, None)
# Score: positive = hard, negative = easy
# ---------------------------------------------------------------------------

def rule_time_spent(text):
    """Detect "spent X hours" patterns where X is high."""
    patterns = [
        r'(?:spent|spend|took|takes?|need)\s+(?:about\s+|around\s+|like\s+)?(\d+)\s*(?:hours?|hrs?)\s*(?:per\s+week|a\s+week|weekly|every\s+week|on)',
        r'(\d+)\s*(?:hours?|hrs?)\s*(?:per\s+week|a\s+week|weekly|every\s+week)',
        r'(?:spent|spend|took)\s+(?:about\s+|around\s+)?(\d+)\s*(?:hours?|hrs?)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            hours = int(m.group(1))
            if hours >= 20:
                return 3.0, f"Time signal: {hours} hours (very high)"
            elif hours >= 10:
                return 2.0, f"Time signal: {hours} hours (high)"
            elif hours >= 5:
                return 1.0, f"Time signal: {hours} hours (moderate)"
            elif hours <= 2:
                return -1.0, f"Time signal: {hours} hours (low)"
    return 0, None


def rule_workload_quantity(text):
    """Detect workload quantity signals like "3 assignments per week"."""
    patterns = [
        r'(\d+)\s*(?:assignments?|problem\s*sets?|psets?|homework|labs?|projects?)\s*(?:per|a|every)\s*(?:week|month)',
        r'(?:weekly|every\s+week)\s*(?:there\s+(?:are|were)\s+)?(\d+)\s*(?:assignments?|problem\s*sets?|psets?|homework|labs?)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            count = int(m.group(1))
            if count >= 3:
                return 2.0, f"Workload quantity: {count} per week (heavy)"
            elif count >= 2:
                return 1.0, f"Workload quantity: {count} per week (moderate)"
    return 0, None


def rule_grade_difficulty(text):
    """Detect grade/scoring difficulty signals."""
    hard_patterns = [
        (r'(?:hard|difficult|tough|impossible)\s+to\s+(?:score|get|achieve)\s+(?:a\s+)?(?:good|high|a\b|a\+)', 2.0, "Hard to score well"),
        (r'(?:low|bad|poor)\s+(?:average|median|mean)', 1.5, "Low average grade"),
        (r'average\s+(?:is|was|around)\s+(?:b-|c|c\+|d)', 2.0, "Low grade average"),
        (r'many\s+(?:people|students)\s+(?:fail|failed|dropped)', 2.0, "High fail/drop rate"),
        (r'(?:bell\s*curve|curved)\s+(?:heavily|a\s+lot|significantly)', 1.5, "Heavy bell curve"),
    ]
    easy_patterns = [
        (r'(?:easy|simple)\s+to\s+(?:score|get|achieve)\s+(?:a\s+)?(?:good|high|a\b|a\+)', -2.0, "Easy to score well"),
        (r'(?:high|good)\s+(?:average|median|mean)', -1.5, "High average grade"),
        (r'most\s+(?:people|students)\s+(?:got|scored|received)\s+(?:a|a\+|good)', -1.5, "Most students did well"),
    ]
    for pat, score, desc in hard_patterns + easy_patterns:
        if re.search(pat, text):
            return score, desc
    return 0, None


def rule_difficulty_adj_academic_noun(text):
    """
    Check if difficulty adjectives appear near academic nouns.
    "brutal exams" → hard signal
    "brutal parking" → ignored
    """
    diff_adjs = r'(?:hard|difficult|tough|brutal|challenging|intense|heavy|overwhelming|insane|crazy|killer|impossible|tricky|complicated|confusing)'
    easy_adjs = r'(?:easy|simple|straightforward|manageable|light|chill|doable|relaxed)'
    academic = r'(?:exam|midterm|final|quiz|assignment|homework|hw|pset|problem\s*set|lab|project|workload|content|material|module|mod|course|paper|tutorial|lecture|topic|concept)'

    hard_score = 0
    easy_score = 0
    explanations = []

    # [adj] [academic] within 3 words
    for m in re.finditer(rf'({diff_adjs})\s+(?:\w+\s+){{0,3}}({academic})', text):
        hard_score += 1.5
        explanations.append(f'"{m.group(1)}" → "{m.group(2)}"')

    # [academic] ... [adj]
    for m in re.finditer(rf'({academic})\s+(?:\w+\s+){{0,3}}(?:is|are|was|were)\s+(?:\w+\s+){{0,2}}({diff_adjs})', text):
        hard_score += 1.5
        explanations.append(f'"{m.group(1)}" is "{m.group(2)}"')

    # Same for easy adjectives
    for m in re.finditer(rf'({easy_adjs})\s+(?:\w+\s+){{0,3}}({academic})', text):
        easy_score += 1.5
        explanations.append(f'"{m.group(1)}" → "{m.group(2)}"')

    for m in re.finditer(rf'({academic})\s+(?:\w+\s+){{0,3}}(?:is|are|was|were)\s+(?:\w+\s+){{0,2}}({easy_adjs})', text):
        easy_score += 1.5
        explanations.append(f'"{m.group(1)}" is "{m.group(2)}"')

    net = hard_score - easy_score
    if explanations:
        return net, "Adj-noun pairs: " + "; ".join(explanations)
    return 0, None


def rule_explicit_rating(text):
    """Detect explicit difficulty ratings like "difficulty: 4/5" or "8/10 difficulty"."""
    patterns = [
        r'(?:difficulty|workload)\s*[:=]\s*(\d+)\s*/\s*(\d+)',
        r'(\d+)\s*/\s*(\d+)\s+(?:difficulty|workload|for\s+difficulty)',
        r'(?:difficulty|workload)\s+(?:is\s+)?(?:about\s+)?(\d+)\s*/\s*(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            num, denom = int(m.group(1)), int(m.group(2))
            if denom > 0:
                ratio = num / denom
                if ratio >= 0.7:
                    return 2.5, f"Explicit rating: {num}/{denom} (hard)"
                elif ratio <= 0.3:
                    return -2.5, f"Explicit rating: {num}/{denom} (easy)"
                else:
                    return 0.5, f"Explicit rating: {num}/{denom} (moderate)"
    return 0, None


def rule_negation(text):
    """Handle negation patterns that flip meaning."""
    neg_hard_patterns = [
        (r'(?:not|isn\'t|wasn\'t|aren\'t|weren\'t)\s+(?:that\s+)?(?:hard|difficult|tough|challenging)', -1.5, "Negated difficulty"),
        (r'(?:don\'t|didn\'t|doesn\'t)\s+(?:find\s+it\s+)?(?:hard|difficult|tough)', -1.5, "Negated difficulty"),
        (r'(?:wouldn\'t|won\'t)\s+(?:say|call)\s+(?:it\s+)?(?:hard|difficult)', -1.5, "Negated difficulty"),
    ]
    neg_easy_patterns = [
        (r'(?:not|isn\'t|wasn\'t)\s+(?:that\s+)?(?:easy|simple|straightforward)', 1.0, "Negated easiness"),
        (r'(?:don\'t|didn\'t)\s+(?:think|find)\s+(?:it\s+)?(?:easy|simple)', 1.0, "Negated easiness"),
    ]
    for pat, score, desc in neg_hard_patterns + neg_easy_patterns:
        if re.search(pat, text):
            return score, desc
    return 0, None


# Collect all rules
ALL_RULES = [
    ("time_spent", rule_time_spent),
    ("workload_qty", rule_workload_quantity),
    ("grade_difficulty", rule_grade_difficulty),
    ("adj_noun", rule_difficulty_adj_academic_noun),
    ("explicit_rating", rule_explicit_rating),
    ("negation", rule_negation),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(reviews):
    """Run all rules on each review. Returns list of results."""
    results = []
    for rev in reviews:
        msg = rev.get("message", "")

        if rev.get("_too_short", False):
            results.append(make_result(
                rev.get("post_id", ""), rev.get("author", ""),
                msg, "neutral", 0.0, "rules", "Too short to score"
            ))
            continue

        text_lower = msg.lower()
        total_score = 0.0
        fired_rules = []

        for rule_name, rule_fn in ALL_RULES:
            score, explanation = rule_fn(text_lower)
            if score != 0 and explanation:
                total_score += score
                fired_rules.append(f"[{rule_name}] {explanation} ({score:+.1f})")

        if total_score >= 0.5:
            label = "hard"
        elif total_score <= -0.5:
            label = "easy"
        else:
            label = "neutral"

        confidence = min(abs(total_score) / 5.0, 1.0) if abs(total_score) >= 0.5 else abs(total_score) / 1.0

        if fired_rules:
            explanation = "; ".join(fired_rules)
        else:
            explanation = "No structural patterns matched"

        results.append(make_result(
            rev.get("post_id", ""), rev.get("author", ""),
            msg, label, confidence, "rules", explanation
        ))

    return results
