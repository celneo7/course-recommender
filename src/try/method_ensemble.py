"""
method_ensemble.py — Weighted ensemble combining multiple methods
==================================================================
Runs available methods and combines via weighted majority vote.
Keyword gets the highest weight (most reliable for this domain).

Weights are configurable. Default:
  keyword:  3  (strong, transparent, good coverage)
  rules:    2  (precise when it fires)
  vader:    1  (weakest — general sentiment ≠ difficulty)
  zeroshot: 3  (if available — strong semantic understanding)
  llm:      4  (if available — most accurate)

Pros:  More robust than any single method; tunable
Cons:  Slower (runs all methods); complexity
"""

from collections import Counter
from shared import make_result

import method_keyword
import method_vader
import method_rules

# Default weights — higher = more trusted
DEFAULT_WEIGHTS = {
    "keyword": 3,
    "vader": 1,
    "rules": 2,
    "zeroshot": 3,
    "llm": 4,
}


def run(reviews, include_zeroshot=False, include_llm=False,
        llm_api_key=None, llm_model=None, llm_provider="openrouter",
        weights=None):
    """
    Run available methods and combine via weighted vote.

    Args:
        reviews: list of review dicts
        include_zeroshot: if True, include zero-shot transformer method
        include_llm: if True, include LLM API method
        llm_api_key: API key for LLM method
        llm_model: model string for LLM method
        llm_provider: "openrouter" or "anthropic"
        weights: dict overriding DEFAULT_WEIGHTS

    Returns list of results.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Always run these three (no external deps beyond pip)
    method_results = {
        "keyword": method_keyword.run(reviews),
        "vader": method_vader.run(reviews),
        "rules": method_rules.run(reviews),
    }

    # Optionally include zero-shot
    if include_zeroshot:
        try:
            import method_zeroshot
            print("  Ensemble: running zero-shot transformer...")
            method_results["zeroshot"] = method_zeroshot.run(reviews)
        except ImportError:
            print("  Ensemble: zero-shot unavailable (need transformers + torch)")

    # Optionally include LLM
    if include_llm and llm_api_key:
        try:
            import method_llm
            print("  Ensemble: running LLM classification...")
            kwargs = {"api_key": llm_api_key, "provider": llm_provider}
            if llm_model:
                kwargs["model"] = llm_model
            method_results["llm"] = method_llm.run(reviews, **kwargs)
        except Exception as e:
            print(f"  Ensemble: LLM method failed: {e}")

    # Combine via weighted vote
    active_methods = list(method_results.keys())
    results = []

    for idx in range(len(reviews)):
        rev = reviews[idx]
        msg = rev.get("message", "")

        # Check if too short
        if method_results["keyword"][idx]["explanation"] == "Too short to score":
            results.append(make_result(
                rev.get("post_id", ""), rev.get("author", ""),
                msg, "neutral", 0.0, "ensemble", "Too short to score"
            ))
            continue

        # Gather votes with weights
        weighted_votes = {"hard": 0, "neutral": 0, "easy": 0}
        vote_details = []

        for method_name in active_methods:
            r = method_results[method_name][idx]
            label = r["label"]
            conf = r["confidence"]
            method_weight = w.get(method_name, 1)

            # Weight = base_weight * confidence (so low-confidence votes count less)
            effective_weight = method_weight * (0.3 + 0.7 * conf)  # floor at 30% of weight
            weighted_votes[label] += effective_weight
            vote_details.append(f"{method_name}={label}({conf})")

        # Pick label with highest weighted vote
        label = max(weighted_votes, key=weighted_votes.get)
        total_weight = sum(weighted_votes.values())

        # Confidence: proportion of weight that went to winning label
        if total_weight > 0:
            confidence = weighted_votes[label] / total_weight
        else:
            confidence = 0.0

        explanation = (
            f"Weighted vote [{label}={weighted_votes[label]:.1f}, "
            f"others={total_weight - weighted_votes[label]:.1f}]: "
            + ", ".join(vote_details)
        )

        results.append(make_result(
            rev.get("post_id", ""), rev.get("author", ""),
            msg, label, round(confidence, 2), "ensemble", explanation
        ))

    return results
