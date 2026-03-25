"""
method_llm.py — LLM-based classification via OpenRouter API
=============================================================
Sends each review to an LLM with a structured prompt asking for
a difficulty label + confidence + explanation.

Supports any model available on OpenRouter:
  - openai/gpt-4o-mini          (cheapest, good enough)
  - anthropic/claude-3.5-haiku   (fast, cheap)
  - google/gemini-flash-1.5      (very cheap)

Prerequisites:
  pip install requests  (or use urllib, included in stdlib)

Setup:
  1. Get an API key from https://openrouter.ai/keys
  2. Pass via --api-key or set OPENROUTER_API_KEY env var

Usage:
  python run_single.py reviews.json --method llm --api-key YOUR_KEY
  python run_single.py reviews.json --method llm --api-key YOUR_KEY --model openai/gpt-4o-mini

Pros:  Most accurate, understands context/sarcasm/implicit signals
Cons:  Costs money (tiny at <100 reviews), requires API key, slower
"""

import json
import re
import time
import os
import urllib.request
from shared import make_result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are classifying university course reviews for DIFFICULTY sentiment.

For each review, determine:
1. label: "hard", "easy", or "neutral"
   - "hard" = reviewer found the course difficult, heavy workload, challenging content
   - "easy" = reviewer found the course easy, light workload, manageable
   - "neutral" = review doesn't clearly express difficulty sentiment, or is mixed/ambiguous
2. confidence: 0.0 to 1.0 (how certain you are)
3. explanation: one sentence explaining your reasoning

Respond ONLY with valid JSON, no markdown fences, no preamble:
{"label": "hard|easy|neutral", "confidence": 0.0, "explanation": "reason"}"""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_openrouter(message, api_key, model=DEFAULT_MODEL):
    """
    Send a review to OpenRouter for classification.
    Returns (label, confidence, explanation).
    """
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this course review:\n\n\"{message[:600]}\""},
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        parsed = json.loads(text)

        label = parsed.get("label", "neutral")
        if label not in ("hard", "easy", "neutral"):
            label = "neutral"
        confidence = float(parsed.get("confidence", 0.5))
        explanation = parsed.get("explanation", "LLM classification")

        return label, confidence, explanation

    except Exception as e:
        return "neutral", 0.0, f"LLM error: {e}"


def _call_anthropic(message, api_key, model="claude-haiku-4-5-20251001"):
    """
    Alternative: call Anthropic API directly (if you have an Anthropic key).
    Returns (label, confidence, explanation).
    """
    body = json.dumps({
        "model": model,
        "max_tokens": 150,
        "messages": [
            {"role": "user", "content": SYSTEM_PROMPT + f"\n\nClassify this course review:\n\n\"{message[:600]}\""},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data["content"][0]["text"].strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        parsed = json.loads(text)

        label = parsed.get("label", "neutral")
        if label not in ("hard", "easy", "neutral"):
            label = "neutral"
        confidence = float(parsed.get("confidence", 0.5))
        explanation = parsed.get("explanation", "LLM classification")

        return label, confidence, explanation

    except Exception as e:
        return "neutral", 0.0, f"LLM error: {e}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(reviews, api_key=None, model=DEFAULT_MODEL, provider="openrouter"):
    """
    Run LLM classification on all reviews.

    Args:
        reviews: list of review dicts
        api_key: API key (OpenRouter or Anthropic)
        model: model string (e.g. "openai/gpt-4o-mini" for OpenRouter)
        provider: "openrouter" or "anthropic"

    Returns list of results.
    """
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key provided. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY, "
            "or pass api_key parameter."
        )

    call_fn = _call_anthropic if provider == "anthropic" else _call_openrouter
    results = []

    for i, rev in enumerate(reviews):
        msg = rev.get("message", "")

        if rev.get("_too_short", False):
            results.append(make_result(
                rev.get("post_id", ""), rev.get("author", ""),
                msg, "neutral", 0.0, "llm", "Too short to score"
            ))
            continue

        print(f"  LLM [{i+1}/{len(reviews)}] ({model})...", end=" ", flush=True)

        if provider == "anthropic":
            label, confidence, explanation = call_fn(msg, api_key, model)
        else:
            label, confidence, explanation = call_fn(msg, api_key, model)

        print(f"{label} ({confidence:.2f})")

        results.append(make_result(
            rev.get("post_id", ""), rev.get("author", ""),
            msg, label, confidence, f"llm_{model.split('/')[-1]}",
            explanation
        ))
        time.sleep(0.3)  # Rate limiting

    return results
