# NUSMods Difficulty Sentiment Analysis

## Overview

This project analyzes course reviews scraped from NUSMods to determine perceived difficulty. Six methods are implemented (four local, two API-based), plus an evaluation framework to compare them.

---

## Project Structure

```
difficulty_analysis/
  shared.py              — Common utilities, output format, loading
  method_keyword.py      — Method 1: Weighted keyword lexicon
  method_vader.py        — Method 2: VADER + academic context filtering (tuned)
  method_rules.py        — Method 3: Structural pattern matching (tuned)
  method_ensemble.py     — Method 4: Weighted ensemble (configurable)
  method_zeroshot.py     — Method 5: Zero-shot transformer (requires torch)
  method_llm.py          — Method 6: LLM via OpenRouter / Anthropic API
  evaluate.py            — Evaluation framework (CLI)
  run_demo.py            — Quick demo with sample reviews
```

### Dependencies

```bash
# Required for all local methods
pip install vaderSentiment textblob scikit-learn

# Required for Method 5 (zero-shot) — run locally
pip install transformers torch

# Method 6 (LLM) uses stdlib urllib — no extra install needed
```

---

## Quick Start

```bash
cd difficulty_analysis

# Demo with sample reviews (no data file needed)
python run_demo.py

# Run evaluation on your actual reviews
python evaluate.py ../data/reviews/CS2040_reviews.json

# Include zero-shot transformer
python evaluate.py ../data/reviews/CS2040_reviews.json --zeroshot

# Include LLM via OpenRouter
python evaluate.py ../data/reviews/CS2040_reviews.json --llm --api-key YOUR_KEY

# Everything + ground truth
python evaluate.py ../data/reviews/CS2040_reviews.json --zeroshot --llm --api-key KEY --ground-truth labels.json

# Save all results
python evaluate.py ../data/reviews/CS2040_reviews.json --output results/
```

---

## Methods Implemented

### Method 1: Weighted Keyword Lexicon (`method_keyword.py`)

**How it works:** Two curated dictionaries (~60 entries each) of hard/easy phrases with weights (0.8–3.0). Phrases matched longest-first with overlap prevention — "not that hard" consumes the span so "hard" won't double-fire. Net score = hard_score - easy_score.

**Classification:** net >= 1.0 → hard, net <= -1.0 → easy, else neutral.

**Strengths:** Fully transparent (traces to exact phrases), zero cost, instant.
**Weaknesses:** Misses implicit difficulty, no sarcasm/context handling.

---

### Method 2: VADER + Academic Context Filtering (`method_vader.py`) — TUNED

**How it works:** Splits review into sentences, keeps only those with academic keywords, runs VADER on the filtered set. Maps negative academic sentiment → "hard".

**Tuning applied:**
- Lowered classification thresholds from ±0.3 → ±0.15 (was too conservative)
- Expanded academic keywords to include difficulty-adjacent terms ("hard", "hours", "study", "revision", "pyp")
- Injected difficulty-specific words into VADER's lexicon with appropriate weights (e.g. "brutal": -2.5, "nightmare": -3.0, "manageable": +1.5)

**Strengths:** Handles vocabulary beyond the keyword lexicon, academic filtering reduces noise.
**Weaknesses:** Negative sentiment ≠ difficult (a poorly taught easy course reads as negative). Still the weakest method for this specific task.

---

### Method 3: Structural Pattern Matching (`method_rules.py`) — TUNED

**How it works:** Six regex-based rule functions that look for structural patterns:

1. **Time spent:** "spent/took N hours" — N >= 20 strong hard, N <= 2 easy
2. **Workload quantity:** "3 assignments per week" patterns
3. **Grade difficulty:** "hard to score well", "many students failed", "low average"
4. **Difficulty adj + academic noun proximity:** "brutal exams" fires, "brutal parking" doesn't. Checks within 3-word window.
5. **Explicit ratings:** "difficulty: 4/5" patterns, mapped by ratio
6. **Negation:** "not hard", "didn't find it difficult" flip meaning

**Tuning applied:**
- Lowered classification threshold from 1.0 → 0.5 (rules fire rarely; when they do, even small scores are meaningful)

**Strengths:** Catches implicit signals (time, quantities, ratings) that keywords miss. Every classification traces to a specific rule.
**Weaknesses:** Brittle to phrasing variations. Informal text (Singlish, abbreviations) can dodge patterns.

---

### Method 4: Weighted Ensemble (`method_ensemble.py`) — REDESIGNED

**How it works:** Runs all available methods and combines via confidence-weighted voting. Each method has a base weight reflecting how much we trust it for this task:

| Method   | Base Weight | Rationale |
|----------|-------------|-----------|
| keyword  | 3           | Best coverage, reliable for this domain |
| rules    | 2           | Precise when it fires |
| vader    | 1           | Weakest — general sentiment ≠ difficulty |
| zeroshot | 3           | Strong semantic understanding (if available) |
| llm      | 4           | Most accurate (if available) |

Effective weight per vote = base_weight × (0.3 + 0.7 × confidence). This means a confident keyword vote (weight 3 × high confidence) outweighs an uncertain VADER vote (weight 1 × low confidence). The 0.3 floor prevents zero-confidence methods from being completely ignored.

**Strengths:** Tunable, adapts to which methods are available, robust.
**Weaknesses:** Complexity; behaviour depends on which methods are included.

---

### Method 5: Zero-Shot Transformer (`method_zeroshot.py`)

**How it works:** Uses `facebook/bart-large-mnli` via HuggingFace's `pipeline("zero-shot-classification")`. The model treats classification as natural language inference — it asks: "Does this review entail that the course is difficult?"

**Candidate labels:**
- "This course is very difficult and has heavy workload"
- "This course is easy and manageable"
- "This review does not discuss course difficulty"

Confidence is based on the margin between the top score and the second-best score.

**Requirements:** `pip install transformers torch` (~2GB download on first run for model + torch)

**Strengths:** Understands context, semantics, and implicit signals. No training data needed. Runs fully locally.
**Weaknesses:** Requires torch (large install). Slower than keyword/VADER (~1-2 sec per review on CPU). Less transparent than keyword methods.

---

### Method 6: LLM Classification (`method_llm.py`)

**How it works:** Sends each review to an LLM with a structured prompt requesting JSON output: `{"label": "hard|easy|neutral", "confidence": 0.0-1.0, "explanation": "reason"}`. Temperature set to 0 for deterministic output.

**Supported providers:**

| Provider | Models | Setup |
|----------|--------|-------|
| OpenRouter | `openai/gpt-4o-mini`, `google/gemini-flash-1.5`, `anthropic/claude-3.5-haiku` | Get key at openrouter.ai/keys |
| Anthropic | `claude-haiku-4-5-20251001` | Get key at console.anthropic.com |

```bash
# OpenRouter (default)
python evaluate.py reviews.json --llm --api-key YOUR_OPENROUTER_KEY --model openai/gpt-4o-mini

# Anthropic direct
python evaluate.py reviews.json --llm --api-key YOUR_ANTHROPIC_KEY --provider anthropic --model claude-haiku-4-5-20251001
```

**Cost estimate:** At < 100 reviews with GPT-4o-mini, total cost is typically under $0.01.

**Strengths:** Most accurate. Understands context, sarcasm, implicit signals. Provides natural-language explanations.
**Weaknesses:** Requires API key and internet. Costs money (tiny). Slower. Less reproducible (model updates may change results).

---

## Evaluation Framework (`evaluate.py`)

### Metrics Computed

**1. Agreement Matrix**
Pairwise agreement rate between all methods. High agreement (>80%) between independent methods suggests reliability.

**2. Per-Method Stats**
For each method: difficulty score (1-5), label distribution (hard/neutral/easy %), coverage (% non-neutral), and average confidence.

**3. Disagreement Audit**
Every review where methods disagree, shown with full explanations from each method. Reading these is the fastest way to understand where each method fails.

**4. Confusion Matrix (with ground truth)**
Per-method accuracy, and per-class precision/recall/F1. Requires a `labels.json` file.

### Ground Truth Format

```json
{
  "3071909979": "hard",
  "3071832456": "neutral",
  "3071755123": "easy"
}
```

### How to Judge If a Method Is Good Enough

- **Agreement > 80%** between keyword and one other method → consistent signal
- **Coverage 40-70%** → method has opinions without overcalling
- **Disagreements look reasonable** → when you read them, the winning method usually feels right
- **With ground truth: F1 > 0.6 per class** → reasonable; F1 > 0.8 → excellent

---

## Other Methods to Consider (Not Implemented)

### Local (No API Required)

**Sentence Embeddings + Cosine Similarity**
Use `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) to embed reviews and "anchor" sentences like "The workload was overwhelming." Classify by which anchor the review is closest to in vector space. Catches semantic similarity beyond keywords. Less transparent — similarity scores aren't as interpretable as keyword matches.

**SetFit (Few-Shot Fine-Tuning)**
Designed for small labeled datasets (8-16 examples). Fine-tunes a sentence transformer for your specific task. If you label 15-20 reviews, SetFit can produce a dedicated classifier that's fast and accurate. Install: `pip install setfit`.

**SVM / Logistic Regression with TF-IDF**
Classical ML. Train on labeled reviews with TF-IDF features (bigrams/trigrams give phrase-level understanding). With < 50 labels, use leave-one-out cross-validation. Fast at inference. Requires `scikit-learn` (already installed).

**Naive Bayes**
Simplest ML classifier. Works well on text with small training sets due to strong independence assumption acting as regularization. Pairs with bag-of-words or TF-IDF.

**spaCy Dependency Parsing**
Full syntactic analysis: parse the sentence tree, find which adjectives modify which nouns grammatically. "The exams were brutal" → `brutal` modifies `exams` → hard. "The parking was brutal" → `brutal` modifies `parking` → ignored. High precision but complex to implement. `pip install spacy && python -m spacy download en_core_web_sm`.

**SentiWordNet / AFINN**
Pre-scored word lists (SentiWordNet uses WordNet; AFINN is a flat list with -5 to +5 scores). More granular than a hand-built lexicon but same core limitation: no context awareness.

**Domain-Adapted Lexicon via PMI**
Learn the word list from data instead of hand-crafting it. Count word frequencies in reviews labeled hard vs easy; words with the biggest frequency difference become your automatic lexicon. Requires labeled data.

**TextBlob**
Similar to VADER — pre-trained sentiment analyzer. Returns polarity (-1 to +1) and subjectivity (0 to 1). Could filter by high subjectivity + negative polarity as a difficulty proxy. Same limitation as VADER (general sentiment ≠ difficulty).

### API-Based

**Hybrid: Keyword + LLM Fallback**
Run keyword first. Only send ambiguous reviews (near-zero net score) to the LLM. Keeps costs minimal while catching what keywords miss.

**LLM-as-Judge for Evaluation**
Use an LLM to label your ground-truth set instead of doing it manually. Faster but less reliable than human labels. Good for quickly generating a larger evaluation set.

**Weak Supervision (Snorkel)**
Write "labeling functions" (your keyword rules, VADER scores, regex patterns) and let Snorkel's generative model resolve conflicts probabilistically. Then train a small model on the resulting labels. Overkill at 50 reviews; powerful at 5,000+.

### Evaluation Methods to Consider

**Inter-Annotator Agreement (Cohen's Kappa)**
Have two people independently label the same reviews. Kappa score tells you the ceiling — if humans only agree 70% of the time, no method will reliably do better.

**Bootstrap Confidence Intervals**
Resample reviews with replacement 1000 times, compute difficulty score each time. The spread gives you a confidence interval accounting for sample size.

**Stratified Analysis**
Break results down by review length, date, or reply status. Long reviews may score differently than short ones. Recent reviews may reflect curriculum changes.

**Error Analysis by Category**
Categorize the disagreements: is the method failing on sarcasm? Implicit signals? Mixed reviews? Knowing the failure mode tells you exactly what to add next.

**Cross-Course Validation**
If you scrape multiple courses, tune on one course and test on another. Tests whether the methods generalize.

**Sensitivity Analysis**
Vary classification thresholds (e.g. keyword net score cutoff from 0.5 to 2.0) and observe how the distribution shifts. If small changes drastically alter results, the method is fragile at the boundary.
