"""
evaluate.py — Evaluation framework for all difficulty sentiment methods
========================================================================
Compares methods against each other and optionally against ground-truth labels.

Metrics:
  - Agreement rate between method pairsdisa
  - Coverage: % of reviews that got a non-neutral label
  - Confidence distribution
  - Disagreement audit: reviews where methods disagree (with full explanations)
  - Confusion matrix with precision/recall/F1 (if ground truth provided)

Usage:
  # Compare local methods only (keyword, vader, rules, ensemble)
  python evaluate.py data/reviews/CS2040_reviews.json

  # Include zero-shot transformer (requires torch + transformers)
  python evaluate.py data/reviews/CS2040_reviews.json --zeroshot

  # Include LLM via OpenRouter
  python evaluate.py data/reviews/CS2040_reviews.json --llm --api-key YOUR_KEY

  # Include everything
  python evaluate.py data/reviews/CS2040_reviews.json --zeroshot --llm --api-key YOUR_KEY

  # Compare against ground truth
  python evaluate.py data/reviews/CS2040_reviews.json --ground-truth labels.json

  # Save all results
  python evaluate.py data/reviews/CS2040_reviews.json --output results/
"""

import json
import argparse
import os
from collections import Counter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars

from shared import load_reviews, summarize
import method_keyword
import method_vader
import method_rules
import method_ensemble


# ============================================================================
# Metrics
# ============================================================================

def agreement_rate(results_a, results_b):
    """% of reviews where two methods give the same label (excluding too-short)."""
    agree, total = 0, 0
    for a, b in zip(results_a, results_b):
        if a["explanation"] == "Too short to score":
            continue
        total += 1
        if a["label"] == b["label"]:
            agree += 1
    return round(agree / total * 100, 1) if total else 0.0


def coverage(results):
    """% of reviews that got a non-neutral label."""
    scorable = [r for r in results if r["explanation"] != "Too short to score"]
    if not scorable:
        return 0.0
    opinionated = sum(1 for r in scorable if r["label"] != "neutral")
    return round(opinionated / len(scorable) * 100, 1)


def avg_confidence(results):
    """Average confidence across scorable reviews."""
    scorable = [r for r in results if r["explanation"] != "Too short to score"]
    if not scorable:
        return 0.0
    return round(sum(r["confidence"] for r in scorable) / len(scorable), 3)


def find_disagreements(all_results, method_names):
    """Find reviews where not all methods agree."""
    disagreements = []
    n = len(list(all_results.values())[0])
    for i in range(n):
        first_method = list(all_results.keys())[0]
        if all_results[first_method][i]["explanation"] == "Too short to score":
            continue
        labels = [all_results[name][i]["label"] for name in method_names]
        if len(set(labels)) > 1:
            row = {
                "post_id": all_results[first_method][i]["post_id"],
                "author": all_results[first_method][i]["author"],
                "message": all_results[first_method][i]["message_full"],
            }
            for name in method_names:
                row[f"{name}_label"] = all_results[name][i]["label"]
                row[f"{name}_conf"] = all_results[name][i]["confidence"]
                row[f"{name}_expl"] = all_results[name][i]["explanation"]
            disagreements.append(row)
    return disagreements


def confusion_matrix(results, ground_truth):
    """
    Build confusion matrix: predicted vs actual.
    ground_truth: dict mapping post_id → label
    """
    labels = ["hard", "neutral", "easy"]
    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}
    correct, total = 0, 0

    for r in results:
        pid = r["post_id"]
        if pid not in ground_truth:
            continue
        actual = ground_truth[pid]
        pred = r["label"]
        if actual not in labels or pred not in labels:
            continue
        matrix[actual][pred] += 1
        total += 1
        if actual == pred:
            correct += 1

    accuracy = round(correct / total * 100, 1) if total else 0.0

    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[a][label] for a in labels if a != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(matrix[label].values()),
        }

    return {"matrix": matrix, "accuracy": accuracy, "total": total, "per_class": per_class}


# ============================================================================
# Main evaluation
# ============================================================================

def load_results_from_dir(results_dir):
    """Load previously saved method results from a directory."""
    all_results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.json"):
            method_name = fname[: -len("_results.json")]
            path = os.path.join(results_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                all_results[method_name] = json.load(f)
            print(f"  Loaded {method_name} from {fname}")
    return all_results


def run_evaluation(reviews, ground_truth=None,
                   include_zeroshot=False, include_llm=False,
                   llm_api_key=None, llm_model=None, llm_provider="openrouter",
                   load_from=None):
    """Run all requested methods and produce comparison report.

    If load_from is set (path to a results directory), previously saved
    method results are loaded from there instead of re-running tagging.
    Methods whose result files are missing will still be run fresh.
    """

    all_results = {}

    if load_from:
        print("\n" + "=" * 65)
        print(f"  LOADING RESULTS FROM {load_from}")
        print("=" * 65 + "\n")
        all_results = load_results_from_dir(load_from)

    # Determine what still needs to be run
    missing_local = [m for m in ("keyword", "vader", "rules") if m not in all_results]
    missing_zeroshot = include_zeroshot and "zeroshot" not in all_results
    missing_llm = include_llm and llm_api_key and "llm" not in all_results
    missing_ensemble = "ensemble" not in all_results

    needs_run = missing_local or missing_zeroshot or missing_llm or missing_ensemble

    if needs_run:
        print("\n" + "=" * 65)
        print("  RUNNING METHODS")
        print("=" * 65)

    # --- Always-available methods ---
    for method_name, method_module in [("keyword", method_keyword), ("vader", method_vader), ("rules", method_rules)]:
        if method_name not in all_results:
            print(f"\n  Running {method_name}...")
            all_results[method_name] = method_module.run(reviews)

    # --- Optional: zero-shot ---
    if missing_zeroshot:
        try:
            import method_zeroshot
            print("  Running zero-shot transformer...")
            all_results["zeroshot"] = method_zeroshot.run(reviews)
        except ImportError as e:
            print(f"  Skipping zero-shot (missing dependency: {e})")
            print("  Install with: pip install transformers torch")

    # --- Optional: LLM ---
    if missing_llm:
        try:
            import method_llm
            print(f"  Running LLM ({llm_model or 'openai/gpt-4o-mini'})...")
            kwargs = {"api_key": llm_api_key, "provider": llm_provider}
            if llm_model:
                kwargs["model"] = llm_model
            all_results["llm"] = method_llm.run(reviews, **kwargs)
        except Exception as e:
            print(f"  LLM method failed: {e}")

    # --- Ensemble (uses whatever methods are available) ---
    if missing_ensemble:
        print("  Running ensemble...")
        all_results["ensemble"] = method_ensemble.run(
            reviews,
            include_zeroshot=include_zeroshot,
            include_llm=include_llm,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_provider=llm_provider,
        )

    names = list(all_results.keys())

    # ---- Agreement Matrix ----
    print("\n" + "=" * 65)
    print("  AGREEMENT MATRIX (% reviews with same label)")
    print("=" * 65)

    col_width = max(len(n) for n in names) + 2
    print(f"\n  {'':>{col_width}}", end="")
    for n in names:
        print(f"{n:>{col_width}}", end="")
    print()

    for n1 in names:
        print(f"  {n1:>{col_width}}", end="")
        for n2 in names:
            if n1 == n2:
                print(f"{'—':>{col_width}}", end="")
            else:
                rate = agreement_rate(all_results[n1], all_results[n2])
                print(f"{rate:>{col_width - 1}.1f}%", end="")
        print()

    # ---- Per-Method Stats ----
    print("\n" + "=" * 65)
    print("  PER-METHOD STATS")
    print("=" * 65)

    print(f"\n  {'Method':12s} {'Score':>7s} {'Hard%':>7s} {'Neut%':>7s} {'Easy%':>7s} {'Cover%':>8s} {'AvgConf':>8s}")
    print(f"  {'─' * 62}")

    for name in names:
        results = all_results[name]
        s = summarize(results)
        cov = coverage(results)
        ac = avg_confidence(results)
        print(f"  {name:12s} {s['difficulty_score']:>6.2f} {s['percentages']['hard']:>6.1f}% "
              f"{s['percentages']['neutral']:>6.1f}% {s['percentages']['easy']:>6.1f}% "
              f"{cov:>7.1f}% {ac:>7.3f}")

    # ---- Disagreements ----
    disagreements = find_disagreements(all_results, names)
    scorable = sum(1 for r in all_results[names[0]] if r["explanation"] != "Too short to score")

    print(f"\n" + "=" * 65)
    print(f"  DISAGREEMENTS: {len(disagreements)}/{scorable} reviews")
    print("=" * 65)

    # if disagreements:
    #     for d in disagreements[:8]:
    #         print(f"\n  [{d['author']}] {d['message_preview']}")
    #         for n in names:
    #             label_key = f"{n}_label"
    #             conf_key = f"{n}_conf"
    #             expl_key = f"{n}_expl"
    #             if label_key in d:
    #                 print(f"    {n:12s}: {d[label_key]:8s} ({d[conf_key]})  {d[expl_key][:65]}")
    #     if len(disagreements) > 8:
    #         print(f"\n  ... and {len(disagreements) - 8} more disagreements")

    # ---- Ground Truth Evaluation ----
    gt_results = {}
    if ground_truth:
        print(f"\n" + "=" * 65)
        print(f"  GROUND TRUTH EVALUATION ({len(ground_truth)} labeled reviews)")
        print("=" * 65)

        for name in names:
            results = all_results[name]
            cm = confusion_matrix(results, ground_truth)
            gt_results[name] = cm

            print(f"\n  --- {name} ---")
            print(f"  Accuracy: {cm['accuracy']}% ({cm['total']} reviews)")
            print(f"\n  {'':12s} {'Pred:hard':>10s} {'Pred:neut':>10s} {'Pred:easy':>10s}")
            for actual in ["hard", "neutral", "easy"]:
                row = cm["matrix"][actual]
                print(f"  Act:{actual:7s} {row['hard']:>10d} {row['neutral']:>10d} {row['easy']:>10d}")
            print(f"\n  {'Class':8s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Support':>8s}")
            for label in ["hard", "neutral", "easy"]:
                pc = cm["per_class"][label]
                print(f"  {label:8s} {pc['precision']:>7.3f} {pc['recall']:>7.3f} {pc['f1']:>7.3f} {pc['support']:>8d}")

    return {
        "all_results": all_results,
        "disagreements": disagreements,
        "ground_truth_results": gt_results,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate difficulty sentiment methods")
    parser.add_argument("input", help="Path to reviews JSON")
    parser.add_argument("--ground-truth", default=None, help="Path to labels JSON: {post_id: label}")
    parser.add_argument("--zeroshot", action="store_true", help="Include zero-shot transformer")
    parser.add_argument("--llm", action="store_true", help="Include LLM classification")
    parser.add_argument("--api-key", default=None, help="API key for LLM (OpenRouter or Anthropic)")
    parser.add_argument("--model", default=None, help="LLM model (e.g. openai/gpt-4o-mini)")
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "anthropic"])
    parser.add_argument("--output", default=None, help="Save results to this directory")
    parser.add_argument("--load-results", default=None, metavar="DIR",
                        help="Load previously saved results from this directory instead of re-running")
    args = parser.parse_args()

    reviews = load_reviews(args.input)

    gt = None
    if args.ground_truth:
        with open(args.ground_truth, "r", encoding="utf-8") as f:
            gt = json.load(f)
        print(f"Loaded {len(gt)} ground-truth labels")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    eval_data = run_evaluation(
        reviews,
        ground_truth=gt,
        include_zeroshot=args.zeroshot,
        include_llm=args.llm,
        llm_api_key=api_key,
        llm_model=args.model,
        llm_provider=args.provider,
        load_from=args.load_results,
    )

    if args.output:
        os.makedirs(args.output, exist_ok=True)

        dis_path = os.path.join(args.output, "disagreements.json")
        with open(dis_path, "w", encoding="utf-8") as f:
            json.dump(eval_data["disagreements"], f, indent=2, ensure_ascii=False)

        for name, results in eval_data["all_results"].items():
            path = os.path.join(args.output, f"{name}_results.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if eval_data["ground_truth_results"]:
            gt_path = os.path.join(args.output, "ground_truth_comparison.json")
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(eval_data["ground_truth_results"], f, indent=2, ensure_ascii=False)

        print(f"\n  Saved all results to {args.output}/")


if __name__ == "__main__":
    import os
    main()
