"""
run_demo.py — Quick demo with sample reviews
==============================================
Usage:  cd difficulty_analysis && python run_demo.py
"""

import json

SAMPLE_REVIEWS = [
    {"post_id": "1", "author": "Alice", "course_code": "CS2040",
     "message": "This module has a very steep learning curve. Spent about 15 hours per week on the assignments. The exams were brutal and the content is extremely heavy. Prepare to suffer."},
    {"post_id": "2", "author": "Bob", "course_code": "CS2040",
     "message": "Not that hard honestly. If you pay attention during lectures and start assignments early, it's very manageable. The content is straightforward if you have a good foundation."},
    {"post_id": "3", "author": "Charlie", "course_code": "CS2040",
     "message": "Prof Tan is amazing. Great teaching style. Would recommend."},
    {"post_id": "4", "author": "Diana", "course_code": "CS2040",
     "message": "The workload is heavy but the material is interesting. I struggled with the later topics but the TAs were helpful. Difficulty: 4/5."},
    {"post_id": "5", "author": "Eve", "course_code": "CS2040",
     "message": "Easy A if you've done competitive programming before. Light workload compared to other CS mods."},
    {"post_id": "6", "author": "Frank", "course_code": "CS2040",
     "message": "ok"},
    {"post_id": "7", "author": "Grace", "course_code": "CS2040",
     "message": "Took 20 hours every week just to keep up. The exams are insanely hard. Many students failed or dropped. This is the hardest module I've taken."},
    {"post_id": "8", "author": "Hank", "course_code": "CS2040",
     "message": "The parking near the lecture hall was terrible. Also the aircon was too cold. Content wise it was alright I guess."},
    {"post_id": "9", "author": "Ivy", "course_code": "CS2040",
     "message": "Pretty easy module if you attend lectures. The assignments are doable and the exam was quite straightforward. Don't believe the hype about it being killer."},
    {"post_id": "10", "author": "Jack", "course_code": "CS2040",
     "message": "Mixed feelings. Some topics were a breeze, others were a nightmare. The bell curve saved me but I wouldn't say it's easy."},
]

for r in SAMPLE_REVIEWS:
    r["_too_short"] = len(r["message"].strip()) < 15


def main():
    from shared import print_summary
    import method_keyword
    import method_vader
    import method_rules
    import method_ensemble

    print("=" * 65)
    print("  DEMO: Running all local methods on 10 sample reviews")
    print("=" * 65)

    for name, fn in [("keyword", method_keyword.run), ("vader", method_vader.run),
                     ("rules", method_rules.run)]:
        results = fn(SAMPLE_REVIEWS)
        print_summary(results, "CS2040", f"[{name}]")
        print()

    # Ensemble (local methods only)
    results = method_ensemble.run(SAMPLE_REVIEWS)
    print_summary(results, "CS2040", "[ensemble]")
    print()

    # Full evaluation
    print("\nRunning full evaluation...\n")
    from evaluate import run_evaluation
    run_evaluation(SAMPLE_REVIEWS)


if __name__ == "__main__":
    main()
