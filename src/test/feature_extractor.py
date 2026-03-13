"""
Feature Extractor
==================
Extracts prediction input features from course reviews.
If extraction fails for a field, it's set to None/NaN.
Prints extraction summary at the end.

Features:
  1. GPA / grades in similar courses
  2. Course characteristics
     a. Content type
     b. Prerequisites
     c. Delivery mode
     d. Attendance / Participation
     e. Assessment structure
     f. Course structure
  3. Year of study
  4. Skills / Prior Knowledge
  5. Instructor reputation (style, clarity, guidance)
  6. Expected effort / time
  7. Preferred assessment type

Usage:
  python feature_extractor.py                  # synthetic data
  python feature_extractor.py reviews.json     # your data
"""

import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# SYNTHETIC DATA
# =============================================================================

SYNTHETIC_REVIEWS = [
    {
        "review_id": "r001", "course_id": "CS201",
        "course_name": "Data Structures & Algorithms",
        "instructor": "Dr. Sarah Chen",
        "reviewer_year": 3, "reviewer_gpa": 3.6,
        "expected_grade": "A-",
        "message": (
            "This course was really challenging but extremely rewarding. "
            "Dr. Chen explains concepts clearly and the lectures are well-organized. "
            "The workload is heavy — expect 10-15 hours per week outside of class. "
            "Weekly coding assignments and two midterms plus a final project. "
            "Attendance isn't mandatory but you'll fall behind without going. "
            "Lectures are recorded which is a lifesaver. "
            "The content is super relevant if you want a software engineering career. "
            "Prerequisites like discrete math really help. Loved the problem-solving aspect."
        ),
    },
    {
        "review_id": "r002", "course_id": "CS201",
        "course_name": "Data Structures & Algorithms",
        "instructor": "Dr. Sarah Chen",
        "reviewer_year": 2, "reviewer_gpa": 3.2,
        "expected_grade": "B+",
        "message": (
            "Tough course. The midterms were brutal and the grading is strict. "
            "I spent about 12 hours a week on assignments alone. "
            "Dr. Chen is knowledgeable but sometimes goes too fast. "
            "Office hours were helpful though. No attendance policy which is nice. "
            "Having a strong math background is essential. "
            "The final project was interesting — we built a search engine. "
            "Definitely useful for interviews and job prep."
        ),
    },
    {
        "review_id": "r003", "course_id": "HIST105",
        "course_name": "Modern World History",
        "instructor": "Prof. James Liu",
        "reviewer_year": 1, "reviewer_gpa": 3.8,
        "expected_grade": "A",
        "message": (
            "Easy A if you do the readings. Prof. Liu is an amazing lecturer — "
            "super engaging and makes history come alive. "
            "Workload is light, maybe 3-4 hours a week. Two essays and a final exam. "
            "Attendance is mandatory and counts for 10% of your grade. "
            "No recorded lectures unfortunately. The class meets MWF at 9am which is rough. "
            "Great for fulfilling gen-ed requirements but also genuinely fascinating. "
            "I'm a freshman and found it very accessible with no prerequisites needed."
        ),
    },
    {
        "review_id": "r004", "course_id": "HIST105",
        "course_name": "Modern World History",
        "instructor": "Prof. James Liu",
        "reviewer_year": 2, "reviewer_gpa": 2.9,
        "expected_grade": "B",
        "message": (
            "Took this as a gen-ed. It was okay but not my thing. "
            "Prof. Liu is a good speaker but the essay grading felt subjective. "
            "Attendance is strictly taken. Early morning classes were painful. "
            "Not much relevance to my business major but interesting enough. "
            "About 5 hours of work per week. The final was memorization-heavy. "
            "Flexible deadline policy for essays was a plus."
        ),
    },
    {
        "review_id": "r005", "course_id": "BIO310",
        "course_name": "Molecular Biology",
        "instructor": "Dr. Priya Patel",
        "reviewer_year": 3, "reviewer_gpa": 3.4,
        "expected_grade": "B+",
        "message": (
            "Very demanding course with a LOT of content to memorize. "
            "Dr. Patel is brilliant but her lectures can be dry and hard to follow. "
            "Expect 15+ hours a week if you want to do well. "
            "Three exams, weekly quizzes, and a lab component. "
            "Attendance is mandatory for labs. Lectures are online and asynchronous "
            "which is nice for flexibility. Essential for pre-med students. "
            "You need a solid foundation in gen chem and intro bio. "
            "The lab work was actually the most enjoyable part."
        ),
    },
    {
        "review_id": "r006", "course_id": "PHIL200",
        "course_name": "Ethics and Society",
        "instructor": "Dr. Maria Santos",
        "reviewer_year": 4, "reviewer_gpa": 3.1,
        "expected_grade": "A-",
        "message": (
            "One of the best courses I've taken. Dr. Santos creates such a welcoming "
            "discussion environment. The workload is moderate — weekly reflection papers "
            "and a final essay. Maybe 6 hours a week. No exams! "
            "Completely in-person, no recordings. Participation is a big part of the grade. "
            "Schedule was T/Th afternoons which worked perfectly. "
            "Really changed how I think about ethical dilemmas. "
            "Highly recommend regardless of your major. No prerequisites required."
        ),
    },
    {
        "review_id": "r007", "course_id": "MATH401",
        "course_name": "Real Analysis",
        "instructor": "Prof. Viktor Novak",
        "reviewer_year": 3, "reviewer_gpa": 3.7,
        "expected_grade": "B",
        "message": (
            "Hardest course I've ever taken. The proofs are incredibly abstract and "
            "Prof. Novak assumes you already know everything. Office hours are useless — "
            "he just restates the theorem. Spent 20 hours a week minimum. "
            "Weekly problem sets that take forever. Two brutal exams. "
            "No attendance policy, lectures not recorded. "
            "You absolutely need linear algebra and advanced calculus first. "
            "Important for grad school but painful. The subject itself is beautiful "
            "once you understand it, which takes a long time."
        ),
    },
    {
        "review_id": "r008", "course_id": "CS201",
        "course_name": "Data Structures & Algorithms",
        "instructor": "Dr. Sarah Chen",
        "reviewer_year": 2, "reviewer_gpa": 2.8,
        "expected_grade": "C+",
        "message": (
            "I struggled a lot in this class. The pace is way too fast and "
            "if you don't have strong coding skills you'll drown. "
            "I wish there was more support for students who aren't CS majors. "
            "The TAs were sometimes helpful but overwhelmed. "
            "Probably 8-10 hours a week of frustrating debugging. "
            "The recorded lectures were the only thing that saved me. "
            "I can see how it's useful but it was just too hard for me. "
            "Dreaded going to class every week."
        ),
    },
]


# =============================================================================
# EXTRACTOR
# =============================================================================

class FeatureExtractor:

    @staticmethod
    def _search(text: str, patterns: list[str]) -> list[str]:
        t = text.lower()
        return [p for p in patterns if p in t]

    # --- Individual extractors: return value or None ---

    def gpa(self, structured_gpa: float | None = None) -> float | None:
        """GPA almost always comes from structured data."""
        return structured_gpa

    def year_of_study(self, text: str, structured_year: int | None = None) -> int | None:
        if structured_year is not None:
            return structured_year
        year_map = {
            "freshman": 1, "first year": 1, "year 1": 1, "1st year": 1,
            "sophomore": 2, "second year": 2, "year 2": 2, "2nd year": 2,
            "junior": 3, "third year": 3, "year 3": 3, "3rd year": 3,
            "senior": 4, "fourth year": 4, "year 4": 4, "4th year": 4,
            "grad student": 5, "graduate": 5, "masters": 5, "phd": 6,
        }
        t = text.lower()
        for label, year in year_map.items():
            if label in t:
                return year
        return None

    def course_content_type(self, text: str) -> str | None:
        """Dominant content type: theory, practical, memorization, discussion, problem_solving."""
        scores = {
            "theory": len(self._search(text, [
                "theory", "theoretical", "abstract", "proofs", "conceptual", "mathematical",
            ])),
            "practical": len(self._search(text, [
                "hands-on", "practical", "applied", "real-world", "coding",
                "programming", "lab", "build", "implement",
            ])),
            "memorization": len(self._search(text, [
                "memorization", "memorize", "rote", "content heavy", "lot of content",
            ])),
            "discussion": len(self._search(text, [
                "discussion", "debate", "seminar", "reading-based", "essays", "writing",
            ])),
            "problem_solving": len(self._search(text, [
                "problem solving", "problem-solving", "analytical", "critical thinking",
            ])),
        }
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return None
        return best

    def prerequisites_mentioned(self, text: str) -> bool | None:
        cues = self._search(text, [
            "prerequisite", "prereq", "need to know", "background in",
            "foundation", "prior course", "should take", "essential",
            "must know", "you need", "strong background", "need a solid",
            "no prerequisites", "no prereq",
        ])
        if not cues:
            return None
        # "no prerequisites" → False, otherwise True
        if self._search(text, ["no prerequisites", "no prereq"]):
            return False
        return True

    def delivery_mode(self, text: str) -> str | None:
        online = self._search(text, ["online", "remote", "virtual", "zoom", "asynchronous"])
        inperson = self._search(text, ["in-person", "in person", "on campus", "face to face", "classroom"])
        if self._search(text, ["hybrid", "blended"]) or (online and inperson):
            return "hybrid"
        if online:
            return "online"
        if inperson:
            return "in_person"
        return None

    def attendance_required(self, text: str) -> bool | None:
        if self._search(text, [
            "attendance mandatory", "attendance is mandatory",
            "required attendance", "strictly taken", "must attend",
            "attendance counts",
        ]):
            return True
        if self._search(text, [
            "no attendance", "not mandatory", "isn't mandatory",
            "no attendance policy",
        ]):
            return False
        return None

    def participation_graded(self, text: str) -> bool | None:
        if self._search(text, [
            "participation", "class discussion", "big part of the grade",
            "counts for", "participation grade",
        ]):
            return True
        return None

    def assessment_types(self, text: str) -> list | None:
        """List of assessment types found, or None if nothing detected."""
        types_map = {
            "exams": r'\b(?:exam|exams|midterm|midterms|final exam)\b',
            "quizzes": r'\b(?:quiz|quizzes)\b',
            "essays": r'\b(?:essay|essays|paper|papers|reflection)\b',
            "projects": r'\b(?:project|projects|final project)\b',
            "assignments": r'\b(?:assignment|assignments|homework|problem set|pset|coding assignment)\b',
            "labs": r'\b(?:lab|labs|lab report|lab component)\b',
            "presentations": r'\b(?:presentation|presentations)\b',
        }
        t = text.lower()
        found = [atype for atype, pat in types_map.items() if re.search(pat, t)]
        return found if found else None

    def has_exams(self, text: str) -> bool | None:
        if re.search(r'\bno\s+exams?\b', text.lower()):
            return False
        if re.search(r'\b(?:exam|exams|midterm|midterms|final exam)\b', text.lower()):
            return True
        return None

    def course_organization(self, text: str) -> str | None:
        organized = self._search(text, [
            "well-organized", "well organized", "structured",
            "clear expectations", "well-structured",
        ])
        disorganized = self._search(text, [
            "disorganized", "all over the place", "no structure",
            "chaotic", "confusing structure",
        ])
        if organized and not disorganized:
            return "organized"
        if disorganized and not organized:
            return "disorganized"
        if organized and disorganized:
            return "mixed"
        return None

    def course_pacing(self, text: str) -> str | None:
        fast = self._search(text, ["fast", "too fast", "fast-paced", "rushed", "cramming"])
        slow = self._search(text, ["slow", "too slow", "drags", "boring pace"])
        if fast and not slow:
            return "fast"
        if slow and not fast:
            return "slow"
        return None

    def instructor_clarity(self, text: str) -> str | None:
        clear = self._search(text, [
            "clear", "explains well", "well-organized", "organized",
            "easy to follow",
        ])
        unclear = self._search(text, [
            "confusing", "unclear", "hard to follow", "goes too fast",
            "disorganized", "assumes you",
        ])
        if clear and not unclear:
            return "clear"
        if unclear and not clear:
            return "unclear"
        if clear and unclear:
            return "mixed"
        return None

    def instructor_engagement(self, text: str) -> str | None:
        engaging = self._search(text, [
            "engaging", "come alive", "interactive", "welcoming",
            "enthusiastic", "passionate", "amazing lecturer",
        ])
        boring = self._search(text, ["boring", "dry", "monotone", "dull"])
        if engaging and not boring:
            return "engaging"
        if boring and not engaging:
            return "boring"
        if engaging and boring:
            return "mixed"
        return None

    def instructor_helpfulness(self, text: str) -> str | None:
        helpful = self._search(text, [
            "helpful", "approachable", "responsive", "accessible",
            "guidance", "feedback", "supportive", "available",
        ])
        unhelpful = self._search(text, [
            "useless", "unhelpful", "never available", "doesn't help",
            "no feedback", "no support",
        ])
        if helpful and not unhelpful:
            return "helpful"
        if unhelpful and not helpful:
            return "unhelpful"
        if helpful and unhelpful:
            return "mixed"
        return None

    def teaching_style(self, text: str) -> str | None:
        if self._search(text, ["discussion", "socratic", "interactive", "collaborative", "seminar"]):
            return "discussion_based"
        if self._search(text, ["hands-on", "practical", "demo", "live coding"]):
            return "hands_on"
        if self._search(text, ["lectures", "lecturing"]):
            return "lecture_based"
        return None

    def effort_vs_expected(self, text: str) -> str | None:
        """Did the student spend more/less time than expected?"""
        over = self._search(text, [
            "more than expected", "way more", "too much time",
            "wasn't worth the time", "excessive",
        ])
        under = self._search(text, [
            "less than expected", "barely had to study",
            "minimal effort",
        ])
        matched = self._search(text, [
            "about what i expected", "reasonable", "fair workload",
        ])
        if over:
            return "more_than_expected"
        if under:
            return "less_than_expected"
        if matched:
            return "as_expected"
        return None

    def preferred_assessment(self, text: str) -> str | None:
        """Rarely extractable — look for explicit preference signals."""
        t = text.lower()
        for atype in ["project", "essay", "exam", "quiz", "lab", "presentation"]:
            if re.search(rf'(?:loved?|enjoyed?|liked?|best part).*{atype}', t):
                return atype
        if re.search(r'no\s+exams?!', t):
            return "no_exams_preferred"
        return None

    # --- Main ---

    def extract(self, review: dict) -> dict:
        """Extract all features from one review. Failed → None."""
        text = review.get("message", "")

        return {
            "review_id": review.get("review_id"),
            "course_id": review.get("course_id"),
            # Demographics
            "gpa": self.gpa(review.get("reviewer_gpa")),
            "year_of_study": self.year_of_study(text, review.get("reviewer_year")),
            # Course characteristics
            "course_content_type": self.course_content_type(text),
            "prerequisites_mentioned": self.prerequisites_mentioned(text),
            "delivery_mode": self.delivery_mode(text),
            "attendance_required": self.attendance_required(text),
            "participation_graded": self.participation_graded(text),
            "assessment_types": self.assessment_types(text),
            "has_exams": self.has_exams(text),
            "course_organization": self.course_organization(text),
            "course_pacing": self.course_pacing(text),
            # Instructor
            "instructor_clarity": self.instructor_clarity(text),
            "instructor_engagement": self.instructor_engagement(text),
            "instructor_helpfulness": self.instructor_helpfulness(text),
            "teaching_style": self.teaching_style(text),
            # Student-side
            "effort_vs_expected": self.effort_vs_expected(text),
            "preferred_assessment": self.preferred_assessment(text),
        }

    def extract_batch(self, reviews: list[dict]) -> pd.DataFrame:
        """Extract all reviews → DataFrame. Prints extraction summary."""
        rows = [self.extract(r) for r in reviews]
        df = pd.DataFrame(rows)

        n = len(df)
        feat_cols = [c for c in df.columns if c not in ("review_id", "course_id")]

        print(f"\n{'─'*60}")
        print(f"  FEATURE EXTRACTION SUMMARY ({n} reviews)")
        print(f"{'─'*60}")
        print(f"  {'Feature':<28} {'Extracted':>10} {'Rate':>8}")
        print(f"  {'─'*51}")

        for col in feat_cols:
            extracted = df[col].notna().sum()
            rate = extracted / n
            icon = "✅" if rate >= 0.5 else "⚠️" if rate >= 0.25 else "❌"
            print(f"  {icon} {col:<26} {extracted:>5}/{n:<4} {rate:>7.0%}")

        dropped = [c for c in feat_cols if df[c].notna().sum() / n < 0.1]
        if dropped:
            print(f"\n  Columns with <10% extraction (consider dropping):")
            for c in dropped:
                print(f"    → {c}")

        print(f"{'─'*60}\n")
        return df


# =============================================================================
# TF-IDF (optional, for downstream ML)
# =============================================================================

def build_tfidf(reviews: list[dict], max_features: int = 200) -> pd.DataFrame:
    """Build TF-IDF matrix from review texts. Useful as extra model features."""
    texts = [r.get("message", "") for r in reviews]
    ids = [r.get("review_id", f"r{i}") for i, r in enumerate(reviews)]
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(texts)
    return pd.DataFrame(matrix.toarray(), columns=tfidf.get_feature_names_out(), index=ids)


# =============================================================================
# MAIN
# =============================================================================

def main():
    reviews = SYNTHETIC_REVIEWS
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            reviews = json.load(f)
        print(f"Loaded {len(reviews)} reviews from {sys.argv[1]}")

    extractor = FeatureExtractor()
    df = extractor.extract_batch(reviews)

    # Optional TF-IDF
    tfidf_df = build_tfidf(reviews, max_features=50)

    out = "/mnt/user-data/outputs"
    df.to_csv(f"{out}/features_extracted.csv", index=False)
    df.to_json(f"{out}/features_extracted.json", orient="records", indent=2)
    tfidf_df.to_csv(f"{out}/tfidf_features.csv")
    print(f"✓ Saved features_extracted.csv")
    print(f"✓ Saved features_extracted.json")
    print(f"✓ Saved tfidf_features.csv")
    print(f"\nSample rows:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
