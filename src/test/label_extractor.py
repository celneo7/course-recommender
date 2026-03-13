"""
Label Extractor
================
Extracts target variables from course reviews.
If extraction fails for a field, it's set to None/NaN.
At the end, prints a summary of what was extractable and what wasn't.

Labels:
  1. Expected Grade
  2. Perceived Difficulty
  3. Workload / Time Intensity
  4. Teaching Quality
  5. Relevance / Usefulness
  6. Interest / Enjoyment
  7. Schedule Timing
  8. Attendance Policy
  9. Recorded Lectures
  10. Flexibility
  11. Delivery Mode

Usage:
  python label_extractor.py                  # synthetic data
  python label_extractor.py reviews.json     # your data
"""

import json
import re
import sys

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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
        "review_text": (
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
        "review_text": (
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
        "review_text": (
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
        "review_text": (
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
        "review_text": (
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
        "review_text": (
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
        "review_text": (
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
        "review_text": (
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

class LabelExtractor:

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    @staticmethod
    def _search(text: str, patterns: list[str]) -> list[str]:
        t = text.lower()
        return [p for p in patterns if p in t]

    def _aspect_sentiment(self, text: str, keywords: list[str]) -> float | None:
        """VADER sentiment on sentences containing any keyword."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        relevant = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
        if not relevant:
            return None
        return round(self.vader.polarity_scores(" ".join(relevant))["compound"], 3)

    # --- Individual extractors: return value or None ---

    def expected_grade(self, text: str, structured: str | None = None) -> str | None:
        if structured:
            return structured
        for pat in [
            r'\b(?:got|received|earned|expect(?:ing)?|hoping for)\s+(?:an?\s+)?([A-D][+-]?|F)\b',
            r'\beasy\s+([A])\b',
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        return None

    def perceived_difficulty(self, text: str) -> str | None:
        hard = self._search(text, [
            "hard", "difficult", "tough", "challenging", "brutal",
            "demanding", "rigorous", "intense", "struggle", "struggled",
            "impossible", "overwhelming", "abstract", "painful", "drown",
            "hardest", "killer",
        ])
        easy = self._search(text, [
            "easy", "manageable", "straightforward", "simple",
            "doable", "accessible", "painless", "chill", "breeze", "light",
        ])
        if not hard and not easy:
            return None
        score = (len(hard) - len(easy)) / (len(hard) + len(easy))
        if score > 0.3:
            return "hard"
        elif score < -0.3:
            return "easy"
        return "moderate"

    def workload_intensity(self, text: str) -> dict:
        """Returns {"intensity": str|None, "hours_per_week": float|None}."""
        hours = []
        for pat in [
            r'(\d{1,2})\s*[-–to]+\s*(\d{1,2})\s*hours?\s*(?:per|a|each)?\s*week',
            r'(\d{1,2})\+?\s*hours?\s*(?:per|a|each)?\s*week',
            r'(?:about|around|maybe|roughly|probably)\s*(\d{1,2})\s*hours',
            r'(\d{1,2})\s*hours?\s*(?:of|on)\s*(?:work|study|assignment|homework|reading)',
            r'spend\w*\s+(\d{1,2})\s*hours',
        ]:
            for m in re.finditer(pat, text, re.IGNORECASE):
                hours.extend(int(g) for g in m.groups() if g)

        est_hours = round(float(np.mean(hours)), 1) if hours else None

        heavy = self._search(text, ["heavy", "a lot of work", "intense", "time consuming", "demanding", "nonstop"])
        light = self._search(text, ["light", "not much work", "minimal", "low workload", "chill"])
        moderate = self._search(text, ["moderate", "reasonable", "fair workload", "manageable"])

        if heavy:
            intensity = "heavy"
        elif light:
            intensity = "light"
        elif moderate:
            intensity = "moderate"
        elif est_hours and est_hours > 12:
            intensity = "heavy"
        elif est_hours and est_hours < 5:
            intensity = "light"
        elif est_hours:
            intensity = "moderate"
        else:
            intensity = None

        return {"intensity": intensity, "hours_per_week": est_hours}

    def teaching_quality(self, text: str) -> str | None:
        kws = [
            "professor", "prof", "instructor", "teacher", "lecturer",
            "dr.", "teaches", "taught", "teaching", "explains",
            "office hours", "ta ", "teaching assistant",
        ]
        sentiment = self._aspect_sentiment(text, kws)

        pos = self._search(text, [
            "clear", "explains well", "well-organized", "organized",
            "engaging", "come alive", "helpful", "approachable",
            "responsive", "welcoming", "amazing lecturer",
        ])
        neg = self._search(text, [
            "confusing", "unclear", "hard to follow", "goes too fast",
            "boring", "dry", "monotone", "useless", "unhelpful",
            "assumes you",
        ])

        if sentiment is None and not pos and not neg:
            return None
        if (sentiment and sentiment > 0.3) or len(pos) > len(neg):
            return "positive"
        elif (sentiment and sentiment < -0.3) or len(neg) > len(pos):
            return "negative"
        return "mixed"

    def relevance_usefulness(self, text: str) -> str | None:
        useful = self._search(text, [
            "relevant", "useful", "career", "job", "industry", "practical",
            "applicable", "real-world", "interview", "valuable", "worthwhile",
            "essential for", "important for", "grad school", "pre-med",
        ])
        useless = self._search(text, [
            "pointless", "useless", "waste of time", "irrelevant",
            "not useful", "no practical",
        ])
        if not useful and not useless:
            return None
        if useful and not useless:
            return "useful"
        if useless and not useful:
            return "not_useful"
        return "mixed"

    def interest_enjoyment(self, text: str) -> str | None:
        pos = self._search(text, [
            "interesting", "fun", "enjoy", "enjoyed", "love", "loved",
            "fascinating", "exciting", "cool", "awesome", "best course",
            "best class", "highly recommend", "come alive", "beautiful",
            "rewarding", "stimulating",
        ])
        neg = self._search(text, [
            "boring", "dull", "tedious", "dreaded", "hated", "painful",
            "miserable", "worst", "not my thing", "slog",
        ])
        if not pos and not neg:
            return None
        score = (len(pos) - len(neg)) / (len(pos) + len(neg))
        if score > 0.3:
            return "enjoyed"
        elif score < -0.3:
            return "not_enjoyed"
        return "mixed"

    def schedule_timing(self, text: str) -> dict:
        """Returns {"time_of_day": str|None, "days": list|None}."""
        time_of_day = None
        if self._search(text, ["morning", "9am", "8am", "early", "9 am", "8 am"]):
            time_of_day = "morning"
        elif self._search(text, ["afternoon"]):
            time_of_day = "afternoon"
        elif self._search(text, ["evening", "night"]):
            time_of_day = "evening"

        days = re.findall(
            r'\b(mwf|m/w/f|t/th|tu/th|tth|monday|tuesday|wednesday|thursday|friday)\b',
            text.lower(),
        )
        return {"time_of_day": time_of_day, "days": days if days else None}

    def attendance_policy(self, text: str) -> str | None:
        if self._search(text, [
            "attendance mandatory", "attendance is mandatory",
            "required attendance", "attendance required",
            "strictly taken", "must attend", "attendance counts",
        ]):
            return "mandatory"
        if self._search(text, [
            "no attendance", "attendance optional", "not mandatory",
            "isn't mandatory", "no attendance policy",
        ]):
            return "optional"
        if self._search(text, ["participation", "class discussion", "big part of the grade"]):
            return "participation_based"
        return None

    def recorded_lectures(self, text: str) -> bool | None:
        no = self._search(text, ["not recorded", "no recording", "no recordings", "aren't recorded"])
        yes = self._search(text, ["recorded", "recordings", "lecture capture", "rewatch", "asynchronous"])
        if no:
            return False
        if yes:
            return True
        return None

    def flexibility(self, text: str) -> str | None:
        flex = self._search(text, [
            "flexible", "flexibility", "lenient deadline", "extension",
            "self-paced", "own pace", "flexible deadline",
        ])
        rigid = self._search(text, [
            "strict deadline", "no extensions", "rigid", "inflexible",
            "no flexibility", "hard deadline",
        ])
        if flex and not rigid:
            return "flexible"
        if rigid and not flex:
            return "rigid"
        if flex and rigid:
            return "mixed"
        return None

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

    # --- Main ---

    def extract(self, review: dict) -> dict:
        """Extract all labels from one review. Failed → None."""
        text = review.get("review_text", "")
        workload = self.workload_intensity(text)
        schedule = self.schedule_timing(text)

        return {
            "review_id": review.get("review_id"),
            "course_id": review.get("course_id"),
            "expected_grade": self.expected_grade(text, review.get("expected_grade")),
            "perceived_difficulty": self.perceived_difficulty(text),
            "workload_intensity": workload["intensity"],
            "hours_per_week": workload["hours_per_week"],
            "teaching_quality": self.teaching_quality(text),
            "relevance_usefulness": self.relevance_usefulness(text),
            "interest_enjoyment": self.interest_enjoyment(text),
            "schedule_time_of_day": schedule["time_of_day"],
            "schedule_days": schedule["days"],
            "attendance_policy": self.attendance_policy(text),
            "recorded_lectures": self.recorded_lectures(text),
            "flexibility": self.flexibility(text),
            "delivery_mode": self.delivery_mode(text),
        }

    def extract_batch(self, reviews: list[dict]) -> pd.DataFrame:
        """Extract all reviews → DataFrame. Prints extraction summary."""
        rows = [self.extract(r) for r in reviews]
        df = pd.DataFrame(rows)

        n = len(df)
        label_cols = [c for c in df.columns if c not in ("review_id", "course_id")]

        print(f"\n{'─'*60}")
        print(f"  LABEL EXTRACTION SUMMARY ({n} reviews)")
        print(f"{'─'*60}")
        print(f"  {'Label':<25} {'Extracted':>10} {'Rate':>8}")
        print(f"  {'─'*48}")

        for col in label_cols:
            extracted = df[col].notna().sum()
            rate = extracted / n
            icon = "✅" if rate >= 0.5 else "⚠️" if rate >= 0.25 else "❌"
            print(f"  {icon} {col:<23} {extracted:>5}/{n:<4} {rate:>7.0%}")

        dropped = [c for c in label_cols if df[c].notna().sum() / n < 0.1]
        if dropped:
            print(f"\n  Columns with <10% extraction (consider dropping):")
            for c in dropped:
                print(f"    → {c}")

        print(f"{'─'*60}\n")
        return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    reviews = SYNTHETIC_REVIEWS
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            reviews = json.load(f)
        print(f"Loaded {len(reviews)} reviews from {sys.argv[1]}")

    extractor = LabelExtractor()
    df = extractor.extract_batch(reviews)

    out = "/mnt/user-data/outputs"
    df.to_csv(f"{out}/labels_extracted.csv", index=False)
    df.to_json(f"{out}/labels_extracted.json", orient="records", indent=2)
    print(f"✓ Saved labels_extracted.csv")
    print(f"✓ Saved labels_extracted.json")
    print(f"\nSample rows:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
