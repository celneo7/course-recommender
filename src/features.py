"""
Feature Extractor
==================
Extracts prediction input features from course reviews.
Each review only needs: {"message": "...", "author": "..."}
One JSON file per course.

Usage:
  python feature_extractor.py                        # synthetic data
  python feature_extractor.py CS201_reviews.json     # your data
"""

import json
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



# =============================================================================
# EXTRACTOR
# =============================================================================

class FeatureExtractor:

    @staticmethod
    def _search(text: str, patterns: list[str]) -> list[str]:
        t = text.lower()
        return [p for p in patterns if p in t]

    # --- Course characteristics ---

    def course_content_type(self, text: str) -> str | None:
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
        return best if scores[best] > 0 else None

    def prerequisites_mentioned(self, text: str) -> bool | None:
        cues = self._search(text, [
            "prerequisite", "prereq", "need to know", "background in",
            "foundation", "prior course", "should take",
            "must know", "strong background", "need a solid",
            "no prerequisites", "no prereq",
            "prior programming experience", "prior experience", "prior knowledge",
            "no programming background", "no coding background",
            "new to programming", "never coded before", "zero experience",
            "no experience", "entirely new to",
        ])
        if not cues:
            return None
        if self._search(text, [
            "no prerequisites", "no prereq",
            "no programming background", "no coding background",
            "new to programming", "never coded before", "zero experience",
            "entirely new to",
        ]):
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
        types_map = {
            "exams": r'\b(?:exam|exams|midterm|midterms|final exam)\b',
            "quizzes": r'\b(?:quiz|quizzes)\b',
            "essays": r'\b(?:essay|essays|paper|papers|reflection)\b',
            "projects": r'\b(?:project|projects|final project)\b',
            "assignments": r'\b(?:assignment|assignments|homework|problem set|pset|coding assignment)\b',
            "labs": r'\b(?:lab|labs|lab report|lab component)\b',
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

    def course_pacing(self, text: str) -> str | None:
        fast = self._search(text, ["fast", "too fast", "fast-paced", "rushed", "cramming"])
        slow = self._search(text, ["slow", "too slow", "drags", "boring pace"])
        if fast and not slow:
            return "fast"
        if slow and not fast:
            return "slow"
        return None

    # --- Instructor ---

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

    # --- Student / reviewer characteristics ---

    def prior_knowledge_level(self, text: str) -> str | None:
        none_kws = self._search(text, [
            "no experience", "zero experience", "no programming experience",
            "new to programming", "never coded", "no background",
            "no prior", "entirely new", "from scratch", "no coding experience",
            "no programming background",
        ])
        experienced_kws = self._search(text, [
            "strong background", "already knew", "prior programming experience",
            "prior experience", "strong coding skills", "familiar with python",
            "background in python", "already know python",
        ])
        some_kws = self._search(text, [
            "some experience", "basic python", "basic programming",
            "little background", "some background",
        ])
        if none_kws and not experienced_kws:
            return "none"
        if experienced_kws and not none_kws:
            return "experienced"
        if some_kws:
            return "some"
        return None

    def student_year(self, text: str) -> int | None:
        m = re.search(r'\b[Yy]ear\s*([1-4])\b', text)
        if m:
            return int(m.group(1))
        m = re.search(r'\b[Yy]([1-4])\b', text)  # NUS shorthand e.g. "Y2"
        if m:
            return int(m.group(1))
        year_map = {
            "freshman": 1, "first year": 1, "1st year": 1,
            "sophomore": 2, "second year": 2, "2nd year": 2,
            "junior": 3, "third year": 3, "3rd year": 3,
            "senior": 4, "fourth year": 4, "4th year": 4,
        }
        t = text.lower()
        for label, year in year_map.items():
            if label in t:
                return year
        return None

    def academic_year(self, text: str) -> str | None:
        """Extracts NUS-style academic year e.g. 'AY20/21 Sem 1'."""
        m = re.search(
            r'\bAY\s*(\d{2,4})[/\-](\d{2,4})\s*(?:Sem(?:ester)?\s*([12]))?\b',
            text, re.IGNORECASE
        )
        if m:
            ay = f"AY{m.group(1)}/{m.group(2)}"
            sem = f" Sem {m.group(3)}" if m.group(3) else ""
            return ay + sem
        return None

    def instructor_names(self, text: str) -> list | None:
        """Extracts instructor names from 'Lecturer: X', 'Prof X', 'Dr X' patterns."""
        names = []
        # Structured header lines: "Lecturer: Prof X", "TA: Name"
        for m in re.finditer(
            r'(?:Lecturer|Tutor|TA|Recitation)\s*:\s*([A-Z][\w].*?)(?:\n|$)',
            text
        ):
            for name in re.split(r'\s*(?:and|&|,|/)\s*', m.group(1).strip()):
                name = name.strip()
                if len(name) > 2 and name not in names:
                    names.append(name)
        # Inline mentions: "Prof Leong", "Dr Nitya"
        for m in re.finditer(
            r'\b(?:Prof(?:essor)?|Dr)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
            text
        ):
            name = m.group(0).strip()
            if name not in names:
                names.append(name)
        return names if names else None

    def course_role(self, text: str) -> str | None:
        """Whether the module is core, elective/UE, or gen-ed."""
        if self._search(text, ["core mod", "core module", "core course", "compulsory"]):
            return "core"
        if re.search(r'\bUE\b', text) or self._search(text, ["unrestricted elective", "free elective"]):
            return "elective_ue"
        if self._search(text, ["gen-ed", "gen ed", "general education", "breadth requirement"]):
            return "gen_ed"
        if self._search(text, ["elective"]):
            return "elective"
        return None

    def has_gamification(self, text: str) -> bool | None:
        """Detects references to gamified learning platforms (e.g. Coursemology)."""
        if self._search(text, [
            "coursemology", "gamification", "gamified", "leaderboard",
            "game-like",
        ]) or re.search(r'\blevel\s+\d+\b', text, re.IGNORECASE):
            return True
        return None

    # --- Grading structure ---

    def finals_weight_pct(self, text: str) -> int | None:
        """Extracts the stated final exam percentage weighting."""
        for pat in [
            r'(\d{2})\s*%\s*(?:for\s+(?:the\s+)?finals?|finals?\s+exam)',
            r'finals?\s*[:\-]\s*(\d{2})\s*%',
            r'finals?\s+(?:is\s+)?(\d{2})\s*%',
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 10 <= val <= 100:
                    return val
        return None

    def progressive_scoring(self, text: str) -> bool | None:
        """True if the review mentions a mechanic where finals can replace/overwrite midterms."""
        cues = self._search(text, [
            "progressive scoring", "overwrite", "replaces midterm",
            "midterms has no weightage", "midterm has no weightage",
            "finals is better than your midterms",
            "finals did better than midterms",
            "0 % progressive", "0% progressive",
        ])
        return True if cues else None

    def su_consideration(self, text: str) -> str | None:
        """Whether the review mentions S/U-ing (satisfactory/unsatisfactory grading option)."""
        t = text.lower()
        discourage = bool(re.search(
            r'\bdo\s+not\s+(?:walk in|take).{0,30}mindset\s+to\s+s/u\b'
            r'|\bnot\b.{0,20}\bwith\s+the\s+mindset\s+to\s+s/u\b',
            t
        ))
        mentioned = bool(re.search(r'\bs/u\b|\bsu\s+it\b|\bsatisfactory\b', t))
        if not mentioned:
            return None
        if discourage:
            return "discouraged"
        return "mentioned"

    # --- Course-specific signals ---

    def warns_plagiarism(self, text: str) -> bool | None:
        """True if the review explicitly warns against plagiarism."""
        cues = self._search(text, [
            "do not plagiarize", "don't plagiarize", "do not plagiarise",
            "don't plagiarise", "strict about it", "plagiarism",
            "plagiarise", "plagiarize",
        ])
        return True if cues else None

    def exam_quality_concerns(self, text: str) -> bool | None:
        """True if the review raises concerns about poorly set or error-prone exams."""
        cues = self._search(text, [
            "error in the paper", "errors in the paper", "error at",
            "typo", "poorly set", "badly conducted", "badly set",
            "errors in exam", "mistake in the exam", "wrong question",
            "debug the paper",
        ])
        return True if cues else None

    def consistent_effort_required(self, text: str) -> bool | None:
        """True if the review emphasises that consistent/regular effort is needed."""
        cues = self._search(text, [
            "consistent effort", "consistently", "don't fall behind",
            "do not fall behind", "keep up", "keep your pace",
            "do your work", "stay on top", "every week",
            "keep on track", "do not procrastinate", "don't procrastinate",
            "week by week",
        ])
        return True if cues else None

    def peer_support_culture(self, text: str) -> bool | None:
        """True if the review highlights peer collaboration or forum-based help."""
        cues = self._search(text, [
            "with friends", "ask around", "ask peers", "study group",
            "forum", "ask in the forum", "peer support",
            "fellow students", "fellow peers",
            "community", "collaborative",
        ])
        return True if cues else None

    def overload_warning(self, text: str) -> bool | None:
        """True if the review explicitly warns against overloading with this course."""
        cues = self._search(text, [
            "do not overload", "don't overload",
            "overloading", "overload",
        ])
        # Require negative framing
        if self._search(text, ["do not overload", "don't overload", "overloading this course is"]):
            return True
        if cues and re.search(r'\b(?:do not|don\'t|avoid|warning|careful)\b.{0,30}\boverload\b', text.lower()):
            return True
        return None

    def major_context(self, text: str) -> str | None:
        """Extracts the reviewer's stated academic major or faculty."""
        m = re.search(
            r'\b((?:Computer Science|CS|Statistics|Mathematics|Math|Engineering|Business|'
            r'Life Science|Biology|Chemistry|Physics|Economics|Finance|Psychology|'
            r'Information Systems|IS|Data Science|Medicine|Law|Arts|Humanities)'
            r'(?:\s+(?:major|student|undergraduate))?)\b',
            text, re.IGNORECASE
        )
        return m.group(1).strip() if m else None

    # --- Main ---

    def extract(self, review: dict) -> dict:
        text = review.get("message", "")

        return {
            "post_id": review.get("post_id"),
            "author": review.get("author"),
            "date": review.get("date"),
            "reply_to_post_id": review.get("reply_to_post_id"),
            # Course / semester context
            "academic_year": self.academic_year(text),
            "instructor_names": self.instructor_names(text),
            "course_content_type": self.course_content_type(text),
            "course_role": self.course_role(text),
            "prerequisites_mentioned": self.prerequisites_mentioned(text),
            "has_gamification": self.has_gamification(text),
            # Delivery & structure
            "delivery_mode": self.delivery_mode(text),
            "attendance_required": self.attendance_required(text),
            "participation_graded": self.participation_graded(text),
            "assessment_types": self.assessment_types(text),
            "has_exams": self.has_exams(text),
            "course_pacing": self.course_pacing(text),
            # Student characteristics
            "prior_knowledge_level": self.prior_knowledge_level(text),
            "student_year": self.student_year(text),
            "major_context": self.major_context(text),
            # Instructor
            "instructor_clarity": self.instructor_clarity(text),
            "instructor_engagement": self.instructor_engagement(text),
            "instructor_helpfulness": self.instructor_helpfulness(text),
            "teaching_style": self.teaching_style(text),
            # Grading structure
            "finals_weight_pct": self.finals_weight_pct(text),
            "progressive_scoring": self.progressive_scoring(text),
            "su_consideration": self.su_consideration(text),
            # Course culture / red flags
            "consistent_effort_required": self.consistent_effort_required(text),
            "peer_support_culture": self.peer_support_culture(text),
            "warns_plagiarism": self.warns_plagiarism(text),
            "exam_quality_concerns": self.exam_quality_concerns(text),
            "overload_warning": self.overload_warning(text),
        }

    def extract_batch(self, reviews: list[dict]) -> pd.DataFrame:
        rows = [self.extract(r) for r in reviews]
        df = pd.DataFrame(rows)

        n = len(df)
        _skip = {"author", "post_id", "date", "reply_to_post_id", "instructor_names"}
        feat_cols = [c for c in df.columns if c not in _skip]

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
# TF-IDF (optional)
# =============================================================================

def build_tfidf(reviews: list[dict], max_features: int = 200) -> pd.DataFrame:
    texts = [r.get("message", "") for r in reviews]
    ids = [r.get("author", f"r{i}") for i, r in enumerate(reviews)]
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(texts)
    return pd.DataFrame(matrix.toarray(), columns=tfidf.get_feature_names_out(), index=ids)


# =============================================================================
# MAIN
# =============================================================================

def main():
    output = 'data/cs2040/'
    input_file = output + 'cs2040_reviews.json'
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    extractor = FeatureExtractor()
    df = extractor.extract_batch(reviews)

    tfidf_df = build_tfidf(reviews, max_features=50)

    df.to_csv(f"{output}/features_extracted.csv", index=False)
    tfidf_df.to_csv(f"{output}/tfidf_features.csv", index=False)


if __name__ == "__main__":
    main()



# =============================================================================
# SYNTHETIC DATA
# =============================================================================

SYNTHETIC_REVIEWS = [
    {
        "author": "student_01",
        "message": (
            "This course was really challenging but extremely rewarding. "
            "Dr. Chen explains concepts clearly and the lectures are well-organized. "
            "The workload is heavy — expect 10-15 hours per week outside of class. "
            "Weekly coding assignments and two midterms plus a final project. "
            "Attendance isn't mandatory but you'll fall behind without going. "
            "Lectures are recorded which is a lifesaver. "
            "The content is super relevant if you want a software engineering career. "
            "Loved the problem-solving aspect."
        ),
    },
    {
        "author": "student_02",
        "message": (
            "Tough course. The midterms were brutal and the grading is strict. "
            "I spent about 12 hours a week on assignments alone. "
            "The prof is knowledgeable but sometimes goes too fast. "
            "Office hours were helpful though. No attendance policy which is nice. "
            "The final project was interesting — we built a search engine. "
            "Definitely useful for interviews and job prep."
        ),
    },
    {
        "author": "student_03",
        "message": (
            "Easy A if you do the readings. The lecturer is amazing — "
            "super engaging and makes history come alive. "
            "Workload is light, maybe 3-4 hours a week. Two essays and a final exam. "
            "Attendance is mandatory and counts for 10% of your grade. "
            "No recorded lectures unfortunately. The class meets MWF at 9am which is rough. "
            "Great for fulfilling gen-ed requirements but also genuinely fascinating."
        ),
    },
    {
        "author": "student_04",
        "message": (
            "Took this as a gen-ed. It was okay but not my thing. "
            "The prof is a good speaker but the essay grading felt subjective. "
            "Attendance is strictly taken. Early morning classes were painful. "
            "Not much relevance to my major but interesting enough. "
            "About 5 hours of work per week. The final was memorization-heavy. "
            "Flexible deadline policy for essays was a plus."
        ),
    },
    {
        "author": "student_05",
        "message": (
            "Very demanding course with a LOT of content to memorize. "
            "The lecturer is brilliant but her talks can be dry and hard to follow. "
            "Expect 15+ hours a week if you want to do well. "
            "Three exams, weekly quizzes, and a lab component. "
            "Attendance is mandatory for labs. Lectures are online and asynchronous "
            "which is nice for flexibility. Essential for pre-med students. "
            "The lab work was actually the most enjoyable part."
        ),
    },
    {
        "author": "student_06",
        "message": (
            "One of the best courses I've taken. The prof creates such a welcoming "
            "discussion environment. The workload is moderate — weekly reflection papers "
            "and a final essay. Maybe 6 hours a week. No exams! "
            "Completely in-person, no recordings. Participation is a big part of the grade. "
            "Schedule was T/Th afternoons which worked perfectly. "
            "Really changed how I think about ethical dilemmas. "
            "Highly recommend regardless of your major."
        ),
    },
    {
        "author": "student_07",
        "message": (
            "Hardest course I've ever taken. The proofs are incredibly abstract and "
            "the prof assumes you already know everything. Office hours are useless — "
            "he just restates the theorem. Spent 20 hours a week minimum. "
            "Weekly problem sets that take forever. Two brutal exams. "
            "No attendance policy, lectures not recorded. "
            "Important for grad school but painful. The subject itself is beautiful "
            "once you understand it, which takes a long time."
        ),
    },
    {
        "author": "student_08",
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
