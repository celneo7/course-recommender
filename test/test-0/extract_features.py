"""
CS1010S Review Feature Extractor
Extracts structured, NLP, and higher-order features from course reviews.
Uses only Python standard library (no external dependencies).
"""

import json
import re
import csv
from collections import defaultdict


# ─────────────────────────────────────────────
# 1. SENTIMENT LEXICONS  (keyword-based)
# ─────────────────────────────────────────────

POSITIVE_WORDS = {
    "great", "good", "excellent", "fantastic", "amazing", "wonderful", "awesome",
    "helpful", "dedicated", "passionate", "fun", "enjoyable", "interesting",
    "clear", "engaging", "responsive", "supportive", "nice", "goated",
    "manageable", "doable", "rewarding", "worth", "useful", "love", "enjoyed",
    "smooth", "fruitful", "concise", "entertaining", "best", "recommend",
    "brilliant", "superb", "pleasant", "appreciate", "motivated", "insightful"
}

NEGATIVE_WORDS = {
    "bad", "hard", "difficult", "tough", "heavy", "terrible", "horrible",
    "disappointing", "frustrated", "error", "mistake", "wrong", "poor",
    "confusing", "unclear", "fast-paced", "overwhelming", "stressful",
    "nightmare", "bonkers", "scared", "dread", "regret", "worst", "painful",
    "unfair", "inconsistent", "unreliable", "frustrating", "annoying"
}

WORKLOAD_HEAVY = {
    "heavy", "lot", "tons", "massive", "overwhelming", "10mc", "time-consuming",
    "consistent effort", "falls behind", "snowball", "killer", "crazy", "crazily"
}

WORKLOAD_LIGHT = {
    "light", "manageable", "easy", "breezy", "not too bad", "not bad"
}

BEGINNER_FRIENDLY_WORDS = {
    "beginner", "zero experience", "no experience", "new to programming",
    "friendly", "basics", "started from basic", "ease into", "never coded"
}

RECOMMEND_SU = {
    "s/u", "su this", "su it", "consider s/u", "recommend su", "gradeless"
}


# ─────────────────────────────────────────────
# 2. EXTRACTION HELPERS
# ─────────────────────────────────────────────

def extract_ay_semester(text: str) -> dict:
    """Extract Academic Year and Semester from review text."""
    result = {"academic_year": None, "semester": None}

    # Match patterns like AY21/22, AY 21/22, AY2021/2022, AY20/21
    ay_match = re.search(
        r"AY\s*(\d{2,4})[/-](\d{2,4})", text, re.IGNORECASE
    )
    if ay_match:
        y1, y2 = ay_match.group(1), ay_match.group(2)
        # Normalise to short form e.g. "21/22"
        if len(y1) == 4:
            y1 = y1[2:]
        if len(y2) == 4:
            y2 = y2[2:]
        result["academic_year"] = f"AY{y1}/{y2}"

    # Semester
    sem_match = re.search(r"[Ss]em(?:ester)?\s*([12])", text)
    if sem_match:
        result["semester"] = int(sem_match.group(1))

    return result


def extract_assessment_weights(text: str) -> dict:
    """Extract percentage weightings for each assessment component."""
    components = {
        "coursemology_pct": None,
        "participation_pct": None,
        "midterm_pct": None,
        "practical_pct": None,
        "finals_pct": None,
        "quiz_pct": None,
        "contest_pct": None,
    }

    patterns = {
        "coursemology_pct": r"(\d+)\s*%\s*[Cc]oursemology",
        "participation_pct": r"(\d+)\s*%\s*[Pp]articipation",
        "midterm_pct":       r"(\d+)\s*%\s*[Mm]id(?:term|s)",
        "practical_pct":     r"(\d+)\s*%\s*[Pp]ractical",
        "finals_pct":        r"(\d+)\s*%\s*[Ff]inal(?:s|s exam)?",
        "quiz_pct":          r"(\d+)\s*%\s*[Qq]ui(?:z|zzes)",
        "contest_pct":       r"(\d+)\s*%\s*[Cc]ontest",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            components[key] = int(match.group(1))

    # Also try reversed order: "Coursemology 25%" or "Finals: 40%"
    reversed_patterns = {
        "coursemology_pct": r"[Cc]oursemology[:\s]+(\d+)\s*%",
        "participation_pct": r"[Pp]articipation[:\s]+(\d+)\s*%",
        "midterm_pct":       r"[Mm]id(?:term|s)[:\s]+(\d+)\s*%",
        "practical_pct":     r"[Pp]ractical[^E\n]*[:\s]+(\d+)\s*%",
        "finals_pct":        r"[Ff]inal[s]?[:\s]+(\d+)\s*%",
        "quiz_pct":          r"[Qq]ui(?:z|zzes)[:\s]+(\d+)\s*%",
    }
    for key, pattern in reversed_patterns.items():
        if components[key] is None:
            match = re.search(pattern, text)
            if match:
                components[key] = int(match.group(1))

    return components


def extract_lecturers(text: str) -> list:
    """Extract lecturer names mentioned in the review."""
    lecturers = []

    # Explicit "Lecturer:" label
    lec_match = re.search(
        r"[Ll]ecturer[s]?[:\s]+([A-Za-z .]+?)(?:\n|,|and |\Z)", text
    )
    if lec_match:
        raw = lec_match.group(1).strip()
        # Split on "/" or "and"
        parts = re.split(r"\s*/\s*|\s+and\s+", raw)
        lecturers.extend([p.strip() for p in parts if p.strip()])

    # Known lecturer names (fallback)
    known = [
        "Ben Leong", "Leong Wai Kay", "Ashish", "Adi",
        "Nitya", "Dr Nitya", "Prof Ashish", "Prof Ben",
        "Prof Leong", "Dr Ashish"
    ]
    for name in known:
        if name.lower() in text.lower() and name not in lecturers:
            lecturers.append(name)

    return list(set(lecturers)) if lecturers else []


def extract_exam_medians(text: str) -> dict:
    """Extract any disclosed exam median scores."""
    medians = {}

    patterns = {
        "practical_median": [
            r"[Pp]ractical[^.]*median[^.]*?(\d+)\s*/\s*(\d+)",
            r"median[^.]*[Pp]ractical[^.]*?(\d+)\s*/\s*(\d+)",
        ],
        "finals_median": [
            r"[Ff]inals?[^.]*median[^.]*?(\d+)\s*/\s*(\d+)",
            r"median[^.]*[Ff]inals?[^.]*?(\d+)\s*/\s*(\d+)",
        ],
    }

    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                medians[key] = f"{m.group(1)}/{m.group(2)}"
                break

    return medians


def extract_grade_disclosed(text: str) -> dict:
    """Extract any final grade or component scores disclosed."""
    result = {"final_grade": None, "scores_disclosed": False}

    grade_match = re.search(
        r"[Ff]inal\s*grade[:\s]+([A-F][+-]?)", text
    )
    if grade_match:
        result["final_grade"] = grade_match.group(1)

    # Detect score disclosure like "Mids: 32/75"
    if re.search(r"\d+\s*/\s*\d+", text):
        result["scores_disclosed"] = True

    return result


def extract_prior_experience(text: str) -> dict:
    """Infer the reviewer's prior programming experience."""
    result = {
        "prior_experience_level": "unknown",
        "languages_mentioned": []
    }

    text_lower = text.lower()

    no_exp_patterns = [
        "zero experience", "no experience", "never coded", "new to programming",
        "no prior", "no background", "entirely new", "never programmed",
        "first time coding", "no coding"
    ]
    some_exp_patterns = [
        "some experience", "basic python", "already know", "prior python",
        "relatively okay", "up till basic", "scratch", "codecademy"
    ]

    if any(p in text_lower for p in no_exp_patterns):
        result["prior_experience_level"] = "none"
    elif any(p in text_lower for p in some_exp_patterns):
        result["prior_experience_level"] = "some"
    elif re.search(r"(year [234]|senior|took.*cs[12])", text_lower):
        result["prior_experience_level"] = "experienced"

    langs = []
    for lang in ["python", "java", "c++", "c ", "r ", "scratch", "javascript"]:
        if lang in text_lower:
            langs.append(lang.strip())
    result["languages_mentioned"] = langs

    return result


def extract_resources_mentioned(text: str) -> list:
    """Extract any external learning resources mentioned."""
    resources = []
    resource_map = {
        "codecademy": "Codecademy",
        "hackerrank": "HackerRank",
        "leetcode": "LeetCode",
        "kattis": "Kattis",
        "youtube": "YouTube",
        "github": "GitHub",
        "coursemology": "Coursemology",
        "cs1010x": "CS1010X lectures",
        "ben leong.*lecture": "Ben Leong lectures",
        "matplotlib": "matplotlib",
    }
    text_lower = text.lower()
    for keyword, label in resource_map.items():
        if re.search(keyword, text_lower):
            resources.append(label)
    return resources


def simple_sentiment(text: str) -> dict:
    """
    Keyword-based sentiment scoring.
    Returns polarity: positive / negative / mixed / neutral
    and a raw score.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    score = pos - neg

    if score > 2:
        polarity = "positive"
    elif score < -2:
        polarity = "negative"
    elif pos > 0 and neg > 0:
        polarity = "mixed"
    else:
        polarity = "neutral"

    return {"sentiment_polarity": polarity, "sentiment_score": score,
            "positive_hits": pos, "negative_hits": neg}


def extract_workload_perception(text: str) -> str:
    """Classify workload perception: heavy / light / moderate / unknown."""
    text_lower = text.lower()

    heavy_hits = sum(1 for w in WORKLOAD_HEAVY if w in text_lower)
    light_hits  = sum(1 for w in WORKLOAD_LIGHT if w in text_lower)

    # Special explicit phrases
    if re.search(r"treat.*10\s*mc|more.*than.*4\s*mc", text_lower):
        return "heavy"
    if "consistent effort" in text_lower or "don't fall behind" in text_lower:
        return "heavy"

    if heavy_hits > light_hits:
        return "heavy"
    elif light_hits > heavy_hits:
        return "light"
    elif heavy_hits > 0 and light_hits > 0:
        return "moderate"
    return "unknown"


def extract_bellcurve(text: str) -> str:
    """Detect bell curve mention: yes / no / unmentioned."""
    text_lower = text.lower()
    if re.search(r"no bell.?curve|no curve", text_lower):
        return "no"
    if re.search(r"bell.?curve", text_lower):
        return "yes"
    return "unmentioned"


def extract_su_advice(text: str) -> str:
    """Detect S/U-related advice."""
    text_lower = text.lower()
    if re.search(r"do not.*s/?u|don['\u2019]t.*s/?u|not.*s/?u it", text_lower):
        return "advised_against"
    if re.search(r"s/?u\s*this|consider.*s/?u|suggest.*s/?u|use.*s/?u", text_lower):
        return "recommended"
    if re.search(r"s/?u", text_lower):
        return "mentioned"
    return "not_mentioned"


def extract_progressive_scoring(text: str) -> bool:
    """Detect mention of finals-overwrites-midterm (progressive scoring) rule."""
    patterns = [
        r"overwrite.*midterm", r"finals.*overwrite", r"progressive.*scor",
        r"if.*finals.*better.*midterm", r"overrid.*midterm",
        r"finals.*55%.*instead", r"midterm.*no weightage"
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def extract_teaching_quality_sentiment(text: str) -> dict:
    """Score sentiment toward lecturers and TAs separately."""
    result = {"lecturer_sentiment": "neutral", "ta_sentiment": "neutral"}

    # Split text into sentences roughly
    sentences = re.split(r"[.!?\n]", text)

    lec_keywords = {"lecturer", "prof", "professor", "lecture"}
    ta_keywords = {"ta", "tutor", "recitation", "teaching assistant"}

    def score_sentences(keywords):
        hits_pos, hits_neg = 0, 0
        for sent in sentences:
            low = sent.lower()
            if any(k in low for k in keywords):
                words = re.findall(r"\b\w+\b", low)
                hits_pos += sum(1 for w in words if w in POSITIVE_WORDS)
                hits_neg += sum(1 for w in words if w in NEGATIVE_WORDS)
        if hits_pos > hits_neg:
            return "positive"
        elif hits_neg > hits_pos:
            return "negative"
        elif hits_pos > 0:
            return "mixed"
        return "neutral"

    result["lecturer_sentiment"] = score_sentences(lec_keywords)
    result["ta_sentiment"] = score_sentences(ta_keywords)
    return result


def extract_topics_covered(text: str) -> list:
    """Extract CS topics mentioned in the review."""
    topic_map = {
        "Python": r"\bpython\b",
        "Recursion": r"\brecursion\b",
        "Iteration": r"\biteration\b",
        "Higher-Order Functions": r"higher.?order function",
        "Data Abstraction": r"data abstraction",
        "OOP": r"object.?oriented|oop\b|\bclass\b",
        "Sorting/Searching": r"\b(sort|search)ing\b",
        "Dynamic Programming": r"dynamic programming|memoization",
        "Time/Space Complexity": r"(time|space) complexity|order of growth",
        "Tuples/Lists/Dicts": r"\b(tuple|list|dict|set)\b",
        "Exceptions": r"exception",
        "Data Visualisation": r"data visuali[sz]",
        "Functional Abstraction": r"functional abstraction",
    }
    found = []
    text_lower = text.lower()
    for topic, pattern in topic_map.items():
        if re.search(pattern, text_lower):
            found.append(topic)
    return found


def extract_gamification_sentiment(text: str) -> str:
    """Sentiment specifically about the Coursemology gamification system."""
    sentences = re.split(r"[.!?\n]", text)
    pos, neg = 0, 0
    for sent in sentences:
        low = sent.lower()
        if any(k in low for k in ["coursemology", "leaderboard", "mission", "gamif", "xp", "level"]):
            words = re.findall(r"\b\w+\b", low)
            pos += sum(1 for w in words if w in POSITIVE_WORDS)
            neg += sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    elif pos > 0:
        return "mixed"
    return "neutral"


def extract_beginner_friendliness(text: str) -> str:
    """Detect whether the review comments on beginner-friendliness."""
    text_lower = text.lower()
    if any(p in text_lower for p in BEGINNER_FRIENDLY_WORDS):
        if any(p in text_lower for p in ["friendly", "ease into", "basics", "from scratch", "started from the basic"]):
            return "beginner_friendly"
        return "beginner_mentioned"
    return "not_mentioned"


def extract_exam_quality_sentiment(text: str) -> str:
    """Detect negative comments about exam quality (errors, fairness)."""
    text_lower = text.lower()
    bad_exam_patterns = [
        r"error.*exam|exam.*error",
        r"mistake.*paper|paper.*mistake",
        r"wrong.*question|question.*wrong",
        r"badly conducted",
        r"debug the paper",
        r"error.prone",
        r"unfair.*exam",
    ]
    if any(re.search(p, text_lower) for p in bad_exam_patterns):
        return "negative"
    if re.search(r"exam.*fair|fair.*exam|well.*set|nicely.*set", text_lower):
        return "positive"
    return "neutral"


def extract_plagiarism_warning(text: str) -> bool:
    """Detect if the review warns against plagiarism."""
    return bool(re.search(r"plagiari[sz]", text, re.IGNORECASE))


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


# ─────────────────────────────────────────────
# 3. MAIN PIPELINE
# ─────────────────────────────────────────────

def extract_features(review: dict) -> dict:
    """Run all extractors on a single review and return a flat feature dict."""
    text = review.get("message", "")
    features = {
        # ── Metadata ──────────────────────────────
        "post_id":          review.get("post_id"),
        "author":           review.get("author"),
        "date":             review.get("date"),
        "likes":            int(review.get("likes", 0) or 0),
        "dislikes":         int(review.get("dislikes", 0) or 0),
        "is_reply":         review.get("reply_to_post_id") is not None,
        "reply_to_author":  review.get("reply_to_author"),
        "word_count":       count_words(text),
    }

    # ── Structured ────────────────────────────────
    features.update(extract_ay_semester(text))
    features.update(extract_assessment_weights(text))
    features.update(extract_exam_medians(text))
    features.update(extract_grade_disclosed(text))
    features["lecturers_mentioned"] = "|".join(extract_lecturers(text))
    features["progressive_scoring_mentioned"] = extract_progressive_scoring(text)
    features["bell_curve"] = extract_bellcurve(text)

    # ── NLP / Content ─────────────────────────────
    features.update(simple_sentiment(text))
    features.update(extract_teaching_quality_sentiment(text))
    features["workload_perception"]        = extract_workload_perception(text)
    features["gamification_sentiment"]     = extract_gamification_sentiment(text)
    features["exam_quality_sentiment"]     = extract_exam_quality_sentiment(text)
    features["beginner_friendliness"]      = extract_beginner_friendliness(text)
    features["su_advice"]                  = extract_su_advice(text)
    features["plagiarism_warning"]         = extract_plagiarism_warning(text)
    features["topics_covered"]             = "|".join(extract_topics_covered(text))
    features["resources_mentioned"]        = "|".join(extract_resources_mentioned(text))
    features["raw_text"]                   = text

    # ── Prior experience ──────────────────────────
    exp = extract_prior_experience(text)
    features["prior_experience_level"]  = exp["prior_experience_level"]
    features["languages_mentioned"]     = "|".join(exp["languages_mentioned"])

    return features


def run(input_path: str, output_csv: str, output_json: str):
    with open(input_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    all_features = [extract_features(r) for r in reviews]

    # ── Write JSON ──────────────────────────────
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_features, f, indent=2, ensure_ascii=False)

    # ── Write CSV ───────────────────────────────
    # All keys from all records (union)
    fieldnames = list(dict.fromkeys(k for feat in all_features for k in feat))
    # Move raw_text to end
    if "raw_text" in fieldnames:
        fieldnames.remove("raw_text")
        fieldnames.append("raw_text")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_features)

    print(f"✅  Processed {len(all_features)} reviews")
    print(f"    Features per review : {len(fieldnames)}")
    print(f"    Output JSON : {output_json}")
    print(f"    Output CSV  : {output_csv}")

    # ── Quick summary ────────────────────────────
    print("\n── Feature summary (first 3 non-reply reviews) ──")
    shown = 0
    for feat in all_features:
        if feat["is_reply"] or shown >= 3:
            continue
        print(f"\n[{feat['post_id']}] {feat['author']} | {feat['academic_year']} S{feat['semester']}")
        print(f"  Workload       : {feat['workload_perception']}")
        print(f"  Sentiment      : {feat['sentiment_polarity']} (score={feat['sentiment_score']})")
        print(f"  Lecturer sent. : {feat['lecturer_sentiment']}")
        print(f"  TA sentiment   : {feat['ta_sentiment']}")
        print(f"  S/U advice     : {feat['su_advice']}")
        print(f"  Bell curve     : {feat['bell_curve']}")
        print(f"  Prog. scoring  : {feat['progressive_scoring_mentioned']}")
        print(f"  Prior exp.     : {feat['prior_experience_level']}")
        print(f"  Topics found   : {feat['topics_covered']}")
        print(f"  Resources      : {feat['resources_mentioned']}")
        print(f"  Lecturers      : {feat['lecturers_mentioned']}")
        assessment = {k: feat[k] for k in [
            "coursemology_pct","midterm_pct","practical_pct","finals_pct","participation_pct"
        ] if feat[k] is not None}
        print(f"  Assessment %   : {assessment}")
        shown += 1


if __name__ == "__main__":
    run(
        input_path  = "/mnt/user-data/uploads/cs1010s_reviews.json",
        output_csv  = "/mnt/user-data/outputs/cs1010s_features.csv",
        output_json = "/mnt/user-data/outputs/cs1010s_features.json",
    )
