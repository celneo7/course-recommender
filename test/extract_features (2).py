"""
CS Course Review Feature Extractor
====================================
Extracts structured and NLP features from NUS CS module reviews.
Designed to work across courses without hardcoding module-specific content.
Uses only Python standard library (no external dependencies).
"""

import json
import re
import csv


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
    """
    Dynamically extract assessment weightage as a single dict keyed by
    component name, e.g. {"Finals": 40, "Midterm": 20, "Lab Assignment": 15}.

    Works by scanning for lines/phrases of the form:
      - "40% Finals"  /  "Finals: 40%"  /  "Finals (40%)"
    No fixed component names are assumed, so it works across any CS module.
    Returns {"assessment_weightage": dict} — stored as JSON string in CSV.
    """
    # Words that appear as section headers and should never be component names
    HEADER_WORDS = {"assessment", "overview", "content", "topics", "grading", "structure"}

    weightage = {}

    # Pattern A: "40% <Component Name>"  — pct comes first
    for m in re.finditer(r"(\d{1,3})\s*%\s+([A-Z][A-Za-z0-9 \-/()]{2,50}?)(?=\n|\(|\d{1,3}\s*%|$)", text):
        pct = int(m.group(1))
        name = m.group(2).strip().rstrip(":-")
        if 1 <= pct <= 100 and name and name.lower() not in HEADER_WORDS:
            weightage[name] = pct

    # Pattern B: "<Component Name>: 40%"  or  "<Component Name> (40%)"
    for m in re.finditer(r"([A-Z][A-Za-z0-9 \-/]{2,50}?)\s*[:(]\s*(\d{1,3})\s*%", text):
        name = m.group(1).strip()
        pct = int(m.group(2))
        if 1 <= pct <= 100 and name and name not in weightage and name.lower() not in HEADER_WORDS:
            weightage[name] = pct

    return {"assessment_weightage": weightage}


def extract_lecturers(text: str) -> list:
    """
    Extract lecturer names from explicit 'Lecturer(s):' labels only.
    No hardcoded names — works across any module.
    """
    lecturers = []

    lec_match = re.search(
        r"[Ll]ecturer[s]?\s*[:\-]\s*(.+?)(?:\n|$)", text
    )
    if lec_match:
        raw = lec_match.group(1).strip()
        parts = re.split(r"\s*/\s*|\s+and\s+|,\s*", raw)
        lecturers.extend([p.strip() for p in parts if p.strip()])

    return list(dict.fromkeys(lecturers))  # dedupe, preserve order


def extract_exam_medians(text: str) -> dict:
    """
    Extract disclosed exam median scores. Handles two formats:
      - Fraction:  "median 7/20"  or  "Practical median ~ 7/20"
      - Decimal:   "median 26.2"  or  "Finals median: 39"
    Always returns all three keys so CSV schema is stable.
    """
    medians = {"midterm_median": None, "finals_median": None, "practical_median": None}
    exams = {
        "midterm_median":   r"[Mm]id(?:term|s?)",
        "finals_median":    r"[Ff]inals?",
        "practical_median": r"[Pp]ractical",
    }
    for key, exam_pat in exams.items():
        # Component then median value
        m = re.search(
            exam_pat + r"[^.\n]{0,40}?median[^.\n]{0,20}?([\d]+(?:[./][\d]+)?)",
            text, re.IGNORECASE
        )
        if not m:
            # Median value then component
            m = re.search(
                r"median[^.\n]{0,20}?" + exam_pat + r"[^.\n]{0,20}?([\d]+(?:[./][\d]+)?)",
                text, re.IGNORECASE
            )
        if m:
            medians[key] = m.group(1)
    return medians


def extract_grade_disclosed(text: str) -> dict:
    """
    Extract final/actual grade and expected grade.

    GRADE_VAL rules:
    - (?<!\w)       : grade letter cannot be the tail of a word (e.g. effor-t, mo-re)
    - [A-F][+-]?    : uppercase only — avoids IGNORECASE matching 'a' as a grade
    - (?:\s*/\s*[A-F][+-]?)? : optional range suffix  e.g. B/B+, A-/A
    - (?:\s*\brange\b)?      : optional descriptor    e.g. A range
    - (?![a-rt-wyz])         : not followed by a word char except 's' (plurals: B+s)
                               and 'x' (which never follows a grade letter meaningfully)
    """
    result = {
        "final_grade": None,
        "expected_grade": None,
        "scores_disclosed": False,
    }

    GRADE_VAL = r"(?<!\w)([A-F][+-]?(?:\s*/\s*[A-F][+-]?)?(?:\s*\brange\b)?)(?![a-rt-wyz])"
    SEP = r"\s*[:=;]\s*"

    # ── Final / actual grade ──────────────────────────────────────
    # Priority 1: explicit label  "Final grade: A+"  /  "Actual Grade = B+"
    m = re.search(
        r"(?:[Ff]inal|[Aa]ctual)\s*[Gg]rade" + SEP + r"(?!\s*\n)" + GRADE_VAL, text
    )
    if m:
        result["final_grade"] = m.group(1).strip().upper()

    # Priority 2: "gotten a C+ for this mod"  /  "got my first A for a cs mod"
    if not result["final_grade"]:
        m = re.search(
            r"\b(?:gotten?|received?)\s+(?:an?\s+)?" + GRADE_VAL + r"[^.]{0,30}?(?:for|overall|\bmod\b)",
            text
        )
        if m:
            result["final_grade"] = m.group(1).strip().upper()

    # Priority 3: "only an A-"  /  "still only an A-"
    if not result["final_grade"]:
        m = re.search(r"\bonly\s+(?:an?\s+)?" + GRADE_VAL, text)
        if m:
            result["final_grade"] = m.group(1).strip().upper()

    # Priority 4: "scored 84+ but still only an A-" already caught above;
    # also catch "ended up with an A" / "in the end ... A-"
    if not result["final_grade"]:
        m = re.search(
            r"\b(?:ended\s+up|in\s+the\s+end)\b[^.\n]{0,30}?" + GRADE_VAL, text
        )
        if m:
            result["final_grade"] = m.group(1).strip().upper()

    # ── Expected grade ────────────────────────────────────────────
    # Explicit label (handles :  =  ;)
    m = re.search(
        r"[Ee]xpected\s*[Gg]rade" + SEP + r"(?!\s*\n)" + GRADE_VAL, text
    )
    if m:
        result["expected_grade"] = m.group(1).strip().upper()

    # Freeform expectation — (?i:...) scopes IGNORECASE to keyword only
    # so 'a' in "score a B" does not match [A-F] (which is case-sensitive here)
    if not result["expected_grade"]:
        m = re.search(
            r"(?i:\b(?:hoping?\s+(?:to\s+(?:get|achieve|score)\s+)?"
            r"|hope(?:d)?\s+for\s+"
            r"|aim(?:ing)?\s+for\s+"
            r"|target(?:ing)?\s+"
            r"|predict(?:ed|ing)?\s+))"
            r"[^.\n]{0,25}?" + GRADE_VAL,
            text,
        )
        if m:
            result["expected_grade"] = m.group(1).strip().upper()

    # ── Numeric scores disclosed ──────────────────────────────────
    if re.search(
        r"\b(?:mids?|finals?|practical|exam|quiz|total|lab|ila|tha|oda)[^.\n]{0,30}?(\d+)\s*/\s*(\d+)",
        text, re.IGNORECASE
    ):
        result["scores_disclosed"] = True
    elif re.search(r"\b(\d{1,3})\s*/\s*(75|100|90|80|50|40|30|20)\b", text):
        result["scores_disclosed"] = True

    return result

    # Scores disclosed: numeric component scores like "32/75", "67/100"
    if re.search(
        r"\b(?:mids?|finals?|practical|exam|quiz|total|lab|ila|tha|oda)[^.\n]{0,30}?(\d+)\s*/\s*(\d+)",
        text, re.IGNORECASE
    ):
        result["scores_disclosed"] = True
    elif re.search(r"\b(\d{1,3})\s*/\s*(75|100|90|80|50|40|30|20)\b", text):
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
    lang_patterns = [
        ("python",     r"\bpython\b"),
        ("java",       r"\bjava\b(?!script)"),  # exclude javascript
        ("javascript", r"\bjavascript\b"),
        ("c++",        r"\bc\+\+"),
        ("c",          r"\blanguage\s+c\b|\bin\s+c\b|\busing\s+c\b|\bc\s+language\b"),
        ("r",          r"\blanguage\s+r\b|\bin\s+r\b|\busing\s+r\b|\br\s+language\b"),
        ("scratch",    r"\bscratch\b(?!\s+which|\s+that|\s+can|\s+to\b)(?<!\bfrom\s)"),
    ]
    for label, pattern in lang_patterns:
        if re.search(pattern, text_lower):
            langs.append(label)
    result["languages_mentioned"] = langs

    return result


def extract_resources_mentioned(text: str) -> list:
    """
    Detect external learning resources mentioned in the review.

    This list will need to grow as new courses are added — add new entries
    to RESOURCE_PATTERNS below whenever a resource appears in a new dataset.
    Each entry is (display_label, regex_pattern).
    """
    RESOURCE_PATTERNS = [
        # Coding practice platforms
        ("Codecademy",        r"codecademy"),
        ("HackerRank",        r"hackerrank"),
        ("LeetCode",          r"leetcode"),
        ("Kattis",            r"kattis"),
        # Course-specific tools
        ("VisuAlgo",          r"visualgo|visuAlgo"),
        ("Coursemology",      r"coursemology"),
        # Study materials
        ("Past Year Papers",  r"past.?year|pyp\b"),
        ("GitHub",            r"github"),
        ("YouTube",           r"youtube"),
        ("NUSMods",           r"nusmods"),
        # Add new entries here as more courses are analysed, e.g.:
        # ("CS1010X Lectures", r"cs1010x"),
        # ("Coursera",         r"coursera"),
    ]
    text_lower = text.lower()
    return [label for label, pattern in RESOURCE_PATTERNS
            if re.search(pattern, text_lower)]


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
    """
    Extract topics only when a student explicitly lists them —
    i.e. a numbered list (1. 2. 3. …) appearing under a header that
    signals a syllabus/content section.

    This approach is course-agnostic: it reads whatever the student wrote,
    rather than matching a fixed vocabulary of topic keywords.
    Returns a list of topic strings as written by the reviewer.
    """
    topics = []

    # Step 1: Find a syllabus-signal header in the text
    # (Topics / Content / Module covers / Syllabus / Covers)
    header_pattern = re.compile(
        r"(?:topics?|content|syllabus|module covers?|covers?|what.{0,10}taught|taught)\s*[:\-]?\s*\n",
        re.IGNORECASE
    )
    header_match = header_pattern.search(text)

    # Step 2: If header found, extract numbered lines that follow it
    if header_match:
        after_header = text[header_match.end():]
        # Grab consecutive numbered items (stop at a blank line or non-numbered line)
        for line in after_header.splitlines():
            line = line.strip()
            m = re.match(r"^\d+[\.)]\s+(.+)", line)
            if m:
                topics.append(m.group(1).strip())
            elif topics and line == "":
                # allow one blank line gap, then stop if no more items
                continue
            elif topics:
                # non-numbered line after we started collecting → stop
                break

    # Step 3: Fallback — if no header, grab any numbered list with 3+ items,
    # but only accept lines that look like short noun phrases (likely topic names),
    # not long opinion sentences. Heuristics: short (<= 60 chars), starts with
    # a capital letter or known CS noun, no first-person verbs like "I/we/you".
    if not topics:
        numbered_lines = re.findall(r"^\s*\d+[\.)]\s+(.+)", text, re.MULTILINE)
        if len(numbered_lines) >= 3:
            opinion_signals = re.compile(
                r"^\s*i |^\s*you |^\s*we |^\s*there |grind|chatgpt|think|feel|suggest|found|made|did|got|took|spent",
                re.IGNORECASE
            )
            topics = [
                l.strip() for l in numbered_lines
                if len(l.strip()) <= 60 and not opinion_signals.match(l.strip())
            ]
            # Require at least 3 clean items to trust this is a topic list
            if len(topics) < 3:
                topics = []

    return topics


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


def parse_date(raw: str) -> str:
    """
    Convert "Wednesday, April 30, 2025 10:02 PM" → "2025-04-30 22:02"
    Removes commas so the date is safe as a CSV field without quoting.
    Returns the original string unchanged if parsing fails.
    """
    if not raw:
        return raw
    try:
        from datetime import datetime
        parts = raw.split(", ", 1)
        date_str = parts[1] if len(parts) == 2 else raw
        dt = datetime.strptime(date_str.strip(), "%B %d, %Y %I:%M %p")
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return raw


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
        "date":             parse_date(review.get("date", "")),
        "likes":            int(review.get("likes", 0) or 0),
        "dislikes":         int(review.get("dislikes", 0) or 0),
        "is_reply":         review.get("reply_to_post_id") is not None,
        "reply_to_author":  review.get("reply_to_author"),
        "word_count":       count_words(text),
    }

    # ── Structured ────────────────────────────────
    features.update(extract_ay_semester(text))
    # assessment_weightage stored as JSON string (dict of component → pct)
    aw = extract_assessment_weights(text)
    features["assessment_weightage"] = "|".join(
        f"{k}={v}" for k, v in aw["assessment_weightage"].items()
    ) if aw["assessment_weightage"] else None
    features.update(extract_exam_medians(text))
    features.update(extract_grade_disclosed(text))
    features["lecturers_mentioned"] = "|".join(extract_lecturers(text))
    features["progressive_scoring_mentioned"] = extract_progressive_scoring(text)
    features["bell_curve"] = extract_bellcurve(text)

    # ── NLP / Content ─────────────────────────────
    features.update(simple_sentiment(text))
    features.update(extract_teaching_quality_sentiment(text))
    features["workload_perception"]    = extract_workload_perception(text)
    features["exam_quality_sentiment"] = extract_exam_quality_sentiment(text)
    features["beginner_friendliness"]  = extract_beginner_friendliness(text)
    features["su_advice"]              = extract_su_advice(text)
    features["plagiarism_warning"]     = extract_plagiarism_warning(text)
    # topics_covered: pipe-delimited list of explicitly listed topic strings
    features["topics_covered"]         = "|".join(extract_topics_covered(text))
    features["topics_explicitly_listed"] = len(extract_topics_covered(text)) > 0
    features["resources_mentioned"]    = "|".join(extract_resources_mentioned(text))
    features["raw_text"]               = text

    # ── Prior experience ──────────────────────────
    exp = extract_prior_experience(text)
    features["prior_experience_level"] = exp["prior_experience_level"]
    features["languages_mentioned"]    = "|".join(exp["languages_mentioned"])

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

    # ── Write CSV ───────────────────────────────
    fieldnames = list(dict.fromkeys(k for feat in all_features for k in feat))
    if "raw_text" in fieldnames:
        fieldnames.remove("raw_text")
        fieldnames.append("raw_text")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for feat in all_features:
            row = feat.copy()
            # Flatten raw_text: replace newlines/carriage returns with a space
            # so each review stays on one CSV row
            if row.get("raw_text"):
                row["raw_text"] = re.sub(r"[\r\n]+", " ", row["raw_text"]).strip()
            writer.writerow(row)

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
        print(f"  Workload         : {feat['workload_perception']}")
        print(f"  Sentiment        : {feat['sentiment_polarity']} (score={feat['sentiment_score']})")
        print(f"  Lecturer sent.   : {feat['lecturer_sentiment']}")
        print(f"  TA sentiment     : {feat['ta_sentiment']}")
        print(f"  S/U advice       : {feat['su_advice']}")
        print(f"  Bell curve       : {feat['bell_curve']}")
        print(f"  Prior exp.       : {feat['prior_experience_level']}")
        print(f"  Topics listed    : {feat['topics_explicitly_listed']}")
        print(f"  Topics           : {feat['topics_covered'][:80]}{'…' if len(feat['topics_covered']) > 80 else ''}")
        print(f"  Resources        : {feat['resources_mentioned']}")
        print(f"  Lecturers        : {feat['lecturers_mentioned']}")
        print(f"  Assessment       : {feat['assessment_weightage']}")
        shown += 1


if __name__ == "__main__":
    import sys
    datasets = [
        ("cs1010s", "/mnt/user-data/uploads/cs1010s_reviews.json"),
        ("cs2040",  "/mnt/user-data/uploads/cs2040_reviews.json"),
    ]
    for name, path in datasets:
        print(f"\n{'='*50}")
        print(f"  Dataset: {name}")
        print(f"{'='*50}")
        run(
            input_path  = path,
            output_csv  = f"/mnt/user-data/outputs/{name}_features.csv",
            output_json = f"/mnt/user-data/outputs/{name}_features.json",
        )
