"""
Course Review Dashboard — Text Mining Prototype
================================================
Extracts aspects from course reviews using keyword-based rules + VADER sentiment,
then generates an HTML dashboard. No model training required.

Usage:
    python dashboard.py cs2040_reviews.json --cross-course cross_course_reviews.json
    
Output:
    dashboard_output.html (open in any browser)

To adapt to your data:
    1. Replace the dummy JSON files with your scraped data
    2. Adjust ASPECT_KEYWORDS if your reviews use different terminology
    3. The cross-course file is optional but enables reviewer profiling
"""

import json
import re
import sys
import os
from collections import Counter, defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Sentiment Analysis (VADER — works well on informal text like reviews)
# ---------------------------------------------------------------------------
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _sia = SentimentIntensityAnalyzer()
    def get_sentiment(text):
        """Returns 'positive', 'negative', or 'neutral' + compound score."""
        scores = _sia.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound
except ImportError:
    print("WARNING: vaderSentiment not installed. Using basic fallback.")
    def get_sentiment(text):
        pos_words = {'great','good','best','love','excellent','amazing','clear','helpful','fair','enjoy','beautiful','elegant','rewarding','passionate','interesting','useful','recommend','fantastic','wonderful'}
        neg_words = {'hard','difficult','tough','struggle','bad','poor','unfair','harsh','confusing','disorganized','overwhelmed','stressed','frustrated','boring','dry','nightmare','insane','brutal'}
        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        pos = len(words & pos_words)
        neg = len(words & neg_words)
        if pos > neg: return 'positive', 0.5
        elif neg > pos: return 'negative', -0.5
        else: return 'neutral', 0.0


# ---------------------------------------------------------------------------
# Aspect Keyword Definitions
# ---------------------------------------------------------------------------
# Each aspect maps to a list of keyword patterns (regex-compatible).
# A review is tagged with an aspect if ANY of its patterns match.
# Patterns are case-insensitive.

ASPECT_KEYWORDS = {
    # Component 1a: Content & Cognitive Demand
    'content_complexity': [
        r'\babstract\b', r'\bcomplex\b', r'\bconceptual\b', r'\btheor(?:y|etical)\b',
        r'\btechnical\b', r'\bdepth\b', r'\badvanced\b', r'\bproof', r'\belegant\b',
        r'\bcognitive\b', r'\bintellectual\b',
    ],
    'assumed_knowledge': [
        r'\bprereq', r'\bbackground\b', r'\bfoundation\b', r'\bprior\s+(?:knowledge|experience)',
        r'\bwish\s+(?:i|I)\s+had\b', r'\bneed\s+to\s+know\b', r'\bassum(?:e|es|ed|ing)\b',
        r'\bprepare[d]?\b', r'\bpreparation\b', r'\bcoming\s+from\b', r'\bbefore\s+tak(?:e|ing)\b',
        r'\bsolid\s+on\b', r'\bcomfortable\s+with\b', r'\bweak\b.*\bfundamental',
    ],
    'learning_pace': [
        r'\bpace\b', r'\bfast\b', r'\brushed?\b', r'\bfall\s+behind\b', r'\bkeep\s+up\b',
        r'\brelentless\b', r'\bevery\s+week\b', r'\bnew\s+(?:topic|concept|data\s+structure)',
        r'\bquick(?:ly)?\b', r'\bspeed\b',
    ],
    'higher_order_thinking': [
        r'\bapply\b', r'\bapplication\b', r'\bproblem[\s-]solv', r'\banalys[ie]s\b',
        r'\bcritical\s+think', r'\bnovel\s+problem', r'\btransfer\b', r'\bmemoriza?tion\b',
    ],

    # Component 1b: Workload & Time Investment
    'workload_hours': [
        r'\b\d+\s*(?:hours?|hrs?)\b', r'\btime\s+(?:spent|needed|required|consuming)',
        r'\bworkload\b', r'\btime[\s-]intensive\b', r'\bheavy\b',
    ],
    'assignment_volume': [
        r'\bassignment', r'\bproblem\s+set', r'\bhomework\b', r'\blab[s]?\b',
        r'\bproject[s]?\b', r'\bsubmission', r'\bPS\s*\d',
    ],
    'deadline_pressure': [
        r'\bdeadline', r'\boverlap', r'\bcluster', r'\btight\b', r'\bspike',
        r'\blast\s+minute\b', r'\bcrunch\b', r'\bdue\s+(?:date|within)',
    ],
    'felt_pressure': [
        r'\boverwhelm', r'\bstress', r'\bburn', r'\bpressur', r'\banxi',
        r'\bexhaust', r'\binsane\b', r'\bbrutal\b', r'\bnightmare\b',
        r'\btoo\s+much\b', r'\bate\s+up\b',
    ],

    # Component 1c: Assessment & Grading
    'scoring_difficulty': [
        r'\bhard\s+to\s+(?:score|get|do)\s+well', r'\btough\s+(?:exam|test|quiz)',
        r'\bexam\b.*\bdifficult', r'\bdifficult\b.*\bexam', r'\bmidterm\b', r'\bfinal\b',
        r'\bgrade[d]?\b', r'\bscor(?:e|ed|ing)\b', r'\bmark[s]?\b',
    ],
    'grading_fairness': [
        r'\bfair\b', r'\bunfair\b', r'\bharsh\b', r'\blenient\b', r'\bstrict\b',
        r'\bbell\s+curve\b', r'\bcurve\b', r'\bblack\s+box\b', r'\breflect',
    ],
    'taught_vs_assessed': [
        r'\bnot\s+(?:covered|taught)\b', r'\bbarely\s+covered\b', r'\bnever\s+(?:covered|taught)',
        r'\bexam\b.*\bnot\b.*\blecture', r'\btested\b.*\bnot\b.*\btaught',
        r'\balign', r'\bmismatch\b', r'\brushed\s+through\b',
    ],
    'grade_distribution': [
        r'\bmost\s+people\s+got\b', r'\baverage\b.*\bgrade', r'\bbell\s+curve\b',
        r'\bcurve\b', r'\brange\b.*\b[A-D][\+\-]?', r'\b[A-D][\+\-]?\s+(?:to|range)',
        r'\bgot\s+(?:an?\s+)?[A-D][\+\-]?\b',
    ],

    # Component 1d: Background Fit
    'background_fit': [
        r'\b(?:CS|IS|math|engineering|science)\s+(?:student|major|background)',
        r'\bnon-CS\b', r'\bcoming\s+from\b', r'\bas\s+a[n]?\s+\w+\s+(?:student|major)',
        r'\bexperience\b', r'\bcomfortable\s+with\b', r'\bstruggl',
    ],

    # Component 2a: Teaching Quality
    'teaching_clarity': [
        r'\bclear(?:ly)?\b', r'\bexplain', r'\bunderstandab', r'\bstructured\b',
        r'\borganiz', r'\bdelivery\b', r'\bslides?\b', r'\blecture[sd]?\b.*\b(?:good|great|well|clear)',
    ],
    'teaching_engagement': [
        r'\bpassion', r'\benthusiasm\b', r'\bengag', r'\bexcit', r'\binteresting\b',
        r'\bboring\b', r'\bdry\b', r'\bmonot', r'\benergy\b',
    ],
    'subject_mastery': [
        r'\bexpert\b', r'\bknowledge(?:able)?\b', r'\bmaster(?:y|ed)?\b',
        r'\bbest\s+(?:lecturer|prof|teacher)', r'\btop\s+tier\b',
    ],
    'rapport': [
        r'\bapproachab', r'\bfriendly\b', r'\bhumou?r', r'\bpersonab', r'\bwit(?:ty)?\b',
        r'\bhelpful\b', r'\bpatient\b', r'\bunderstanding\b',
    ],
    'encourages_participation': [
        r'\bparticipat', r'\binteractiv', r'\bdiscussion\b', r'\btutorial',
    ],

    # Component 2b: Support & Responsiveness
    'availability': [
        r'\boffice\s+hours?\b', r'\bemail\b', r'\brespons(?:e|ive)\b',
        r'\bavailab', r'\bpacked\b',
    ],
    'feedback_quality': [
        r'\bfeedback\b', r'\bcomment[s]?\b.*\b(?:on|about)\b', r'\bexplanation\b.*\bwrong',
        r'\bno\s+(?:explanation|feedback)\b', r'\bdidn\'t\s+know\b.*\bmistake',
    ],
    'helpfulness': [
        r'\bhelpful\b', r'\bsupport', r'\bwhen\s+(?:i|students?)\s+(?:struggle|stuck)',
        r'\bguidance\b', r'\bdirection\b',
    ],

    # Component 2c: Course Design
    'course_structure': [
        r'\bstructur', r'\borganiz', r'\blogical\b', r'\bwell[\s-](?:designed|structured|organized)',
        r'\bmodule[s]?\b.*\bflow', r'\bbuild[s]?\s+(?:on|logically)',
    ],
    'pacing': [
        r'\bpacing\b', r'\bfront[\s-]loaded\b', r'\bback[\s-]loaded\b',
        r'\bweek\s+\d', r'\bsemester\b', r'\bmidterm\b.*\bfinal\b',
    ],
    'assessment_format': [
        r'\bexam[s]?\b', r'\bquiz(?:zes)?\b', r'\bproject[s]?\b', r'\bessay',
        r'\bpresentation\b', r'\bformat\b', r'\bweighting\b', r'\b\d+%\b',
        r'\bmidterm\b', r'\bfinal\b',
    ],

    # Component 3a: Career & Learning Relevance
    'career_relevance': [
        r'\bcareer\b', r'\bjob[s]?\b', r'\binterview', r'\bindustry\b', r'\bSWE\b',
        r'\bFAANG\b', r'\boccupation\b', r'\bprofessional\b', r'\bpractical\b',
    ],
    'skill_building': [
        r'\bskill', r'\blearn(?:ed|ing|t)?\b', r'\bgain(?:ed)?\b',
        r'\bproblem[\s-]decompos', r'\balgorithmic\s+thinking\b', r'\btransfer\b',
    ],
    'intellectual_value': [
        r'\binsight', r'\beye[\s-]open', r'\bperspective', r'\bnew\s+knowledge\b',
        r'\bbeautiful\b', r'\belegant\b', r'\bintellectual\b',
    ],
    'delivers_on_description': [
        r'\bexpect(?:ed|ation)?\b', r'\bdescription\b', r'\bpromis',
        r'\bnot\s+what\s+(?:i|I)\b', r'\bmislead', r'\bdifferent\s+from\b',
    ],

    # Component 3b: Interest & Enjoyment
    'topic_interest': [
        r'\binteresting\b', r'\benjoy', r'\bfun\b', r'\bfascinat', r'\bloved?\b',
        r'\bboring\b', r'\bdry\b', r'\bdull\b', r'\btedious\b',
    ],
    'overall_satisfaction': [
        r'\brecommend', r'\bworth\b', r'\boverall\b', r'\bbest\b.*\b(?:module|course)',
        r'\bworst\b', r'\bregret\b', r'\bwould\s+(?:not\s+)?(?:take|recommend)',
    ],

    # Contextual Layer: Logistics
    'logistics_attendance': [
        r'\battendance\b', r'\bmandatory\b', r'\brequired\b.*\b(?:attend|class|lecture)',
        r'\bskip\b', r'\bcompulsory\b',
    ],
    'logistics_recording': [
        r'\brecord(?:ed|ing)?\b', r'\bwebcast\b', r'\bonline\b.*\blecture',
        r'\breplay\b', r'\brewind\b',
    ],
    'logistics_format': [
        r'\bonline\b', r'\bin[\s-]person\b', r'\bhybrid\b', r'\bclass\s+size\b',
        r'\bformat\b',
    ],
}

# Map aspects to components for dashboard grouping
COMPONENT_MAP = {
    '1a_content_demand': ['content_complexity', 'assumed_knowledge', 'learning_pace', 'higher_order_thinking'],
    '1b_workload': ['workload_hours', 'assignment_volume', 'deadline_pressure', 'felt_pressure'],
    '1c_assessment': ['scoring_difficulty', 'grading_fairness', 'taught_vs_assessed', 'grade_distribution'],
    '1d_background_fit': ['background_fit'],
    '2a_teaching_quality': ['teaching_clarity', 'teaching_engagement', 'subject_mastery', 'rapport', 'encourages_participation'],
    '2b_support': ['availability', 'feedback_quality', 'helpfulness'],
    '2c_course_design': ['course_structure', 'pacing', 'assessment_format'],
    '3a_relevance': ['career_relevance', 'skill_building', 'intellectual_value', 'delivers_on_description'],
    '3b_enjoyment': ['topic_interest', 'overall_satisfaction'],
    'logistics': ['logistics_attendance', 'logistics_recording', 'logistics_format'],
}

COMPONENT_LABELS = {
    '1a_content_demand': ('Component 1a', 'Content & Cognitive Demand', 'How intellectually challenging is the material?'),
    '1b_workload': ('Component 1b', 'Workload & Time Investment', 'How much time and effort does this course demand?'),
    '1c_assessment': ('Component 1c', 'Assessment & Grading', 'How hard is it to do well?'),
    '1d_background_fit': ('Component 1d', 'Background Fit', 'How hard was it for students like me?'),
    '2a_teaching_quality': ('Component 2a', 'Teaching Quality & Engagement', 'Is the instructor clear, engaging, and knowledgeable?'),
    '2b_support': ('Component 2b', 'Support & Responsiveness', 'Will I get help when I need it?'),
    '2c_course_design': ('Component 2c', 'Course Design & Organisation', 'Is the course well-structured?'),
    '3a_relevance': ('Component 3a', 'Career & Learning Relevance', 'Is this course useful for my goals?'),
    '3b_enjoyment': ('Component 3b', 'Interest & Enjoyment', 'Will I actually enjoy this course?'),
    'logistics': ('Contextual', 'Logistics', 'Attendance, recordings, format (varies by semester/instructor)'),
}

ASPECT_LABELS = {
    'content_complexity': 'Complexity of material',
    'assumed_knowledge': 'Assumed prior knowledge',
    'learning_pace': 'Learning pace',
    'higher_order_thinking': 'Higher-order thinking',
    'workload_hours': 'Hours per week',
    'assignment_volume': 'Assignment frequency & volume',
    'deadline_pressure': 'Deadline pressure',
    'felt_pressure': 'Overall felt pressure',
    'scoring_difficulty': 'Difficulty of scoring well',
    'grading_fairness': 'Grading fairness',
    'taught_vs_assessed': 'Taught vs. assessed alignment',
    'grade_distribution': 'Grade distribution signals',
    'background_fit': 'Background fit',
    'teaching_clarity': 'Clarity of explanation',
    'teaching_engagement': 'Enthusiasm & engagement',
    'subject_mastery': 'Subject mastery',
    'rapport': 'Rapport with students',
    'encourages_participation': 'Encourages participation',
    'availability': 'Availability outside class',
    'feedback_quality': 'Quality of feedback',
    'helpfulness': 'Helpfulness',
    'course_structure': 'Overall structure',
    'pacing': 'Pacing across semester',
    'assessment_format': 'Assessment format & structure',
    'career_relevance': 'Career usefulness',
    'skill_building': 'Skill-building',
    'intellectual_value': 'Intellectual value',
    'delivers_on_description': 'Delivers on description',
    'topic_interest': 'Topic interest',
    'overall_satisfaction': 'Overall satisfaction',
    'logistics_attendance': 'Attendance policy',
    'logistics_recording': 'Recorded lectures',
    'logistics_format': 'Delivery format',
}


# ---------------------------------------------------------------------------
# Extraction Functions
# ---------------------------------------------------------------------------

def tag_aspects(review_text):
    """Multi-label aspect tagging using keyword matching. Returns list of matched aspect keys."""
    matched = []
    text_lower = review_text.lower()
    for aspect, patterns in ASPECT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matched.append(aspect)
                break  # one match is enough per aspect
    return matched


def extract_hours(text):
    """Extract reported hours per week from review text (Pattern C)."""
    patterns = [
        r'(\d+)\s*[-–to]+\s*(\d+)\s*(?:hours?|hrs?)\s*(?:per|a|each|every)?\s*(?:week|wk)',
        r'(?:about|around|maybe|roughly|approximately)?\s*(\d+)\s*(?:\+)?\s*(?:hours?|hrs?)\s*(?:per|a|each|every)?\s*(?:week|wk)',
        r'(\d+)\s*(?:hours?|hrs?)\s*(?:per|a|each|every)\s*(?:week|wk)',
    ]
    hours = []
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                hours.extend([int(x) for x in m if x.isdigit()])
            else:
                hours.append(int(m))
    return hours


def extract_grades(text):
    """Extract grade mentions from review text (Pattern C)."""
    pattern = r'\b([A-D][\+\-]?)\b'
    # Filter out common false positives
    grades = re.findall(pattern, text)
    valid_grades = [g for g in grades if g not in ['I', 'A', 'B'] or re.search(r'got\s+(?:an?\s+)?' + re.escape(g), text, re.IGNORECASE)]
    # More targeted extraction
    got_pattern = r'(?:got|received|ended\s+with|scored)\s+(?:an?\s+)?([A-D][\+\-]?)\b'
    return re.findall(got_pattern, text, re.IGNORECASE)


def extract_instructor(text):
    """Extract instructor/professor mentions."""
    patterns = [
        r'(?:Prof(?:essor)?|Dr)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    instructors = []
    for p in patterns:
        instructors.extend(re.findall(p, text))
    return instructors


def extract_semester(text):
    """Extract academic year / semester mentions."""
    patterns = [
        r'AY\s*(\d{2}/\d{2})\s*Sem\s*(\d)',
        r'(?:Sem(?:ester)?)\s*(\d)\s*.*?(\d{4})',
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def get_top_keywords(texts, n=10):
    """Extract top keywords from a collection of texts (for word frequency display)."""
    STOPWORDS = {
        'the','a','an','is','was','were','be','been','being','this','that','these','those',
        'and','or','but','in','on','at','to','for','of','i','my','me','we','our','it','its',
        'with','from','by','as','so','if','not','no','do','did','had','has','have','very',
        'too','also','just','than','then','when','how','what','which','who','will','would',
        'could','should','can','may','much','many','some','all','each','every','per','you',
        'your','they','them','their','about','more','really','get','got','take','took',
        'think','thought','like','know','make','go','went','one','two','well','even',
        'still','thing','things','way','lot','bit','quite','pretty','don','didn','doesn',
        'isn','wasn','aren','won','haven','wouldn','couldn','shouldn','there','here',
        'course','module','class','semester','week','student','students','prof','professor',
    }
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        all_words.extend([w for w in words if w not in STOPWORDS and len(w) > 2])
    return Counter(all_words).most_common(n)


# ---------------------------------------------------------------------------
# Analysis Pipeline
# ---------------------------------------------------------------------------

def analyze_reviews(reviews, cross_course_reviews=None):
    """Main analysis pipeline. Returns structured results for dashboard rendering."""

    # Filter to top-level posts only (exclude replies for main analysis)
    top_level = [r for r in reviews if r.get('reply_to_post_id') is None]
    replies = [r for r in reviews if r.get('reply_to_post_id') is not None]

    total_reviews = len(top_level)

    # --- Step 1: Tag each review with aspects ---
    tagged_reviews = []
    for review in top_level:
        aspects = tag_aspects(review['message'])
        sentiment, score = get_sentiment(review['message'])
        hours = extract_hours(review['message'])
        grades = extract_grades(review['message'])
        instructors = extract_instructor(review['message'])
        semester = extract_semester(review['message'])

        tagged_reviews.append({
            **review,
            'aspects': aspects,
            'sentiment': sentiment,
            'sentiment_score': score,
            'extracted_hours': hours,
            'extracted_grades': grades,
            'extracted_instructors': instructors,
            'extracted_semester': semester,
        })

    # --- Step 2: Aggregate per aspect ---
    aspect_results = {}
    for aspect_key in ASPECT_KEYWORDS:
        matching = [r for r in tagged_reviews if aspect_key in r['aspects']]
        if not matching:
            aspect_results[aspect_key] = {
                'count': 0, 'total': total_reviews, 'pct': 0,
                'sentiment_breakdown': {}, 'excerpts': [], 'keywords': [],
                'extracted_hours': [], 'extracted_grades': [],
            }
            continue

        sentiments = [r['sentiment'] for r in matching]
        sent_counts = Counter(sentiments)
        sent_total = len(sentiments)

        # Per-aspect sentiment (Pattern A)
        sentiment_breakdown = {
            'positive': round(sent_counts.get('positive', 0) / sent_total * 100),
            'negative': round(sent_counts.get('negative', 0) / sent_total * 100),
            'neutral': round(sent_counts.get('neutral', 0) / sent_total * 100),
            'total_tagged': sent_total,
        }

        # Excerpts (sorted by likes)
        excerpts = sorted(matching, key=lambda x: int(x.get('likes', 0)), reverse=True)
        excerpt_list = [{
            'text': r['message'],
            'author': r['author'],
            'date': r['date'],
            'likes': r['likes'],
            'sentiment': r['sentiment'],
            'semester': r.get('extracted_semester', ''),
            'instructors': r.get('extracted_instructors', []),
        } for r in excerpts]  # top 5 by likes

        # Keywords for this aspect
        keywords = get_top_keywords([r['message'] for r in matching], n=8)

        # Extracted hours / grades (Pattern C)
        all_hours = []
        all_grades = []
        for r in matching:
            all_hours.extend(r['extracted_hours'])
            all_grades.extend(r['extracted_grades'])

        aspect_results[aspect_key] = {
            'count': len(matching),
            'total': total_reviews,
            'pct': round(len(matching) / total_reviews * 100),
            'sentiment_breakdown': sentiment_breakdown,
            'excerpts': excerpt_list,
            'keywords': keywords,
            'extracted_hours': sorted(all_hours),
            'extracted_grades': all_grades,
        }

    # --- Step 3: Cross-course reviewer profiling ---
    reviewer_profiles = {}
    if cross_course_reviews:
        # Group all reviews (target + cross-course) by author
        all_by_author = defaultdict(list)
        for r in tagged_reviews:
            all_by_author[r['author']].append({
                'course': 'TARGET',
                'message': r['message'],
                'sentiment': r['sentiment'],
                'sentiment_score': r['sentiment_score'],
                'grades': r['extracted_grades'],
                'hours': r['extracted_hours'],
            })
        for r in cross_course_reviews:
            sent, score = get_sentiment(r['message'])
            grades = extract_grades(r['message'])
            hours = extract_hours(r['message'])
            all_by_author[r['author']].append({
                'course': r['course_code'],
                'message': r['message'],
                'sentiment': sent,
                'sentiment_score': score,
                'grades': grades,
                'hours': hours,
            })

        for author, author_reviews in all_by_author.items():
            other_courses = [r for r in author_reviews if r['course'] != 'TARGET']
            target_reviews = [r for r in author_reviews if r['course'] == 'TARGET']
            if not other_courses:
                continue

            # Calibration: average sentiment across all courses vs. target
            avg_other_sentiment = sum(r['sentiment_score'] for r in other_courses) / len(other_courses)
            avg_target_sentiment = sum(r['sentiment_score'] for r in target_reviews) / len(target_reviews) if target_reviews else 0

            reviewer_profiles[author] = {
                'other_courses': [r['course'] for r in other_courses],
                'other_grades': [g for r in other_courses for g in r['grades']],
                'avg_other_sentiment': round(avg_other_sentiment, 2),
                'avg_target_sentiment': round(avg_target_sentiment, 2),
                'courses_reviewed': len(other_courses),
                'sentiment_diff': round(avg_target_sentiment - avg_other_sentiment, 2),
            }

    # --- Step 4: Instructor summary ---
    all_instructors = Counter()
    for r in tagged_reviews:
        for inst in r['extracted_instructors']:
            all_instructors[inst] += 1

    # --- Step 5: Overall summary stats ---
    all_hours = []
    all_grades = []
    for r in tagged_reviews:
        all_hours.extend(r['extracted_hours'])
        all_grades.extend(r['extracted_grades'])

    overall_sentiments = Counter(r['sentiment'] for r in tagged_reviews)

    summary = {
        'total_reviews': total_reviews,
        'total_replies': len(replies),
        'overall_sentiment': {
            'positive': overall_sentiments.get('positive', 0),
            'negative': overall_sentiments.get('negative', 0),
            'neutral': overall_sentiments.get('neutral', 0),
        },
        'hours_range': (min(all_hours), max(all_hours)) if all_hours else None,
        'hours_median': sorted(all_hours)[len(all_hours)//2] if all_hours else None,
        'grade_distribution': Counter(all_grades),
        'instructors': dict(all_instructors),
    }

    return {
        'summary': summary,
        'aspects': aspect_results,
        'reviewer_profiles': reviewer_profiles,
        'tagged_reviews': tagged_reviews,
    }


# ---------------------------------------------------------------------------
# HTML Dashboard Generator
# ---------------------------------------------------------------------------

def generate_dashboard_html(results, course_code="CS2040"):
    """Generate a self-contained HTML dashboard from analysis results."""

    summary = results['summary']
    aspects = results['aspects']
    profiles = results['reviewer_profiles']

    # --- Helper functions for HTML generation ---
    def sentiment_bar(breakdown, min_count=10):
        """Generate a sentiment bar or fallback text."""
        total = breakdown.get('total_tagged', 0)
        if total == 0:
            return '<span class="muted">No reviews tagged</span>'
        if total < min_count:
            return f'<span class="muted">Only {total} reviews tagged — showing excerpts instead</span>'
        pos = breakdown['positive']
        neg = breakdown['negative']
        neu = breakdown['neutral']
        return f'''
        <div class="sentiment-bar-container">
            <div class="sentiment-bar">
                <div class="sentiment-pos" style="width:{pos}%" title="Positive: {pos}%"></div>
                <div class="sentiment-neu" style="width:{neu}%" title="Neutral: {neu}%"></div>
                <div class="sentiment-neg" style="width:{neg}%" title="Negative: {neg}%"></div>
            </div>
            <div class="sentiment-labels">
                <span class="label-pos">{pos}% positive</span>
                <span class="label-neg">{neg}% negative</span>
                <span class="label-count">({total} reviews)</span>
            </div>
        </div>'''

    def excerpt_html(excerpts):
        """Generate excerpt cards."""
        if not excerpts:
            return '<p class="muted">No relevant excerpts found.</p>'
        html = ''
        for e in excerpts:
            sem_info = ''
            if e.get('instructors'):
                sem_info += f' · Prof {", ".join(e["instructors"])}'
            if e.get('semester'):
                sem_info += f' · {e["semester"]}'
            sent_class = e['sentiment']
            html += f'''
            <div class="excerpt {sent_class}">
                <p class="excerpt-text">"{e['text']}"</p>
                <div class="excerpt-meta">
                    <span class="author">— {e['author']}</span>
                    <span class="likes">👍 {e['likes']}</span>
                    {f'<span class="sem-info">{sem_info}</span>' if sem_info else ''}
                </div>
            </div>'''
        return html

    def keyword_pills(keywords):
        """Generate keyword pills."""
        if not keywords:
            return ''
        pills = ''.join(f'<span class="pill">{word} ({count})</span>' for word, count in keywords[:6])
        return f'<div class="keywords">{pills}</div>'

    def aspect_section(aspect_key, data):
        """Generate a single aspect section."""
        label = ASPECT_LABELS.get(aspect_key, aspect_key)
        count = data['count']
        total = data['total']
        pct = data['pct']

        mention_badge = f'<span class="mention-badge">{count}/{total} reviews ({pct}%)</span>' if count > 0 else '<span class="mention-badge zero">0 mentions</span>'

        # Special: hours extraction
        hours_html = ''
        if aspect_key == 'workload_hours' and data['extracted_hours']:
            hrs = data['extracted_hours']
            if len(hrs) >= 2:
                hours_html = f'<div class="extracted-fact">📊 Reported hours/week: {min(hrs)}–{max(hrs)} hrs (median: {sorted(hrs)[len(hrs)//2]})</div>'
            elif len(hrs) == 1:
                hours_html = f'<div class="extracted-fact">📊 Reported: ~{hrs[0]} hrs/week</div>'

        # Special: grade extraction
        grades_html = ''
        if aspect_key == 'grade_distribution' and data['extracted_grades']:
            grade_counts = Counter(data['extracted_grades'])
            grades_str = ', '.join(f'{g}: {c}' for g, c in grade_counts.most_common())
            grades_html = f'<div class="extracted-fact">📊 Grades mentioned: {grades_str}</div>'

        return f'''
        <div class="aspect-block">
            <div class="aspect-header">
                <h4>{label}</h4>
                {mention_badge}
            </div>
            {hours_html}
            {grades_html}
            {sentiment_bar(data['sentiment_breakdown'])}
            {keyword_pills(data['keywords'])}
            <details>
                <summary>View {count} excerpts)</summary>
                {excerpt_html(data['excerpts'])} 
            </details>
        </div>'''

    # --- Build component sections ---
    components_html = ''
    for comp_key, aspect_keys in COMPONENT_MAP.items():
        comp_num, comp_name, comp_question = COMPONENT_LABELS[comp_key]

        # Determine parent component heading
        if comp_key.startswith('1') and '1a' in comp_key:
            components_html += '<div class="component-group"><h2>Component 1: How Hard Will This Be (For Me)?</h2>'
        elif comp_key.startswith('2') and '2a' in comp_key:
            components_html += '</div><div class="component-group"><h2>Component 2: How\'s the Teaching?</h2>'
        elif comp_key.startswith('3') and '3a' in comp_key:
            components_html += '</div><div class="component-group"><h2>Component 3: Is This Course Worth Taking?</h2>'
        elif comp_key == 'logistics':
            components_html += '</div><div class="component-group"><h2>Contextual Layer: Logistics</h2>'

        components_html += f'''
        <div class="component-card">
            <div class="component-title">
                <span class="comp-num">{comp_num}</span>
                <div>
                    <h3>{comp_name}</h3>
                    <p class="comp-question">{comp_question}</p>
                </div>
            </div>'''

        for ak in aspect_keys:
            data = aspects.get(ak, {'count':0, 'total':0, 'pct':0, 'sentiment_breakdown':{}, 'excerpts':[], 'keywords':[], 'extracted_hours':[], 'extracted_grades':[]})
            components_html += aspect_section(ak, data)

        components_html += '</div>'

    if components_html.count('<div class="component-group">') > 0:
        components_html += '</div>'

    # --- Reviewer profiles section ---
    profiles_html = ''
    if profiles:
        profiles_html = '<div class="component-group"><h2>Reviewer Profiles (Cross-Course)</h2><div class="component-card">'
        profiles_html += '<p class="comp-question">How do reviewers of this course rate other courses? Helps calibrate whether a reviewer is generally harsh/generous or specifically so for this course.</p>'
        for author, profile in sorted(profiles.items(), key=lambda x: x[1]['courses_reviewed'], reverse=True):
            diff = profile['sentiment_diff']
            if diff < -0.2:
                calibration = '⚠️ Rated this course notably lower than their other courses'
            elif diff > 0.2:
                calibration = '✅ Rated this course notably higher than their other courses'
            else:
                calibration = '➖ Similar sentiment to their other course reviews'

            other_courses_str = ', '.join(profile['other_courses'])
            grades_str = ', '.join(profile['other_grades']) if profile['other_grades'] else 'N/A'

            profiles_html += f'''
            <div class="profile-card">
                <div class="profile-header">
                    <strong>{author}</strong>
                    <span class="mention-badge">{profile['courses_reviewed']} other courses reviewed</span>
                </div>
                <div class="profile-body">
                    <div>Also reviewed: {other_courses_str}</div>
                    <div>Grades in other courses: {grades_str}</div>
                    <div>{calibration}</div>
                </div>
            </div>'''
        profiles_html += '</div></div>'

    # --- Summary section ---
    sent = summary['overall_sentiment']
    sent_total = sent['positive'] + sent['negative'] + sent['neutral']
    hours_info = f"{summary['hours_range'][0]}–{summary['hours_range'][1]} hrs/week (median: {summary['hours_median']})" if summary['hours_range'] else 'Not enough data'
    grade_dist = ', '.join(f'{g}: {c}' for g, c in summary['grade_distribution'].most_common()) if summary['grade_distribution'] else 'Not enough data'
    instructors_info = ', '.join(f'{name} ({count} mentions)' for name, count in summary['instructors'].items()) if summary['instructors'] else 'Not mentioned'

    # --- Full HTML ---
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Course Dashboard — {course_code}</title>
<style>
    :root {{
        --bg: #0f1117;
        --surface: #1a1d27;
        --surface2: #232733;
        --border: #2e3344;
        --text: #e2e4e9;
        --text-muted: #8b8fa3;
        --accent: #6c8cff;
        --accent-dim: #6c8cff22;
        --pos: #4ade80;
        --pos-bg: #4ade8018;
        --neg: #f87171;
        --neg-bg: #f8717118;
        --neu: #fbbf24;
        --neu-bg: #fbbf2418;
        --radius: 10px;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.6;
        padding: 2rem;
        max-width: 900px;
        margin: 0 auto;
    }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
    h2 {{
        font-size: 1.3rem;
        color: var(--accent);
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }}
    h3 {{ font-size: 1.1rem; margin: 0; }}
    h4 {{ font-size: 0.95rem; margin: 0; color: var(--text); }}
    .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; }}

    /* Summary cards */
    .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }}
    .summary-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem;
    }}
    .summary-card .label {{ color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .summary-card .value {{ font-size: 1.3rem; font-weight: 600; margin-top: 0.25rem; }}

    /* Component cards */
    .component-group {{ margin-bottom: 1rem; }}
    .component-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem;
        margin-bottom: 1rem;
    }}
    .component-title {{
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }}
    .comp-num {{
        background: var(--accent-dim);
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        white-space: nowrap;
        margin-top: 0.1rem;
    }}
    .comp-question {{ color: var(--text-muted); font-size: 0.9rem; margin-top: 0.15rem; }}

    /* Aspect blocks */
    .aspect-block {{
        border-top: 1px solid var(--border);
        padding: 1rem 0;
    }}
    .aspect-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }}
    .mention-badge {{
        background: var(--surface2);
        color: var(--text-muted);
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
    }}
    .mention-badge.zero {{ opacity: 0.5; }}

    /* Sentiment bar */
    .sentiment-bar-container {{ margin: 0.5rem 0; }}
    .sentiment-bar {{
        display: flex;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        background: var(--surface2);
    }}
    .sentiment-pos {{ background: var(--pos); }}
    .sentiment-neu {{ background: var(--neu); }}
    .sentiment-neg {{ background: var(--neg); }}
    .sentiment-labels {{
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        margin-top: 0.25rem;
        color: var(--text-muted);
    }}
    .label-pos {{ color: var(--pos); }}
    .label-neg {{ color: var(--neg); }}
    .label-count {{ margin-left: auto; }}

    /* Extracted facts */
    .extracted-fact {{
        background: var(--accent-dim);
        border: 1px solid var(--accent);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }}

    /* Keywords */
    .keywords {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.5rem 0; }}
    .pill {{
        background: var(--surface2);
        color: var(--text-muted);
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
    }}

    /* Excerpts */
    details {{ margin-top: 0.5rem; }}
    summary {{
        cursor: pointer;
        color: var(--accent);
        font-size: 0.85rem;
        padding: 0.25rem 0;
    }}
    summary:hover {{ text-decoration: underline; }}
    .excerpt {{
        background: var(--surface2);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 3px solid var(--border);
    }}
    .excerpt.positive {{ border-left-color: var(--pos); }}
    .excerpt.negative {{ border-left-color: var(--neg); }}
    .excerpt-text {{ font-size: 0.85rem; line-height: 1.5; font-style: italic; }}
    .excerpt-meta {{
        display: flex;
        gap: 0.75rem;
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.4rem;
        flex-wrap: wrap;
    }}

    /* Profiles */
    .profile-card {{
        border-top: 1px solid var(--border);
        padding: 0.75rem 0;
    }}
    .profile-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }}
    .profile-body {{
        font-size: 0.85rem;
        color: var(--text-muted);
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }}

    .muted {{ color: var(--text-muted); font-size: 0.85rem; font-style: italic; }}

    /* Method note */
    .method-note {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: var(--text-muted);
    }}
    .method-note h3 {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.5rem; }}
</style>
</head>
<body>
    <h1>📊 {course_code} Course Dashboard</h1>
    <p class="subtitle">Text-mining prototype · {summary['total_reviews']} top-level reviews analysed · {summary['total_replies']} replies</p>

    <div class="summary-grid">
        <div class="summary-card">
            <div class="label">Overall Sentiment</div>
            <div class="value" style="color:var(--pos)">{round(sent['positive']/sent_total*100) if sent_total else 0}% positive</div>
        </div>
        <div class="summary-card">
            <div class="label">Reported Hours/Week</div>
            <div class="value">{hours_info}</div>
        </div>
        <div class="summary-card">
            <div class="label">Grades Mentioned</div>
            <div class="value" style="font-size:1rem">{grade_dist}</div>
        </div>
        <div class="summary-card">
            <div class="label">Instructors</div>
            <div class="value" style="font-size:1rem">{instructors_info}</div>
        </div>
    </div>

    {components_html}

    {profiles_html}

    <div class="method-note">
        <h3>Methodology</h3>
        <p>Aspect tagging: keyword/regex matching (no model training). Sentiment: VADER (rule-based, designed for informal text). 
        Hours & grades: regex numeric extraction. Reviewer profiles: cross-course sentiment comparison by author.
        This is a prototype to validate whether the dataset contains extractable signal. Results should be interpreted as directional, not precise.</p>
    </div>
</body>
</html>'''

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Course Review Dashboard Generator')
    parser.add_argument('reviews_file', help='Path to reviews JSON file')
    parser.add_argument('--cross-course', help='Path to cross-course reviews JSON file', default=None)
    parser.add_argument('--course-code', help='Course code label', default='CS2040')
    parser.add_argument('--output', help='Output HTML file path', default='dashboard_output.html')
    args = parser.parse_args()

    # Load reviews
    with open(args.reviews_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    print(f"Loaded {len(reviews)} reviews from {args.reviews_file}")

    # Load cross-course reviews if provided
    cross_course = None
    if args.cross_course:
        with open(args.cross_course, 'r', encoding='utf-8') as f:
            cross_course = json.load(f)
        print(f"Loaded {len(cross_course)} cross-course reviews from {args.cross_course}")

    # Analyze
    results = analyze_reviews(reviews, cross_course)

    # Print summary to console
    print(f"\n{'='*50}")
    print(f"ANALYSIS SUMMARY — {args.course_code}")
    print(f"{'='*50}")
    print(f"Total reviews: {results['summary']['total_reviews']}")
    print(f"Overall sentiment: {results['summary']['overall_sentiment']}")
    if results['summary']['hours_range']:
        print(f"Hours/week range: {results['summary']['hours_range'][0]}–{results['summary']['hours_range'][1]} (median: {results['summary']['hours_median']})")
    if results['summary']['grade_distribution']:
        print(f"Grades mentioned: {dict(results['summary']['grade_distribution'])}")
    print(f"Instructors: {results['summary']['instructors']}")
    print(f"\nAspect coverage:")
    for comp_key, aspect_keys in COMPONENT_MAP.items():
        label = COMPONENT_LABELS[comp_key][1]
        total_mentions = sum(results['aspects'].get(ak, {}).get('count', 0) for ak in aspect_keys)
        print(f"  {label}: {total_mentions} total aspect mentions")

    # Generate dashboard
    html = generate_dashboard_html(results, args.course_code)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nDashboard saved to {args.output}")


if __name__ == '__main__':
    main()
