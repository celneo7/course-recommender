"""
Course Review EDA Dashboard — Plotly Edition
=============================================
Reads the extraction results from dashboard.py and generates an interactive
HTML dashboard with plotly charts.

Usage:
    python eda_dashboard.py cs2040_reviews.json --cross-course cross_course_reviews.json

Output:
    eda_dashboard.html (open in any browser, fully interactive)
"""

import json
import re
import sys
import os
from collections import Counter, defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ---------------------------------------------------------------------------
# Import extraction logic from dashboard.py
# ---------------------------------------------------------------------------
from dashboard import (
    ASPECT_KEYWORDS, COMPONENT_MAP, COMPONENT_LABELS, ASPECT_LABELS,
    tag_aspects, get_sentiment, extract_hours, extract_grades,
    extract_instructor, extract_semester, get_top_keywords,
)

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
BG = "#12141f"
SURFACE = "#1a1d2e"
BORDER = "#2a2f45"
TEXT = "#e2e8f0"
TEXT_MUTED = "#64748b"
POS = "#34d399"
NEG = "#f87171"
NEU = "#fbbf24"
PALETTE = ["#2dd4bf","#22d3ee","#60a5fa","#818cf8","#a78bfa","#f472b6",
           "#fb7185","#fbbf24","#a3e635","#34d399","#fb923c","#38bdf8"]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(family="DM Sans, system-ui, sans-serif", color=TEXT, size=12),
    margin=dict(l=60, r=30, t=50, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)

def styled_layout(**kwargs):
    layout = {**LAYOUT_DEFAULTS}
    layout.update(kwargs)
    return layout


# ---------------------------------------------------------------------------
# Analysis pipeline (reuses dashboard.py logic)
# ---------------------------------------------------------------------------

def analyze(reviews_file, cross_course_file=None):
    with open(reviews_file, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    cross_course = None
    if cross_course_file:
        with open(cross_course_file, "r", encoding="utf-8") as f:
            cross_course = json.load(f)

    # Filter top-level
    top_level = [r for r in reviews if r.get("reply_to_post_id") is None]
    total = len(top_level)

    # Tag + sentiment
    tagged = []
    for r in top_level:
        aspects = tag_aspects(r["message"])
        sent, score = get_sentiment(r["message"])
        hours = extract_hours(r["message"])
        grades = extract_grades(r["message"])
        instructors = extract_instructor(r["message"])
        semester = extract_semester(r["message"])
        tagged.append({
            **r, "aspects": aspects, "sentiment": sent, "score": score,
            "hours": hours, "grades": grades, "instructors": instructors, "semester": semester,
        })

    # Per-aspect stats
    aspect_stats = {}
    for key in ASPECT_KEYWORDS:
        matching = [r for r in tagged if key in r["aspects"]]
        if not matching:
            aspect_stats[key] = {"count": 0, "pos": 0, "neg": 0, "neu": 0, "avg_score": 0, "scores": []}
            continue
        pos = sum(1 for r in matching if r["sentiment"] == "positive")
        neg = sum(1 for r in matching if r["sentiment"] == "negative")
        neu = len(matching) - pos - neg
        avg = sum(r["score"] for r in matching) / len(matching)
        aspect_stats[key] = {
            "count": len(matching), "pos": pos, "neg": neg, "neu": neu,
            "avg_score": round(avg, 3), "scores": [r["score"] for r in matching],
        }

    # Co-occurrence
    aspect_keys = list(ASPECT_KEYWORDS.keys())
    review_aspects = {r.get("post_id", i): set(r["aspects"]) for i, r in enumerate(tagged)}
    cooccur = []
    for i, a in enumerate(aspect_keys):
        for j, b in enumerate(aspect_keys):
            if i >= j:
                continue
            overlap = sum(1 for aspects in review_aspects.values() if a in aspects and b in aspects)
            if overlap >= 2:
                cooccur.append((ASPECT_LABELS.get(a, a), ASPECT_LABELS.get(b, b), overlap))
    cooccur.sort(key=lambda x: x[2], reverse=True)

    # Instructor grouping
    instr_data = defaultdict(list)
    for r in tagged:
        for inst in r["instructors"]:
            instr_data[inst].append(r)

    # Semester grouping
    sem_data = defaultdict(list)
    for r in tagged:
        s = r["semester"] or "Unknown"
        sem_data[s].append(r)

    # Hours vs sentiment
    hours_sent = []
    for r in tagged:
        if r["hours"]:
            avg_h = sum(r["hours"]) / len(r["hours"])
            hours_sent.append({"author": r["author"], "hours": avg_h, "score": r["score"], "sentiment": r["sentiment"]})

    # Cross-course profiles
    profiles = {}
    if cross_course:
        by_author = defaultdict(list)
        for r in tagged:
            by_author[r["author"]].append({"course": "TARGET", "score": r["score"]})
        for r in cross_course:
            s, sc = get_sentiment(r["message"])
            by_author[r["author"]].append({"course": r.get("course_code", "?"), "score": sc})
        for author, revs in by_author.items():
            target = [x for x in revs if x["course"] == "TARGET"]
            other = [x for x in revs if x["course"] != "TARGET"]
            if other:
                profiles[author] = {
                    "other_courses": list(set(x["course"] for x in other)),
                    "avg_target": round(sum(x["score"] for x in target) / len(target), 2) if target else 0,
                    "avg_other": round(sum(x["score"] for x in other) / len(other), 2),
                }

    return {
        "tagged": tagged, "total": total, "aspect_stats": aspect_stats,
        "cooccur": cooccur, "instr_data": dict(instr_data), "sem_data": dict(sem_data),
        "hours_sent": hours_sent, "profiles": profiles,
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def fig_overall_sentiment(tagged, total):
    pos = sum(1 for r in tagged if r["sentiment"] == "positive")
    neg = sum(1 for r in tagged if r["sentiment"] == "negative")
    neu = total - pos - neg
    fig = go.Figure(go.Pie(
        labels=["Positive", "Negative", "Neutral"], values=[pos, neg, neu],
        marker=dict(colors=[POS, NEG, NEU]),
        hole=0.55, textinfo="label+percent", textfont=dict(size=12),
    ))
    fig.update_layout(**styled_layout(title="Overall Sentiment Distribution", height=350))
    return fig


def fig_sentiment_histogram(tagged):
    scores = [r["score"] for r in tagged]
    fig = go.Figure(go.Histogram(
        x=scores, nbinsx=20,
        marker=dict(color=PALETTE[2], line=dict(width=1, color=BORDER)),
    ))
    fig.update_layout(**styled_layout(
        title="Sentiment Score Distribution (VADER Compound)",
        xaxis_title="Compound Score", yaxis_title="Count", height=350,
    ))
    return fig


def fig_aspect_coverage(aspect_stats, total):
    items = sorted(aspect_stats.items(), key=lambda x: x[1]["count"], reverse=True)
    labels = [ASPECT_LABELS.get(k, k) for k, _ in items]
    counts = [v["count"] for _, v in items]
    colors = [PALETTE[i % len(PALETTE)] if c >= 3 else "#4a5068" for i, c in enumerate(counts)]

    fig = go.Figure(go.Bar(
        y=labels, x=counts, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{c}/{total}" for c in counts], textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(**styled_layout(
        title=f"Aspect Mention Frequency (out of {total} reviews)",
        xaxis_title="Number of Reviews", height=max(450, len(labels) * 22),
        yaxis=dict(autorange="reversed", gridcolor=BORDER),
        margin=dict(l=140, r=50, t=50, b=40),
    ))
    return fig


def fig_aspect_sentiment(aspect_stats):
    items = [(k, v) for k, v in aspect_stats.items() if v["count"] >= 3]
    items.sort(key=lambda x: x[1]["avg_score"])
    labels = [ASPECT_LABELS.get(k, k) for k, _ in items]
    scores = [v["avg_score"] for _, v in items]
    colors = [POS if s >= 0 else NEG for s in scores]

    fig = go.Figure(go.Bar(
        y=labels, x=scores, orientation="h",
        marker=dict(color=colors),
        text=[f"{s:.2f}" for s in scores], textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(**styled_layout(
        title="Average Sentiment per Aspect (3+ reviews)",
        xaxis_title="Avg VADER Score", height=max(350, len(labels) * 22),
        xaxis=dict(range=[-1, 1], gridcolor=BORDER, zerolinecolor="#4a5068", zerolinewidth=2),
        yaxis=dict(gridcolor=BORDER),
        margin=dict(l=140, r=50, t=50, b=40),
    ))
    return fig


def fig_sentiment_stacked(aspect_stats):
    items = [(k, v) for k, v in aspect_stats.items() if v["count"] >= 3]
    items.sort(key=lambda x: x[1]["count"], reverse=True)
    labels = [ASPECT_LABELS.get(k, k) for k, _ in items[:15]]
    pos = [v["pos"] for _, v in items[:15]]
    neg = [v["neg"] for _, v in items[:15]]
    neu = [v["neu"] for _, v in items[:15]]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=pos, name="Positive", orientation="h", marker_color=POS))
    fig.add_trace(go.Bar(y=labels, x=neu, name="Neutral", orientation="h", marker_color=NEU))
    fig.add_trace(go.Bar(y=labels, x=neg, name="Negative", orientation="h", marker_color=NEG))
    fig.update_layout(**styled_layout(
        title="Sentiment Breakdown per Aspect",
        barmode="stack", height=max(380, len(labels) * 26),
        yaxis=dict(autorange="reversed", gridcolor=BORDER),
        margin=dict(l=140, r=30, t=50, b=40),
        legend=dict(orientation="h", y=1.08),
    ))
    return fig


def fig_temporal(sem_data):
    semesters = sorted(sem_data.keys())
    avg_scores = []
    counts = []
    pos_counts = []
    neg_counts = []
    for s in semesters:
        revs = sem_data[s]
        avg_scores.append(round(sum(r["score"] for r in revs) / len(revs), 2))
        counts.append(len(revs))
        pos_counts.append(sum(1 for r in revs if r["sentiment"] == "positive"))
        neg_counts.append(sum(1 for r in revs if r["sentiment"] == "negative"))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Avg Sentiment Over Time", "Review Volume Over Time"))
    fig.add_trace(go.Scatter(
        x=semesters, y=avg_scores, mode="lines+markers",
        marker=dict(size=10, color=PALETTE[1]), line=dict(width=3, color=PALETTE[1]),
        name="Avg Sentiment",
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=semesters, y=pos_counts, name="Positive", marker_color=POS), row=1, col=2)
    fig.add_trace(go.Bar(x=semesters, y=neg_counts, name="Negative", marker_color=NEG), row=1, col=2)
    fig.update_layout(**styled_layout(
        height=350, barmode="stack",
        legend=dict(orientation="h", y=-0.15),
    ))
    fig.update_yaxes(gridcolor=BORDER)
    fig.update_xaxes(gridcolor=BORDER)
    return fig


def fig_instructor_comparison(instr_data, aspect_stats):
    instructors = sorted(instr_data.keys())
    if len(instructors) < 2:
        return None

    # Overall stats
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Overall Sentiment by Instructor", "Top Aspect Sentiment by Instructor"),
        specs=[[{"type": "bar"}, {"type": "bar"}]])

    avgs = [round(sum(r["score"] for r in instr_data[i]) / len(instr_data[i]), 2) for i in instructors]
    fig.add_trace(go.Bar(
        x=[f"Prof {i}" for i in instructors], y=avgs,
        marker_color=[POS if a >= 0 else NEG for a in avgs],
        text=[f"{a:.2f}" for a in avgs], textposition="outside",
    ), row=1, col=1)

    # Per-aspect comparison for top aspects
    top_aspects = sorted(aspect_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:6]
    for idx, inst in enumerate(instructors):
        inst_reviews = {r.get("post_id", i): r for i, r in enumerate(instr_data[inst])}
        aspect_avgs = []
        for ak, av in top_aspects:
            matching = [r for r in instr_data[inst] if ak in r["aspects"]]
            if matching:
                aspect_avgs.append(round(sum(r["score"] for r in matching) / len(matching), 2))
            else:
                aspect_avgs.append(0)
        fig.add_trace(go.Bar(
            x=[ASPECT_LABELS.get(k, k) for k, _ in top_aspects],
            y=aspect_avgs, name=f"Prof {inst}",
            marker_color=PALETTE[idx * 3 % len(PALETTE)],
        ), row=1, col=2)

    fig.update_layout(**styled_layout(
        height=380, barmode="group",
        legend=dict(orientation="h", y=-0.15),
    ))
    fig.update_yaxes(gridcolor=BORDER)
    return fig


def fig_hours_vs_sentiment(hours_sent):
    if not hours_sent:
        return None
    fig = go.Figure(go.Scatter(
        x=[h["hours"] for h in hours_sent],
        y=[h["score"] for h in hours_sent],
        mode="markers+text",
        text=[h["author"][:12] for h in hours_sent],
        textposition="top center", textfont=dict(size=9, color=TEXT_MUTED),
        marker=dict(
            size=12,
            color=[POS if h["score"] >= 0 else NEG for h in hours_sent],
            line=dict(width=1, color=BORDER),
        ),
    ))
    fig.update_layout(**styled_layout(
        title="Hours per Week vs Sentiment Score",
        xaxis_title="Reported Hours/Week",
        yaxis_title="VADER Compound Score",
        height=380,
        yaxis=dict(range=[-1, 1], gridcolor=BORDER, zerolinecolor="#4a5068", zerolinewidth=2),
    ))
    return fig


def fig_cooccurrence(cooccur):
    if not cooccur:
        return None
    top = cooccur[:15]
    labels = [f"{a} + {b}" for a, b, _ in top]
    values = [v for _, _, v in top]

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=PALETTE[4]),
        text=values, textposition="outside",
    ))
    fig.update_layout(**styled_layout(
        title="Aspect Co-occurrence (Top Pairs)",
        xaxis_title="Shared Reviews", height=max(350, len(labels) * 28),
        yaxis=dict(autorange="reversed", gridcolor=BORDER),
        margin=dict(l=220, r=40, t=50, b=40),
    ))
    return fig


def fig_extraction_quality(aspect_stats, total):
    items = sorted(aspect_stats.items(), key=lambda x: x[1]["count"], reverse=True)
    labels = [ASPECT_LABELS.get(k, k) for k, _ in items]
    counts = [v["count"] for _, v in items]

    def quality(c):
        if c >= 10: return "Strong"
        if c >= 5: return "Moderate"
        if c >= 2: return "Weak"
        return "Insufficient"

    def qcolor(c):
        if c >= 10: return POS
        if c >= 5: return NEU
        if c >= 2: return "#fb923c"
        return NEG

    fig = go.Figure(go.Bar(
        y=labels, x=counts, orientation="h",
        marker=dict(color=[qcolor(c) for c in counts]),
        text=[f"{c} — {quality(c)}" for c in counts],
        textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(**styled_layout(
        title="Extraction Quality per Aspect",
        xaxis_title="Tagged Reviews",
        height=max(500, len(labels) * 22),
        yaxis=dict(autorange="reversed", gridcolor=BORDER),
        margin=dict(l=140, r=100, t=50, b=40),
    ))
    return fig


def fig_reviewer_profiles(profiles):
    if not profiles:
        return None
    authors = sorted(profiles.keys(), key=lambda a: profiles[a]["avg_target"])
    target_scores = [profiles[a]["avg_target"] for a in authors]
    other_scores = [profiles[a]["avg_other"] for a in authors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=authors, x=target_scores, name="This Course", orientation="h",
        marker_color=PALETTE[2],
    ))
    fig.add_trace(go.Bar(
        y=authors, x=other_scores, name="Other Courses", orientation="h",
        marker_color=PALETTE[5],
    ))
    fig.update_layout(**styled_layout(
        title="Reviewer Calibration: This Course vs Other Courses",
        xaxis_title="Avg Sentiment Score", barmode="group",
        height=max(300, len(authors) * 40),
        yaxis=dict(gridcolor=BORDER),
        margin=dict(l=120, r=30, t=50, b=40),
        legend=dict(orientation="h", y=1.08),
    ))
    return fig


def fig_component_coverage(aspect_stats, total):
    comp_data = []
    for comp_key, aspects in COMPONENT_MAP.items():
        label = COMPONENT_LABELS[comp_key][1]
        all_ids = set()
        for a in aspects:
            matching = [r for r in range(total)]  # placeholder
        count = len(set().union(*(
            set(i for i, r in enumerate(tagged_global) if a in r["aspects"])
            for a in aspects
        )))
        comp_data.append({"name": label, "count": count})
    return comp_data

# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

tagged_global = []  # will be set in main

def build_html(results, course_code):
    global tagged_global
    tagged_global = results["tagged"]

    figures = []

    # Section 1: Overview
    figures.append(("Overview", "overall_sentiment", fig_overall_sentiment(results["tagged"], results["total"])))
    figures.append(("Overview", "sentiment_dist", fig_sentiment_histogram(results["tagged"])))

    # Section 2: Aspect Analysis
    figures.append(("Aspect Analysis", "coverage", fig_aspect_coverage(results["aspect_stats"], results["total"])))
    figures.append(("Aspect Analysis", "sent_stacked", fig_sentiment_stacked(results["aspect_stats"])))
    figures.append(("Aspect Analysis", "avg_sentiment", fig_aspect_sentiment(results["aspect_stats"])))

    co = fig_cooccurrence(results["cooccur"])
    if co:
        figures.append(("Aspect Analysis", "cooccurrence", co))

    # Section 3: Temporal & Instructor
    if results["sem_data"]:
        figures.append(("Temporal & Instructor", "temporal", fig_temporal(results["sem_data"])))
    inst = fig_instructor_comparison(results["instr_data"], results["aspect_stats"])
    if inst:
        figures.append(("Temporal & Instructor", "instructor", inst))

    # Section 4: Deep Dives
    hrs = fig_hours_vs_sentiment(results["hours_sent"])
    if hrs:
        figures.append(("Deep Dives", "hours_sent", hrs))

    prof = fig_reviewer_profiles(results["profiles"])
    if prof:
        figures.append(("Deep Dives", "profiles", prof))

    # Section 5: Extraction Quality
    figures.append(("Extraction Quality", "quality", fig_extraction_quality(results["aspect_stats"], results["total"])))

    # Build HTML
    sections = {}
    for section, name, fig in figures:
        if section not in sections:
            sections[section] = []
        sections[section].append((name, fig))

    chart_divs = []
    nav_links = []
    for section, charts in sections.items():
        section_id = section.lower().replace(" ", "_").replace("&", "and")
        nav_links.append(f'<a href="#{section_id}" class="nav-link">{section}</a>')
        chart_divs.append(f'<h2 id="{section_id}" class="section-title">{section}</h2>')
        for name, fig in charts:
            chart_divs.append(f'<div class="chart-card">{fig.to_html(full_html=False, include_plotlyjs=False, config=dict(displayModeBar=True, displaylogo=False))}</div>')

    # Summary stats
    pos = sum(1 for r in results["tagged"] if r["sentiment"] == "positive")
    neg = sum(1 for r in results["tagged"] if r["sentiment"] == "negative")
    total = results["total"]
    all_hours = [h for r in results["tagged"] for h in r["hours"]]
    hours_str = f"{min(all_hours)}–{max(all_hours)} hrs/wk" if all_hours else "N/A"
    aspects_per_review = sum(len(r["aspects"]) for r in results["tagged"]) / total if total else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Dashboard — {course_code}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'DM Sans', system-ui, -apple-system, sans-serif;
        background: {BG}; color: {TEXT};
        line-height: 1.6;
    }}
    .container {{ max-width: 1000px; margin: 0 auto; padding: 24px 20px; }}
    .header {{ margin-bottom: 24px; }}
    .header .label {{ font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.12em; }}
    .header h1 {{
        font-size: 28px; font-weight: 700; margin: 4px 0;
        background: linear-gradient(135deg, #22d3ee, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .header .sub {{ font-size: 13px; color: {TEXT_MUTED}; }}
    .nav {{ display: flex; gap: 6px; margin-bottom: 24px; flex-wrap: wrap; }}
    .nav-link {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
        padding: 6px 14px; color: {TEXT_MUTED}; text-decoration: none; font-size: 12px; font-weight: 600;
        transition: all 0.2s;
    }}
    .nav-link:hover {{ color: {TEXT}; border-color: #818cf8; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 24px; }}
    .stat-card {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px;
    }}
    .stat-card .label {{ font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.06em; }}
    .stat-card .value {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
    .stat-card .sub {{ font-size: 11px; color: {TEXT_MUTED}; }}
    .section-title {{
        font-size: 18px; font-weight: 600; color: #818cf8;
        margin: 32px 0 14px; padding-bottom: 8px; border-bottom: 1px solid {BORDER};
    }}
    .chart-card {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 12px; margin-bottom: 16px; overflow: hidden;
    }}
    .footer {{
        margin-top: 40px; padding: 16px 0; border-top: 1px solid {BORDER};
        font-size: 11px; color: #475569; text-align: center;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="label">Course Review EDA</div>
        <h1>{course_code} — Exploratory Analysis</h1>
        <div class="sub">Interactive dashboard · {total} reviews · Keyword extraction + VADER sentiment · Plotly charts</div>
    </div>

    <nav class="nav">
        {''.join(nav_links)}
    </nav>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Reviews Analysed</div>
            <div class="value">{total}</div>
            <div class="sub">top-level posts</div>
        </div>
        <div class="stat-card">
            <div class="label">Positive</div>
            <div class="value" style="color:{POS}">{round(pos/total*100)}%</div>
            <div class="sub">{pos} reviews</div>
        </div>
        <div class="stat-card">
            <div class="label">Negative</div>
            <div class="value" style="color:{NEG}">{round(neg/total*100)}%</div>
            <div class="sub">{neg} reviews</div>
        </div>
        <div class="stat-card">
            <div class="label">Hours/Week</div>
            <div class="value" style="font-size:20px">{hours_str}</div>
            <div class="sub">from {len(all_hours)} mentions</div>
        </div>
        <div class="stat-card">
            <div class="label">Aspects/Review</div>
            <div class="value">{aspects_per_review:.1f}</div>
            <div class="sub">avg aspects tagged</div>
        </div>
    </div>

    {''.join(chart_divs)}

    <div class="footer">
        Prototype EDA dashboard · Keyword-based aspect extraction + VADER sentiment · Data may be simulated
    </div>
</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Course Review EDA Dashboard (Plotly)")
    parser.add_argument("reviews_file", help="Path to reviews JSON")
    parser.add_argument("--cross-course", help="Path to cross-course reviews JSON", default=None)
    parser.add_argument("--course-code", help="Course code", default="CS2040")
    parser.add_argument("--output", help="Output HTML path", default="eda_dashboard.html")
    args = parser.parse_args()

    print(f"Analyzing {args.reviews_file}...")
    results = analyze(args.reviews_file, args.cross_course)

    print("Generating dashboard...")
    html = build_html(results, args.course_code)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved to {args.output}")
    print(f"Open in browser: file://{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
