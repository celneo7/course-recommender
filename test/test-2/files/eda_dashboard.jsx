import { useState, useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, LineChart, Line, CartesianGrid, Legend, ScatterChart, Scatter, ZAxis } from "recharts";

// ============================================================
// SAMPLE DATA — Replace with your actual extracted data
// ============================================================
const REVIEWS_RAW = [
  { id:"1001", author:"cs_student_22", msg:"Took this in AY22/23 Sem 2 with Prof Tan. The content is really interesting but the workload is no joke. I spent around 12-15 hours per week on problem sets alone. The topics build on each other fast so if you fall behind it's hard to catch up. Prof Tan explains things clearly though and his slides are well organized. Ended with a B+, which I think is fair for the effort I put in. Would recommend if you genuinely enjoy algorithms.", date:"2023-06", semester:"AY22/23 S2", instructor:"Tan", sentiment:0.72, sentLabel:"positive" },
  { id:"1002", author:"struggling_eng", msg:"This was the hardest module I've taken so far. I came from an engineering background with minimal coding experience and the prerequisites were NOT enough. The pace is relentless - new data structure every week. I spent 20+ hours some weeks. The tutorials helped but office hours were always packed. Ended with a C+.", date:"2023-06", semester:"AY22/23 S2", instructor:"Tan", sentiment:-0.65, sentLabel:"negative" },
  { id:"1003", author:"algo_lover", msg:"Best CS module hands down. Prof Tan is incredibly passionate and makes even the dry topics feel exciting. The problem sets are tough but very rewarding. Assessments tested understanding rather than memorization. Grading was fair. Around 10 hours per week.", date:"2023-07", semester:"AY22/23 S2", instructor:"Tan", sentiment:0.88, sentLabel:"positive" },
  { id:"1004", author:"pragmatic_senior", msg:"Essential module for SWE interviews. Almost everything comes up in technical interviews at FAANG companies. Content isn't that abstract if you've done well in CS1101S. Prof Tan's teaching is top tier. Deadline clustering was my only complaint - PS 3, 4 and 5 all came within 2 weeks.", date:"2023-07", semester:"AY22/23 S2", instructor:"Tan", sentiment:0.61, sentLabel:"positive" },
  { id:"1005", author:"night_owl_coder", msg:"Took this with Prof Lee in AY23/24 Sem 1. Very different experience. Prof Lee's lectures were a bit disorganized and he tends to jump between topics. Tutorials were hit or miss. Workload was manageable though, maybe 8-10 hours per week. Grading felt harsh.", date:"2024-01", semester:"AY23/24 S1", instructor:"Lee", sentiment:-0.31, sentLabel:"negative" },
  { id:"1007", author:"math_major_here", msg:"As a math major taking this as an elective, the theoretical parts were manageable but programming assignments were challenging. Graph algorithms section was beautiful. About 8 hours per week. Prof Tan was very helpful during office hours.", date:"2023-08", semester:"AY22/23 S2", instructor:"Tan", sentiment:0.45, sentLabel:"positive" },
  { id:"1008", author:"second_timer", msg:"Retook after getting D+ the first time. Second time around much more manageable with better preparation. The jump around week 5-6 with balanced BSTs is where most struggle. The curve helped. Most people got B- to B+ range.", date:"2023-09", semester:"AY23/24 S1", instructor:"Tan", sentiment:0.22, sentLabel:"positive" },
  { id:"1009", author:"quiet_student", msg:"Well-structured module. Each week builds logically. But I wish there was more feedback on assignments - just got a grade with no explanation. The final had questions on topics barely covered in lectures. 12 hours per week.", date:"2023-10", semester:"AY23/24 S1", instructor:"Tan", sentiment:-0.15, sentLabel:"negative" },
  { id:"1010", author:"exchange_student", msg:"Coming from a European university, this course is very well taught. The emphasis on practical implementation alongside theory is great. However workload is significantly higher than what I'm used to. Group project was stressful. Extremely relevant for software engineering.", date:"2023-11", semester:"AY23/24 S1", instructor:"Tan", sentiment:0.38, sentLabel:"positive" },
  { id:"1011", author:"burned_out", msg:"I was taking 5 modules and this one ate up most of my time. The assignments are long and you can't start them last minute. Deadlines overlapped constantly. I felt overwhelmed most of the semester. Attendance wasn't mandatory but tutorials counted for grades.", date:"2023-11", semester:"AY23/24 S1", instructor:"Tan", sentiment:-0.72, sentLabel:"negative" },
  { id:"1012", author:"overachiever_99", msg:"Honestly not that hard if you have a strong CS1101S foundation. 6-8 hours per week and got an A-. Prof Tan is one of the best lecturers. Start problem sets early and go to office hours. Very useful for interviews.", date:"2023-12", semester:"AY23/24 S1", instructor:"Tan", sentiment:0.81, sentLabel:"positive" },
  { id:"1013", author:"career_focused", msg:"Taking this purely because every SWE job requires DSA knowledge. Content can be dry at times but knowing it opens doors. Skills transfer to literally every other CS module. Got a B which I'm fine with.", date:"2024-01", semester:"AY23/24 S1", instructor:"Tan", sentiment:0.19, sentLabel:"positive" },
  { id:"1014", author:"frustrated_student", msg:"Grading is a black box. I studied harder than any other module and still got B-. Meanwhile my friend who put in less effort got an A. Bell curve is brutal. Exam had questions on topics the prof rushed through in the last lecture. Not fair.", date:"2024-02", semester:"AY23/24 S2", instructor:"Lee", sentiment:-0.82, sentLabel:"negative" },
  { id:"1015", author:"ta_perspective", msg:"Was a TA for this course. Common struggles: students underestimate time for problem sets, don't practice before tutorials, and many have weak recursion fundamentals from CS1101S. If you're considering this, make sure you can write recursive functions in your sleep.", date:"2024-02", semester:"AY23/24 S2", instructor:"Tan", sentiment:0.25, sentLabel:"positive" },
  { id:"1017", author:"is_student", msg:"I'm an IS student who took this. The difficulty was higher than expected based on prereqs. The module assumes you're comfortable with complex code but IS students don't get as much programming practice. Prof Tan was helpful. About 15 hours per week. Got a B.", date:"2024-03", semester:"AY23/24 S2", instructor:"Tan", sentiment:0.05, sentLabel:"neutral" },
  { id:"1018", author:"night_owl_coder2", msg:"If you get Prof Lee, recorded lectures are essential because his live delivery can be confusing. The recordings let you pause and rewind. Also the textbook CLRS is way too dense, use VisuAlgo instead.", date:"2024-04", semester:"AY23/24 S2", instructor:"Lee", sentiment:-0.18, sentLabel:"negative" },
];

// Aspect tagging results (simulated extraction)
const ASPECT_TAGS = {
  content_complexity: ["1002","1003","1007","1013","1017"],
  assumed_knowledge: ["1002","1004","1007","1008","1015","1017"],
  learning_pace: ["1001","1002","1008"],
  workload_hours: ["1001","1002","1003","1004","1005","1007","1009","1010","1011","1012","1017"],
  assignment_volume: ["1001","1004","1009","1011","1015"],
  deadline_pressure: ["1004","1010","1011"],
  felt_pressure: ["1002","1010","1011","1014"],
  scoring_difficulty: ["1002","1005","1008","1012","1014"],
  grading_fairness: ["1001","1003","1005","1009","1014"],
  taught_vs_assessed: ["1009","1014"],
  grade_distribution: ["1001","1002","1003","1005","1008","1012","1013","1014","1017"],
  background_fit: ["1002","1004","1007","1008","1012","1015","1017"],
  teaching_clarity: ["1001","1003","1004","1005","1012","1018"],
  teaching_engagement: ["1001","1003","1004","1013"],
  subject_mastery: ["1003","1004","1012"],
  rapport: ["1003","1007"],
  availability: ["1002","1007","1012"],
  feedback_quality: ["1009"],
  helpfulness: ["1007","1015","1017"],
  course_structure: ["1001","1005","1008","1009"],
  assessment_format: ["1003","1009","1014"],
  career_relevance: ["1004","1010","1012","1013"],
  skill_building: ["1003","1013","1015"],
  intellectual_value: ["1003","1007"],
  topic_interest: ["1001","1003","1013"],
  overall_satisfaction: ["1001","1003","1004","1012"],
  logistics_attendance: ["1004","1011"],
  logistics_recording: ["1001","1018"],
};

const ASPECT_LABELS = {
  content_complexity:"Complexity", assumed_knowledge:"Prior Knowledge", learning_pace:"Pace",
  workload_hours:"Hours/Week", assignment_volume:"Assignments", deadline_pressure:"Deadlines",
  felt_pressure:"Felt Pressure", scoring_difficulty:"Scoring Difficulty", grading_fairness:"Grading Fairness",
  taught_vs_assessed:"Taught vs Assessed", grade_distribution:"Grade Dist.", background_fit:"Background Fit",
  teaching_clarity:"Clarity", teaching_engagement:"Engagement", subject_mastery:"Mastery",
  rapport:"Rapport", availability:"Availability", feedback_quality:"Feedback",
  helpfulness:"Helpfulness", course_structure:"Structure", assessment_format:"Assessment Format",
  career_relevance:"Career Relevance", skill_building:"Skill Building", intellectual_value:"Intellectual Value",
  topic_interest:"Topic Interest", overall_satisfaction:"Overall Satisfaction",
  logistics_attendance:"Attendance", logistics_recording:"Recordings",
};

const COMPONENT_GROUPS = {
  "Difficulty": ["content_complexity","assumed_knowledge","learning_pace","workload_hours","assignment_volume","deadline_pressure","felt_pressure"],
  "Assessment": ["scoring_difficulty","grading_fairness","taught_vs_assessed","grade_distribution"],
  "Background": ["background_fit"],
  "Teaching": ["teaching_clarity","teaching_engagement","subject_mastery","rapport"],
  "Support": ["availability","feedback_quality","helpfulness"],
  "Design": ["course_structure","assessment_format"],
  "Value": ["career_relevance","skill_building","intellectual_value","topic_interest","overall_satisfaction"],
  "Logistics": ["logistics_attendance","logistics_recording"],
};

const COLORS = {
  teal: "#2dd4bf", cyan: "#22d3ee", blue: "#60a5fa", indigo: "#818cf8",
  violet: "#a78bfa", pink: "#f472b6", rose: "#fb7185", amber: "#fbbf24",
  lime: "#a3e635", emerald: "#34d399", red: "#f87171", orange: "#fb923c",
  sky: "#38bdf8", fuchsia: "#e879f9",
};
const PALETTE = Object.values(COLORS);
const POS_COLOR = "#34d399";
const NEG_COLOR = "#f87171";
const NEU_COLOR = "#fbbf24";

// ============================================================
// COMPUTED DATA
// ============================================================
function computeAll() {
  const total = REVIEWS_RAW.length;
  
  // Aspect coverage
  const aspectCoverage = Object.entries(ASPECT_TAGS).map(([key, ids]) => {
    const reviews = ids.map(id => REVIEWS_RAW.find(r => r.id === id)).filter(Boolean);
    const pos = reviews.filter(r => r.sentLabel === "positive").length;
    const neg = reviews.filter(r => r.sentLabel === "negative").length;
    const neu = reviews.filter(r => r.sentLabel === "neutral").length;
    const avgSent = reviews.length > 0 ? reviews.reduce((s, r) => s + r.sentiment, 0) / reviews.length : 0;
    return {
      aspect: key, label: ASPECT_LABELS[key], count: ids.length, pct: Math.round(ids.length / total * 100),
      pos, neg, neu, avgSent: Math.round(avgSent * 100) / 100,
      posPct: reviews.length > 0 ? Math.round(pos / reviews.length * 100) : 0,
      negPct: reviews.length > 0 ? Math.round(neg / reviews.length * 100) : 0,
    };
  }).sort((a, b) => b.count - a.count);

  // Component-level aggregation
  const componentData = Object.entries(COMPONENT_GROUPS).map(([name, aspects]) => {
    const allIds = new Set(aspects.flatMap(a => ASPECT_TAGS[a] || []));
    const reviews = [...allIds].map(id => REVIEWS_RAW.find(r => r.id === id)).filter(Boolean);
    const avgSent = reviews.length > 0 ? reviews.reduce((s, r) => s + r.sentiment, 0) / reviews.length : 0;
    return { name, mentions: allIds.size, avgSent: Math.round(avgSent * 100) / 100, aspects: aspects.length };
  });

  // Temporal (by semester)
  const semesters = [...new Set(REVIEWS_RAW.map(r => r.semester))].sort();
  const temporal = semesters.map(sem => {
    const reviews = REVIEWS_RAW.filter(r => r.semester === sem);
    const avg = reviews.reduce((s, r) => s + r.sentiment, 0) / reviews.length;
    return {
      semester: sem, count: reviews.length,
      avgSentiment: Math.round(avg * 100) / 100,
      pos: reviews.filter(r => r.sentLabel === "positive").length,
      neg: reviews.filter(r => r.sentLabel === "negative").length,
      neu: reviews.filter(r => r.sentLabel === "neutral").length,
    };
  });

  // Instructor comparison
  const instructors = [...new Set(REVIEWS_RAW.map(r => r.instructor))];
  const instructorData = instructors.map(inst => {
    const reviews = REVIEWS_RAW.filter(r => r.instructor === inst);
    const avg = reviews.reduce((s, r) => s + r.sentiment, 0) / reviews.length;
    // Per-aspect sentiment for this instructor
    const aspectSentiments = Object.entries(ASPECT_TAGS).map(([key, ids]) => {
      const instReviews = ids.map(id => REVIEWS_RAW.find(r => r.id === id)).filter(r => r && r.instructor === inst);
      return { aspect: ASPECT_LABELS[key], value: instReviews.length > 0 ? Math.round((instReviews.reduce((s, r) => s + r.sentiment, 0) / instReviews.length + 1) * 50) : 0 };
    }).filter(a => a.value > 0);
    return {
      name: `Prof ${inst}`, count: reviews.length,
      avgSentiment: Math.round(avg * 100) / 100,
      pos: reviews.filter(r => r.sentLabel === "positive").length,
      neg: reviews.filter(r => r.sentLabel === "negative").length,
      radarData: aspectSentiments.sort((a,b) => b.value - a.value).slice(0, 8),
    };
  });

  // Sentiment distribution
  const sentBuckets = [
    { range: "Very Neg\n(-1 to -0.6)", min: -1, max: -0.6 },
    { range: "Negative\n(-0.6 to -0.2)", min: -0.6, max: -0.2 },
    { range: "Neutral\n(-0.2 to 0.2)", min: -0.2, max: 0.2 },
    { range: "Positive\n(0.2 to 0.6)", min: 0.2, max: 0.6 },
    { range: "Very Pos\n(0.6 to 1)", min: 0.6, max: 1.01 },
  ];
  const sentDist = sentBuckets.map(b => ({
    range: b.range,
    count: REVIEWS_RAW.filter(r => r.sentiment >= b.min && r.sentiment < b.max).length,
  }));

  // Hours extraction
  const hoursData = [
    {author:"cs_student_22", hours:13.5}, {author:"struggling_eng", hours:20},
    {author:"algo_lover", hours:10}, {author:"night_owl_coder", hours:9},
    {author:"math_major_here", hours:8}, {author:"quiet_student", hours:12},
    {author:"overachiever_99", hours:7}, {author:"is_student", hours:15},
    {author:"burned_out", hours:18},
  ].map(h => {
    const r = REVIEWS_RAW.find(rv => rv.author === h.author);
    return { ...h, sentiment: r ? r.sentiment : 0, grade: r ? r.sentLabel : "neutral" };
  });

  // Aspect co-occurrence
  const aspectKeys = Object.keys(ASPECT_TAGS);
  const cooccurrence = [];
  for (let i = 0; i < aspectKeys.length; i++) {
    for (let j = i + 1; j < aspectKeys.length; j++) {
      const setA = new Set(ASPECT_TAGS[aspectKeys[i]]);
      const setB = new Set(ASPECT_TAGS[aspectKeys[j]]);
      const overlap = [...setA].filter(x => setB.has(x)).length;
      if (overlap >= 2) {
        cooccurrence.push({
          a: ASPECT_LABELS[aspectKeys[i]], b: ASPECT_LABELS[aspectKeys[j]],
          overlap, sizeA: setA.size, sizeB: setB.size,
        });
      }
    }
  }
  cooccurrence.sort((a, b) => b.overlap - a.overlap);

  return { aspectCoverage, componentData, temporal, instructorData, sentDist, hoursData, cooccurrence, total };
}

// ============================================================
// COMPONENTS
// ============================================================
const TABS = ["Overview", "Aspects", "Sentiment", "Instructors", "Extraction"];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#1e2130", border: "1px solid #363b52", borderRadius: 8, padding: "8px 12px", fontSize: 12 }}>
      <div style={{ color: "#94a3b8", marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || "#e2e8f0" }}>
          {p.name}: <strong>{typeof p.value === 'number' ? (Number.isInteger(p.value) ? p.value : p.value.toFixed(2)) : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ background: "#1a1d2e", border: "1px solid #2a2f45", borderRadius: 12, padding: "16px 20px", flex: "1 1 160px" }}>
      <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || "#e2e8f0", marginTop: 4 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function SectionTitle({ children, sub }) {
  return (
    <div style={{ marginBottom: 16, marginTop: 28 }}>
      <h3 style={{ fontSize: 16, fontWeight: 600, color: "#e2e8f0", margin: 0 }}>{children}</h3>
      {sub && <p style={{ fontSize: 12, color: "#64748b", margin: "4px 0 0" }}>{sub}</p>}
    </div>
  );
}

// --- TAB: Overview ---
function OverviewTab({ data }) {
  const { aspectCoverage, componentData, sentDist, total } = data;
  const posCount = REVIEWS_RAW.filter(r => r.sentLabel === "positive").length;
  const negCount = REVIEWS_RAW.filter(r => r.sentLabel === "negative").length;
  const avgSent = (REVIEWS_RAW.reduce((s, r) => s + r.sentiment, 0) / total).toFixed(2);

  const pieData = [
    { name: "Positive", value: posCount, color: POS_COLOR },
    { name: "Negative", value: negCount, color: NEG_COLOR },
    { name: "Neutral", value: total - posCount - negCount, color: NEU_COLOR },
  ];

  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 8 }}>
        <StatCard label="Total Reviews" value={total} sub="top-level only" />
        <StatCard label="Avg Sentiment" value={avgSent} sub="VADER compound" color={parseFloat(avgSent) > 0 ? POS_COLOR : NEG_COLOR} />
        <StatCard label="Positive" value={`${Math.round(posCount/total*100)}%`} sub={`${posCount} reviews`} color={POS_COLOR} />
        <StatCard label="Negative" value={`${Math.round(negCount/total*100)}%`} sub={`${negCount} reviews`} color={NEG_COLOR} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div>
          <SectionTitle sub="How many reviews mention each dashboard component?">Component Coverage</SectionTitle>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={componentData} layout="vertical" margin={{ left: 60, right: 20, top: 5, bottom: 5 }}>
              <XAxis type="number" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} width={55} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="mentions" radius={[0, 6, 6, 0]}>
                {componentData.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <SectionTitle sub="Distribution of VADER compound scores">Sentiment Spread</SectionTitle>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={sentDist} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
              <XAxis dataKey="range" tick={{ fill: "#64748b", fontSize: 10 }} interval={0} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {sentDist.map((_, i) => <Cell key={i} fill={[NEG_COLOR, "#fb923c", NEU_COLOR, "#a3e635", POS_COLOR][i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <SectionTitle sub="Positive vs negative sentiment breakdown per component">Component Sentiment</SectionTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={componentData} margin={{ left: 60, right: 20, top: 5, bottom: 5 }}>
          <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="avgSent" name="Avg Sentiment" radius={[6, 6, 0, 0]}>
            {componentData.map((d, i) => <Cell key={i} fill={d.avgSent >= 0 ? POS_COLOR : NEG_COLOR} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- TAB: Aspects ---
function AspectsTab({ data }) {
  const { aspectCoverage, cooccurrence, total } = data;

  return (
    <div>
      <SectionTitle sub={`How many of ${total} reviews mention each sub-aspect? Sorted by frequency.`}>
        Aspect Mention Frequency
      </SectionTitle>
      <ResponsiveContainer width="100%" height={520}>
        <BarChart data={aspectCoverage} layout="vertical" margin={{ left: 100, right: 30, top: 5, bottom: 5 }}>
          <XAxis type="number" tick={{ fill: "#64748b", fontSize: 11 }} />
          <YAxis type="category" dataKey="label" tick={{ fill: "#94a3b8", fontSize: 11 }} width={95} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="count" name="Reviews" radius={[0, 6, 6, 0]}>
            {aspectCoverage.map((d, i) => <Cell key={i} fill={d.count >= 5 ? PALETTE[i % PALETTE.length] : "#4a5068"} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <SectionTitle sub="Which aspects tend to appear together in the same review? (min 2 co-occurrences)">
        Aspect Co-occurrence (Top Pairs)
      </SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 8 }}>
        {cooccurrence.slice(0, 12).map((pair, i) => (
          <div key={i} style={{ background: "#1a1d2e", border: "1px solid #2a2f45", borderRadius: 10, padding: "10px 14px" }}>
            <div style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 600 }}>{pair.a} + {pair.b}</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: PALETTE[i % PALETTE.length], marginTop: 4 }}>{pair.overlap}</div>
            <div style={{ fontSize: 11, color: "#64748b" }}>shared reviews</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// --- TAB: Sentiment ---
function SentimentTab({ data }) {
  const { aspectCoverage, temporal, hoursData } = data;
  const sentAspects = aspectCoverage.filter(a => a.count >= 3).slice(0, 14);

  return (
    <div>
      <SectionTitle sub="Positive vs negative ratio for each aspect (aspects with 3+ reviews)">
        Sentiment Breakdown by Aspect
      </SectionTitle>
      <ResponsiveContainer width="100%" height={380}>
        <BarChart data={sentAspects} layout="vertical" margin={{ left: 100, right: 20, top: 5, bottom: 5 }} stackOffset="expand" barSize={18}>
          <XAxis type="number" tickFormatter={v => `${Math.round(v * 100)}%`} tick={{ fill: "#64748b", fontSize: 11 }} />
          <YAxis type="category" dataKey="label" tick={{ fill: "#94a3b8", fontSize: 11 }} width={95} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
          <Bar dataKey="pos" name="Positive" stackId="s" fill={POS_COLOR} radius={[0, 0, 0, 0]} />
          <Bar dataKey="neu" name="Neutral" stackId="s" fill={NEU_COLOR} />
          <Bar dataKey="neg" name="Negative" stackId="s" fill={NEG_COLOR} radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div>
          <SectionTitle sub="Average sentiment score per semester">Sentiment Over Time</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={temporal} margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2f45" />
              <XAxis dataKey="semester" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} domain={[-1, 1]} />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="avgSentiment" name="Avg Sentiment" stroke={COLORS.cyan} strokeWidth={2} dot={{ fill: COLORS.cyan, r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <SectionTitle sub="Reviews per semester and sentiment split">Volume Over Time</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={temporal} margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
              <XAxis dataKey="semester" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="pos" name="Positive" stackId="s" fill={POS_COLOR} />
              <Bar dataKey="neu" name="Neutral" stackId="s" fill={NEU_COLOR} />
              <Bar dataKey="neg" name="Negative" stackId="s" fill={NEG_COLOR} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <SectionTitle sub="Does more time spent correlate with sentiment? Each dot is a reviewer.">Hours vs Sentiment</SectionTitle>
      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2f45" />
          <XAxis type="number" dataKey="hours" name="Hours/wk" tick={{ fill: "#64748b", fontSize: 11 }} label={{ value: "Hours per week", position: "bottom", fill: "#64748b", fontSize: 11 }} />
          <YAxis type="number" dataKey="sentiment" name="Sentiment" tick={{ fill: "#64748b", fontSize: 11 }} domain={[-1, 1]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload;
            return (
              <div style={{ background: "#1e2130", border: "1px solid #363b52", borderRadius: 8, padding: "8px 12px", fontSize: 12 }}>
                <div style={{ color: "#e2e8f0", fontWeight: 600 }}>{d.author}</div>
                <div style={{ color: "#94a3b8" }}>{d.hours} hrs/wk · Sentiment: {d.sentiment.toFixed(2)}</div>
              </div>
            );
          }} />
          <Scatter data={hoursData} fill={COLORS.violet}>
            {hoursData.map((d, i) => <Cell key={i} fill={d.sentiment >= 0 ? POS_COLOR : NEG_COLOR} />)}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- TAB: Instructors ---
function InstructorsTab({ data }) {
  const { instructorData, aspectCoverage } = data;

  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 8 }}>
        {instructorData.map((inst, i) => (
          <StatCard key={i} label={inst.name} value={inst.avgSentiment}
            sub={`${inst.count} reviews · ${inst.pos} pos / ${inst.neg} neg`}
            color={inst.avgSentiment >= 0 ? POS_COLOR : NEG_COLOR} />
        ))}
      </div>

      <SectionTitle sub="Normalized aspect sentiment per instructor (0-100 scale, higher = more positive)">
        Instructor Radar Comparison
      </SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${instructorData.length}, 1fr)`, gap: 16 }}>
        {instructorData.map((inst, i) => (
          <div key={i}>
            <div style={{ textAlign: "center", fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 4 }}>{inst.name}</div>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={inst.radarData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                <PolarGrid stroke="#2a2f45" />
                <PolarAngleAxis dataKey="aspect" tick={{ fill: "#94a3b8", fontSize: 9 }} />
                <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 100]} />
                <Radar dataKey="value" stroke={PALETTE[i * 3]} fill={PALETTE[i * 3]} fillOpacity={0.25} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      <SectionTitle sub="Side-by-side sentiment for top aspects">Instructor Sentiment Comparison</SectionTitle>
      {(() => {
        const topAspects = aspectCoverage.filter(a => a.count >= 4).slice(0, 8);
        const compData = topAspects.map(a => {
          const row = { aspect: a.label };
          instructorData.forEach(inst => {
            const ids = ASPECT_TAGS[a.aspect] || [];
            const reviews = ids.map(id => REVIEWS_RAW.find(r => r.id === id)).filter(r => r && r.instructor === inst.name.replace("Prof ", ""));
            row[inst.name] = reviews.length > 0 ? Math.round(reviews.reduce((s, r) => s + r.sentiment, 0) / reviews.length * 100) / 100 : 0;
          });
          return row;
        });
        return (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={compData} margin={{ left: 80, right: 20, top: 5, bottom: 5 }} layout="vertical">
              <XAxis type="number" tick={{ fill: "#64748b", fontSize: 11 }} domain={[-1, 1]} />
              <YAxis type="category" dataKey="aspect" tick={{ fill: "#94a3b8", fontSize: 11 }} width={75} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {instructorData.map((inst, i) => (
                <Bar key={inst.name} dataKey={inst.name} fill={PALETTE[i * 3]} radius={4} barSize={12} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );
      })()}
    </div>
  );
}

// --- TAB: Extraction ---
function ExtractionTab({ data }) {
  const { aspectCoverage, total } = data;

  const qualityData = aspectCoverage.map(a => ({
    ...a,
    quality: a.count >= 10 ? "Strong" : a.count >= 5 ? "Moderate" : a.count >= 2 ? "Weak" : "Insufficient",
    qualityColor: a.count >= 10 ? POS_COLOR : a.count >= 5 ? NEU_COLOR : a.count >= 2 ? "#fb923c" : NEG_COLOR,
    canShowPct: a.count >= 10,
    canShowSent: a.count >= 5,
    level: a.count >= 10 ? "Pattern A+B+C" : a.count >= 5 ? "Pattern A+B" : a.count >= 2 ? "Pattern B + excerpts" : "Excerpts only",
  }));

  const strongCount = qualityData.filter(q => q.quality === "Strong").length;
  const modCount = qualityData.filter(q => q.quality === "Moderate").length;
  const weakCount = qualityData.filter(q => q.quality === "Weak").length;
  const insuffCount = qualityData.filter(q => q.quality === "Insufficient").length;

  const untaggedCount = REVIEWS_RAW.filter(r => {
    return !Object.values(ASPECT_TAGS).some(ids => ids.includes(r.id));
  }).length;

  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 8 }}>
        <StatCard label="Aspects Tracked" value={Object.keys(ASPECT_TAGS).length} />
        <StatCard label="Strong Signal (10+)" value={strongCount} color={POS_COLOR} />
        <StatCard label="Moderate (5-9)" value={modCount} color={NEU_COLOR} />
        <StatCard label="Weak / Insufficient" value={weakCount + insuffCount} color={NEG_COLOR} />
      </div>

      <SectionTitle sub="For each aspect: how many reviews were tagged, and what display patterns are viable at that sample size?">
        Extraction Quality Assessment
      </SectionTitle>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #2a2f45" }}>
              {["Aspect", "Tagged", "% of Reviews", "Signal", "Viable Display", "Avg Sentiment"].map(h => (
                <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: "#64748b", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {qualityData.map((q, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #1e2235" }}>
                <td style={{ padding: "8px 10px", color: "#e2e8f0", fontWeight: 500 }}>{q.label}</td>
                <td style={{ padding: "8px 10px", color: "#e2e8f0" }}>{q.count}</td>
                <td style={{ padding: "8px 10px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 60, height: 6, background: "#1e2235", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ width: `${q.pct}%`, height: "100%", background: q.qualityColor, borderRadius: 3 }} />
                    </div>
                    <span style={{ color: "#94a3b8" }}>{q.pct}%</span>
                  </div>
                </td>
                <td style={{ padding: "8px 10px" }}>
                  <span style={{ background: q.qualityColor + "22", color: q.qualityColor, padding: "2px 8px", borderRadius: 10, fontSize: 11, fontWeight: 600 }}>
                    {q.quality}
                  </span>
                </td>
                <td style={{ padding: "8px 10px", color: "#94a3b8", fontSize: 11 }}>{q.level}</td>
                <td style={{ padding: "8px 10px", color: q.avgSent >= 0 ? POS_COLOR : NEG_COLOR, fontWeight: 600 }}>{q.avgSent}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <SectionTitle sub="What percentage of reviews are captured by the keyword-based extraction?">Coverage Gaps</SectionTitle>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ background: "#1a1d2e", border: "1px solid #2a2f45", borderRadius: 12, padding: 20 }}>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em" }}>Untagged Reviews</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: untaggedCount > 0 ? NEG_COLOR : POS_COLOR, marginTop: 4 }}>{untaggedCount} / {total}</div>
          <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>
            {untaggedCount === 0 ? "All reviews matched at least one aspect" : `${untaggedCount} review(s) didn't match any keyword pattern — may need manual review or pattern expansion`}
          </div>
        </div>
        <div style={{ background: "#1a1d2e", border: "1px solid #2a2f45", borderRadius: 12, padding: 20 }}>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em" }}>Avg Aspects per Review</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: COLORS.cyan, marginTop: 4 }}>
            {(Object.values(ASPECT_TAGS).reduce((s, ids) => s + ids.length, 0) / total).toFixed(1)}
          </div>
          <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>Higher = richer reviews. Below 2.0 suggests reviews are short or keywords are too narrow.</div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================
export default function Dashboard() {
  const [tab, setTab] = useState("Overview");
  const data = useMemo(() => computeAll(), []);

  return (
    <div style={{ background: "#12141f", minHeight: "100vh", color: "#e2e8f0", fontFamily: "'DM Sans', 'Outfit', system-ui, sans-serif" }}>
      <div style={{ maxWidth: 960, margin: "0 auto", padding: "24px 20px" }}>
        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 4 }}>Course Review Analysis</div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, background: "linear-gradient(135deg, #22d3ee, #818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            CS2040 — Data Structures & Algorithms
          </h1>
          <p style={{ fontSize: 13, color: "#64748b", margin: "6px 0 0" }}>
            EDA Dashboard · {data.total} reviews · Text-mining prototype (keyword extraction + VADER sentiment)
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 20, background: "#1a1d2e", borderRadius: 10, padding: 4 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              flex: 1, padding: "8px 12px", border: "none", borderRadius: 8, cursor: "pointer",
              fontSize: 13, fontWeight: 600, transition: "all 0.2s",
              background: tab === t ? "#2a2f45" : "transparent",
              color: tab === t ? "#e2e8f0" : "#64748b",
            }}>
              {t}
            </button>
          ))}
        </div>

        {/* Content */}
        {tab === "Overview" && <OverviewTab data={data} />}
        {tab === "Aspects" && <AspectsTab data={data} />}
        {tab === "Sentiment" && <SentimentTab data={data} />}
        {tab === "Instructors" && <InstructorsTab data={data} />}
        {tab === "Extraction" && <ExtractionTab data={data} />}

        {/* Footer */}
        <div style={{ marginTop: 32, padding: "16px 0", borderTop: "1px solid #2a2f45", fontSize: 11, color: "#475569", textAlign: "center" }}>
          Prototype dashboard for validating extraction pipeline · Data is simulated · Keyword-based aspect tagging + VADER sentiment
        </div>
      </div>
    </div>
  );
}
