"""
view_disagreements.py — Generate an HTML viewer for disagreements.json
======================================================================
Usage:
  python view_disagreements.py
  python view_disagreements.py path/to/disagreements.json
  python view_disagreements.py --output my_report.html
"""

import json
import argparse
import os
import webbrowser

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Disagreements Viewer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f0f2f5;
    color: #1a1a2e;
    min-height: 100vh;
  }

  /* ── Header ── */
  .header {
    background: #1a1a2e;
    color: #fff;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 24px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 12px rgba(0,0,0,.3);
  }
  .header h1 { font-size: 1.2rem; font-weight: 600; flex: 1; }
  .stat-pill {
    background: rgba(255,255,255,.1);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    white-space: nowrap;
  }
  .search-box {
    background: rgba(255,255,255,.1);
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 6px;
    color: #fff;
    padding: 6px 12px;
    font-size: 0.85rem;
    width: 220px;
    outline: none;
  }
  .search-box::placeholder { color: rgba(255,255,255,.45); }
  .search-box:focus { border-color: rgba(255,255,255,.5); background: rgba(255,255,255,.15); }

  /* ── Filter bar ── */
  .filter-bar {
    background: #fff;
    border-bottom: 1px solid #e2e5ea;
    padding: 10px 32px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }
  .filter-label { font-size: 0.78rem; color: #888; margin-right: 4px; }
  .filter-btn {
    border: 1px solid #d0d5dd;
    background: #fff;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    cursor: pointer;
    transition: all .15s;
    color: #555;
  }
  .filter-btn:hover  { border-color: #6366f1; color: #6366f1; }
  .filter-btn.active { background: #6366f1; border-color: #6366f1; color: #fff; }

  /* ── Main content ── */
  .main { max-width: 960px; margin: 0 auto; padding: 28px 20px 60px; }

  .count-label { font-size: 0.82rem; color: #888; margin-bottom: 16px; }

  /* ── Card ── */
  .card {
    background: #fff;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    overflow: hidden;
    transition: box-shadow .15s;
  }
  .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.12); }

  .card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 18px;
    background: #f8f9fb;
    border-bottom: 1px solid #eef0f3;
  }
  .card-index {
    background: #6366f1;
    color: #fff;
    border-radius: 6px;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700; flex-shrink: 0;
  }
  .card-author { font-weight: 600; font-size: 0.9rem; }
  .card-post-id { font-size: 0.75rem; color: #aaa; }
  .conflict-summary { margin-left: auto; display: flex; gap: 5px; }

  /* ── Message ── */
  .message-box {
    padding: 14px 18px;
    font-size: 0.88rem;
    line-height: 1.65;
    color: #333;
    border-bottom: 1px solid #eef0f3;
    max-height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    background: #fefefe;
  }
  .message-box::-webkit-scrollbar { width: 6px; }
  .message-box::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }

  /* ── Methods grid ── */
  .methods-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 1px;
    background: #eef0f3;
  }
  .method-cell {
    background: #fff;
    padding: 12px 14px;
  }
  .method-name {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: #888;
    margin-bottom: 6px;
  }
  .label-badge {
    display: inline-block;
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 6px;
  }
  .label-hard    { background: #fdecea; color: #c0392b; }
  .label-neutral { background: #fef9e7; color: #b7770d; }
  .label-easy    { background: #eafaf1; color: #1e8449; }
  .conf-bar-track {
    background: #eee;
    border-radius: 3px;
    height: 4px;
    margin-bottom: 5px;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: #6366f1;
  }
  .method-expl {
    font-size: 0.75rem;
    color: #666;
    line-height: 1.45;
  }

  /* ── Empty state ── */
  .empty { text-align: center; padding: 60px 20px; color: #aaa; font-size: 0.95rem; }
</style>
</head>
<body>

<div class="header">
  <h1>&#9878; Disagreements Viewer</h1>
  <span class="stat-pill" id="total-stat"></span>
  <span class="stat-pill" id="method-stat"></span>
  <input class="search-box" type="text" id="search" placeholder="Search by author or text…">
</div>

<div class="filter-bar">
  <span class="filter-label">Filter where:</span>
  <button class="filter-btn active" data-filter="all">All</button>
  <span class="filter-label" style="margin-left:8px">method says:</span>
  <button class="filter-btn" data-filter="hard">hard</button>
  <button class="filter-btn" data-filter="neutral">neutral</button>
  <button class="filter-btn" data-filter="easy">easy</button>
</div>

<div class="main">
  <div class="count-label" id="count-label"></div>
  <div id="cards"></div>
  <div class="empty" id="empty-state" style="display:none">No matching reviews.</div>
</div>

<script>
const DATA = __DATA__;

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Detect methods
const methods = DATA.length
  ? Object.keys(DATA[0]).filter(k => k.endsWith('_label')).map(k => k.slice(0,-6))
  : [];

document.getElementById('total-stat').textContent  = DATA.length + ' reviews';
document.getElementById('method-stat').textContent = methods.join(' · ');

function labelBadge(label) {
  return `<span class="label-badge label-${label}">${label}</span>`;
}

function renderCard(r, idx) {
  const conflictBadges = methods.map(m => {
    const lbl = r[m+'_label'] || 'neutral';
    return `<span class="label-badge label-${lbl}" style="padding:2px 7px;font-size:0.7rem">${m[0].toUpperCase()}: ${lbl}</span>`;
  }).join('');

  const methodCells = methods.map(m => {
    const lbl  = r[m+'_label']  || '—';
    const conf = (r[m+'_conf']  ?? 0);
    const expl = r[m+'_expl']   || '';
    const pct  = Math.round(conf * 100);
    return `
      <div class="method-cell">
        <div class="method-name">${escHtml(m)}</div>
        ${labelBadge(lbl)}
        <div class="conf-bar-track"><div class="conf-bar-fill" style="width:${pct}%"></div></div>
        <div class="method-expl">${escHtml(expl)}</div>
      </div>`;
  }).join('');

  const msg = escHtml(r.message || '');

  return `
    <div class="card" data-labels="${methods.map(m=>r[m+'_label']).join(' ')}"
         data-text="${escHtml((r.message||'').toLowerCase())} ${escHtml((r.author||'').toLowerCase())}">
      <div class="card-header">
        <div class="card-index">${idx+1}</div>
        <div>
          <div class="card-author">${escHtml(r.author || 'Unknown')}</div>
          <div class="card-post-id">post ${escHtml(r.post_id || '')}</div>
        </div>
        <div class="conflict-summary">${conflictBadges}</div>
      </div>
      <div class="message-box">${msg}</div>
      <div class="methods-grid">${methodCells}</div>
    </div>`;
}

let activeFilter = 'all';
let searchTerm   = '';

function update() {
  const cards  = document.getElementById('cards');
  const empty  = document.getElementById('empty-state');
  const label  = document.getElementById('count-label');
  let visible  = 0;

  const html = DATA.map((r, i) => {
    const labels = (r._labels = methods.map(m => r[m+'_label'] || 'neutral'));
    const text   = ((r.message||'') + ' ' + (r.author||'')).toLowerCase();

    const filterOk = activeFilter === 'all' || labels.includes(activeFilter);
    const searchOk = !searchTerm  || text.includes(searchTerm);

    if (filterOk && searchOk) { visible++; return renderCard(r, i); }
    return '';
  }).join('');

  cards.innerHTML = html;
  label.textContent = visible + ' of ' + DATA.length + ' reviews';
  empty.style.display = visible ? 'none' : 'block';
}

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.dataset.filter;
    update();
  });
});

document.getElementById('search').addEventListener('input', e => {
  searchTerm = e.target.value.trim().toLowerCase();
  update();
});

update();
</script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.input:
        input_path = args.input
    else:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "results", "disagreements.json"),
            os.path.join(os.getcwd(), "results", "disagreements.json"),
            os.path.join(os.getcwd(), "disagreements.json"),
        ]
        input_path = next((p for p in candidates if os.path.exists(p)), None)
        if not input_path:
            print("Could not find disagreements.json. Pass the path explicitly.")
            return

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(records, ensure_ascii=False))

    out_path = args.output or os.path.splitext(input_path)[0] + "_viewer.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {out_path}")
    webbrowser.open(f"file:///{os.path.abspath(out_path).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
