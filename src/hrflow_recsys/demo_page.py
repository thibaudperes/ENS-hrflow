DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HrFlow job recommender</title>
<style>
  :root {
    --bg: #f7f7f5; --card: #ffffff; --ink: #1d1d1b; --muted: #6b6b66;
    --line: #e4e4df; --accent: #2f5bd3; --apply: #1f8a4c;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #141413; --card: #1e1e1c; --ink: #ececea; --muted: #9a9a94;
      --line: #2e2e2b; --accent: #7d9cf0; --apply: #4cc283;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 48px; }
  h1 { font-size: 26px; margin: 0 0 6px; }
  .lead { color: var(--muted); margin: 0 0 20px; max-width: 760px; }
  .bar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 20px; }
  button {
    background: var(--accent); color: #fff; border: 0; border-radius: 8px;
    padding: 10px 18px; font: inherit; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: .6; cursor: wait; }
  .links a { color: var(--accent); margin-right: 14px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  section {
    background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px;
  }
  h2 { font-size: 15px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin: 0 0 12px; }
  ol { margin: 0; padding-left: 0; list-style: none; counter-reset: item; }
  li {
    counter-increment: item; display: grid; grid-template-columns: 28px 1fr;
    gap: 8px; padding: 8px 0; border-top: 1px solid var(--line);
  }
  li:first-child { border-top: 0; }
  li::before { content: counter(item); color: var(--muted); font-variant-numeric: tabular-nums; }
  .id { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
  .tag {
    display: inline-block; font-size: 11px; font-weight: 600; border-radius: 4px;
    padding: 0 6px; margin-left: 6px; border: 1px solid var(--line); color: var(--muted);
  }
  .tag.apply { color: var(--apply); border-color: var(--apply); }
  .action { font-weight: 600; }
  .title { font-weight: 600; }
  .summary { color: var(--muted); font-size: 13px; }
  .note { color: var(--muted); font-size: 13px; margin-top: 16px; }
  .error { color: #c0392b; }
</style>
</head>
<body>
<main>
  <h1>Next-job recommendation from a browsing session</h1>
  <p class="lead">
    A real test session from the ENS × HrFlow.ai challenge is sent to the model.
    It returns the 10 offers the candidate is most likely to open next, and whether
    they will view or apply. Hybrid model: Markov transitions, offer-id proximity and
    TF-IDF text similarity (MRR@10 0.094 on a validation grouped by session chain).
  </p>
  <div class="bar">
    <button id="go">Another session</button>
    <span class="links">
      <a href="/docs">API docs</a>
      <a href="https://github.com/thibaudperes/ens-challenge-hrflow-recsys">Code on GitHub</a>
    </span>
  </div>
  <div class="grid">
    <section>
      <h2>Session: offers opened, in order</h2>
      <p class="sub" id="session-sub"></p>
      <ol id="session"></ol>
    </section>
    <section>
      <h2>Recommended next offers</h2>
      <p class="sub" id="rec-sub"></p>
      <ol id="recs"></ol>
    </section>
  </div>
  <p class="note">The first call can take about 15 s: the service scales to zero when idle.</p>
</main>
<script>
const btn = document.getElementById("go");

function item(job, tag) {
  const li = document.createElement("li");
  const body = document.createElement("div");
  const [title, ...rest] = (job.text || "(no text)").split("\\n");
  const head = document.createElement("div");
  head.className = "title";
  head.textContent = title;
  body.appendChild(head);
  if (rest.join("").trim()) {
    const summary = document.createElement("div");
    summary.className = "summary";
    summary.textContent = rest.join(" ");
    body.appendChild(summary);
  }
  const meta = document.createElement("div");
  meta.className = "id";
  meta.textContent = "offer " + job.job_id;
  if (tag) {
    const t = document.createElement("span");
    t.className = "tag" + (tag === "apply" ? " apply" : "");
    t.textContent = tag;
    meta.appendChild(t);
  }
  body.appendChild(meta);
  li.appendChild(body);
  return li;
}

async function load() {
  btn.disabled = true;
  try {
    const res = await fetch("/demo/random");
    if (!res.ok) throw new Error((await res.json()).detail || res.status);
    const d = await res.json();
    document.getElementById("session-sub").className = "sub";
    const session = document.getElementById("session");
    const recs = document.getElementById("recs");
    session.replaceChildren(...d.session.map((job, i) => item(job, d.actions[i])));
    recs.replaceChildren(...d.recommendations.map((job) => item(job)));
    document.getElementById("session-sub").textContent = d.session.length + " offers";
    const sub = document.getElementById("rec-sub");
    sub.textContent = "Predicted next action: ";
    const a = document.createElement("span");
    a.className = "action";
    a.textContent = d.action;
    sub.appendChild(a);
  } catch (e) {
    const sub = document.getElementById("session-sub");
    sub.className = "sub error";
    sub.textContent = "Could not load a session: " + String(e.message);
  } finally {
    btn.disabled = false;
  }
}

btn.addEventListener("click", load);
load();
</script>
</body>
</html>
"""
