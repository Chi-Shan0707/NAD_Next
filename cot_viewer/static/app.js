const CONFIG = window.COT_VIEWER_CONFIG || {};

function detectScriptRoot() {
  const configured = (CONFIG.scriptRoot || "").replace(/\/$/, "");
  const pathname = (window.location.pathname || "").replace(/\/+$/, "");
  const proxyMatch = pathname.match(/^(.*\/proxy\/\d+)(?:\/.*)?$/);
  if (proxyMatch && proxyMatch[1]) return proxyMatch[1];
  if (configured === "/hub") return "";
  return configured;
}

const SCRIPT_ROOT = detectScriptRoot();

const STATE = {
  cache: null,
  problemId: null,
  methodId: "svd_slot100",
  anchor: 100,
  svdDomain: "math",
  mode: "fixed",
  datasets: {},
  methods: [],
  trajectory: null,
  scores: null,
  featurePanel: null,
  runContributions: null,
  modelSummary: null,
  tokenEvidence: null,
  focusMetric: "conf",
  selectedExplainSampleId: null,
  neuronLoaded: null,
};

const CANONICAL_SVD_METHODS = new Set([
  "es_svd_math_rr_r1",
  "es_svd_science_rr_r1",
  "es_svd_ms_rr_r1",
]);

const C = {
  green: "#16a34a", red: "#dc2626", blue: "#2563eb",
  purple: "#7c3aed", orange: "#f59e0b", gray: "#6b7280",
  conf: "#2563eb", entropy: "#dc2626", gini: "#16a34a",
  selfcert: "#7c3aed", logprob: "#f59e0b",
};

const METRIC_LABEL = {
  conf: "Conf", entropy: "Entropy", gini: "Gini",
  selfcert: "Self-Cert", logprob: "LogProb",
};

const $ = (id) => document.getElementById(id);

function isCanonicalSvdMethod(methodId = STATE.methodId) {
  return CANONICAL_SVD_METHODS.has(methodId);
}

function currentCacheDomain() {
  const sel = $("sel-dataset");
  const label = sel && sel.selectedOptions && sel.selectedOptions[0]
    ? String(sel.selectedOptions[0].textContent || "").toLowerCase()
    : "";
  if (label.includes("gpqa")) return "science";
  if (label.includes("lcb")) return "coding";
  return "math";
}

function withRoot(path) {
  if (!path.startsWith("/")) return `${SCRIPT_ROOT}/${path}`;
  return `${SCRIPT_ROOT}${path}`;
}

function setStatus(msg) { $("status").textContent = msg; }

function showAlert(message) {
  const box = $("global-alert");
  box.style.display = "";
  box.textContent = message;
}

function clearAlert() {
  const box = $("global-alert");
  box.style.display = "none";
  box.textContent = "";
}

function fmt(x, d = 4) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(d);
}

async function apiFetch(path, params = {}) {
  const query = new URLSearchParams();
  if (STATE.cache) query.set("cache", STATE.cache);
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined && `${v}` !== "") query.set(k, v);
  }
  const qs = query.toString();
  const url = `${withRoot(path)}${qs ? `?${qs}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

// Error-isolated parallel fetch: failed promises return null
async function loadAll(promises) {
  return Promise.all(
    promises.map((p) =>
      p.catch((e) => {
        console.warn("isolated fetch error:", e.message);
        return null;
      })
    )
  );
}

function setPlotEmpty(targetId, msg) {
  const node = $(targetId);
  if (node) node.innerHTML = `<div class="plot-empty">${msg}</div>`;
}

function safePlot(targetId, traces, layout = {}, config = {}) {
  if (!window.Plotly) { setPlotEmpty(targetId, "Plotly not loaded."); return; }
  const node = $(targetId);
  if (!node) return;
  Plotly.newPlot(node, traces, layout, { displayModeBar: false, responsive: true, ...config });
}

/* ---- Health ---- */
async function updateHealth() {
  try {
    const res = await fetch(withRoot("/api/health_viewer"));
    const data = await res.json();
    const pill = $("health-pill");
    const arts = data.artifacts || {};
    const ok = arts.earlystop_svd_model && arts.bridge_model;
    pill.classList.toggle("ok", ok);
    pill.classList.toggle("bad", !ok);
    pill.textContent = `Health: ${ok ? "OK" : "degraded"} · ds=${data.datasets}`;
  } catch {
    const pill = $("health-pill");
    pill.classList.add("bad");
    pill.textContent = "Health: unavailable";
  }
}

/* ---- Init ---- */
async function initDatasets() {
  setStatus("Loading datasets...");
  const res = await fetch(withRoot("/api/datasets"));
  const data = await res.json();
  STATE.datasets = data || {};
  const sel = $("sel-dataset");
  sel.innerHTML = "";
  Object.entries(STATE.datasets).forEach(([name, path]) => {
    const opt = document.createElement("option");
    opt.value = path;
    opt.textContent = name;
    sel.appendChild(opt);
  });
  STATE.cache = Object.values(STATE.datasets)[0] || null;
  if (STATE.cache) sel.value = STATE.cache;
}

async function initMethods() {
  const res = await fetch(withRoot("/api/method_catalog"));
  const data = await res.json();
  STATE.methods = data.methods || [];
  STATE.methodId = data.primary_method_id || "svd_slot100";
  const sel = $("sel-method");
  sel.innerHTML = "";
  STATE.methods.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.primary ? `${m.label} ★` : m.label;
    sel.appendChild(opt);
  });
  sel.value = STATE.methodId;
}

function updateMethodControls() {
  const anchorSel = $("sel-anchor");
  const domainSel = $("sel-svd-domain");
  const canonical = isCanonicalSvdMethod();
  const cacheDomain = currentCacheDomain();

  if (!canonical) {
    anchorSel.disabled = true;
    domainSel.disabled = true;
    domainSel.value = cacheDomain;
    STATE.svdDomain = cacheDomain;
    return;
  }

  anchorSel.disabled = false;
  if (STATE.methodId === "es_svd_math_rr_r1") {
    domainSel.disabled = true;
    domainSel.value = "math";
    STATE.svdDomain = "math";
    return;
  }
  if (STATE.methodId === "es_svd_science_rr_r1") {
    domainSel.disabled = true;
    domainSel.value = "science";
    STATE.svdDomain = "science";
    return;
  }

  domainSel.disabled = false;
  if (cacheDomain === "coding") {
    domainSel.value = STATE.svdDomain === "science" ? "science" : "math";
  } else {
    domainSel.value = cacheDomain;
    STATE.svdDomain = cacheDomain;
  }
}

async function loadProblems() {
  if (!STATE.cache) return;
  setStatus("Loading problems...");
  const rows = await apiFetch("/api/problems");
  const sel = $("sel-problem");
  sel.innerHTML = "";
  (rows || []).forEach((r) => {
    const opt = document.createElement("option");
    opt.value = r.problem_id;
    opt.textContent = `${r.problem_id} · ${r.num_runs} runs · acc ${r.accuracy}%`;
    sel.appendChild(opt);
  });
  STATE.problemId = rows && rows.length ? rows[0].problem_id : null;
  if (STATE.problemId) sel.value = STATE.problemId;
}

/* ---- 1. renderTrajectory — Hero chart (all N runs + Top1/2 highlighted) ---- */
function renderTrajectory() {
  const traj = STATE.trajectory;
  if (!traj || !traj.success) {
    setPlotEmpty("plot-trajectory", "Trajectory unavailable.");
    $("trajectory-subtitle").textContent = "—";
    return;
  }

  const slotScores = traj.slot_scores || [];
  const runInfos = traj.run_infos || [];
  const positions = traj.positions || [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

  if (!slotScores.length) {
    setPlotEmpty("plot-trajectory", traj.fallback || "No slot scores for this problem.");
    $("trajectory-subtitle").textContent = traj.fallback || "—";
    return;
  }

  const topRuns = (STATE.scores && STATE.scores.top_runs) || [];
  const top1SampleId = topRuns[0] ? Number(topRuns[0].sample_id) : -1;
  const top2SampleId = topRuns[1] ? Number(topRuns[1].sample_id) : -1;

  // Map sample_id → local index in run_infos
  const sampleToLocal = {};
  runInfos.forEach((r, i) => { sampleToLocal[Number(r.sample_id)] = i; });
  const top1Idx = sampleToLocal[top1SampleId] ?? -1;
  const top2Idx = sampleToLocal[top2SampleId] ?? -1;

  // Anchor zone backgrounds at 10/40/70/100
  const shapes = [
    { type: "rect", x0: 10, x1: 30, y0: 0, y1: 1, xref: "x", yref: "paper",
      fillcolor: "rgba(37,99,235,0.07)", line: { width: 0 } },
    { type: "rect", x0: 40, x1: 60, y0: 0, y1: 1, xref: "x", yref: "paper",
      fillcolor: "rgba(124,58,237,0.07)", line: { width: 0 } },
    { type: "rect", x0: 70, x1: 90, y0: 0, y1: 1, xref: "x", yref: "paper",
      fillcolor: "rgba(245,158,11,0.07)", line: { width: 0 } },
  ];

  const traces = [];

  // Background runs: thin, low-opacity (green=correct, red=incorrect)
  slotScores.forEach((ys, i) => {
    if (i === top1Idx || i === top2Idx) return;
    const ri = runInfos[i] || {};
    traces.push({
      type: "scatter", mode: "lines",
      x: positions, y: ys,
      name: `Run ${ri.run_index ?? i}`,
      showlegend: false,
      line: { color: ri.is_correct ? C.green : C.red, width: 1 },
      opacity: 0.28,
    });
  });

  // Top2: orange dashed
  if (top2Idx >= 0 && top2Idx < slotScores.length) {
    const ri = runInfos[top2Idx] || {};
    traces.push({
      type: "scatter", mode: "lines+markers",
      x: positions, y: slotScores[top2Idx],
      name: `Top2 Run ${ri.run_index ?? top2Idx} ${ri.is_correct ? "✓" : "✗"}`,
      line: { color: C.orange, width: 2, dash: "dash" },
    });
  }

  // Top1: thick blue solid
  if (top1Idx >= 0 && top1Idx < slotScores.length) {
    const ri = runInfos[top1Idx] || {};
    traces.push({
      type: "scatter", mode: "lines+markers",
      x: positions, y: slotScores[top1Idx],
      name: `Top1 Run ${ri.run_index ?? top1Idx} ${ri.is_correct ? "✓" : "✗"}`,
      line: { color: C.blue, width: 3 },
    });
  }

  safePlot("plot-trajectory", traces, {
    margin: { l: 50, r: 20, t: 20, b: 50 },
    xaxis: { title: "Official Slot (%)", tickvals: positions },
    yaxis: { title: "Verifier Score" },
    shapes,
    legend: { orientation: "h", y: -0.26, x: 0 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
  });

  // Subtitle
  const top1Run = topRuns[0];
  if (top1Run) {
    const margin = top1Run.margin_vs_next != null ? fmt(top1Run.margin_vs_next, 4) : "—";
    const correct = top1Run.is_correct ? "✓" : "✗";
    const shape = (traj.trajectory_shapes || [])[top1Idx] || "";
    $("trajectory-subtitle").textContent =
      `Selected: Run ${top1Run.run_index} ${correct}  margin=${margin}${shape ? `  shape=${shape}` : ""}`;
  } else {
    $("trajectory-subtitle").textContent = "—";
  }
}

/* ---- 2. renderDecision — Card 2 ---- */
function rowKey(row) {
  return String(row.key || row.feature || row.family || row.label || "");
}

function rowLabel(row) {
  return String(row.label || row.feature || row.family || row.key || "").slice(0, 28);
}

function renderFeatureRowsHtml(rows, emptyText = "No rows.") {
  if (!rows || !rows.length) {
    return `<div style='color:var(--text-muted);font-size:12px'>${emptyText}</div>`;
  }
  const maxAdv = Math.max(...rows.map((r) => Math.abs(Number(r.advantage ?? r.delta ?? r.total_contribution ?? 0))), 1e-9);
  return rows.map((r) => {
    const adv = Number(r.advantage ?? r.delta ?? r.total_contribution ?? 0);
    const pct = Math.round((Math.abs(adv) / maxAdv) * 100);
    const barCls = adv >= 0 ? "pos" : "neg";
    const fam = r.family || "other";
    const key = rowKey(r);
    const label = rowLabel(r);
    const sign = adv >= 0 ? "+" : "";
    return `
      <div class="feat-row">
        <div class="feat-label fam-${fam}" title="${key}">${label}</div>
        <div class="feat-track">
          <div class="feat-bar ${barCls}" style="width:${pct}%"></div>
        </div>
        <div class="feat-delta">${sign}${fmt(adv, 3)}</div>
      </div>`;
  }).join("");
}

async function loadCanonicalRunContributions(sampleId) {
  if (!isCanonicalSvdMethod() || !STATE.problemId) return;
  const data = await apiFetch("/api/svd/explain/run_contributions", {
    method: STATE.methodId,
    problem_id: STATE.problemId,
    sample_id: sampleId,
    anchor: STATE.anchor,
  });
  STATE.runContributions = data;
  STATE.selectedExplainSampleId = Number(sampleId);
  renderFeaturePanel();
  renderDecision();
}

function renderDecision() {
  const scores = STATE.scores || {};
  const topRuns = scores.top_runs || [];

  const cardsEl = $("decision-top-runs");
  const cardsHtml = !topRuns.length
    ? "<div style='color:var(--text-muted);font-size:13px'>No runs available.</div>"
    : topRuns.slice(0, 3).map((r, i) => {
        const rankCls = i === 0 ? "rank-1" : i === 1 ? "rank-2" : "";
        const dotCls = r.is_correct ? "dot-correct" : "dot-incorrect";
        const margin = r.margin_vs_next != null ? fmt(r.margin_vs_next, 4) : "—";
        const clickableCls = isCanonicalSvdMethod() ? "clickable" : "";
        const activeCls = Number(r.sample_id) === Number(STATE.selectedExplainSampleId) ? "active-run" : "";
        return `
          <div class="top-run-card ${rankCls} ${clickableCls} ${activeCls}" data-sample="${r.sample_id}">
            <div class="rank-badge">Top${i + 1}</div>
            <div class="run-info">
              <div class="run-label">Run ${r.run_index} · sample ${r.sample_id}</div>
              <div class="run-score">score ${fmt(r.score, 4)} · margin ${margin}</div>
            </div>
            <div class="correctness-dot ${dotCls}"></div>
          </div>`;
      }).join("");
  cardsEl.innerHTML = `<div class="top-run-cards">${cardsHtml}</div>`;

  if (isCanonicalSvdMethod()) {
    cardsEl.querySelectorAll(".top-run-card[data-sample]").forEach((el) => {
      el.addEventListener("click", async () => {
        await loadCanonicalRunContributions(Number(el.dataset.sample));
      });
    });
  }

  $("why-text").textContent = scores.why_selected || "—";

  const famEl = $("decision-family-deltas");
  const familyRows = (scores.top_family_deltas || []).slice(0, 4);
  if (familyRows.length) {
    famEl.innerHTML = familyRows.map((row) => {
      const delta = Number(row.delta || 0);
      const cls = delta >= 0 ? "pos" : "neg";
      const sign = delta >= 0 ? "+" : "";
      return `<div class="family-chip ${cls}">${row.family}: ${sign}${fmt(delta, 3)}</div>`;
    }).join("");
  } else {
    famEl.innerHTML = "";
  }

  const diag = scores.diagnostics || {};
  const rows = [];
  (diag.blockers || []).forEach((x) => rows.push(`<div class="diag-row bad">⛔ ${x}</div>`));
  (diag.warnings || []).forEach((x) => rows.push(`<div class="diag-row warn">⚠ ${x}</div>`));
  (diag.notes || []).forEach((x) => rows.push(`<div class="diag-row note">ℹ ${x}</div>`));
  if (!rows.length) rows.push("<div class='diag-row note'>✓ Status normal.</div>");
  $("diag-block").innerHTML = rows.join("");
}

/* ---- 3. renderFeaturePanel — Card 3 (pure HTML bars, no Plotly) ---- */
function renderFeaturePanel() {
  const fp = STATE.featurePanel;
  const panelEl = $("plot-features");
  if (!fp || !fp.success) {
    panelEl.innerHTML = "<div style='color:var(--text-muted);font-size:12px'>Feature panel unavailable.</div>";
    return;
  }

  if (isCanonicalSvdMethod()) {
    const decisionRows = (fp.feature_rows || []).slice(0, 10);
    const familyRows = (fp.family_rows || []).slice(0, 6);
    const runData = STATE.runContributions || {};
    const runRows = (runData.feature_contributions || []).slice(0, 10).map((row) => ({
      ...row,
      key: row.feature,
      label: row.feature,
      advantage: row.total_contribution,
    }));
    const modelRows = ((STATE.modelSummary && STATE.modelSummary.top_positive_features) || []).slice(0, 6).map((row) => ({
      ...row,
      key: row.feature,
      label: row.feature,
      advantage: row.signed_weight,
    }));
    const runMeta = runData.run || {};
    const metaNote = runData.success
      ? `Selected run: Run ${runMeta.run_index} · score ${fmt(runData.score, 4)} · recon err ${fmt(runData.reconstruction_error, 6)}`
      : "Select a top run to inspect per-run contributions.";
    panelEl.innerHTML = `
      <div class="feat-section">
        <div class="feat-section-title">Top1 vs Top2 Feature Deltas</div>
        ${renderFeatureRowsHtml(decisionRows, "No decision deltas.")}
      </div>
      <div class="feat-section">
        <div class="feat-section-title">Top Family Deltas</div>
        ${renderFeatureRowsHtml(familyRows.map((row) => ({
          ...row,
          key: row.family,
          label: row.family,
          advantage: row.delta,
          family: row.family,
        })), "No family deltas.")}
      </div>
      <div class="feat-section">
        <div class="feat-section-title">Selected Run Contributions</div>
        <div class="feat-meta-note">${metaNote}</div>
        ${renderFeatureRowsHtml(runRows, "No run contribution rows.")}
      </div>
      <div class="feat-section">
        <div class="feat-section-title">Model Anchor Priors</div>
        <div class="feat-meta-note">${STATE.modelSummary?.route_meta ? `anchor=${STATE.modelSummary.anchor_pct}% · domain=${STATE.modelSummary.domain}` : "—"}</div>
        ${renderFeatureRowsHtml(modelRows, "No model summary rows.")}
      </div>`;
    return;
  }

  const rows = (fp.feature_rows || []).slice(0, 12);
  panelEl.innerHTML = renderFeatureRowsHtml(rows, "No feature rows.");
}

/* ---- 4. renderGroupContext — Card 4 (sorted horizontal bar) ---- */
function renderGroupContext() {
  const scores = STATE.scores || {};
  const runs = (scores.group_context && scores.group_context.runs) || [];

  if (!runs.length) {
    setPlotEmpty("plot-group", "No group context.");
    $("group-subtitle").textContent = "—";
    return;
  }

  const topIds = new Set((scores.top_runs || []).map((r) => Number(r.sample_id)));
  const sorted = [...runs].sort((a, b) => Number(a.score) - Number(b.score));

  safePlot(
    "plot-group",
    [{
      type: "bar",
      orientation: "h",
      x: sorted.map((r) => Number(r.score)),
      y: sorted.map((r) => `Run ${r.run_index}`),
      text: sorted.map((r) => topIds.has(Number(r.sample_id)) ? "★" : ""),
      textposition: "outside",
      marker: {
        color: sorted.map((r) => r.is_correct ? C.green : C.red),
        line: {
          color: sorted.map((r) => topIds.has(Number(r.sample_id)) ? "#111827" : "transparent"),
          width: sorted.map((r) => topIds.has(Number(r.sample_id)) ? 1.5 : 0),
        },
      },
      hovertemplate: "%{y}<br>score=%{x:.4f}<extra></extra>",
    }],
    {
      margin: { l: 70, r: 40, t: 16, b: 36 },
      xaxis: { title: "Method Score" },
      yaxis: { automargin: true, tickfont: { size: 10 } },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
    }
  );

  const nCorrect = runs.filter((r) => r.is_correct).length;
  $("group-subtitle").textContent =
    `${runs.length} runs · ${nCorrect} correct · green=correct, red=incorrect, ★=selected`;
}

/* ---- 5. renderTokenEvidence — Card 5 ---- */
function renderTokenEvidence() {
  const ev = STATE.tokenEvidence;
  if (!ev || !ev.success) {
    setPlotEmpty("plot-token-metric", "Token evidence unavailable.");
    $("text-left").innerHTML = "";
    $("text-right").innerHTML = "";
    return;
  }

  const focus = STATE.focusMetric;
  const metrics = ["conf", "entropy", "gini", "selfcert", "logprob"];
  const traces = [];

  function addRun(runObj, suffix, dash) {
    if (!runObj) return;
    const x = [...Array(runObj.num_slices || 0).keys()];
    metrics.forEach((m) => {
      const y = (runObj.metrics && runObj.metrics[m]) || [];
      traces.push({
        type: "scatter", mode: "lines", x, y,
        name: `${METRIC_LABEL[m] || m} ${suffix}`,
        line: { color: C[m], width: m === focus ? 2.7 : 1.2, dash },
        opacity: m === focus ? 1.0 : 0.2,
      });
    });
  }

  addRun(ev.primary, "Top1", "solid");
  addRun(ev.compare, "Top2", "dot");

  safePlot("plot-token-metric", traces, {
    margin: { l: 45, r: 12, t: 16, b: 46 },
    xaxis: { title: "Slice Index" },
    yaxis: { title: "Metric Value" },
    legend: { orientation: "h", y: -0.32, x: 0 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
  });

  const highlights = ev.highlights || {};
  const hlSet = new Set(highlights[focus] || []);

  function renderTextPanel(runObj, titleId, boxId) {
    if (!runObj || !runObj.run) {
      $(boxId).innerHTML = "";
      return;
    }
    $(titleId).textContent = `Run ${runObj.run.run_index} · ${runObj.run.is_correct ? "✓" : "✗"}`;
    const html = (runObj.slices || []).map((s) => {
      const cls = hlSet.has(Number(s.idx)) ? "slice highlight" : "slice";
      return `<span class="${cls}" data-sample="${runObj.run.sample_id}" data-idx="${s.idx}" data-start="${s.tok_start}" data-end="${s.tok_end}">[${s.idx}] ${s.text}</span>`;
    }).join("\n");
    $(boxId).innerHTML = html || "";
  }

  renderTextPanel(ev.primary, "text-left-title", "text-left");
  renderTextPanel(ev.compare, "text-right-title", "text-right");
  bindSliceClicks();
}

function bindSliceClicks() {
  document.querySelectorAll(".slice").forEach((el) => {
    el.addEventListener("click", async () => {
      document.querySelectorAll(".slice.active").forEach((n) => n.classList.remove("active"));
      el.classList.add("active");
      const sid = Number(el.dataset.sample);
      const idx = Number(el.dataset.idx);
      const start = Number(el.dataset.start);
      const end = Number(el.dataset.end);
      try {
        const data = await apiFetch(`/api/slice/${sid}/${idx}`, { tok_start: start, tok_end: end });
        const rows = data.tokens || [];
        if (!rows.length) { $("token-detail-body").textContent = "No tokens."; return; }
        const head = `<div class="token-row header"><span>pos</span><span>token</span><span>conf</span><span>H</span><span>gini</span><span>self</span><span>logp</span></div>`;
        const body = rows.map((t) =>
          `<div class="token-row"><span>${t.pos}</span><span>${(t.text || "").replace(/</g, "&lt;")}</span><span>${fmt(t.conf, 4)}</span><span>${fmt(t.entropy, 4)}</span><span>${fmt(t.gini, 4)}</span><span>${fmt(t.selfcert, 4)}</span><span>${fmt(t.logprob, 4)}</span></div>`
        ).join("");
        $("token-detail-body").innerHTML = head + body;
      } catch (e) {
        $("token-detail-body").textContent = `Error: ${e.message}`;
      }
    });
  });
}

/* ---- Neuron: lazy-load on Advanced <details> toggle ---- */
async function loadAndRenderNeuron() {
  const scores = STATE.scores || {};
  const top1 = (scores.top_runs || [])[0];
  if (!top1) { setPlotEmpty("plot-neuron", "No top run available."); return; }
  try {
    const data = await apiFetch(`/api/neuron_heatmap/${top1.sample_id}`, { mode: STATE.mode });
    const hm = (data.heatmap && data.heatmap.count) || [];
    if (!hm.length) { setPlotEmpty("plot-neuron", "No neuron heatmap data."); return; }
    const x = [...Array(data.num_slices || 0).keys()];
    safePlot("plot-neuron", [{
      type: "heatmap", z: hm, x, y: data.layers || [],
      colorscale: "Blues", colorbar: { title: "Count" },
    }], {
      title: { text: "Neuron Activation (Top1)", font: { size: 13 } },
      margin: { l: 60, r: 30, t: 32, b: 36 },
      xaxis: { title: "Slice" },
      yaxis: { title: "Layer" },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
    });
  } catch (e) {
    setPlotEmpty("plot-neuron", `Load error: ${e.message}`);
  }
}

function renderAll() {
  renderTrajectory();
  renderDecision();
  renderFeaturePanel();
  renderGroupContext();
  renderTokenEvidence();
}

async function loadCanonicalDashboard(pid, method, mode) {
  const problemSummary = await apiFetch("/api/svd/explain/problem_top1_vs_top2", {
    method,
    problem_id: STATE.problemId,
    anchor: STATE.anchor,
  });
  const modelSummary = await apiFetch("/api/svd/explain/model_summary", {
    method,
    anchor: STATE.anchor,
    domain: STATE.svdDomain,
  });

  STATE.scores = problemSummary;
  STATE.trajectory = {
    success: true,
    problem_id: problemSummary.problem_id,
    positions: problemSummary.positions || [10, 40, 70, 100],
    slot_scores: problemSummary.slot_scores || [],
    run_infos: problemSummary.run_infos || [],
    trajectory_shapes: [],
  };
  STATE.featurePanel = {
    success: true,
    feature_rows: problemSummary.top_feature_deltas || [],
    family_rows: problemSummary.top_family_deltas || [],
  };
  STATE.modelSummary = modelSummary || problemSummary.model_summary || null;

  const topRuns = problemSummary.top_runs || [];
  const top1 = topRuns[0];
  const top2 = topRuns[1];

  if (top1) {
    STATE.selectedExplainSampleId = Number(top1.sample_id);
    const [runContrib, tokenEvidence] = await loadAll([
      apiFetch("/api/svd/explain/run_contributions", {
        method,
        problem_id: STATE.problemId,
        sample_id: top1.sample_id,
        anchor: STATE.anchor,
      }),
      apiFetch(`/api/token_evidence/${top1.sample_id}`, {
        compare_sample_id: top2 ? top2.sample_id : top1.sample_id,
        mode,
      }),
    ]);
    STATE.runContributions = runContrib;
    STATE.tokenEvidence = tokenEvidence;
  } else {
    STATE.selectedExplainSampleId = null;
    STATE.runContributions = null;
    STATE.tokenEvidence = null;
  }
}

/* ---- Main dashboard load: parallel fetch with error isolation ---- */
async function loadDashboard() {
  if (!STATE.cache || !STATE.problemId) return;
  clearAlert();
  STATE.neuronLoaded = null;
  STATE.runContributions = null;
  STATE.modelSummary = null;

  const pid = encodeURIComponent(STATE.problemId);
  const method = STATE.methodId;
  const mode = STATE.mode;

  setStatus("Loading dashboard...");

  if (isCanonicalSvdMethod(method)) {
    await loadCanonicalDashboard(pid, method, mode);
  } else {
    const [trajectory, scores, featurePanel, bootstrap] = await loadAll([
      apiFetch(`/api/svd_trajectory/${pid}`),
      apiFetch(`/api/method_scores/${pid}`, { method }),
      apiFetch(`/api/feature_panel/${pid}`, { method }),
      apiFetch(`/api/dashboard_bootstrap/${pid}`, { method, mode }),
    ]);

    STATE.trajectory = trajectory;
    STATE.scores = scores;
    STATE.featurePanel = featurePanel;
    STATE.tokenEvidence = bootstrap?.token_evidence ?? null;

    if (!STATE.scores && bootstrap?.scores) STATE.scores = bootstrap.scores;
  }

  renderAll();

  const renderReady = isCanonicalSvdMethod(method)
    ? Boolean(STATE.scores && STATE.scores.applicable !== false)
    : Boolean(STATE.scores && STATE.scores.render_ready);
  if (!renderReady) {
    showAlert("Limited renderable data for this problem. Showing available diagnostics.");
  }
  setStatus("Ready");
}

/* ---- Event bindings ---- */
function bindEvents() {
  $("sel-dataset").addEventListener("change", async (e) => {
    STATE.cache = e.target.value;
    updateMethodControls();
    STATE.neuronLoaded = null;
    await loadProblems();
    await loadDashboard();
  });

  $("sel-problem").addEventListener("change", async (e) => {
    STATE.problemId = e.target.value;
    STATE.neuronLoaded = null;
    await loadDashboard();
  });

  $("sel-method").addEventListener("change", async (e) => {
    STATE.methodId = e.target.value;
    updateMethodControls();
    STATE.neuronLoaded = null;
    await loadDashboard();
  });

  $("sel-anchor").addEventListener("change", async (e) => {
    STATE.anchor = Number(e.target.value);
    STATE.neuronLoaded = null;
    await loadDashboard();
  });

  $("sel-svd-domain").addEventListener("change", async (e) => {
    STATE.svdDomain = e.target.value;
    if (isCanonicalSvdMethod()) {
      STATE.modelSummary = await apiFetch("/api/svd/explain/model_summary", {
        method: STATE.methodId,
        anchor: STATE.anchor,
        domain: STATE.svdDomain,
      });
      renderFeaturePanel();
    }
  });

  $("sel-mode").addEventListener("change", async (e) => {
    STATE.mode = e.target.value;
    STATE.neuronLoaded = null;
    await loadDashboard();
  });

  $("token-metric-focus").addEventListener("change", (e) => {
    STATE.focusMetric = e.target.value;
    if (STATE.tokenEvidence) renderTokenEvidence();
  });

  document.getElementById("advanced-fold")
    .addEventListener("toggle", async (ev) => {
      if (ev.target.open && STATE.neuronLoaded !== STATE.problemId) {
        await loadAndRenderNeuron();
        STATE.neuronLoaded = STATE.problemId;
      }
    });
}

/* ---- Entry point ---- */
async function main() {
  try {
    await initDatasets();
    await Promise.all([updateHealth(), initMethods()]);
    STATE.anchor = Number($("sel-anchor").value);
    STATE.svdDomain = currentCacheDomain();
    $("sel-svd-domain").value = STATE.svdDomain;
    updateMethodControls();
    await loadProblems();
    bindEvents();
    STATE.mode = $("sel-mode").value;
    STATE.focusMetric = $("token-metric-focus").value;
    await loadDashboard();
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
    showAlert(`Initialization failed: ${err.message}`);
  }
}

main();
