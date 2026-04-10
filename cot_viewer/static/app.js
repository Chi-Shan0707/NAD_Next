const CONFIG = window.COT_VIEWER_CONFIG || {};
const SCRIPT_ROOT = (CONFIG.scriptRoot || "").replace(/\/$/, "");

const state = {
  cache: null,
  problemId: null,
  methodId: "svd_slot100",
  mode: "fixed",
  datasets: {},
  methods: [],
  scores: null,
  lens: null,
  tokenEvidence: null,
  runCompare: null,
  top1SampleId: null,
  top2SampleId: null,
  neuronLoadedFor: null,
};

const COLORS = {
  positive: "#16a34a",
  risky: "#dc2626",
  confidence: "#2563eb",
  trajectory: "#7c3aed",
  instability: "#f59e0b",
  inactive: "#6b7280",
  green: "#16a34a",
  red: "#dc2626",
  blue: "#2563eb",
  purple: "#7c3aed",
  orange: "#f59e0b",
  conf: "#2563eb",
  entropy: "#dc2626",
  gini: "#16a34a",
  selfcert: "#7c3aed",
  logprob: "#f59e0b",
};

const METRIC_LABEL = {
  conf: "置信度",
  entropy: "熵",
  gini: "Gini",
  selfcert: "Self-Cert",
  logprob: "LogProb",
};

const $ = (id) => document.getElementById(id);

function withRoot(path) {
  if (!path.startsWith("/")) return `${SCRIPT_ROOT}/${path}`;
  return `${SCRIPT_ROOT}${path}`;
}

function setStatus(msg) {
  $("status").textContent = msg;
}

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

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status}`);
  }
  return res.json();
}

async function api(path, params = {}, includeCache = true) {
  const query = new URLSearchParams();
  if (includeCache && state.cache) query.set("cache", state.cache);
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined && `${v}` !== "") query.set(k, v);
  }
  const qs = query.toString();
  const url = `${withRoot(path)}${qs ? `?${qs}` : ""}`;
  return fetchJSON(url);
}

function setPlotEmpty(targetId, msg) {
  const node = $(targetId);
  node.innerHTML = `<div class="plot-empty">${msg}</div>`;
}

function safePlot(targetId, traces, layout = {}, config = {}) {
  if (!window.Plotly) {
    setPlotEmpty(targetId, "Plotly 未加载（已降级到文本解释）。");
    return;
  }
  Plotly.newPlot(
    $(targetId),
    traces,
    layout,
    {
      displayModeBar: false,
      responsive: true,
      ...config,
    }
  );
}

async function updateHealth() {
  try {
    const data = await api("/api/health_viewer", {}, false);
    const pill = $("health-pill");
    const artifacts = data.artifacts || {};
    const critical =
      artifacts.earlystop_svd_model &&
      artifacts.bridge_model &&
      artifacts.code_v2_metrics &&
      artifacts.gpqa_pairwise_model;
    pill.classList.remove("ok", "bad");
    if (critical) {
      pill.classList.add("ok");
      pill.textContent = `Health: OK · datasets=${data.datasets}`;
    } else {
      pill.classList.add("bad");
      pill.textContent = `Health: degraded · datasets=${data.datasets}`;
    }
  } catch (err) {
    const pill = $("health-pill");
    pill.classList.remove("ok");
    pill.classList.add("bad");
    pill.textContent = "Health: unavailable";
  }
}

async function initDatasets() {
  setStatus("Loading datasets...");
  const datasets = await api("/api/datasets", {}, false);
  state.datasets = datasets || {};
  const sel = $("sel-dataset");
  sel.innerHTML = "";
  Object.entries(state.datasets).forEach(([name, path]) => {
    const opt = document.createElement("option");
    opt.value = path;
    opt.textContent = name;
    sel.appendChild(opt);
  });
  const first = Object.values(state.datasets)[0];
  state.cache = first || null;
  if (state.cache) sel.value = state.cache;
}

async function initMethods() {
  const data = await api("/api/method_catalog", {}, false);
  state.methods = data.methods || [];
  state.methodId = data.primary_method_id || "svd_slot100";
  const sel = $("sel-method");
  sel.innerHTML = "";
  state.methods.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.primary ? `${m.label}（主线）` : m.label;
    sel.appendChild(opt);
  });
  sel.value = state.methodId;
}

async function loadProblems() {
  if (!state.cache) return;
  setStatus("Loading problems...");
  const rows = await api("/api/problems");
  const sel = $("sel-problem");
  sel.innerHTML = "";
  rows.forEach((r) => {
    const opt = document.createElement("option");
    opt.value = r.problem_id;
    opt.textContent = `${r.problem_id} · ${r.num_runs} runs · acc ${r.accuracy}%`;
    sel.appendChild(opt);
  });
  state.problemId = rows.length ? rows[0].problem_id : null;
  if (state.problemId) sel.value = state.problemId;
}

function renderMiniTable(rows, columns) {
  if (!rows || !rows.length) return "<div class='subtle-note'>No rows.</div>";
  const head = columns.map((c) => `<th>${c.label}</th>`).join("");
  const body = rows
    .map((r) => {
      const tds = columns
        .map((c) => {
          const v = typeof c.render === "function" ? c.render(r[c.key], r) : r[c.key];
          return `<td>${v ?? "—"}</td>`;
        })
        .join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderTopCards(topRuns = []) {
  const box = $("top-cards");
  if (!topRuns.length) {
    box.innerHTML = "<div class='subtle-note'>无可用 run。</div>";
    return;
  }
  box.innerHTML = topRuns
    .map((r, i) => {
      const cls = i === 0 ? "top-card top1" : "top-card";
      const okCls = r.is_correct ? "ok" : "bad";
      const margin = r.margin_vs_next == null ? "—" : fmt(r.margin_vs_next, 4);
      return `
      <div class="${cls}">
        <div class="rank">Top${i + 1}</div>
        <div class="run">Run ${r.run_index} · sample ${r.sample_id}</div>
        <div class="score">score: <b>${fmt(r.score, 4)}</b></div>
        <div class="${okCls}">${r.correctness_mark} ${r.is_correct ? "correct" : "incorrect"}</div>
        <div class="rank">rank #${r.rank} · margin vs next: ${margin}</div>
      </div>`;
    })
    .join("");
}

function renderDiagnostics(diag = {}) {
  const box = $("diagnostics-box");
  const rows = [];
  const blockers = diag.blockers || [];
  const warnings = diag.warnings || [];
  const notes = diag.notes || [];

  blockers.forEach((x) => rows.push(`<div class="diag-row bad">⛔ ${x}</div>`));
  warnings.forEach((x) => rows.push(`<div class="diag-row warn">⚠ ${x}</div>`));
  notes.forEach((x) => rows.push(`<div class="diag-row note">ℹ ${x}</div>`));
  if (!rows.length) {
    rows.push("<div class='diag-row note'>✅ 当前方法与数据状态正常。</div>");
  }
  box.innerHTML = rows.join("");
}

function renderSummary(scores = {}) {
  renderTopCards(scores.top_runs || []);
  $("why-selected").textContent = scores.why_selected || "—";
  $("method-meta").textContent = [
    `Method: ${scores.method_label || "—"}`,
    `Domain: ${scores.domain || "—"}`,
    `Cache: ${scores.cache_key || "—"}`,
    scores.applicable ? "" : "⚠ 当前方法不是该域主方法（诊断模式）",
    scores.failure_mode ? `status=${scores.failure_mode}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  renderDiagnostics(scores.diagnostics || {});
}

function plotHorizontalBars(targetId, rows, title) {
  if (!rows || !rows.length) {
    setPlotEmpty(targetId, "No feature diff.");
    return;
  }
  const labels = rows.map((r) => r.label).reverse();
  const vals = rows.map((r) => Number(r.advantage)).reverse();
  const colors = rows
    .map((r) => (Number(r.advantage) >= 0 ? COLORS.positive : COLORS.risky))
    .reverse();

  safePlot(
    targetId,
    [
      {
        type: "bar",
        orientation: "h",
        x: vals,
        y: labels,
        marker: { color: colors },
        text: vals.map((v) => fmt(v, 4)),
        textposition: "auto",
        hovertemplate: "%{y}<br>advantage=%{x:.4f}<extra></extra>",
      },
    ],
    {
      title: { text: title, font: { size: 12 } },
      margin: { l: 150, r: 20, t: 30, b: 28 },
      xaxis: { zeroline: true, zerolinecolor: "#94a3b8" },
      yaxis: { automargin: true },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
    }
  );
}

function renderGroupContext(scores = {}) {
  const runs = (scores.group_context && scores.group_context.runs) || [];
  if (!runs.length) {
    setPlotEmpty("group-context-plot", "No group context.");
    setPlotEmpty("diff-top12", "No feature diff.");
    setPlotEmpty("diff-median", "No feature diff.");
    return;
  }

  const topIds = new Set((scores.top_runs || []).map((r) => Number(r.sample_id)));
  const headCount = scores.group_context.head_count || 1;
  const headScores = runs
    .filter((r) => r.rank <= headCount)
    .map((r) => Number(r.score))
    .sort((a, b) => a - b);
  const headThreshold = headScores.length ? headScores[0] : null;

  const shapes = [];
  if (headThreshold != null) {
    shapes.push({
      type: "line",
      x0: headThreshold,
      x1: headThreshold,
      y0: -0.45,
      y1: 0.45,
      line: { color: "#94a3b8", width: 1.2, dash: "dot" },
    });
  }

  safePlot(
    "group-context-plot",
    [
      {
        type: "scatter",
        mode: "markers",
        x: runs.map((r) => Number(r.score)),
        y: runs.map((r) => Number(r.density_jitter)),
        text: runs.map(
          (r) =>
            `Run ${r.run_index} · sample ${r.sample_id}<br>rank #${r.rank}<br>${r.is_correct ? "✓ correct" : "✗ incorrect"}`
        ),
        hovertemplate: "%{text}<br>score=%{x:.4f}<extra></extra>",
        marker: {
          color: runs.map((r) => (r.is_correct ? COLORS.positive : COLORS.risky)),
          size: runs.map((r) => (topIds.has(Number(r.sample_id)) ? 14 : 8)),
          line: {
            color: runs.map((r) => (topIds.has(Number(r.sample_id)) ? "#111827" : "white")),
            width: runs.map((r) => (topIds.has(Number(r.sample_id)) ? 1.1 : 0.8)),
          },
          opacity: 0.92,
        },
      },
    ],
    {
      title: { text: "64-run 组内分布（X=method score, Y=density jitter）", font: { size: 13 } },
      margin: { l: 50, r: 20, t: 34, b: 38 },
      xaxis: { title: "Method Score" },
      yaxis: { title: "Density Jitter", showticklabels: false, zeroline: false },
      shapes,
      paper_bgcolor: "white",
      plot_bgcolor: "white",
    }
  );

  plotHorizontalBars("diff-top12", (scores.compare_bars && scores.compare_bars.top1_vs_top2) || [], "Top1 vs Top2");
  plotHorizontalBars(
    "diff-median",
    (scores.compare_bars && scores.compare_bars.top1_vs_median) || [],
    "Top1 vs Group Median"
  );
}

function renderLensTable(rows) {
  $("method-lens-table").innerHTML = renderMiniTable(rows, [
    { key: "label", label: "Feature" },
    { key: "top1_value", label: "Top1", render: (v) => fmt(v, 4) },
    { key: "top2_value", label: "Top2", render: (v) => fmt(v, 4) },
    { key: "delta", label: "Δ", render: (v) => fmt(v, 4) },
    { key: "top1_percentile", label: "Top1 pct", render: (v) => `${fmt(v, 1)}%` },
  ]);
}

function renderMethodLens(lens) {
  $("method-lens-note").textContent = lens && lens.explanation ? lens.explanation : "";
  $("method-lens-table").innerHTML = "";

  if (!lens || !lens.success) {
    setPlotEmpty("method-lens-plot", "Lens unavailable.");
    return;
  }

  if (lens.lens_type === "code_v2") {
    const rows = lens.features || [];
    const labels = rows.map((r) => r.label);
    const top1Vals = rows.map((r) => Number(r.top1_contribution));
    const top2Vals = rows.map((r) => Number(r.top2_contribution));
    safePlot(
      "method-lens-plot",
      [
        { type: "bar", x: labels, y: top1Vals, name: "Top1 contrib", marker: { color: COLORS.blue } },
        { type: "bar", x: labels, y: top2Vals, name: "Top2 contrib", marker: { color: COLORS.orange } },
      ],
      {
        barmode: "group",
        margin: { l: 40, r: 20, t: 30, b: 70 },
        title: { text: "code_v2: 特征贡献对比", font: { size: 13 } },
        yaxis: { title: "Weighted Contribution" },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
      }
    );
    renderLensTable(rows);
    const blind = lens.cache_blind_shape || {};
    if (Object.keys(blind).length) {
      $("method-lens-note").textContent += ` · blind agreement=${fmt(blind.top1_agreement, 4)}`;
    }
    return;
  }

  if (lens.lens_type === "slot100_verifier") {
    const x = lens.positions || [];
    const y1 = lens.slot_scores_top1 || [];
    const y2 = lens.slot_scores_top2 || [];
    const shapes = [
      { type: "rect", x0: 10, x1: 30, y0: 0, y1: 1, xref: "x", yref: "paper", fillcolor: "rgba(37,99,235,0.08)", line: { width: 0 } },
      { type: "rect", x0: 40, x1: 60, y0: 0, y1: 1, xref: "x", yref: "paper", fillcolor: "rgba(124,58,237,0.08)", line: { width: 0 } },
      { type: "rect", x0: 70, x1: 90, y0: 0, y1: 1, xref: "x", yref: "paper", fillcolor: "rgba(245,158,11,0.08)", line: { width: 0 } },
    ];
    safePlot(
      "method-lens-plot",
      [
        { type: "scatter", mode: "lines+markers", x, y: y1, name: "Top1", line: { color: COLORS.blue, width: 3 } },
        { type: "scatter", mode: "lines+markers", x, y: y2, name: "Top2", line: { color: COLORS.purple, width: 2, dash: "dot" } },
      ],
      {
        title: { text: "slot trajectory（10/40/70/100 anchors）", font: { size: 13 } },
        margin: { l: 45, r: 20, t: 32, b: 42 },
        xaxis: { title: "Official Slot" },
        yaxis: { title: "Verifier Score" },
        shapes,
        paper_bgcolor: "white",
        plot_bgcolor: "white",
      }
    );
    const rowData = [
      { label: "source", top1_value: lens.source || "slot100", top2_value: "", delta: "", top1_percentile: "" },
      { label: "anchor note", top1_value: lens.anchor_note || "", top2_value: "", delta: "", top1_percentile: "" },
    ];
    $("method-lens-table").innerHTML = renderMiniTable(rowData, [
      { key: "label", label: "Field" },
      { key: "top1_value", label: "Value" },
    ]);
    return;
  }

  if (lens.lens_type === "science_hybrid_round3") {
    const dec = lens.decision || {};
    const baseline = dec.baseline_scores || [];
    const pairwise = dec.pairwise_scores || [];
    const hybrid = dec.hybrid_scores || [];
    const x = baseline.map((_, i) => `run#${i}`);
    safePlot(
      "method-lens-plot",
      [
        { type: "scatter", mode: "lines+markers", x, y: baseline, name: "baseline", line: { color: COLORS.blue } },
        { type: "scatter", mode: "lines+markers", x, y: pairwise, name: "pairwise", line: { color: COLORS.purple } },
        { type: "scatter", mode: "lines+markers", x, y: hybrid, name: "hybrid", line: { color: COLORS.green, width: 3 } },
      ],
      {
        title: { text: "science_hybrid: baseline / pairwise / hybrid", font: { size: 13 } },
        margin: { l: 45, r: 20, t: 32, b: 60 },
        yaxis: { title: "Score" },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
      }
    );
    $("method-lens-table").innerHTML = renderMiniTable(
      [
        { key: "trigger", value: lens.triggered ? "Yes" : "No", note: `shortlist=${(lens.shortlist_indices || []).join(",")}` },
        { key: "override", value: lens.overridden ? "Yes" : "No", note: "" },
        { key: "baseline_gap", value: fmt(lens.baseline_gap, 4), note: "" },
        { key: "pairwise_margin", value: fmt(lens.pairwise_margin_vs_baseline, 4), note: "" },
        { key: "reason", value: lens.gate_reason || "", note: "" },
      ],
      [
        { key: "key", label: "Field" },
        { key: "value", label: "Value" },
        { key: "note", label: "Note" },
      ]
    );
    return;
  }

  const rows = lens.features || [];
  if (!rows.length) {
    setPlotEmpty("method-lens-plot", "该方法当前无可解释特征。");
    $("method-lens-table").innerHTML = "<div class='subtle-note'>No feature rows.</div>";
    return;
  }

  const labels = rows.map((r) => r.label);
  const values = rows.map((r) => Number(r.advantage || 0));
  safePlot(
    "method-lens-plot",
    [
      {
        type: "bar",
        x: labels,
        y: values,
        marker: { color: values.map((v) => (v >= 0 ? COLORS.green : COLORS.red)) },
      },
    ],
    {
      title: { text: `${lens.lens_type || "method"} feature deltas`, font: { size: 13 } },
      margin: { l: 40, r: 20, t: 30, b: 70 },
      yaxis: { title: "Advantage" },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
    }
  );
  $("method-lens-table").innerHTML = renderMiniTable(rows, [
    { key: "label", label: "Feature" },
    { key: "advantage", label: "Advantage", render: (v) => fmt(v, 4) },
    { key: "delta", label: "Δ", render: (v) => fmt(v, 4) },
  ]);
}

function renderTrajectory(scores, compare) {
  const top = scores && scores.top_runs ? scores.top_runs : [];
  if (!top.length || !compare || !compare.success) {
    setPlotEmpty("trajectory-plot", "该问题暂无可用轨迹。");
    $("trajectory-note").textContent = "";
    return;
  }

  const extras = compare.extras || {};
  const slotScores = extras.slot_scores || null;
  if (!slotScores || !slotScores.length) {
    setPlotEmpty("trajectory-plot", "该方法无 early-stop 轨迹，主要看 token 证据。");
    $("trajectory-note").textContent = "";
    return;
  }

  const leftIdx = Number(compare.left.local_index);
  const rightIdx = Number(compare.right.local_index);
  const positions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
  const y1 = slotScores[leftIdx] || [];
  const y2 = slotScores[rightIdx] || [];
  safePlot(
    "trajectory-plot",
    [
      { type: "scatter", mode: "lines+markers", x: positions, y: y1, name: `Top1 Run ${compare.left.run_index}`, line: { color: COLORS.blue, width: 3 } },
      { type: "scatter", mode: "lines+markers", x: positions, y: y2, name: `Top2 Run ${compare.right.run_index}`, line: { color: COLORS.purple, width: 2, dash: "dot" } },
    ],
    {
      margin: { l: 45, r: 20, t: 18, b: 36 },
      xaxis: { title: "Official Slots" },
      yaxis: { title: "Score" },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
    }
  );
  $("trajectory-note").textContent =
    "route复用：10/20/30→10%，40/50/60→40%，70/80/90→70%，100为独立final。";
}

function renderTokenText(side, runPayload, highlights, titleId, boxId) {
  if (!runPayload || !runPayload.run) {
    $(boxId).innerHTML = "<div class='subtle-note'>No text.</div>";
    return;
  }
  $(titleId).textContent = `Run ${runPayload.run.run_index} · sample ${runPayload.run.sample_id} · ${
    runPayload.run.is_correct ? "✓" : "✗"
  }`;
  const highlightSet = new Set(highlights || []);
  const html = (runPayload.slices || [])
    .map((s) => {
      const cls = highlightSet.has(Number(s.idx)) ? "slice highlight" : "slice";
      return `<span class="${cls}" data-side="${side}" data-sample="${runPayload.run.sample_id}" data-idx="${s.idx}" data-start="${s.tok_start}" data-end="${s.tok_end}">[${s.idx}] ${s.text}</span>`;
    })
    .join("\n");
  $(boxId).innerHTML = html || "<div class='subtle-note'>No slices.</div>";
}

async function renderTokenDetail(sampleId, sliceIdx, tokStart, tokEnd) {
  const data = await api(`/api/slice/${sampleId}/${sliceIdx}`, {
    tok_start: tokStart,
    tok_end: tokEnd,
  });
  const rows = data.tokens || [];
  if (!rows.length) {
    $("token-detail-body").innerHTML = "No token rows.";
    return;
  }
  const head =
    "<div class=\"token-row header\"><span>pos</span><span>token</span><span>conf</span><span>H</span><span>gini</span><span>self</span><span>logp</span></div>";
  const body = rows
    .map(
      (t) =>
        `<div class="token-row"><span>${t.pos}</span><span>${(t.text || "").replaceAll("<", "&lt;")}</span><span>${fmt(
          t.conf,
          4
        )}</span><span>${fmt(t.entropy, 4)}</span><span>${fmt(t.gini, 4)}</span><span>${fmt(t.selfcert, 4)}</span><span>${fmt(
          t.logprob,
          4
        )}</span></div>`
    )
    .join("");
  $("token-detail-body").innerHTML = head + body;
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
      await renderTokenDetail(sid, idx, start, end);
    });
  });
}

function renderTokenMetricPlot(evidence) {
  if (!evidence || !evidence.primary) {
    setPlotEmpty("token-metric-plot", "Token evidence unavailable.");
    return;
  }
  const focus = $("token-metric-focus").value;
  const metrics = ["conf", "entropy", "gini", "selfcert", "logprob"];
  const traces = [];

  const addRun = (runObj, suffix, dash = "solid") => {
    if (!runObj) return;
    const x = [...Array(runObj.num_slices || 0).keys()];
    metrics.forEach((m) => {
      const y = (runObj.metrics && runObj.metrics[m]) || [];
      traces.push({
        type: "scatter",
        mode: "lines",
        x,
        y,
        name: `${METRIC_LABEL[m] || m} ${suffix}`,
        line: { color: COLORS[m], width: m === focus ? 2.7 : 1.4, dash },
        opacity: m === focus ? 1.0 : 0.22,
      });
    });
  };

  addRun(evidence.primary, "Top1", "solid");
  addRun(evidence.compare, "Top2", "dot");

  safePlot(
    "token-metric-plot",
    traces,
    {
      margin: { l: 45, r: 12, t: 16, b: 34 },
      xaxis: { title: "Slice Index" },
      yaxis: { title: "Metric Value" },
      legend: { orientation: "h", y: -0.22, x: 0 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
    }
  );
}

function renderTokenEvidence(evidence) {
  state.tokenEvidence = evidence;
  if (!evidence || !evidence.success) {
    setPlotEmpty("token-metric-plot", "Token evidence unavailable.");
    $("text-left").innerHTML = "<div class='subtle-note'>No text.</div>";
    $("text-right").innerHTML = "<div class='subtle-note'>No text.</div>";
    return;
  }

  renderTokenMetricPlot(evidence);
  const focus = $("token-metric-focus").value;
  const highlights = evidence.highlights || {};
  renderTokenText("left", evidence.primary, highlights[focus], "text-left-title", "text-left");
  renderTokenText("right", evidence.compare, highlights[focus], "text-right-title", "text-right");
  bindSliceClicks();
}

async function renderDerivative(scores) {
  const top = scores.top_runs || [];
  const left = top[0];
  const right = top[1] || top[0];
  if (!left) {
    setPlotEmpty("derivative-plot", "No derivative.");
    return;
  }

  const [d1, d2] = await Promise.all([
    api(`/api/derivatives/${left.sample_id}`, { mode: state.mode }),
    api(`/api/derivatives/${right.sample_id}`, { mode: state.mode }),
  ]);
  const x1 = [...Array(d1.num_slices || 0).keys()];
  const x2 = [...Array(d2.num_slices || 0).keys()];
  const traces = [
    { type: "scatter", mode: "lines", x: x1, y: (d1.d1 && d1.d1.conf) || [], name: "Top1 d1(conf)", line: { color: COLORS.conf, width: 2 } },
    { type: "scatter", mode: "lines", x: x1, y: (d1.d1 && d1.d1.entropy) || [], name: "Top1 d1(entropy)", line: { color: COLORS.entropy, width: 2 } },
    { type: "scatter", mode: "lines", x: x2, y: (d2.d1 && d2.d1.conf) || [], name: "Top2 d1(conf)", line: { color: COLORS.conf, dash: "dot" } },
    { type: "scatter", mode: "lines", x: x2, y: (d2.d1 && d2.d1.entropy) || [], name: "Top2 d1(entropy)", line: { color: COLORS.entropy, dash: "dot" } },
  ];
  safePlot(
    "derivative-plot",
    traces,
    {
      margin: { l: 45, r: 12, t: 18, b: 34 },
      xaxis: { title: "Slice" },
      yaxis: { title: "d1" },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
    }
  );
}

function buildMethodHighlightShapes(methodId, nSlices) {
  const shapes = [];
  if (nSlices <= 0) return shapes;
  const x = (p) => Math.max(0, Math.min(nSlices - 1, Math.round((p / 100) * (nSlices - 1))));
  if (methodId === "svd_slot100" || methodId === "slot100_verifier") {
    [10, 40, 70, 100].forEach((p) => {
      const xi = x(p);
      shapes.push({
        type: "line",
        x0: xi,
        x1: xi,
        y0: 0,
        y1: 1,
        xref: "x",
        yref: "paper",
        line: { color: COLORS.blue, width: 1, dash: "dot" },
      });
    });
  } else if (methodId === "code_v2") {
    shapes.push({
      type: "rect",
      x0: Math.floor(nSlices * 0.8),
      x1: nSlices - 1,
      y0: 0,
      y1: 1,
      xref: "x",
      yref: "paper",
      fillcolor: "rgba(245,158,11,0.12)",
      line: { width: 0 },
    });
  } else if (methodId === "extreme8_reflection") {
    shapes.push({
      type: "rect",
      x0: Math.floor(nSlices * 0.45),
      x1: Math.floor(nSlices * 0.7),
      y0: 0,
      y1: 1,
      xref: "x",
      yref: "paper",
      fillcolor: "rgba(124,58,237,0.10)",
      line: { width: 0 },
    });
  }
  return shapes;
}

async function renderNeuron(scores) {
  const top1 = scores.top_runs && scores.top_runs[0];
  if (!top1) return;

  const details = $("neuron-fold");
  if (!details || !details.open) return;
  if (state.neuronLoadedFor === `${state.cache}:${state.problemId}:${state.mode}:${state.methodId}`) return;

  const data = await api(`/api/neuron_heatmap/${top1.sample_id}`, { mode: state.mode });
  const hm = (data.heatmap && data.heatmap.count) || [];
  if (!hm.length) {
    setPlotEmpty("neuron-plot", "No neuron heatmap.");
    return;
  }
  const layers = data.layers || [];
  const x = [...Array(data.num_slices || 0).keys()];
  const shapes = buildMethodHighlightShapes(state.methodId, x.length);
  safePlot(
    "neuron-plot",
    [
      {
        type: "heatmap",
        z: hm,
        x,
        y: layers,
        colorscale: "Blues",
        colorbar: { title: "Count" },
      },
    ],
    {
      title: { text: "Neuron Activation (Top1)", font: { size: 13 } },
      margin: { l: 60, r: 30, t: 32, b: 36 },
      xaxis: { title: "Slice" },
      yaxis: { title: "Layer" },
      shapes,
      paper_bgcolor: "white",
      plot_bgcolor: "white",
    }
  );
  state.neuronLoadedFor = `${state.cache}:${state.problemId}:${state.mode}:${state.methodId}`;
}

async function refreshDashboard() {
  if (!state.cache || !state.problemId) return;
  clearAlert();
  try {
    setStatus("Loading decision dashboard...");
    const boot = await api(`/api/dashboard_bootstrap/${encodeURIComponent(state.problemId)}`, {
      method: state.methodId,
      mode: state.mode,
    });
    if (!boot.success) {
      throw new Error(boot.error || "bootstrap failed");
    }

    state.scores = boot.scores || {};
    state.lens = boot.lens || {};
    state.runCompare = boot.run_compare || {};

    renderSummary(state.scores);
    renderMethodLens(state.lens);
    renderGroupContext(state.scores);
    renderTrajectory(state.scores, state.runCompare);
    renderTokenEvidence(boot.token_evidence || {});

    setStatus("Loading derivatives...");
    await renderDerivative(state.scores);

    await renderNeuron(state.scores);

    if (!state.scores.render_ready) {
      showAlert("当前问题可渲染数据不足，已展示可用诊断信息。");
    }
    setStatus("Ready");
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
    showAlert(`页面加载失败：${err.message}`);
    setPlotEmpty("method-lens-plot", "加载失败");
    setPlotEmpty("group-context-plot", "加载失败");
    setPlotEmpty("trajectory-plot", "加载失败");
    setPlotEmpty("token-metric-plot", "加载失败");
  }
}

function bindEvents() {
  $("sel-dataset").addEventListener("change", async (e) => {
    state.cache = e.target.value;
    state.neuronLoadedFor = null;
    await loadProblems();
    await refreshDashboard();
  });

  $("sel-problem").addEventListener("change", async (e) => {
    state.problemId = e.target.value;
    state.neuronLoadedFor = null;
    await refreshDashboard();
  });

  $("sel-method").addEventListener("change", async (e) => {
    state.methodId = e.target.value;
    state.neuronLoadedFor = null;
    await refreshDashboard();
  });

  $("sel-mode").addEventListener("change", async (e) => {
    state.mode = e.target.value;
    state.neuronLoadedFor = null;
    await refreshDashboard();
  });

  $("token-metric-focus").addEventListener("change", () => {
    if (!state.tokenEvidence) return;
    renderTokenMetricPlot(state.tokenEvidence);
    const focus = $("token-metric-focus").value;
    const highlights = state.tokenEvidence.highlights || {};
    renderTokenText("left", state.tokenEvidence.primary, highlights[focus], "text-left-title", "text-left");
    renderTokenText("right", state.tokenEvidence.compare, highlights[focus], "text-right-title", "text-right");
    bindSliceClicks();
  });

  const neuronDetails = $("neuron-fold");
  if (neuronDetails) {
    neuronDetails.addEventListener("toggle", async () => {
      if (neuronDetails.open && state.scores) {
        await renderNeuron(state.scores);
      }
    });
  }
}

async function main() {
  try {
    await updateHealth();
    await initDatasets();
    await initMethods();
    await loadProblems();
    bindEvents();
    state.mode = $("sel-mode").value;
    await refreshDashboard();
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
    showAlert(`初始化失败：${err.message}`);
  }
}

main();
