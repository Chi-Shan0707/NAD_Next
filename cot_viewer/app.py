#!/usr/bin/env python3
"""Chain-of-Thought Viewer – browse decoded reasoning chains and per-token metrics."""

import json
import os
import sys
import math
from typing import Any, Optional

sys.path.insert(0, "/home/jovyan/work/NAD_Next")

from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file, url_for

from nad.core.views.reader import Agg, CacheReader, ViewSpec, CutSpec, CutType, Order
from nad.io.loader import NadNextLoader
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Optional analysis deps (scatter / MDS / selector simulation)
# ---------------------------------------------------------------------------

try:
    from sklearn.manifold import MDS as _MDS
    _HAS_MDS = True
except ImportError:
    _HAS_MDS = False

try:
    from nad.core.distance.engine import DistanceEngine, DistanceSpec
    from nad.core.selectors.base import SelectorContext
    from nad.core.selectors.extreme8_impl import (
        extract_extreme8_raw_values,
        build_extreme8_features,
    )
    _HAS_ANALYSIS = True
except ImportError:
    _HAS_ANALYSIS = False

try:
    from nad.ops.earlystop import EARLY_STOP_POSITIONS
    from nad.ops.earlystop_svd import (
        extract_earlystop_signals_for_sample,
        get_domain as _earlystop_domain,
        load_earlystop_svd_bundle,
        _build_representation as _earlystop_build_representation,
        _predict_svd_lr as _earlystop_predict_svd_lr,
        _rank_transform_matrix as _earlystop_rank_transform_matrix,
    )
    from nad.ops.earlystop_svm import load_earlystop_svm_bundle
    from nad.core.selectors.code_v2_impl import (
        DEFAULT_CODE_V2_WEIGHTS,
        compute_code_v2_primary_scores,
    )
    from nad.core.selectors.gpqa_pairwise_impl import (
        GPQAPairwiseScorer,
        build_gpqa_pairwise_features_configurable,
        extract_gpqa_pairwise_raw,
    )
    from nad.core.selectors.science_dynamic_impl import (
        compute_science_dynamic_primary_scores,
    )
    from nad.core.selectors.science_hybrid_impl import (
        ScienceHybridConfig,
        compute_pairwise_probability_matrix,
        compute_science_hybrid_decision,
    )
    _HAS_DECISION_METHODS = True
except ImportError:
    _HAS_DECISION_METHODS = False

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

MUI_HUB_ROOT = Path("/home/jovyan/public-ro/MUI_HUB")
MODEL_NAME   = "DeepSeek-R1-0528-Qwen3-8B"
MODEL_PATH   = "/home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B"

DATASETS: dict[str, str] = {}      # name -> cache_dir path
LOADERS: dict[str, NadNextLoader] = {}
READERS: dict[str, CacheReader] = {}
EVAL_REPORTS: dict[str, dict] = {}
META_CACHE: dict[str, dict] = {}
TOKENIZER = None
BOUNDARY_IDS: np.ndarray = None    # token IDs for \n . ? : — set at startup
SCATTER_CACHE: dict = {}           # (problem_id, cache_path) -> {"result":…, "raw":…}
GROUP_COMPARE_CACHE: dict = {}     # (problem_id, cache_path, metric, mode) -> response dict
NGC_CACHE: dict = {}               # (problem_id, cache_path) -> neuron_group_compare response
DECISION_CONTEXT_CACHE: dict = {}  # (cache_path, problem_id) -> reusable problem context
METHOD_RESULT_CACHE: dict = {}     # (cache_path, problem_id, method_id) -> decision payload
METHOD_LENS_CACHE: dict = {}       # (cache_path, problem_id, method_id) -> method lens payload
RUN_COMPARE_CACHE: dict = {}       # (cache_path, problem_id, method_id) -> run compare payload
TOKEN_EVIDENCE_CACHE: dict = {}    # (cache_path, sample_id, compare_id, mode) -> token evidence payload

REPO_ROOT = Path("/home/jovyan/work/NAD_Next")
EARLYSTOP_SVD_MODEL = REPO_ROOT / "models/ml_selectors/earlystop_prefix10_svd_round1.pkl"
BRIDGE_MODEL = REPO_ROOT / "models/ml_selectors/bestofn_svm_bridge_v1.pkl"
GPQA_PAIRWISE_MODEL = REPO_ROOT / "models/ml_selectors/gpqa_pairwise_round1.pkl"
CODE_V2_METRICS_PATH = REPO_ROOT / "result/code_v2_candidate_20260406_exhaustive/code_v2_metrics.json"
SCIENCE_HYBRID_RESULT_GLOB = str(REPO_ROOT / "result/science_hybrid_round3_*/science_hybrid_round3.json")

MODEL_ALIAS_MAP = {
    "DeepSeek-R1-0528-Qwen3-8B": "DS-R1",
    "Qwen3-4B": "Qwen3-4B",
}

OFFICIAL_SLOT_TO_ANCHOR = {
    0.1: 0.1,
    0.2: 0.1,
    0.3: 0.1,
    0.4: 0.4,
    0.5: 0.4,
    0.6: 0.4,
    0.7: 0.7,
    0.8: 0.7,
    0.9: 0.7,
    1.0: 1.0,
}

ANCHOR_POSITIONS = (0.1, 0.4, 0.7, 1.0)
ANCHOR_SLOT_INDICES = [0, 3, 6, 9]
METHOD_COLORS = {
    "positive": "#16a34a",
    "risky": "#dc2626",
    "confidence": "#2563eb",
    "trajectory": "#7c3aed",
    "instability": "#f59e0b",
    "inactive": "#6b7280",
}

METHOD_CATALOG = [
    {
        "id": "svd_slot100",
        "label": "SVD Slot100 主线",
        "family": "earlystop_verifier",
        "applies_to": ["math", "science", "coding"],
        "primary": True,
        "description": "earlystop_prefix10_svd_round1_slot100 + coding bridge",
    },
    {
        "id": "slot100_verifier",
        "label": "Slot100 Verifier",
        "family": "earlystop_verifier",
        "applies_to": ["math", "science", "coding"],
        "primary": False,
        "description": "earlystop_prefix10_svd_round1 slot trajectory",
    },
    {
        "id": "svm_bridge_lcb",
        "label": "SVM Bridge",
        "family": "bridge",
        "applies_to": ["math", "science", "coding"],
        "primary": False,
        "description": "bestofn_svm_bridge_v1 final-slot verifier",
    },
    {
        "id": "code_v2",
        "label": "Code v2",
        "family": "coding",
        "applies_to": ["coding"],
        "primary": False,
        "description": "prefix saturation v2 coding selector",
    },
    {
        "id": "science_hybrid_round3",
        "label": "Science Hybrid Round3",
        "family": "science",
        "applies_to": ["science"],
        "primary": False,
        "description": "shortlist_blend + pairwise rerank",
    },
    {
        "id": "science_baseline_v1",
        "label": "Science Baseline v1",
        "family": "science",
        "applies_to": ["science"],
        "primary": False,
        "description": "recency confidence baseline",
    },
    {
        "id": "extreme8_reflection",
        "label": "Extreme8 Reflection",
        "family": "reflection",
        "applies_to": ["math", "science", "coding"],
        "primary": False,
        "description": "dc_z / dc_r / reflection_count_r lens",
    },
    {
        "id": "gpqa_pairwise_round1",
        "label": "GPQA Pairwise Round1",
        "family": "science",
        "applies_to": ["science"],
        "primary": False,
        "description": "pairwise scorer only",
    },
]


def _scan_datasets():
    """Scan all cache* splits under MUI_HUB_ROOT for the configured model."""
    for split_dir in sorted(MUI_HUB_ROOT.iterdir()):
        if not split_dir.is_dir() or not split_dir.name.startswith("cache"):
            continue
        model_dir = split_dir / MODEL_NAME
        if not model_dir.is_dir():
            continue
        split_label = split_dir.name  # "cache", "cache_train", "cache_test"
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            cache_dirs = sorted(
                [d for d in ds_dir.iterdir()
                 if d.is_dir() and d.name.startswith("cache_neuron_")]
            )
            if cache_dirs:
                key = f"{ds_dir.name}  [{split_label}]"
                DATASETS[key] = str(cache_dirs[-1])


def _get_loader(cache_path: str) -> NadNextLoader:
    if cache_path not in LOADERS:
        LOADERS[cache_path] = NadNextLoader(cache_path)
    return LOADERS[cache_path]


def _get_reader(cache_path: str) -> CacheReader:
    if cache_path not in READERS:
        READERS[cache_path] = CacheReader(cache_path)
    return READERS[cache_path]


def _get_eval_report(cache_path: str) -> dict:
    if cache_path not in EVAL_REPORTS:
        p = os.path.join(cache_path, "evaluation_report_compact.json")
        if os.path.exists(p):
            with open(p) as f:
                EVAL_REPORTS[cache_path] = json.load(f)
        else:
            EVAL_REPORTS[cache_path] = {}
    return EVAL_REPORTS[cache_path]


def _get_meta(cache_path: str) -> dict:
    if cache_path not in META_CACHE:
        p = os.path.join(cache_path, "meta.json")
        with open(p) as f:
            META_CACHE[cache_path] = json.load(f)
    return META_CACHE[cache_path]


def _ensure_runtime_loaded() -> None:
    global TOKENIZER, BOUNDARY_IDS

    if not DATASETS:
        _scan_datasets()
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if BOUNDARY_IDS is None and TOKENIZER is not None:
        BOUNDARY_IDS = _compute_boundary_ids(TOKENIZER)


def _require_cache() -> str:
    _ensure_runtime_loaded()
    cache = request.args.get("cache")
    if not cache or cache not in DATASETS.values():
        return None
    return cache


def _compute_boundary_ids(tokenizer) -> np.ndarray:
    """Encode boundary characters and return the union of their token IDs."""
    ids: set[int] = set()
    for ch in ['\n', '.', '?', ':']:
        ids.update(tokenizer.encode(ch, add_special_tokens=False))
    return np.array(sorted(ids), dtype=np.int32)


def _plotly_vendor_path() -> Optional[Path]:
    """Return local plotly.min.js path from installed plotly package if available."""
    try:
        import plotly  # type: ignore

        p = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
        if p.exists():
            return p
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Slice-building helpers
# ---------------------------------------------------------------------------

def _build_fixed_slices(offsets: list[int], token_ids) -> list[dict]:
    """One output slice per raw row (original fixed-32-token behaviour)."""
    slices = []
    for i in range(len(offsets) - 1):
        tok_start = offsets[i]
        tok_end = offsets[i + 1]
        if tok_start >= len(token_ids):
            break
        text = TOKENIZER.decode(token_ids[tok_start:tok_end].tolist(), skip_special_tokens=False)
        slices.append({"idx": i, "text": text, "tok_start": tok_start, "tok_end": tok_end})
    return slices


def _build_smart_slices(offsets: list[int], token_ids) -> list[dict]:
    """Merge raw rows into smart super-slices aligned to language boundaries."""
    from nad.ops.smart_slice import smart_slice_grouping

    offs_arr = np.array(offsets, dtype=np.int64)
    tok_arr = np.asarray(token_ids, dtype=np.int32)
    grouping = smart_slice_grouping(tok_arr, offs_arr, BOUNDARY_IDS)

    # Collapse consecutive raw rows that share the same group ID
    group_ranges: dict[int, dict] = {}
    for raw_i, gid in enumerate(grouping.tolist()):
        if gid not in group_ranges:
            group_ranges[gid] = {"tok_start": offsets[raw_i], "tok_end": offsets[raw_i + 1]}
        else:
            group_ranges[gid]["tok_end"] = offsets[raw_i + 1]

    slices = []
    for out_idx, gid in enumerate(sorted(group_ranges)):
        g = group_ranges[gid]
        tok_start = g["tok_start"]
        tok_end = min(g["tok_end"], len(token_ids))
        if tok_start >= len(token_ids):
            break
        text = TOKENIZER.decode(token_ids[tok_start:tok_end].tolist(), skip_special_tokens=False)
        slices.append({"idx": out_idx, "text": text, "tok_start": tok_start, "tok_end": tok_end})
    return slices


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    _ensure_runtime_loaded()
    return render_template(
        "index.html",
        script_root=request.script_root or "",
        plotly_vendor_url=url_for("api_plotly_vendor_js"),
    )


@app.route("/vendor/plotly.min.js")
def api_plotly_vendor_js():
    path = _plotly_vendor_path()
    if path is None:
        return Response(
            "window.Plotly = window.Plotly || undefined;",
            mimetype="application/javascript",
        )
    return send_file(path, mimetype="application/javascript")


@app.route("/api/datasets")
def api_datasets():
    _ensure_runtime_loaded()
    return jsonify(DATASETS)


@app.route("/api/problems")
def api_problems():
    cache = _require_cache()
    if not cache:
        return jsonify({"error": "invalid cache"}), 400

    meta = _get_meta(cache)
    report = _get_eval_report(cache)

    # Build per-problem run count from meta samples
    samples = meta["samples"]
    problem_runs: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        pid = str(s["problem_id"])
        problem_runs.setdefault(pid, []).append(i)

    # Build per-problem accuracy from eval report
    problem_accuracy: dict[str, float] = {}
    if "results" in report:
        for r in report["results"]:
            pid = str(r["problem_id"])
            stats = r.get("statistics", {})
            problem_accuracy[pid] = stats.get("accuracy", 0.0)

    result = []
    for pid in sorted(problem_runs, key=lambda x: int(x) if x.isdigit() else x):
        result.append({
            "problem_id": pid,
            "num_runs": len(problem_runs[pid]),
            "accuracy": round(problem_accuracy.get(pid, 0.0), 1),
        })
    return jsonify(result)


@app.route("/api/runs/<problem_id>")
def api_runs(problem_id: str):
    cache = _require_cache()
    if not cache:
        return jsonify({"error": "invalid cache"}), 400

    meta = _get_meta(cache)
    report = _get_eval_report(cache)

    # Build quick lookup: (problem_id, run_index) -> is_correct
    correctness: dict[tuple[str, int], bool] = {}
    if "results" in report:
        for r in report["results"]:
            if str(r["problem_id"]) == problem_id:
                for run in r.get("runs", []):
                    correctness[(problem_id, run["run_index"])] = run.get("is_correct", False)
                break

    runs = []
    for sample_id, s in enumerate(meta["samples"]):
        if str(s["problem_id"]) == problem_id:
            ri = s["run_index"]
            runs.append({
                "sample_id": sample_id,
                "run_index": ri,
                "is_correct": correctness.get((problem_id, ri), None),
            })
    return jsonify(runs)


def _get_slices(sample_id: int, mode: str, cache: str):
    """Shared helper: build slices and return (slices_list, token_view)."""
    reader = _get_reader(cache)
    loader = _get_loader(cache)

    tv = reader.get_token_view(sample_id)
    if tv.token_ids is None or len(tv.token_ids) == 0:
        return [], tv

    token_ids = tv.token_ids

    # Compute per-row token boundaries (response rows only)
    row_lo, row_hi = loader.get_row_range_for_sample(sample_id)
    rows_trp = reader.rows_token_row_ptr
    if rows_trp is not None:
        base = int(rows_trp[row_lo + 1])
        offsets = [min(int(rows_trp[r]) - base, len(token_ids))
                   for r in range(row_lo + 1, row_hi + 2)]
    else:
        offsets = [0, len(token_ids)]

    if mode == "smart" and BOUNDARY_IDS is not None and rows_trp is not None and len(offsets) > 1:
        try:
            slices = _build_smart_slices(offsets, token_ids)
        except Exception:
            slices = _build_fixed_slices(offsets, token_ids)
    else:
        slices = _build_fixed_slices(offsets, token_ids)

    return slices, tv


def _slice_averages(tv, slices):
    """Compute per-slice nanmean for each metric. Returns {metric: [mean_per_slice]}."""
    metric_sources = {
        "entropy": tv.tok_neg_entropy,
        "conf": tv.tok_conf,
        "gini": tv.tok_gini,
        "selfcert": tv.tok_selfcert,
        "logprob": tv.tok_logprob,
    }
    result = {m: [] for m in metric_sources}
    for s in slices:
        ts, te = s["tok_start"], s["tok_end"]
        for name, arr in metric_sources.items():
            if arr is None or ts >= len(arr):
                result[name].append(None)
                continue
            vals = np.array(arr[ts:min(te, len(arr))], dtype=np.float64)
            if name == "entropy":
                vals = -vals  # neg_entropy → entropy
            if len(vals) == 0 or np.all(np.isnan(vals)):
                result[name].append(None)
            else:
                result[name].append(round(float(np.nanmean(vals)), 6))
    return result


@app.route("/api/chain/<int:sample_id>")
def api_chain(sample_id: int):
    cache = _require_cache()
    if not cache:
        return jsonify({"error": "invalid cache"}), 400

    mode = request.args.get("mode", "fixed")  # "fixed" | "smart"
    slices, tv = _get_slices(sample_id, mode, cache)

    if not slices:
        return jsonify({"num_slices": 0, "slices": []})

    return jsonify({"num_slices": len(slices), "slices": slices})


@app.route("/api/derivatives/<int:sample_id>")
def api_derivatives(sample_id: int):
    cache = _require_cache()
    if not cache:
        return jsonify({"error": "invalid cache"}), 400

    mode = request.args.get("mode", "fixed")
    slices, tv = _get_slices(sample_id, mode, cache)

    if not slices:
        return jsonify({"num_slices": 0, "metrics": [], "averages": {}, "d1": {}, "d2": {}, "d3": {}})

    metrics = ["entropy", "conf", "gini", "selfcert", "logprob"]
    averages = _slice_averages(tv, slices)

    def _diff(values):
        """np.diff with None→NaN handling, returns list with None for NaN."""
        arr = np.array([v if v is not None else np.nan for v in values], dtype=np.float64)
        if len(arr) < 2:
            return []
        d = np.diff(arr)
        return [None if np.isnan(v) else round(float(v), 6) for v in d]

    d1, d2, d3 = {}, {}, {}
    for m in metrics:
        d1[m] = _diff(averages[m])
        d2[m] = _diff(d1[m])
        d3[m] = _diff(d2[m])

    return jsonify({
        "num_slices": len(slices),
        "metrics": metrics,
        "averages": averages,
        "d1": d1,
        "d2": d2,
        "d3": d3,
    })


@app.route("/api/slice/<int:sample_id>/<int:slice_idx>")
def api_slice(sample_id: int, slice_idx: int):
    cache = _require_cache()
    if not cache:
        return jsonify({"error": "invalid cache"}), 400

    reader = _get_reader(cache)
    loader = _get_loader(cache)

    tv = reader.get_token_view(sample_id)
    if tv.token_ids is None:
        return jsonify({"error": "no token data"}), 404

    # Optional direct token-range override (used by smart mode super-slices)
    tok_start_param = request.args.get("tok_start")
    tok_end_param = request.args.get("tok_end")
    if tok_start_param is not None and tok_end_param is not None:
        tok_start = int(tok_start_param)
        tok_end = min(int(tok_end_param), len(tv.token_ids))
    else:
        # Row-based lookup for fixed mode
        rows_trp = reader.rows_token_row_ptr
        if rows_trp is None:
            return jsonify({"error": "no row data"}), 404

        row_lo, row_hi = loader.get_row_range_for_sample(sample_id)
        base = int(rows_trp[row_lo + 1])
        num_slices = row_hi - row_lo
        if slice_idx < 0 or slice_idx >= num_slices:
            return jsonify({"error": "slice_idx out of range"}), 400

        row = row_lo + 1 + slice_idx
        tok_start = int(rows_trp[row]) - base
        tok_end = min(int(rows_trp[row + 1]) - base, len(tv.token_ids))

    tokens = []
    for pos in range(tok_start, tok_end):
        tid = int(tv.token_ids[pos])
        text = TOKENIZER.decode([tid], skip_special_tokens=False)
        tok = {
            "pos": pos,
            "token_id": tid,
            "text": text,
            "conf": _f(tv.tok_conf, pos),
            "entropy": -_f(tv.tok_neg_entropy, pos) if tv.tok_neg_entropy is not None else None,
            "gini": _f(tv.tok_gini, pos),
            "selfcert": _f(tv.tok_selfcert, pos),
            "logprob": _f(tv.tok_logprob, pos),
        }
        tokens.append(tok)

    return jsonify({"slice_idx": slice_idx, "tok_start": tok_start, "tok_end": tok_end, "tokens": tokens})


def _f(arr, idx):
    """Extract a float from an optional numpy array, returning None if unavailable."""
    if arr is None or idx >= len(arr):
        return None
    v = float(arr[idx])
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, 6)


# ---------------------------------------------------------------------------
# Neuron heatmap
# ---------------------------------------------------------------------------

def _build_row_to_slice(num_rows: int, offsets: list[int], token_ids, mode: str):
    """Map each raw row index (0..num_rows-1) to an output slice index.

    For 'fixed' mode this is identity; for 'smart' mode rows are grouped via
    smart_slice_grouping and the mapping reflects the merged super-slices.
    Returns (row_to_slice dict, num_output_slices).
    """
    if mode == "smart" and BOUNDARY_IDS is not None and len(offsets) > 1:
        try:
            from nad.ops.smart_slice import smart_slice_grouping

            offs_arr = np.array(offsets, dtype=np.int64)
            tok_arr = np.asarray(token_ids, dtype=np.int32)
            grouping = smart_slice_grouping(tok_arr, offs_arr, BOUNDARY_IDS)
            # grouping[i] = group id for raw row i; remap to dense 0..N-1
            unique_gids = sorted(set(grouping.tolist()))
            gid_to_slice = {g: idx for idx, g in enumerate(unique_gids)}
            mapping = {i: gid_to_slice[g] for i, g in enumerate(grouping.tolist())}
            return mapping, len(unique_gids)
        except Exception:
            pass
    # Fixed: identity mapping
    return {i: i for i in range(num_rows)}, num_rows


@app.route("/api/neuron_heatmap/<int:sample_id>")
def api_neuron_heatmap(sample_id: int):
    cache = _require_cache()
    if not cache:
        return jsonify({"error": "invalid cache"}), 400

    mode = request.args.get("mode", "fixed")
    reader = _get_reader(cache)
    loader = _get_loader(cache)

    # Check rows/ bank availability
    rows_srp = reader.rows_sample_row_ptr
    rows_rp = reader.rows_row_ptr
    rows_keys = reader.rows_keys
    if rows_srp is None or rows_rp is None or rows_keys is None:
        return jsonify({"error": "no rows/ bank in this cache"}), 404

    rows_w_sum = reader.rows_weights_for(Agg.SUM)
    rows_w_max = reader.rows_weights_for(Agg.MAX)

    # Row range for sample
    row_start = int(rows_srp[sample_id])
    row_end = int(rows_srp[sample_id + 1])
    num_raw_rows = row_end - row_start

    if num_raw_rows == 0:
        return jsonify({"num_slices": 0, "layers": [], "heatmap": {}, "jaccard": [], "token_metrics": {}})

    # Token offsets for smart slicing
    tv = reader.get_token_view(sample_id)
    token_ids = tv.token_ids if tv.token_ids is not None else np.array([], dtype=np.int32)
    rows_trp = reader.rows_token_row_ptr
    if rows_trp is not None and len(token_ids) > 0:
        row_lo, row_hi = loader.get_row_range_for_sample(sample_id)
        base = int(rows_trp[row_lo + 1])
        offsets = [min(int(rows_trp[r]) - base, len(token_ids))
                   for r in range(row_lo + 1, row_hi + 2)]
    else:
        offsets = [0, len(token_ids)] if len(token_ids) > 0 else [0]

    row_to_slice, num_slices = _build_row_to_slice(num_raw_rows, offsets, token_ids, mode)

    # First pass: discover layers and collect per-slice key sets
    layer_set = set()
    slice_key_sets = [set() for _ in range(num_slices)]

    # Accumulators: {(layer_idx, slice_idx) -> [count, w_sum_total, w_max_max]}
    # We'll use dicts first then fill arrays after discovering all layers
    from collections import defaultdict
    cell_count = defaultdict(int)
    cell_wsum = defaultdict(float)
    cell_wmax = defaultdict(float)

    for raw_i in range(num_raw_rows):
        row_idx = row_start + raw_i
        sl = row_to_slice.get(raw_i, raw_i)
        k_start = int(rows_rp[row_idx])
        k_end = int(rows_rp[row_idx + 1])
        if k_start == k_end:
            continue

        keys = rows_keys[k_start:k_end]
        layers = (keys >> 16).astype(np.int32)

        # Collect key set for Jaccard
        slice_key_sets[sl].update(keys.tolist())

        # Per-layer aggregation
        unique_layers = np.unique(layers)
        for lay in unique_layers:
            layer_set.add(int(lay))
            mask = layers == lay
            cnt = int(mask.sum())
            cell_count[(int(lay), sl)] += cnt
            if rows_w_sum is not None:
                cell_wsum[(int(lay), sl)] += float(np.sum(rows_w_sum[k_start:k_end][mask]))
            if rows_w_max is not None:
                cell_wmax[(int(lay), sl)] = max(cell_wmax[(int(lay), sl)],
                                                 float(np.max(rows_w_max[k_start:k_end][mask])))

    layers_sorted = sorted(layer_set)
    layer_to_idx = {l: i for i, l in enumerate(layers_sorted)}
    n_layers = len(layers_sorted)

    # Build 2D arrays [n_layers][num_slices]
    hm_count = [[0] * num_slices for _ in range(n_layers)]
    hm_wsum = [[0.0] * num_slices for _ in range(n_layers)]
    hm_wmax = [[0.0] * num_slices for _ in range(n_layers)]

    for (lay, sl), cnt in cell_count.items():
        li = layer_to_idx[lay]
        hm_count[li][sl] = cnt
        hm_wsum[li][sl] = round(cell_wsum[(lay, sl)], 4)
        hm_wmax[li][sl] = round(cell_wmax[(lay, sl)], 4)

    # Inter-slice Jaccard similarity (consecutive slices)
    jaccard = [None]  # first slice has no predecessor
    for i in range(1, num_slices):
        a, b = slice_key_sets[i - 1], slice_key_sets[i]
        if not a and not b:
            jaccard.append(None)
        else:
            inter = len(a & b)
            union = len(a | b)
            jaccard.append(round(inter / union, 4) if union > 0 else None)

    # Token metrics (reuse existing helpers)
    slices_list, tv2 = _get_slices(sample_id, mode, cache)
    token_metrics = {}
    if slices_list:
        avgs = _slice_averages(tv2, slices_list)
        token_metrics["entropy"] = avgs.get("entropy", [])
        token_metrics["conf"] = avgs.get("conf", [])

    return jsonify({
        "num_slices": num_slices,
        "layers": layers_sorted,
        "heatmap": {
            "count": hm_count,
            "w_sum_total": hm_wsum,
            "w_max_max": hm_wmax,
        },
        "jaccard": jaccard,
        "token_metrics": token_metrics,
    })


# ---------------------------------------------------------------------------
# Analysis helpers — scatter / MDS / KNN / selector simulation
# ---------------------------------------------------------------------------

def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros(len(arr), dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _rank01_arr(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64) / (n - 1)
    return ranks


def _get_problem_run_ids(cache: str, problem_id: str):
    """Return (run_ids, correct_labels) for a problem from meta + eval report."""
    meta = _get_meta(cache)
    report = _get_eval_report(cache)

    correctness: dict[int, bool] = {}
    if "results" in report:
        for r in report["results"]:
            if str(r["problem_id"]) == problem_id:
                for run in r.get("runs", []):
                    correctness[run["run_index"]] = bool(run.get("is_correct", False))
                break

    run_ids, correct_labels = [], []
    for sample_id, s in enumerate(meta["samples"]):
        if str(s["problem_id"]) == problem_id:
            run_ids.append(sample_id)
            correct_labels.append(correctness.get(s.get("run_index", 0), False))

    return run_ids, correct_labels


def _compute_scatter_for_problem(cache: str, problem_id: str):
    """Compute MDS, Extreme8 features, and KNN purity. Returns (result, raw_data)."""
    run_ids, correct_labels = _get_problem_run_ids(cache, problem_id)
    n = len(run_ids)
    empty = {"problem_id": problem_id, "runs": [], "n_runs": 0, "knn_purity_mean": None}
    if n == 0:
        return empty, {}

    reader = _get_reader(cache)
    vspec = ViewSpec(agg=Agg.MAX, cut=CutSpec(CutType.MASS, 0.98), order=Order.BY_KEY)
    views = [reader.get_run_view(rid, vspec) for rid in run_ids]
    lengths = [int(len(v.keys)) for v in views]

    engine = DistanceEngine(DistanceSpec("ja", num_threads=4))
    D = engine.dense_matrix(views).astype(np.float64)

    mds_xy = np.zeros((n, 2), dtype=np.float64)
    if _HAS_MDS and n >= 2:
        try:
            D_sym = np.clip((D + D.T) / 2.0, 0.0, None)
            np.fill_diagonal(D_sym, 0.0)
            mds_xy = _MDS(n_components=2, dissimilarity="precomputed",
                          random_state=42, n_init=1, max_iter=300,
                          normalized_stress="auto").fit_transform(D_sym)
        except Exception:
            pass

    ctx = SelectorContext(cache=reader, problem_id=problem_id,
                          run_ids=run_ids, views=views)
    raw = extract_extreme8_raw_values(ctx)
    feats = build_extreme8_features(raw)  # (n,3): dc_z, dc_r, reflection_count_r

    k = min(5, n - 1)
    runs_list, knn_sum = [], 0.0
    for i in range(n):
        di = D[i].copy(); di[i] = np.inf
        nn = np.argsort(di)[:k]
        knn5, hits = [], 0
        for j in nn:
            knn5.append({"sample_id": int(run_ids[j]),
                          "distance": round(float(di[j]), 4),
                          "correct": bool(correct_labels[j]),
                          "dc_r": round(float(feats[j, 1]), 4),
                          "rc_r": round(float(feats[j, 2]), 4)})
            if correct_labels[j]: hits += 1
        purity = hits / k if k > 0 else 0.0
        knn_sum += purity
        runs_list.append({
            "sample_id": int(run_ids[i]),
            "correct": bool(correct_labels[i]),
            "dc_z": round(float(feats[i, 0]), 4),
            "dc_r": round(float(feats[i, 1]), 4),
            "reflection_count_r": round(float(feats[i, 2]), 4),
            "reflection_count": round(float(raw["reflection_count"][i]), 1),
            "mds_x": round(float(mds_xy[i, 0]), 4),
            "mds_y": round(float(mds_xy[i, 1]), 4),
            "knn5": knn5,
            "knn_purity": round(purity, 2),
        })

    result = {"problem_id": problem_id, "runs": runs_list,
              "n_runs": n, "knn_purity_mean": round(knn_sum / n, 3)}
    raw_data = {"D": D, "feats": feats, "raw": raw,
                "run_ids": run_ids, "correct_labels": correct_labels,
                "lengths": lengths, "mds_xy": mds_xy,
                "cache_path": cache, "problem_id": problem_id}
    return result, raw_data


_SELECTOR_CATALOG = [
    {"id": "dc_r",               "label": "dc_r  (conf. rank)",         "group": "Feature"},
    {"id": "reflection_count_r", "label": "reflection_count_r",         "group": "Feature"},
    {"id": "deepconf",           "label": "DeepConf  (min tok_conf)",   "group": "Feature"},
    {"id": "medoid",             "label": "Medoid  (Jaccard)",          "group": "Distance"},
    {"id": "knn-medoid-3",       "label": "KNN-Medoid  k=3",            "group": "Distance"},
    {"id": "knn-medoid-5",       "label": "KNN-Medoid  k=5",            "group": "Distance"},
    {"id": "min-activation",     "label": "Min-Activation",             "group": "Baseline"},
    {"id": "max-activation",     "label": "Max-Activation",             "group": "Baseline"},
    {"id": "extreme8-best",      "label": "Extreme8-Best  (ML model)",  "group": "ML"},
    {"id": "extreme8-mixed",     "label": "Extreme8-Mixed  (ML model)", "group": "ML"},
]
_SEL_NAME_MAP   = {"knn-medoid-3": "knn-medoid", "knn-medoid-5": "knn-medoid"}
_SEL_PARAMS_MAP = {"knn-medoid-3": {"k": 3}, "knn-medoid-5": {"k": 5}}


def _run_selector(raw_data: dict, selector_id: str) -> dict:
    D = raw_data["D"];  feats = raw_data["feats"]
    raw = raw_data["raw"];  run_ids = raw_data["run_ids"]
    correct_labels = raw_data["correct_labels"];  lengths = raw_data["lengths"]
    cache_path = raw_data["cache_path"];  problem_id = raw_data["problem_id"]
    n = len(run_ids)
    scores = np.zeros(n, dtype=np.float64)
    sel_idx, score_label, error = 0, "score", None

    try:
        if selector_id == "dc_r":
            scores = feats[:, 1].copy(); sel_idx = int(np.argmax(scores)); score_label = "dc_r"
        elif selector_id == "reflection_count_r":
            scores = feats[:, 2].copy(); sel_idx = int(np.argmax(scores)); score_label = "rc_r"
        elif selector_id == "min-activation":
            la = np.array(lengths, dtype=np.float64)
            scores = _norm01(-la); sel_idx = int(np.argmin(la)); score_label = "−length (norm)"
        elif selector_id == "max-activation":
            la = np.array(lengths, dtype=np.float64)
            scores = _norm01(la); sel_idx = int(np.argmax(la)); score_label = "length (norm)"
        elif selector_id == "medoid":
            md = D.mean(axis=1); scores = _norm01(-md)
            sel_idx = int(np.argmin(md)); score_label = "−mean_dist (norm)"
        elif selector_id in ("knn-medoid-3", "knn-medoid-5"):
            kv = min(3 if selector_id == "knn-medoid-3" else 5, n - 1)
            S = 1.0 - D
            rs = np.array([np.mean(np.partition(np.delete(S[i], i), -kv)[-kv:])
                           for i in range(n)])
            scores = _norm01(rs); sel_idx = int(np.argmax(rs))
            score_label = f"knn-sim k={kv} (norm)"
        else:
            from nad.core.selectors.registry import build_selector
            from nad.core.selectors.base import SelectorSpec as _SS
            sel_name = _SEL_NAME_MAP.get(selector_id, selector_id)
            sel_params = _SEL_PARAMS_MAP.get(selector_id, {})
            sel = build_selector(_SS(sel_name, sel_params))
            reader = _get_reader(cache_path)
            vspec = ViewSpec(agg=Agg.MAX, cut=CutSpec(CutType.MASS, 0.98), order=Order.BY_KEY)
            views = [reader.get_run_view(rid, vspec) for rid in run_ids]
            ctx = SelectorContext(cache=reader, problem_id=problem_id,
                                  run_ids=run_ids, views=views)
            sel.bind(ctx)
            run_stats = {"lengths": np.array(lengths, dtype=np.int32)}
            sel_idx = int(sel.select(D, run_stats))
            if selector_id == "deepconf":
                scores = _norm01(-raw["dc_raw"]); score_label = "deepconf quality (norm)"
            elif selector_id in ("extreme8-best", "extreme8-mixed"):
                scores = _norm01(feats[:, 1] + feats[:, 2]); score_label = "dc_r+rc_r proxy"
            else:
                scores[sel_idx] = 1.0
    except Exception as exc:
        error = str(exc)

    return {"selected_idx": sel_idx,
            "selected_sample_id": int(run_ids[sel_idx]),
            "selected_correct": bool(correct_labels[sel_idx]),
            "scores": scores, "score_label": score_label, "error": error}


@app.route("/api/scatter_data/<problem_id>")
def api_scatter_data(problem_id: str):
    cache = _require_cache()
    if not cache:
        return jsonify({"success": False, "error": "invalid cache"}), 400
    if not _HAS_ANALYSIS:
        return jsonify({"success": False, "error": "analysis deps unavailable"}), 503

    key = (problem_id, cache)
    if key not in SCATTER_CACHE:
        try:
            result, raw_data = _compute_scatter_for_problem(cache, problem_id)
            SCATTER_CACHE[key] = {"result": result, "raw": raw_data}
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify({"success": True, **SCATTER_CACHE[key]["result"]})


@app.route("/api/selector_catalog")
def api_selector_catalog():
    return jsonify({"selectors": _SELECTOR_CATALOG})


@app.route("/api/selector_pick/<problem_id>")
def api_selector_pick(problem_id: str):
    cache = _require_cache()
    if not cache:
        return jsonify({"success": False, "error": "invalid cache"}), 400
    if not _HAS_ANALYSIS:
        return jsonify({"success": False, "error": "analysis deps unavailable"}), 503

    selector_id = request.args.get("selector", "medoid").strip()

    key = (problem_id, cache)
    if key not in SCATTER_CACHE:
        try:
            result, raw_data = _compute_scatter_for_problem(cache, problem_id)
            SCATTER_CACHE[key] = {"result": result, "raw": raw_data}
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

    raw_data = SCATTER_CACHE[key]["raw"]
    scatter_result = SCATTER_CACHE[key]["result"]
    if not raw_data:
        return jsonify({"success": False, "error": "no runs"}), 404

    sim = _run_selector(raw_data, selector_id)
    if sim["error"]:
        return jsonify({"success": False, "error": sim["error"],
                        "selector": selector_id}), 500

    scores = sim["scores"]
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int); ranks[order] = np.arange(len(scores))

    all_runs = []
    for i, run_info in enumerate(scatter_result["runs"]):
        all_runs.append({
            "sample_id": run_info["sample_id"],
            "correct": run_info["correct"],
            "score": round(float(scores[i]), 4),
            "rank": int(ranks[i]),
            "is_selected": (i == sim["selected_idx"]),
            "dc_r": run_info["dc_r"],
            "dc_z": run_info["dc_z"],
            "reflection_count_r": run_info["reflection_count_r"],
            "reflection_count": run_info["reflection_count"],
            "mds_x": run_info["mds_x"],
            "mds_y": run_info["mds_y"],
        })
    all_runs.sort(key=lambda r: r["rank"])

    return jsonify({
        "success": True,
        "problem_id": problem_id,
        "selector": selector_id,
        "selected_sample_id": sim["selected_sample_id"],
        "selected_correct": sim["selected_correct"],
        "score_label": sim["score_label"],
        "n_runs": len(all_runs),
        "all_runs": all_runs,
    })


# ---------------------------------------------------------------------------
# Group Compare
# ---------------------------------------------------------------------------

def _compute_group_mean(runs_data):
    if not runs_data:
        return []
    max_len = max(len(r) for r in runs_data)
    result = []
    for i in range(max_len):
        vals = [r[i] for r in runs_data if i < len(r) and r[i] is not None]
        result.append(round(float(np.mean(vals)), 6) if vals else None)
    return result


@app.route("/api/group_compare/<problem_id>")
def api_group_compare(problem_id: str):
    cache = _require_cache()
    if not cache:
        return jsonify({"error": "invalid cache"}), 400

    metric = request.args.get("metric", "entropy")
    mode = request.args.get("mode", "fixed")

    cache_key = (problem_id, cache, metric, mode)
    if cache_key in GROUP_COMPARE_CACHE:
        return jsonify(GROUP_COMPARE_CACHE[cache_key])

    meta = _get_meta(cache)
    report = _get_eval_report(cache)

    # Build correctness lookup for this problem
    correctness: dict[int, bool] = {}
    if "results" in report:
        for r in report["results"]:
            if str(r["problem_id"]) == problem_id:
                for run in r.get("runs", []):
                    correctness[run["run_index"]] = bool(run.get("is_correct", False))
                break

    correct_runs, incorrect_runs = [], []
    for sample_id, s in enumerate(meta["samples"]):
        if str(s["problem_id"]) != problem_id:
            continue
        slices, tv = _get_slices(sample_id, mode, cache)
        if not slices:
            continue
        avgs = _slice_averages(tv, slices)
        vals = avgs.get(metric, [])
        is_correct = correctness.get(s.get("run_index", 0), False)
        if is_correct:
            correct_runs.append(vals)
        else:
            incorrect_runs.append(vals)

    max_slices = max(
        (max((len(r) for r in correct_runs), default=0)),
        (max((len(r) for r in incorrect_runs), default=0)),
    )

    response = {
        "problem_id": problem_id,
        "metric": metric,
        "correct": {
            "runs": correct_runs,
            "mean": _compute_group_mean(correct_runs),
            "n": len(correct_runs),
        },
        "incorrect": {
            "runs": incorrect_runs,
            "mean": _compute_group_mean(incorrect_runs),
            "n": len(incorrect_runs),
        },
        "max_slices": max_slices,
    }
    GROUP_COMPARE_CACHE[cache_key] = response
    return jsonify(response)


# ---------------------------------------------------------------------------
# Neuron Group Compare
# ---------------------------------------------------------------------------

@app.route("/api/neuron_group_compare/<problem_id>")
def api_neuron_group_compare(problem_id: str):
    cache = _require_cache()
    if not cache:
        return jsonify({"error": "invalid cache"}), 400

    ngc_key = (problem_id, cache)
    if ngc_key in NGC_CACHE:
        return jsonify(NGC_CACHE[ngc_key])

    run_ids, correct_labels = _get_problem_run_ids(cache, problem_id)
    n = len(run_ids)
    if n == 0:
        return jsonify({"error": "no runs"}), 404

    reader = _get_reader(cache)
    rows_srp = reader.rows_sample_row_ptr
    rows_rp  = reader.rows_row_ptr
    rows_keys = reader.rows_keys

    if rows_srp is None or rows_rp is None or rows_keys is None:
        return jsonify({"error": "no rows/ bank in this cache"}), 404

    # For each run: collect aggregate key set and per-layer neuron sets
    run_key_sets   = []   # list[set[int]]  — full encoded keys (layer<<16|neuron)
    run_layer_sets = []   # list[dict[int, set[int]]]  — {layer: {neuron_ids}}
    all_layers: set[int] = set()

    for sample_id in run_ids:
        row_start = int(rows_srp[sample_id])
        row_end   = int(rows_srp[sample_id + 1])

        full_set: set[int] = set()
        layer_sets: dict[int, set[int]] = {}

        for row_idx in range(row_start, row_end):
            k_start = int(rows_rp[row_idx])
            k_end   = int(rows_rp[row_idx + 1])
            if k_start == k_end:
                continue
            keys = rows_keys[k_start:k_end]
            full_set.update(keys.tolist())
            layers_arr  = (keys >> 16).astype(np.int32)
            neurons_arr = (keys & 0xFFFF).astype(np.int32)
            for lay, neu in zip(layers_arr.tolist(), neurons_arr.tolist()):
                if lay not in layer_sets:
                    layer_sets[lay] = set()
                layer_sets[lay].add(neu)
                all_layers.add(lay)

        run_key_sets.append(full_set)
        run_layer_sets.append(layer_sets)

    layers_sorted = sorted(all_layers)
    n_correct   = sum(correct_labels)
    n_incorrect = n - n_correct

    # --- A: Layer activation density (unique neurons per layer per run) ---
    correct_dens   = {l: [] for l in layers_sorted}
    incorrect_dens = {l: [] for l in layers_sorted}
    for layer_sets, is_correct in zip(run_layer_sets, correct_labels):
        target = correct_dens if is_correct else incorrect_dens
        for l in layers_sorted:
            target[l].append(len(layer_sets.get(l, set())))

    def _mean_std(vals):
        if not vals:
            return 0.0, 0.0
        a = np.array(vals, dtype=np.float64)
        return round(float(a.mean()), 2), round(float(a.std()), 2)

    layer_density = {"correct": {"mean": [], "std": []},
                     "incorrect": {"mean": [], "std": []}}
    for l in layers_sorted:
        cm, cs = _mean_std(correct_dens[l])
        im, is_ = _mean_std(incorrect_dens[l])
        layer_density["correct"]["mean"].append(cm)
        layer_density["correct"]["std"].append(cs)
        layer_density["incorrect"]["mean"].append(im)
        layer_density["incorrect"]["std"].append(is_)

    # --- B: Pairwise Jaccard distance matrix (sort: correct first) ---
    # Reorder so correct runs come first
    order = [i for i, c in enumerate(correct_labels) if c] + \
            [i for i, c in enumerate(correct_labels) if not c]
    ord_run_ids  = [run_ids[i]  for i in order]
    ord_keys     = [run_key_sets[i] for i in order]
    ord_labels   = ["C" if correct_labels[i] else "I" for i in order]

    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0.0)
            elif j < i:
                row.append(matrix[j][i])
            else:
                a, b = ord_keys[i], ord_keys[j]
                union = len(a | b)
                jac = round(1.0 - len(a & b) / union, 4) if union else 1.0
                row.append(jac)
        matrix.append(row)

    # --- C: Per-layer Venn (union of activated neurons across group) ---
    correct_union   = {}   # {layer: set of neuron ids}
    incorrect_union = {}
    for layer_sets, is_correct in zip(run_layer_sets, correct_labels):
        target = correct_union if is_correct else incorrect_union
        for l, neurons in layer_sets.items():
            if l not in target:
                target[l] = set()
            target[l].update(neurons)

    only_correct, shared, only_incorrect = [], [], []
    for l in layers_sorted:
        c_set = correct_union.get(l, set())
        i_set = incorrect_union.get(l, set())
        only_correct.append(len(c_set - i_set))
        shared.append(len(c_set & i_set))
        only_incorrect.append(len(i_set - c_set))

    response = {
        "problem_id": problem_id,
        "layers": layers_sorted,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "layer_density": layer_density,
        "jaccard_matrix": {
            "matrix": matrix,
            "labels": ord_labels,
            "sample_ids": ord_run_ids,
        },
        "layer_venn": {
            "only_correct": only_correct,
            "shared": shared,
            "only_incorrect": only_incorrect,
        },
    }
    NGC_CACHE[ngc_key] = response
    return jsonify(response)


# ---------------------------------------------------------------------------
# Decision-first dashboard helpers / APIs
# ---------------------------------------------------------------------------

_BUNDLE_CACHE: dict[str, Any] = {}
_CODE_V2_ARTIFACT: Optional[dict[str, Any]] = None
_SCIENCE_HYBRID_ARTIFACT: Optional[dict[str, Any]] = None
_PAIRWISE_SCORER: Optional[Any] = None


def _to_py(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _to_py(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    return v


def _load_bundle_cached(path: Path, loader_fn):
    key = str(path)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    if not path.exists():
        return None
    try:
        _BUNDLE_CACHE[key] = loader_fn(path)
        return _BUNDLE_CACHE[key]
    except Exception:
        return None


def _load_code_v2_artifact() -> dict[str, Any]:
    global _CODE_V2_ARTIFACT
    if _CODE_V2_ARTIFACT is not None:
        return _CODE_V2_ARTIFACT
    if CODE_V2_METRICS_PATH.exists():
        try:
            with CODE_V2_METRICS_PATH.open("r", encoding="utf-8") as f:
                _CODE_V2_ARTIFACT = json.load(f)
        except Exception:
            _CODE_V2_ARTIFACT = {}
    else:
        _CODE_V2_ARTIFACT = {}
    return _CODE_V2_ARTIFACT


def _latest_science_hybrid_path() -> Optional[Path]:
    root = REPO_ROOT / "result"
    paths = sorted(root.glob("science_hybrid_round3_*/science_hybrid_round3.json"))
    if not paths:
        return None
    return paths[-1]


def _load_science_hybrid_artifact() -> dict[str, Any]:
    global _SCIENCE_HYBRID_ARTIFACT
    if _SCIENCE_HYBRID_ARTIFACT is not None:
        return _SCIENCE_HYBRID_ARTIFACT
    p = _latest_science_hybrid_path()
    if p and p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                _SCIENCE_HYBRID_ARTIFACT = json.load(f)
        except Exception:
            _SCIENCE_HYBRID_ARTIFACT = {}
    else:
        _SCIENCE_HYBRID_ARTIFACT = {}
    return _SCIENCE_HYBRID_ARTIFACT


def _load_pairwise_scorer() -> Optional[Any]:
    global _PAIRWISE_SCORER
    if _PAIRWISE_SCORER is not None:
        return _PAIRWISE_SCORER
    if not _HAS_DECISION_METHODS:
        return None
    if not GPQA_PAIRWISE_MODEL.exists():
        return None
    try:
        _PAIRWISE_SCORER = GPQAPairwiseScorer.load(GPQA_PAIRWISE_MODEL)
    except Exception:
        _PAIRWISE_SCORER = None
    return _PAIRWISE_SCORER


def _dataset_name_from_cache(cache: str) -> str:
    try:
        return Path(cache).parent.name
    except Exception:
        return ""


def _model_name_from_cache(cache: str) -> str:
    try:
        return Path(cache).parents[1].name
    except Exception:
        return MODEL_NAME


def _cache_key_from_cache(cache: str) -> str:
    ds = _dataset_name_from_cache(cache)
    model_name = _model_name_from_cache(cache)
    alias = MODEL_ALIAS_MAP.get(model_name, model_name)
    return f"{alias}/{ds}"


def _domain_from_cache(cache: str) -> str:
    ds = _dataset_name_from_cache(cache)
    if _HAS_DECISION_METHODS:
        try:
            return _earlystop_domain(ds)
        except Exception:
            pass
    if ds == "gpqa":
        return "science"
    if "lcb" in ds:
        return "coding"
    return "math"


def _method_def(method_id: str) -> dict[str, Any]:
    for item in METHOD_CATALOG:
        if item["id"] == method_id:
            return item
    return METHOD_CATALOG[0]


def _problem_run_infos(cache: str, problem_id: str) -> list[dict[str, Any]]:
    meta = _get_meta(cache)
    report = _get_eval_report(cache)

    correctness: dict[int, bool] = {}
    if "results" in report:
        for r in report["results"]:
            if str(r["problem_id"]) == str(problem_id):
                for run in r.get("runs", []):
                    correctness[int(run["run_index"])] = bool(run.get("is_correct", False))
                break

    rows = []
    for sample_id, s in enumerate(meta["samples"]):
        if str(s["problem_id"]) != str(problem_id):
            continue
        run_index = int(s.get("run_index", sample_id))
        rows.append({
            "sample_id": int(sample_id),
            "run_index": run_index,
            "is_correct": bool(correctness.get(run_index, False)),
        })
    rows.sort(key=lambda x: x["run_index"])
    return rows


def _build_problem_context(cache: str, problem_id: str) -> dict[str, Any]:
    key = (cache, str(problem_id))
    if key in DECISION_CONTEXT_CACHE:
        return DECISION_CONTEXT_CACHE[key]

    run_infos = _problem_run_infos(cache, str(problem_id))
    run_ids = [int(r["sample_id"]) for r in run_infos]
    n = len(run_ids)

    context = {
        "cache": cache,
        "problem_id": str(problem_id),
        "dataset_name": _dataset_name_from_cache(cache),
        "cache_key": _cache_key_from_cache(cache),
        "domain": _domain_from_cache(cache),
        "run_infos": run_infos,
        "run_ids": run_ids,
        "n_runs": n,
        "reader": _get_reader(cache),
    }

    if n == 0:
        context["D"] = np.zeros((0, 0), dtype=np.float64)
        DECISION_CONTEXT_CACHE[key] = context
        return context

    vspec = ViewSpec(agg=Agg.MAX, cut=CutSpec(CutType.MASS, 0.98), order=Order.BY_KEY)
    views = []
    if _HAS_ANALYSIS:
        try:
            views = [context["reader"].get_run_view(rid, vspec) for rid in run_ids]
        except Exception:
            views = []
    context["views"] = views

    if _HAS_ANALYSIS and views:
        try:
            engine = DistanceEngine(DistanceSpec("ja", num_threads=4))
            D = engine.dense_matrix(views).astype(np.float64)
        except Exception:
            D = np.zeros((n, n), dtype=np.float64)
    else:
        D = np.zeros((n, n), dtype=np.float64)
    context["D"] = D

    if _HAS_ANALYSIS and _HAS_DECISION_METHODS and views:
        try:
            selector_ctx = SelectorContext(
                cache=context["reader"],
                problem_id=str(problem_id),
                run_ids=run_ids,
                views=views,
            )
            context["selector_ctx"] = selector_ctx
            ext_raw = extract_extreme8_raw_values(selector_ctx)
            ext_feat = build_extreme8_features(ext_raw)
            context["extreme_raw"] = ext_raw
            context["extreme_feat"] = ext_feat
        except Exception:
            context["selector_ctx"] = None
            context["extreme_raw"] = None
            context["extreme_feat"] = None
    else:
        context["selector_ctx"] = None
        context["extreme_raw"] = None
        context["extreme_feat"] = None

    DECISION_CONTEXT_CACHE[key] = context
    return context


def _trajectory_shape(anchor_values: np.ndarray) -> str:
    arr = np.asarray(anchor_values, dtype=np.float64)
    if arr.size == 0:
        return "unknown"
    span = float(np.max(arr) - np.min(arr))
    delta = float(arr[-1] - arr[0])
    if span < 1e-8:
        return "平稳"
    if delta > 0.20 * span:
        return "上升"
    if delta < -0.20 * span:
        return "下降"
    return "平稳"


def _score_with_route(route: dict[str, Any], x_raw: np.ndarray, feature_to_idx: dict[str, int]) -> np.ndarray:
    route_type = str(route.get("route_type", "baseline"))
    if route_type == "baseline":
        score_col = feature_to_idx[str(route["signal_name"])]
        return np.asarray(x_raw[:, score_col], dtype=np.float64)

    x_rank = _earlystop_rank_transform_matrix(x_raw)
    feat_indices = [int(v) for v in route["feature_indices"]]
    rep = str(route["representation"])
    x_rep = _earlystop_build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=feat_indices,
        representation=rep,
    )

    if route_type == "svd":
        return np.asarray(_earlystop_predict_svd_lr(route["model"], x_rep), dtype=np.float64)

    scorer = route.get("scorer")
    if scorer is not None and hasattr(scorer, "score_group"):
        return np.asarray(scorer.score_group(x_rep), dtype=np.float64)

    return np.zeros(x_raw.shape[0], dtype=np.float64)


def _build_signal_tensor(
    reader: CacheReader,
    run_ids: list[int],
    feature_names: list[str],
    required_features: Optional[set[str]] = None,
) -> np.ndarray:
    n = len(run_ids)
    n_pos = len(EARLY_STOP_POSITIONS)
    tensor = np.zeros((n, n_pos, len(feature_names)), dtype=np.float64)
    req = set(feature_names) if required_features is None else set(required_features)

    for row_i, sample_id in enumerate(run_ids):
        signal_map = extract_earlystop_signals_for_sample(
            reader,
            int(sample_id),
            required_features=req,
        )
        for feat_i, feat_name in enumerate(feature_names):
            vals = np.asarray(signal_map.get(feat_name, [0.0] * n_pos), dtype=np.float64)
            if vals.size >= n_pos:
                tensor[row_i, :, feat_i] = vals[:n_pos]
    return tensor


def _ensure_earlystop_payload(context: dict[str, Any]) -> Optional[dict[str, Any]]:
    if "earlystop_payload" in context:
        return context["earlystop_payload"]
    if not _HAS_DECISION_METHODS:
        context["earlystop_payload"] = None
        return None

    bundle = _load_bundle_cached(EARLYSTOP_SVD_MODEL, load_earlystop_svd_bundle)
    if bundle is None:
        context["earlystop_payload"] = None
        return None

    domain = context["domain"]
    domain_bundle = bundle["domains"].get(domain)
    if not domain_bundle:
        context["earlystop_payload"] = None
        return None

    routes = domain_bundle["routes"]
    feature_names = [str(v) for v in bundle["feature_names"]]
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    required: set[str] = set()
    for route in routes:
        if str(route.get("route_type", "")) == "baseline":
            required.add(str(route["signal_name"]))
        else:
            required.update(str(x) for x in route["feature_names"])

    tensor = _build_signal_tensor(
        context["reader"],
        context["run_ids"],
        feature_names,
        required_features=required,
    )

    slot_scores = np.zeros((context["n_runs"], len(EARLY_STOP_POSITIONS)), dtype=np.float64)
    positions = [float(p) for p in bundle.get("positions", list(EARLY_STOP_POSITIONS))]
    route_anchor_positions = []

    for pos_i, route in enumerate(routes):
        x_raw = tensor[:, pos_i, :]
        slot_scores[:, pos_i] = _score_with_route(route, x_raw, feature_to_idx)
        p = float(positions[pos_i])
        route_anchor_positions.append(float(route.get("training_position", OFFICIAL_SLOT_TO_ANCHOR.get(p, p))))

    anchor_scores = slot_scores[:, ANCHOR_SLOT_INDICES]
    trajectory_shapes = [_trajectory_shape(anchor_scores[i]) for i in range(context["n_runs"])]

    payload = {
        "bundle": bundle,
        "feature_names": feature_names,
        "feature_to_idx": feature_to_idx,
        "routes": routes,
        "tensor": tensor,
        "slot_scores": slot_scores,
        "anchor_scores": anchor_scores,
        "route_anchor_positions": route_anchor_positions,
        "trajectory_shapes": trajectory_shapes,
        "positions": positions,
    }
    context["earlystop_payload"] = payload
    return payload


def _ensure_bridge_payload(context: dict[str, Any]) -> Optional[dict[str, Any]]:
    if "bridge_payload" in context:
        return context["bridge_payload"]
    if not _HAS_DECISION_METHODS:
        context["bridge_payload"] = None
        return None

    bundle = _load_bundle_cached(BRIDGE_MODEL, load_earlystop_svm_bundle)
    if bundle is None:
        context["bridge_payload"] = None
        return None

    domain = context["domain"]
    domain_bundle = bundle["domains"].get(domain, {})
    route = domain_bundle.get("route")
    if route is None:
        context["bridge_payload"] = None
        return None

    feature_names = [str(v) for v in bundle["feature_names"]]
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    es = _ensure_earlystop_payload(context)
    if es is not None and es["feature_names"] == feature_names:
        tensor = es["tensor"]
    else:
        required: set[str] = set()
        if str(route.get("route_type", "")) == "baseline":
            required.add(str(route["signal_name"]))
        else:
            required.update(str(x) for x in route["feature_names"])
        tensor = _build_signal_tensor(
            context["reader"],
            context["run_ids"],
            feature_names,
            required_features=required,
        )

    pos_i = int(bundle.get("position_index", 9))
    pos_i = max(0, min(pos_i, tensor.shape[1] - 1))
    x_raw = tensor[:, pos_i, :]
    scores = _score_with_route(route, x_raw, feature_to_idx)

    payload = {
        "bundle": bundle,
        "feature_names": feature_names,
        "feature_to_idx": feature_to_idx,
        "route": route,
        "position_index": pos_i,
        "scores": np.asarray(scores, dtype=np.float64),
    }
    context["bridge_payload"] = payload
    return payload


def _build_ranking(scores: np.ndarray, run_infos: list[dict[str, Any]]) -> dict[str, Any]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(run_infos)
    if n == 0:
        return {"order": np.zeros((0,), dtype=np.int64), "ranks": np.zeros((0,), dtype=np.int64), "rows": []}

    run_idx = np.asarray([int(r["run_index"]) for r in run_infos], dtype=np.int64)
    order = np.lexsort((run_idx, -scores_arr))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1, dtype=np.int64)

    seed = sum(ord(ch) for ch in str(run_infos[0]["sample_id"]) + str(n))
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, 0.12, size=n)

    rows = []
    for local_i in range(n):
        r = run_infos[local_i]
        rank = int(ranks[local_i])
        pct = 1.0 if n <= 1 else float((n - rank) / float(n - 1))
        rows.append({
            "local_index": int(local_i),
            "sample_id": int(r["sample_id"]),
            "run_index": int(r["run_index"]),
            "is_correct": bool(r["is_correct"]),
            "score": float(scores_arr[local_i]),
            "rank": rank,
            "percentile": round(float(100.0 * pct), 2),
            "density_jitter": float(jitter[local_i]),
        })
    return {"order": order, "ranks": ranks, "rows": rows}


def _compute_feature_percentiles(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n <= 1:
        return np.ones(n, dtype=np.float64) * 100.0

    work = arr.copy()
    finite = np.isfinite(work)
    if not finite.any():
        return np.ones(n, dtype=np.float64) * 50.0
    fill = float(np.nanmedian(work[finite]))
    work = np.where(finite, work, fill)
    if not higher_is_better:
        work = -work
    order = np.argsort(work, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return 100.0 * ranks / float(max(1, n - 1))


def _feature_diff_rows(
    feature_specs: list[dict[str, Any]],
    feature_table: dict[str, np.ndarray],
    idx_a: int,
    idx_b: int,
) -> list[dict[str, Any]]:
    rows = []
    for spec in feature_specs:
        key = str(spec["key"])
        if key not in feature_table:
            continue
        vals = np.asarray(feature_table[key], dtype=np.float64)
        if idx_a >= len(vals) or idx_b >= len(vals):
            continue
        higher_is_better = bool(spec.get("higher_is_better", True))
        sign = 1.0 if higher_is_better else -1.0
        va = float(vals[idx_a])
        vb = float(vals[idx_b])
        rows.append({
            "key": key,
            "label": str(spec["label"]),
            "value_a": va,
            "value_b": vb,
            "delta": va - vb,
            "advantage": (va - vb) * sign,
            "higher_is_better": higher_is_better,
            "color": spec.get("color", METHOD_COLORS["inactive"]),
        })
    rows.sort(key=lambda x: abs(float(x["advantage"])), reverse=True)
    return rows


def _compute_method_result(method_id: str, context: dict[str, Any]) -> dict[str, Any]:
    n = context["n_runs"]
    if n == 0:
        return {
            "scores": np.zeros((0,), dtype=np.float64),
            "feature_specs": [],
            "feature_table": {},
            "extras": {"fallback_reason": "no_runs"},
        }

    selector_ctx = context.get("selector_ctx")
    domain = context["domain"]

    if method_id in {"slot100_verifier", "svd_slot100"}:
        es = _ensure_earlystop_payload(context)
        br = _ensure_bridge_payload(context)
        if es is None:
            return {
                "scores": np.zeros((n,), dtype=np.float64),
                "feature_specs": [],
                "feature_table": {},
                "extras": {"fallback_reason": "earlystop_payload_unavailable"},
            }

        slot_scores = np.asarray(es["slot_scores"], dtype=np.float64)
        use_source = "slot100"
        final_scores = slot_scores[:, 9].copy()
        if method_id == "svd_slot100" and domain == "coding" and br is not None:
            final_scores = np.asarray(br["scores"], dtype=np.float64)
            use_source = "svm_bridge_slot100"

        feature_table = {
            "slot10": slot_scores[:, 0],
            "slot40": slot_scores[:, 3],
            "slot70": slot_scores[:, 6],
            "slot100": slot_scores[:, 9],
        }
        feature_specs = [
            {"key": "slot10", "label": "Slot10 信号", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
            {"key": "slot40", "label": "Slot40 信号", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
            {"key": "slot70", "label": "Slot70 信号", "higher_is_better": True, "color": METHOD_COLORS["trajectory"]},
            {"key": "slot100", "label": "Slot100 Final Verifier", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
        ]
        if br is not None:
            feature_table["bridge_slot100"] = np.asarray(br["scores"], dtype=np.float64)
            feature_specs.append({
                "key": "bridge_slot100",
                "label": "SVM Bridge Final",
                "higher_is_better": True,
                "color": METHOD_COLORS["confidence"],
            })
        return {
            "scores": final_scores,
            "feature_specs": feature_specs,
            "feature_table": feature_table,
            "extras": {
                "slot_scores": slot_scores,
                "anchor_scores": np.asarray(es["anchor_scores"], dtype=np.float64),
                "route_anchor_positions": list(es["route_anchor_positions"]),
                "trajectory_shapes": list(es["trajectory_shapes"]),
                "positions": list(es["positions"]),
                "source": use_source,
                "bridge_available": br is not None,
            },
        }

    if method_id == "svm_bridge_lcb":
        br = _ensure_bridge_payload(context)
        if br is None:
            return {
                "scores": np.zeros((n,), dtype=np.float64),
                "feature_specs": [],
                "feature_table": {},
                "extras": {"fallback_reason": "bridge_payload_unavailable"},
            }
        scores = np.asarray(br["scores"], dtype=np.float64)
        return {
            "scores": scores,
            "feature_specs": [
                {
                    "key": "bridge_slot100",
                    "label": "SVM Bridge Final",
                    "higher_is_better": True,
                    "color": METHOD_COLORS["confidence"],
                }
            ],
            "feature_table": {"bridge_slot100": scores},
            "extras": {"route_type": str(br["route"].get("route_type", "unknown"))},
        }

    if method_id == "extreme8_reflection":
        feats = context.get("extreme_feat")
        raw = context.get("extreme_raw")
        if feats is None or raw is None:
            return {
                "scores": np.zeros((n,), dtype=np.float64),
                "feature_specs": [],
                "feature_table": {},
                "extras": {"fallback_reason": "extreme8_features_unavailable"},
            }
        scores = np.asarray(feats[:, 1] + feats[:, 2], dtype=np.float64)
        return {
            "scores": scores,
            "feature_specs": [
                {"key": "dc_z", "label": "dc_z", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
                {"key": "dc_r", "label": "dc_r", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
                {"key": "reflection_count_r", "label": "reflection_count_r", "higher_is_better": True, "color": METHOD_COLORS["trajectory"]},
            ],
            "feature_table": {
                "dc_z": np.asarray(feats[:, 0], dtype=np.float64),
                "dc_r": np.asarray(feats[:, 1], dtype=np.float64),
                "reflection_count_r": np.asarray(feats[:, 2], dtype=np.float64),
                "reflection_count": np.asarray(raw["reflection_count"], dtype=np.float64),
            },
            "extras": {},
        }

    if method_id == "code_v2" and selector_ctx is not None:
        scores, rank_feat, raw = compute_code_v2_primary_scores(
            selector_ctx,
            weights=DEFAULT_CODE_V2_WEIGHTS,
        )
        weights = dict(DEFAULT_CODE_V2_WEIGHTS)
        contributions = {
            "prefix_best_window_quality": rank_feat[:, 0] * weights["prefix_best_window_quality"],
            "head_tail_gap": rank_feat[:, 1] * weights["head_tail_gap"],
            "tail_variance": rank_feat[:, 2] * weights["tail_variance"],
            "post_reflection_recovery": rank_feat[:, 3] * weights["post_reflection_recovery"],
            "last_block_instability": rank_feat[:, 4] * weights["last_block_instability"],
        }
        return {
            "scores": np.asarray(scores, dtype=np.float64),
            "feature_specs": [
                {"key": "prefix_best_window_quality", "label": "Prefix Window Quality", "higher_is_better": False, "color": METHOD_COLORS["confidence"]},
                {"key": "head_tail_gap", "label": "Head-Tail Gap", "higher_is_better": True, "color": METHOD_COLORS["trajectory"]},
                {"key": "tail_variance", "label": "Tail Variance", "higher_is_better": False, "color": METHOD_COLORS["instability"]},
                {"key": "post_reflection_recovery", "label": "Post-Reflection Recovery", "higher_is_better": True, "color": METHOD_COLORS["trajectory"]},
                {"key": "last_block_instability", "label": "Last-Block Instability", "higher_is_better": False, "color": METHOD_COLORS["instability"]},
            ],
            "feature_table": {
                "prefix_best_window_quality": np.asarray(raw["prefix_best_window_quality"], dtype=np.float64),
                "head_tail_gap": np.asarray(raw["head_tail_gap"], dtype=np.float64),
                "tail_variance": np.asarray(raw["tail_variance"], dtype=np.float64),
                "post_reflection_recovery": np.asarray(raw["post_reflection_recovery"], dtype=np.float64),
                "last_block_instability": np.asarray(raw["last_block_instability"], dtype=np.float64),
            },
            "extras": {
                "weights": weights,
                "rank_features": np.asarray(rank_feat, dtype=np.float64),
                "contributions": contributions,
                "blind_shapes": _load_code_v2_artifact().get("blind_shapes", {}),
            },
        }

    if method_id == "code_v2" and selector_ctx is None:
        return {
            "scores": np.zeros((n,), dtype=np.float64),
            "feature_specs": [],
            "feature_table": {},
            "extras": {"fallback_reason": "selector_ctx_unavailable"},
        }

    if method_id in {"science_baseline_v1", "science_hybrid_round3", "gpqa_pairwise_round1"} and selector_ctx is not None:
        baseline_scores, _, sci_raw = compute_science_dynamic_primary_scores(selector_ctx)
        scorer = _load_pairwise_scorer()
        include_margin = bool(getattr(scorer, "include_margin", False)) if scorer is not None else False
        include_dominance = bool(getattr(scorer, "include_dominance", False)) if scorer is not None else False

        gpqa_raw = extract_gpqa_pairwise_raw(selector_ctx)
        X = build_gpqa_pairwise_features_configurable(
            gpqa_raw,
            include_margin=include_margin,
            include_dominance=include_dominance,
        )

        if scorer is None:
            pairwise_scores = np.zeros((n,), dtype=np.float64)
            prob_matrix = np.full((n, n), 0.5, dtype=np.float64)
        else:
            prob_matrix = compute_pairwise_probability_matrix(scorer, X)
            pairwise_scores = np.mean(prob_matrix - np.eye(n), axis=1)

        if method_id == "science_baseline_v1":
            return {
                "scores": np.asarray(baseline_scores, dtype=np.float64),
                "feature_specs": [
                    {"key": "recency_conf_mean", "label": "Recency Confidence", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
                    {"key": "prefix_conf_mean", "label": "Prefix Confidence", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
                    {"key": "late_recovery", "label": "Late Recovery", "higher_is_better": True, "color": METHOD_COLORS["trajectory"]},
                ],
                "feature_table": {
                    "recency_conf_mean": np.asarray(sci_raw["recency_conf_mean"], dtype=np.float64),
                    "prefix_conf_mean": np.asarray(sci_raw["prefix_conf_mean"], dtype=np.float64),
                    "late_recovery": np.asarray(sci_raw["late_recovery"], dtype=np.float64),
                },
                "extras": {"baseline_scores": np.asarray(baseline_scores, dtype=np.float64)},
            }

        if method_id == "gpqa_pairwise_round1":
            return {
                "scores": np.asarray(pairwise_scores, dtype=np.float64),
                "feature_specs": [
                    {"key": "dc_z", "label": "dc_z", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
                    {"key": "dc_r", "label": "dc_r", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
                    {"key": "reflection_count_r", "label": "reflection_count_r", "higher_is_better": True, "color": METHOD_COLORS["trajectory"]},
                ],
                "feature_table": {
                    "dc_z": np.asarray(X[:, 0], dtype=np.float64),
                    "dc_r": np.asarray(X[:, 1], dtype=np.float64),
                    "reflection_count_r": np.asarray(X[:, 2], dtype=np.float64),
                    "pairwise_score": np.asarray(pairwise_scores, dtype=np.float64),
                },
                "extras": {"prob_matrix": np.asarray(prob_matrix, dtype=np.float64)},
            }

        art = _load_science_hybrid_artifact()
        cfg_raw = (((art.get("selected_candidate") or {}).get("config")) or {})
        cfg = ScienceHybridConfig(
            family=str(cfg_raw.get("family", "shortlist_blend")),
            backend=str(cfg_raw.get("backend", "win_count")),
            tau=float(cfg_raw.get("tau", 0.0)),
            k=int(cfg_raw.get("k", 2)),
            alpha=float(cfg_raw.get("alpha", 0.25)),
            m=float(cfg_raw.get("m", 0.0)),
            temperature=float(cfg_raw.get("temperature", 1.0)),
        ).validate()

        decision = compute_science_hybrid_decision(
            baseline_scores=np.asarray(baseline_scores, dtype=np.float64),
            pairwise_prob_matrix=np.asarray(prob_matrix, dtype=np.float64),
            D=np.asarray(context["D"], dtype=np.float64),
            run_ids=[int(v) for v in context["run_ids"]],
            baseline_gate_scores=np.asarray(gpqa_raw["recency_conf_mean"], dtype=np.float64),
            config=cfg,
        )
        return {
            "scores": np.asarray(decision.hybrid_scores, dtype=np.float64),
            "feature_specs": [
                {"key": "baseline_score", "label": "Baseline Score", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
                {"key": "pairwise_score", "label": "Pairwise Score", "higher_is_better": True, "color": METHOD_COLORS["trajectory"]},
                {"key": "hybrid_score", "label": "Hybrid Score", "higher_is_better": True, "color": METHOD_COLORS["confidence"]},
            ],
            "feature_table": {
                "baseline_score": np.asarray(decision.baseline_scores, dtype=np.float64),
                "pairwise_score": np.asarray(decision.pairwise_scores, dtype=np.float64),
                "hybrid_score": np.asarray(decision.hybrid_scores, dtype=np.float64),
            },
            "extras": {
                "decision": decision.to_dict(),
                "science_raw": {k: np.asarray(v, dtype=np.float64) for k, v in sci_raw.items()},
                "gpqa_raw": {k: np.asarray(v, dtype=np.float64) for k, v in gpqa_raw.items()},
                "pairwise_prob_matrix": np.asarray(prob_matrix, dtype=np.float64),
            },
        }

    if method_id in {"science_baseline_v1", "science_hybrid_round3", "gpqa_pairwise_round1"} and selector_ctx is None:
        return {
            "scores": np.zeros((n,), dtype=np.float64),
            "feature_specs": [],
            "feature_table": {},
            "extras": {"fallback_reason": "selector_ctx_unavailable"},
        }

    # Fallback: stable but non-informative
    return {
        "scores": np.zeros((n,), dtype=np.float64),
        "feature_specs": [],
        "feature_table": {},
        "extras": {"fallback_reason": "unknown_method_or_unavailable"},
    }


def _build_method_diagnostics(
    context: dict[str, Any],
    method_id: str,
    method_result: dict[str, Any],
    applicable: bool,
) -> dict[str, Any]:
    warnings: list[str] = []
    blockers: list[str] = []
    notes: list[str] = []

    n_runs = int(context.get("n_runs", 0))
    reader: CacheReader = context.get("reader")
    has_rows_bank = bool(
        reader is not None
        and reader.rows_sample_row_ptr is not None
        and reader.rows_row_ptr is not None
        and reader.rows_keys is not None
    )

    has_token_bank = False
    run_ids = context.get("run_ids", [])
    if reader is not None and run_ids:
        try:
            tv = reader.get_token_view(int(run_ids[0]))
            has_token_bank = bool(tv is not None and tv.token_ids is not None and len(tv.token_ids) > 0)
        except Exception:
            has_token_bank = False

    artifacts = {
        "earlystop_svd_model": bool(EARLYSTOP_SVD_MODEL.exists()),
        "bridge_model": bool(BRIDGE_MODEL.exists()),
        "gpqa_pairwise_model": bool(GPQA_PAIRWISE_MODEL.exists()),
        "code_v2_metrics": bool(CODE_V2_METRICS_PATH.exists()),
        "science_hybrid_result": bool(_latest_science_hybrid_path() is not None),
    }

    if n_runs <= 0:
        blockers.append("当前 problem 没有可用 run。")
    if not applicable:
        warnings.append("当前方法与该任务域不匹配，处于诊断模式。")

    if method_id in {"svd_slot100", "slot100_verifier"} and not artifacts["earlystop_svd_model"]:
        blockers.append("缺少 early-stop SVD 模型文件。")

    if method_id in {"svd_slot100", "svm_bridge_lcb"} and context.get("domain") == "coding":
        if not artifacts["bridge_model"]:
            blockers.append("Coding 域需要 SVM bridge 模型，但当前文件缺失。")

    if method_id in {"code_v2", "science_hybrid_round3", "science_baseline_v1", "gpqa_pairwise_round1"} and not has_rows_bank:
        warnings.append("rows bank 不可用，部分 trajectory/reflection 特征会降级。")

    if method_id == "science_hybrid_round3":
        if not artifacts["science_hybrid_result"]:
            warnings.append("science_hybrid_round3 结果产物缺失，使用在线重算路径。")
        if not artifacts["gpqa_pairwise_model"]:
            warnings.append("pairwise 模型缺失，science hybrid 将退化为 baseline。")

    if method_id == "code_v2":
        art = _load_code_v2_artifact()
        blind = art.get("blind_shapes", {}) if isinstance(art, dict) else {}
        cache_key = str(context.get("cache_key", ""))
        if cache_key and cache_key not in blind:
            notes.append(f"未找到 {cache_key} 的 blind-shape 诊断（例如 DS-R1/lcb_v5）。")

    fallback_reason = str(method_result.get("extras", {}).get("fallback_reason", "")).strip()
    if fallback_reason:
        warnings.append(f"方法已降级：{fallback_reason}")

    scores = np.asarray(method_result.get("scores", []), dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size > 1 and float(np.std(finite)) < 1e-10:
        warnings.append("方法分数几乎无区分度（flat scores）。")

    severity = "ok"
    if blockers:
        severity = "blocked"
    elif warnings:
        severity = "degraded"

    return {
        "severity": severity,
        "blockers": blockers,
        "warnings": warnings,
        "notes": notes,
        "has_rows_bank": has_rows_bank,
        "has_token_bank": has_token_bank,
        "artifacts": artifacts,
    }


def _why_selected(method_id: str, method_result: dict[str, Any], top1_idx: int, top2_idx: int, domain: str) -> str:
    specs = method_result.get("feature_specs", [])
    table = method_result.get("feature_table", {})
    diff_rows = _feature_diff_rows(specs, table, top1_idx, top2_idx)
    good = [r for r in diff_rows if float(r["advantage"]) > 0][:2]

    if method_id == "science_hybrid_round3":
        decision = method_result.get("extras", {}).get("decision", {})
        if decision.get("triggered", False):
            return (
                f"shortlist/rerank 已触发，top1 在 hybrid 排序中领先；"
                f"配置为 family={decision.get('config', {}).get('family', 'shortlist_blend')}。"
            )
        return "science hybrid 未触发重排，沿用 baseline 顶槽候选。"

    if method_id in {"slot100_verifier", "svd_slot100"}:
        source = str(method_result.get("extras", {}).get("source", "slot100"))
        if method_id == "svd_slot100" and source == "svm_bridge_slot100":
            return "该题属于 Coding 域，主线切到 SVM bridge final verifier，top1 在 final 分数上领先。"
        return "selected because slot100 final verifier dominates，且 anchor 轨迹更稳。"

    if method_id == "code_v2":
        if len(good) >= 2:
            return f"selected because {good[0]['label']} 更优，且 {good[1]['label']} 更稳。"
        if good:
            return f"selected because {good[0]['label']} 优势明显。"
        return "code_v2 分数组合最优（prefix 与 recovery 贡献领先）。"

    if method_id == "extreme8_reflection":
        return "selected because dc_r 与 reflection_count_r 在组内同时靠前。"

    if method_id == "svm_bridge_lcb":
        return "selected because bridge verifier 在 final-slot 分数领先。"

    if method_id == "science_baseline_v1":
        return "selected because recency confidence baseline score is highest。"

    if method_id == "gpqa_pairwise_round1":
        return "selected because pairwise 胜率均值最高。"

    if good:
        return f"selected because {good[0]['label']} 与 {good[-1]['label']} 更占优。"
    return f"{domain} 域下该方法分数最高。"


def _build_method_payload(cache: str, problem_id: str, method_id: str) -> dict[str, Any]:
    cache_key = (cache, str(problem_id), method_id)
    if cache_key in METHOD_RESULT_CACHE:
        return METHOD_RESULT_CACHE[cache_key]

    context = _build_problem_context(cache, str(problem_id))
    method_meta = _method_def(method_id)
    result = _compute_method_result(method_id, context)

    ranking = _build_ranking(result["scores"], context["run_infos"])
    order = ranking["order"]
    rows = ranking["rows"]
    n = len(rows)

    top_indices = order[:min(3, n)].tolist() if n > 0 else []
    top_runs = []
    for rank_pos, local_idx in enumerate(top_indices):
        row = rows[int(local_idx)]
        next_score = None
        if rank_pos + 1 < len(top_indices):
            next_idx = int(top_indices[rank_pos + 1])
            next_score = float(rows[next_idx]["score"])
        margin = None if next_score is None else float(row["score"] - next_score)
        top_runs.append({
            **row,
            "rank": int(rank_pos + 1),
            "correctness_mark": "✓" if row["is_correct"] else "✗",
            "margin_vs_next": margin,
        })

    top1_idx = int(top_indices[0]) if top_indices else 0
    top2_idx = int(top_indices[1]) if len(top_indices) > 1 else top1_idx
    compare_top12 = _feature_diff_rows(result.get("feature_specs", []), result.get("feature_table", {}), top1_idx, top2_idx)

    top1_vs_median = []
    for spec in result.get("feature_specs", []):
        key = str(spec["key"])
        vals = np.asarray(result.get("feature_table", {}).get(key, []), dtype=np.float64)
        if vals.size == 0 or top1_idx >= vals.size:
            continue
        higher = bool(spec.get("higher_is_better", True))
        sign = 1.0 if higher else -1.0
        finite = vals[np.isfinite(vals)]
        med = float(np.median(finite)) if finite.size > 0 else 0.0
        v1 = float(vals[top1_idx])
        top1_vs_median.append({
            "key": key,
            "label": str(spec["label"]),
            "value_a": v1,
            "value_b": med,
            "delta": v1 - med,
            "advantage": (v1 - med) * sign,
            "higher_is_better": higher,
            "color": spec.get("color", METHOD_COLORS["inactive"]),
        })
    top1_vs_median.sort(key=lambda x: abs(float(x["advantage"])), reverse=True)

    applicable = context["domain"] in set(method_meta.get("applies_to", []))
    diagnostics = _build_method_diagnostics(context, method_id, result, bool(applicable))
    why = _why_selected(method_id, result, top1_idx, top2_idx, context["domain"]) if n > 0 else "该题无可用 run。"
    if not applicable:
        why = f"[域外诊断模式] {why}"

    scores = np.asarray(result["scores"], dtype=np.float64)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size > 0:
        stats = {
            "min": float(np.min(finite_scores)),
            "max": float(np.max(finite_scores)),
            "mean": float(np.mean(finite_scores)),
            "median": float(np.median(finite_scores)),
            "std": float(np.std(finite_scores)),
        }
    else:
        stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}

    payload = {
        "success": True,
        "problem_id": str(problem_id),
        "method_id": method_id,
        "method_label": method_meta["label"],
        "method_family": method_meta["family"],
        "method_description": method_meta["description"],
        "domain": context["domain"],
        "dataset_name": context["dataset_name"],
        "cache_key": context["cache_key"],
        "applicable": bool(applicable),
        "top_runs": top_runs,
        "selected_sample_id": int(top_runs[0]["sample_id"]) if top_runs else None,
        "selected_run_index": int(top_runs[0]["run_index"]) if top_runs else None,
        "why_selected": why,
        "group_context": {
            "runs": rows,
            "n_runs": int(n),
            "head_count": max(1, int(math.ceil(0.10 * max(1, n)))) if n > 0 else 0,
        },
        "compare_bars": {
            "top1_vs_top2": compare_top12[:8],
            "top1_vs_median": top1_vs_median[:8],
        },
        "score_stats": stats,
        "diagnostics": diagnostics,
        "render_ready": bool(n > 0 and diagnostics.get("severity") != "blocked"),
        "failure_mode": diagnostics.get("severity") if diagnostics.get("severity") != "ok" else None,
    }

    METHOD_RESULT_CACHE[cache_key] = payload
    METHOD_LENS_CACHE[cache_key] = {
        "method_id": method_id,
        "feature_specs": result.get("feature_specs", []),
        "feature_table": result.get("feature_table", {}),
        "extras": result.get("extras", {}),
        "top_indices": top_indices,
        "domain": context["domain"],
        "cache_key": context["cache_key"],
    }
    RUN_COMPARE_CACHE[cache_key] = {
        "feature_specs": result.get("feature_specs", []),
        "feature_table": result.get("feature_table", {}),
        "rows": rows,
        "top_indices": top_indices,
        "extras": result.get("extras", {}),
    }
    return payload


def _build_method_lens_payload(cache: str, problem_id: str, method_id: str) -> dict[str, Any]:
    key = (cache, str(problem_id), method_id)
    method_payload = _build_method_payload(cache, str(problem_id), method_id)
    diagnostics = method_payload.get("diagnostics", {})

    if key not in METHOD_LENS_CACHE:
        _build_method_payload(cache, str(problem_id), method_id)
    base = METHOD_LENS_CACHE.get(key, {})
    if not base:
        return {
            "success": False,
            "error": "method lens unavailable",
            "diagnostics": diagnostics,
        }

    top_indices = base.get("top_indices", [])
    top1 = int(top_indices[0]) if top_indices else 0
    top2 = int(top_indices[1]) if len(top_indices) > 1 else top1
    feature_specs = base.get("feature_specs", [])
    feature_table = {k: np.asarray(v, dtype=np.float64) for k, v in base.get("feature_table", {}).items()}
    extras = base.get("extras", {})

    def _attach(payload: dict[str, Any]) -> dict[str, Any]:
        payload["method_id"] = method_id
        payload["domain"] = method_payload.get("domain")
        payload["diagnostics"] = diagnostics
        return payload

    lens_type = "generic"
    if method_id in {"slot100_verifier", "svd_slot100", "svm_bridge_lcb"}:
        lens_type = "slot100_verifier"
        payload = {
            "success": True,
            "lens_type": lens_type,
            "positions": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "slot_scores_top1": _to_py(np.asarray(extras.get("slot_scores", np.zeros((0, 10))))[top1].tolist() if extras.get("slot_scores") is not None and len(np.asarray(extras.get("slot_scores"))) > top1 else []),
            "slot_scores_top2": _to_py(np.asarray(extras.get("slot_scores", np.zeros((0, 10))))[top2].tolist() if extras.get("slot_scores") is not None and len(np.asarray(extras.get("slot_scores"))) > top2 else []),
            "route_anchor_positions": _to_py(extras.get("route_anchor_positions", [])),
            "trajectory_shapes": _to_py(extras.get("trajectory_shapes", [])),
            "source": extras.get("source", "slot100"),
            "anchor_note": "10/20/30 复用 10% route；40/50/60 复用 40%；70/80/90 复用 70%；100 为 final slot。",
            "explanation": "使用 early-stop 4-anchor（10/40/70/100）对比 Top1 与 Top2 的轨迹强弱。",
        }
        return _attach(payload)

    if method_id == "code_v2":
        lens_type = "code_v2"
        features = []
        contribs = extras.get("contributions", {})
        for spec in feature_specs:
            key_name = str(spec["key"])
            vals = np.asarray(feature_table.get(key_name, []), dtype=np.float64)
            if vals.size == 0:
                continue
            pcts = _compute_feature_percentiles(vals, bool(spec.get("higher_is_better", True)))
            cvals = np.asarray(contribs.get(key_name, np.zeros_like(vals)), dtype=np.float64)
            features.append({
                "key": key_name,
                "label": str(spec["label"]),
                "higher_is_better": bool(spec.get("higher_is_better", True)),
                "top1_value": float(vals[top1]),
                "top2_value": float(vals[top2]),
                "top1_percentile": float(pcts[top1]),
                "top2_percentile": float(pcts[top2]),
                "top1_contribution": float(cvals[top1]) if cvals.size > top1 else 0.0,
                "top2_contribution": float(cvals[top2]) if cvals.size > top2 else 0.0,
                "delta": float(vals[top1] - vals[top2]),
                "color": spec.get("color", METHOD_COLORS["inactive"]),
            })
        features.sort(key=lambda r: abs(float(r["top1_contribution"] - r["top2_contribution"])), reverse=True)
        blind = extras.get("blind_shapes", {})
        cache_key = _cache_key_from_cache(cache)
        return {
            "success": True,
            "lens_type": lens_type,
            "weights": _to_py(extras.get("weights", DEFAULT_CODE_V2_WEIGHTS)),
            "features": features,
            "push_up": [f for f in features if f["top1_contribution"] >= f["top2_contribution"]][:3],
            "push_down": [f for f in features if f["top1_contribution"] < f["top2_contribution"]][:3],
            "cache_blind_shape": _to_py(blind.get(cache_key, {})),
            "explanation": "对比 5 个 coding 特征在 Top1/Top2 的贡献与组内分位。",
        }
        return _attach(payload)

    if method_id == "science_hybrid_round3":
        lens_type = "science_hybrid_round3"
        decision = extras.get("decision", {})
        payload = {
            "success": True,
            "lens_type": lens_type,
            "decision": _to_py(decision),
            "triggered": bool(decision.get("triggered", False)),
            "overridden": bool(decision.get("overridden", False)),
            "shortlist_indices": _to_py(decision.get("shortlist_indices", [])),
            "baseline_gap": float(decision.get("baseline_gap", 0.0)),
            "pairwise_margin_vs_baseline": float(decision.get("pairwise_margin_vs_baseline", 0.0)),
            "gate_reason": (
                "shortlist_blend 触发：先按 baseline shortlist，再用 pairwise 融合 rerank。"
                if bool(decision.get("triggered", False))
                else "本题未触发 rerank，沿用 baseline 排序。"
            ),
            "explanation": "展示 baseline → pairwise → hybrid 的 shortlist/rerank 决策链。",
        }
        return _attach(payload)

    if method_id in {"science_baseline_v1", "gpqa_pairwise_round1"}:
        payload = {
            "success": True,
            "lens_type": method_id,
            "features": _to_py(_feature_diff_rows(feature_specs, feature_table, top1, top2)),
            "explanation": "展示 science 基线/GPQA pairwise 的 Top1 vs Top2 特征差异。",
        }
        return _attach(payload)

    if method_id == "extreme8_reflection":
        payload = {
            "success": True,
            "lens_type": "reflection",
            "features": _to_py(_feature_diff_rows(feature_specs, feature_table, top1, top2)),
            "explanation": "突出 dc_z / dc_r / reflection_count_r 三个 reflection 关键量。",
        }
        return _attach(payload)

    payload = {
        "success": True,
        "lens_type": lens_type,
        "features": _to_py(_feature_diff_rows(feature_specs, feature_table, top1, top2)),
        "explanation": "通用方法透镜：展示 Top1 与 Top2 的关键特征差异。",
    }
    return _attach(payload)


def _sample_brief(cache: str, sample_id: int) -> dict[str, Any]:
    meta = _get_meta(cache)
    if sample_id < 0 or sample_id >= len(meta.get("samples", [])):
        return {}
    s = meta["samples"][sample_id]
    problem_id = str(s.get("problem_id"))
    run_index = int(s.get("run_index", sample_id))

    correctness = False
    report = _get_eval_report(cache)
    if "results" in report:
        for r in report["results"]:
            if str(r.get("problem_id")) != problem_id:
                continue
            for run in r.get("runs", []):
                if int(run.get("run_index", -1)) == run_index:
                    correctness = bool(run.get("is_correct", False))
                    break
            break
    return {
        "sample_id": int(sample_id),
        "problem_id": problem_id,
        "run_index": run_index,
        "is_correct": correctness,
    }


def _build_token_evidence(cache: str, sample_id: int, compare_id: Optional[int], mode: str) -> dict[str, Any]:
    key = (cache, int(sample_id), None if compare_id is None else int(compare_id), mode)
    if key in TOKEN_EVIDENCE_CACHE:
        return TOKEN_EVIDENCE_CACHE[key]

    def _run_payload(sid: int) -> dict[str, Any]:
        slices, tv = _get_slices(int(sid), mode, cache)
        avgs = _slice_averages(tv, slices) if slices else {
            "entropy": [], "conf": [], "gini": [], "selfcert": [], "logprob": []
        }
        return {
            "run": _sample_brief(cache, int(sid)),
            "num_slices": len(slices),
            "slices": slices,
            "metrics": avgs,
        }

    primary = _run_payload(sample_id)
    compare = _run_payload(compare_id) if compare_id is not None else None

    highlights = {}
    for metric in ("conf", "entropy", "gini", "selfcert", "logprob"):
        vals = np.asarray(primary["metrics"].get(metric, []), dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, np.nan)
        if vals.size == 0:
            highlights[metric] = []
            continue

        if compare is not None:
            vals2 = np.asarray(compare["metrics"].get(metric, []), dtype=np.float64)
            m = min(vals.size, vals2.size)
            if m > 0:
                score = np.abs(vals[:m] - vals2[:m])
            else:
                score = np.abs(vals - np.nanmedian(vals))
        else:
            score = np.abs(vals - np.nanmedian(vals))

        if score.size == 0:
            highlights[metric] = []
            continue
        score = np.where(np.isfinite(score), score, -np.inf)
        k = min(4, int(np.sum(np.isfinite(score) & (score > -np.inf))))
        if k <= 0:
            highlights[metric] = []
            continue
        idx = np.argsort(-score)[:k]
        highlights[metric] = [int(v) for v in idx.tolist()]

    out = {
        "success": True,
        "mode": mode,
        "primary": primary,
        "compare": compare,
        "highlights": highlights,
    }
    TOKEN_EVIDENCE_CACHE[key] = out
    return out


def _build_run_compare_payload(
    cache: str,
    problem_id: str,
    method_id: str,
    left_sample_id: Optional[int] = None,
    right_sample_id: Optional[int] = None,
) -> tuple[dict[str, Any], int]:
    cache_key = (cache, str(problem_id), method_id)
    if cache_key not in RUN_COMPARE_CACHE:
        _build_method_payload(cache, str(problem_id), method_id)
    base = RUN_COMPARE_CACHE.get(cache_key, {})
    if not base:
        return {"success": False, "error": "no compare data"}, 404

    rows = base.get("rows", [])
    if not rows:
        return {"success": False, "error": "no rows"}, 404

    sample_to_local = {int(r["sample_id"]): int(r["local_index"]) for r in rows}
    top_indices = base.get("top_indices", [])
    default_left = int(rows[int(top_indices[0])]["sample_id"]) if top_indices else int(rows[0]["sample_id"])
    default_right = int(rows[int(top_indices[1])]["sample_id"]) if len(top_indices) > 1 else default_left

    left_sample = int(default_left if left_sample_id is None else left_sample_id)
    right_sample = int(default_right if right_sample_id is None else right_sample_id)

    if left_sample not in sample_to_local or right_sample not in sample_to_local:
        return {"success": False, "error": "sample not in problem"}, 400

    left_idx = sample_to_local[left_sample]
    right_idx = sample_to_local[right_sample]
    diffs = _feature_diff_rows(
        base.get("feature_specs", []),
        {k: np.asarray(v, dtype=np.float64) for k, v in base.get("feature_table", {}).items()},
        left_idx,
        right_idx,
    )

    method_payload = _build_method_payload(cache, str(problem_id), method_id)
    out = {
        "success": True,
        "problem_id": str(problem_id),
        "method_id": method_id,
        "left": rows[left_idx],
        "right": rows[right_idx],
        "feature_diffs": diffs[:10],
        "extras": _to_py(base.get("extras", {})),
        "diagnostics": method_payload.get("diagnostics", {}),
    }
    return out, 200


@app.route("/api/method_catalog")
def api_method_catalog():
    return jsonify({
        "primary_method_id": "svd_slot100",
        "methods": METHOD_CATALOG,
    })


@app.route("/api/health_viewer")
def api_health_viewer():
    _ensure_runtime_loaded()
    plotly_vendor = _plotly_vendor_path()
    out = {
        "success": True,
        "datasets": len(DATASETS),
        "tokenizer_loaded": TOKENIZER is not None,
        "boundary_ids_ready": BOUNDARY_IDS is not None,
        "plotly_vendor_js": str(plotly_vendor) if plotly_vendor else None,
        "artifacts": {
            "earlystop_svd_model": bool(EARLYSTOP_SVD_MODEL.exists()),
            "bridge_model": bool(BRIDGE_MODEL.exists()),
            "gpqa_pairwise_model": bool(GPQA_PAIRWISE_MODEL.exists()),
            "code_v2_metrics": bool(CODE_V2_METRICS_PATH.exists()),
            "science_hybrid_result": bool(_latest_science_hybrid_path() is not None),
        },
    }
    return jsonify(out)


@app.route("/api/method_scores/<problem_id>")
def api_method_scores(problem_id: str):
    cache = _require_cache()
    if not cache:
        return jsonify({"success": False, "error": "invalid cache"}), 400
    method_id = request.args.get("method", "svd_slot100").strip()
    payload = _build_method_payload(cache, str(problem_id), method_id)
    return jsonify(_to_py(payload))


@app.route("/api/method_lens/<problem_id>")
def api_method_lens(problem_id: str):
    cache = _require_cache()
    if not cache:
        return jsonify({"success": False, "error": "invalid cache"}), 400
    method_id = request.args.get("method", "svd_slot100").strip()
    payload = _build_method_lens_payload(cache, str(problem_id), method_id)
    return jsonify(_to_py(payload))


@app.route("/api/dashboard_bootstrap/<problem_id>")
def api_dashboard_bootstrap(problem_id: str):
    cache = _require_cache()
    if not cache:
        return jsonify({"success": False, "error": "invalid cache"}), 400

    method_id = request.args.get("method", "svd_slot100").strip()
    mode = request.args.get("mode", "fixed")

    scores = _build_method_payload(cache, str(problem_id), method_id)
    lens = _build_method_lens_payload(cache, str(problem_id), method_id)

    top_runs = scores.get("top_runs", [])
    left_sample = int(top_runs[0]["sample_id"]) if top_runs else None
    right_sample = int(top_runs[1]["sample_id"]) if len(top_runs) > 1 else left_sample

    compare_payload = {"success": False, "error": "no runs"}
    token_payload = {"success": False, "error": "no runs"}
    if left_sample is not None:
        compare_payload, _ = _build_run_compare_payload(
            cache=cache,
            problem_id=str(problem_id),
            method_id=method_id,
            left_sample_id=left_sample,
            right_sample_id=right_sample,
        )
        token_payload = _build_token_evidence(
            cache=cache,
            sample_id=left_sample,
            compare_id=right_sample,
            mode=mode,
        )

    out = {
        "success": True,
        "problem_id": str(problem_id),
        "method_id": method_id,
        "mode": mode,
        "scores": _to_py(scores),
        "lens": _to_py(lens),
        "run_compare": _to_py(compare_payload),
        "token_evidence": _to_py(token_payload),
        "render_ready": bool(scores.get("render_ready", False)),
        "diagnostics": _to_py(scores.get("diagnostics", {})),
    }
    return jsonify(out)


@app.route("/api/run_compare/<problem_id>")
def api_run_compare(problem_id: str):
    cache = _require_cache()
    if not cache:
        return jsonify({"success": False, "error": "invalid cache"}), 400

    method_id = request.args.get("method", "svd_slot100").strip()
    left_sample_raw = request.args.get("left_sample_id")
    right_sample_raw = request.args.get("right_sample_id")
    left_sample = int(left_sample_raw) if left_sample_raw is not None and left_sample_raw != "" else None
    right_sample = int(right_sample_raw) if right_sample_raw is not None and right_sample_raw != "" else None

    out, status = _build_run_compare_payload(
        cache=cache,
        problem_id=str(problem_id),
        method_id=method_id,
        left_sample_id=left_sample,
        right_sample_id=right_sample,
    )
    return jsonify(_to_py(out)), int(status)


@app.route("/api/token_evidence/<int:sample_id>")
def api_token_evidence(sample_id: int):
    cache = _require_cache()
    if not cache:
        return jsonify({"success": False, "error": "invalid cache"}), 400
    mode = request.args.get("mode", "fixed")
    compare_id_raw = request.args.get("compare_sample_id")
    compare_id = int(compare_id_raw) if compare_id_raw is not None and compare_id_raw != "" else None
    payload = _build_token_evidence(cache, int(sample_id), compare_id, mode)
    return jsonify(_to_py(payload))

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Scanning datasets...")
    _scan_datasets()
    print(f"Found {len(DATASETS)} datasets: {list(DATASETS.keys())}")

    print("Loading tokenizer...")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Tokenizer loaded.")

    BOUNDARY_IDS = _compute_boundary_ids(TOKENIZER)
    print(f"Boundary token IDs: {BOUNDARY_IDS.tolist()}")

    app.run(host="0.0.0.0", port=5002, debug=False)
