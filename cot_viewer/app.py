#!/usr/bin/env python3
"""Chain-of-Thought Viewer – browse decoded reasoning chains and per-token metrics."""

import json
import os
import sys

sys.path.insert(0, "/home/jovyan/work/NAD_Next")

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from nad.core.views.reader import Agg, CacheReader
from nad.io.loader import NadNextLoader
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

CACHE_BASE = Path("/home/jovyan/public-ro/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B")
MODEL_PATH = "/home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B"

DATASETS: dict[str, str] = {}      # name -> cache_dir path
LOADERS: dict[str, NadNextLoader] = {}
READERS: dict[str, CacheReader] = {}
EVAL_REPORTS: dict[str, dict] = {}
META_CACHE: dict[str, dict] = {}
TOKENIZER = None
BOUNDARY_IDS: np.ndarray = None    # token IDs for \n . ? : — set at startup


def _scan_datasets():
    """Find all datasets and pick the lexicographically last cache dir for each."""
    for ds_dir in sorted(CACHE_BASE.iterdir()):
        if not ds_dir.is_dir():
            continue
        cache_dirs = sorted(
            [d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("cache_neuron_")]
        )
        if cache_dirs:
            DATASETS[ds_dir.name] = str(cache_dirs[-1])


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


def _require_cache() -> str:
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
    return render_template("index.html")


@app.route("/api/datasets")
def api_datasets():
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
