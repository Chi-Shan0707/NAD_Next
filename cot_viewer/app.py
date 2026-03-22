#!/usr/bin/env python3
"""Chain-of-Thought Viewer – browse decoded reasoning chains and per-token metrics."""

import json
import os
import sys

sys.path.insert(0, "/home/jovyan/work/NAD_Next")

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from nad.core.views.reader import CacheReader
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


@app.route("/api/chain/<int:sample_id>")
def api_chain(sample_id: int):
    cache = _require_cache()
    if not cache:
        return jsonify({"error": "invalid cache"}), 400

    reader = _get_reader(cache)
    loader = _get_loader(cache)

    tv = reader.get_token_view(sample_id)
    if tv.token_ids is None or len(tv.token_ids) == 0:
        return jsonify({"num_slices": 0, "slices": []})

    token_ids = tv.token_ids

    # Compute slice boundaries from rows_token_row_ptr.
    # Row 0 is the prompt row (skipped by get_token_view), so response
    # slices start at row_lo+1.
    row_lo, row_hi = loader.get_row_range_for_sample(sample_id)
    rows_trp = reader.rows_token_row_ptr
    if rows_trp is not None:
        # base = start of response tokens in global array
        base = int(rows_trp[row_lo + 1])
        # Slice boundaries: rows row_lo+1 .. row_hi (inclusive)
        offsets = []
        for r in range(row_lo + 1, row_hi + 2):
            offsets.append(int(rows_trp[r]) - base)
        # Clamp to actual token count
        offsets = [min(o, len(token_ids)) for o in offsets]
    else:
        # Fallback: treat entire response as one slice
        offsets = [0, len(token_ids)]

    slices = []
    for i in range(len(offsets) - 1):
        tok_start = offsets[i]
        tok_end = offsets[i + 1]
        if tok_start >= len(token_ids):
            break
        ids = token_ids[tok_start:tok_end].tolist()
        text = TOKENIZER.decode(ids, skip_special_tokens=False)
        slices.append({
            "idx": i,
            "text": text,
            "tok_start": tok_start,
            "tok_end": tok_end,
        })

    return jsonify({"num_slices": len(slices), "slices": slices})


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

    # Recompute slice boundaries (same logic as /api/chain)
    row_lo, row_hi = loader.get_row_range_for_sample(sample_id)
    rows_trp = reader.rows_token_row_ptr
    if rows_trp is None:
        return jsonify({"error": "no row data"}), 404

    base = int(rows_trp[row_lo + 1])
    num_slices = row_hi - row_lo  # excluding prompt row
    if slice_idx < 0 or slice_idx >= num_slices:
        return jsonify({"error": "slice_idx out of range"}), 400

    row = row_lo + 1 + slice_idx
    tok_start = int(rows_trp[row]) - base
    tok_end = int(rows_trp[row + 1]) - base
    tok_end = min(tok_end, len(tv.token_ids))

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
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Scanning datasets...")
    _scan_datasets()
    print(f"Found {len(DATASETS)} datasets: {list(DATASETS.keys())}")

    print("Loading tokenizer...")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Tokenizer loaded.")

    app.run(host="0.0.0.0", port=5002, debug=False)
