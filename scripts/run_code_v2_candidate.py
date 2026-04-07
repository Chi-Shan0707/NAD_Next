#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.code_dynamic_impl import (
    DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    compute_code_dynamic_primary_scores_from_raw,
    order_code_dynamic_group_indices,
    prepare_code_dynamic_run_state,
)
from nad.core.selectors.code_v2_impl import (
    CODE_V2_WEIGHT_NAMES,
    DEFAULT_CODE_V2_WEIGHTS,
    compute_code_v2_primary_scores_from_raw,
)
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import build_problem_groups
from scripts.run_code_baseline_v1_phase2 import (
    BLIND_CODE_KEYS,
    CODE_CACHE_KEY,
    DEFAULT_VIEW,
    MetricAccumulator,
    _build_code_raw_from_precomputed,
    _compute_copeland_scores,
    _compute_derived_trace_stats,
    _fmt_pct,
    _load_entry_map,
    _precompute_run_summary,
    _problem_sort_key,
    _summarize_metrics_table,
)

REFLECTION_THRESHOLDS = [0.20, 0.25, 0.30, 0.35]
LOOKBACKS = [8, 16, 24]
PREFIX_FRACS = [0.10, 0.20, 0.30]
BASELINE_PARAMS = {
    "reflection_threshold": DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    "reflection_lookback_slices": DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    "prefix_fraction": 0.20,
}
CANDIDATE_PARAMS = {
    "reflection_threshold": DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    "reflection_lookback_slices": DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    "prefix_fraction": 0.30,
}


@dataclass
class CodeV2ProblemData:
    problem_id: str
    run_ids: list[int]
    labels: np.ndarray
    D: np.ndarray
    min_conf_scores: np.ndarray
    copeland_scores: np.ndarray
    run_summaries: list[dict[str, Any]]
    derived_rows: list[dict[str, float]]
    baseline_scores: np.ndarray


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _precompute_code_v2_raw(
    run_summaries: list[dict[str, Any]],
    derived_rows: list[dict[str, float]],
    *,
    reflection_threshold: float,
    reflection_lookback_slices: int,
    prefix_fraction: float,
) -> dict[str, np.ndarray]:
    n = len(run_summaries)
    raw = {
        "prefix_best_window_quality": np.full(n, np.nan, dtype=np.float64),
        "head_tail_gap": np.full(n, np.nan, dtype=np.float64),
        "tail_variance": np.full(n, np.nan, dtype=np.float64),
        "post_reflection_recovery": np.full(n, np.nan, dtype=np.float64),
        "last_block_instability": np.full(n, np.nan, dtype=np.float64),
    }
    for idx, summary in enumerate(run_summaries):
        refl = summary["reflection_lookup"][(float(reflection_threshold), int(reflection_lookback_slices))]
        raw["prefix_best_window_quality"][idx] = float(summary["prefix_best_by_fraction"][float(prefix_fraction)])
        raw["head_tail_gap"][idx] = float(summary["base_head_tail_gap"])
        raw["tail_variance"][idx] = float(summary["base_tail_variance"])
        raw["post_reflection_recovery"][idx] = float(refl["post_reflection_recovery"])
        raw["last_block_instability"][idx] = float(derived_rows[idx]["last_block_instability_score"])
    return raw


def _mean_tok_conf_from_state(run_state: dict[str, Any]) -> float:
    tok_conf = np.asarray(run_state.get("tok_conf", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if tok_conf.size == 0:
        return float("-inf")
    return float(-np.mean(tok_conf))


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in ("auroc", "hit@1", "pairwise", "selacc@10%", "n_problems", "n_samples", "top10_count")
    }


def _compact_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, **_compact_metrics(metrics)}


def _grid_row_name(thr: float, lb: int, pf: float) -> str:
    return f"grid__thr{thr:.2f}__lb{lb}__pf{pf:.2f}".replace(".", "p")


def _gate_code_v2(compare_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    baseline = compare_metrics["code_baseline_v1"]
    candidate = compare_metrics["code_baseline_v2_candidate"]
    failed: list[str] = []
    if float(candidate["selacc@10%"] or 0.0) <= float(baseline["selacc@10%"] or 0.0):
        failed.append(
            f"SelAcc@10 {_fmt_pct(candidate['selacc@10%'])} ≤ baseline {_fmt_pct(baseline['selacc@10%'])}"
        )
    if float(candidate["pairwise"] or 0.0) < 0.50:
        failed.append(f"Pairwise {_fmt_pct(candidate['pairwise'])} < 50.00%")
    if float(candidate["hit@1"] or 0.0) < float(baseline["hit@1"] or 0.0) - 0.01:
        failed.append(
            f"Hit@1 {_fmt_pct(candidate['hit@1'])} < guardrail {_fmt_pct(float(baseline['hit@1'] or 0.0) - 0.01)}"
        )
    return len(failed) == 0, failed


def _weights_name(weights: dict[str, float]) -> str:
    return (
        f"pfx{weights['prefix_best_window_quality']:.2f}_"
        f"gap{weights['head_tail_gap']:.2f}_"
        f"tail{weights['tail_variance']:.2f}_"
        f"rec{weights['post_reflection_recovery']:.2f}_"
        f"inst{weights['last_block_instability']:.2f}"
    ).replace(".", "p")


def _weight_search_space() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for head_tail_gap in (0.00, 0.06):
        for tail_variance in (0.08, 0.16):
            for post_reflection_recovery in (0.22, 0.28):
                for last_block_instability in (0.16, 0.24):
                    prefix = 1.0 - (
                        float(head_tail_gap)
                        + float(tail_variance)
                        + float(post_reflection_recovery)
                        + float(last_block_instability)
                    )
                    if prefix < 0.36 or prefix > 0.60:
                        continue
                    weights = {
                        "prefix_best_window_quality": round(float(prefix), 2),
                        "head_tail_gap": round(float(head_tail_gap), 2),
                        "tail_variance": round(float(tail_variance), 2),
                        "post_reflection_recovery": round(float(post_reflection_recovery), 2),
                        "last_block_instability": round(float(last_block_instability), 2),
                    }
                    if all(abs(weights[k] - float(DEFAULT_CODE_V2_WEIGHTS[k])) < 1e-9 for k in DEFAULT_CODE_V2_WEIGHTS):
                        continue
                    candidates.append({
                        "name": _weights_name(weights),
                        "weights": weights,
                    })
    candidates.sort(
        key=lambda row: (
            -float(row["weights"]["prefix_best_window_quality"]),
            float(row["weights"]["head_tail_gap"]),
            -float(row["weights"]["post_reflection_recovery"]),
            -float(row["weights"]["last_block_instability"]),
            float(row["weights"]["tail_variance"]),
            row["name"],
        )
    )
    return candidates


def _preload_problem_data(
    cache_root: Path,
    *,
    distance_threads: int,
    prefix_window_tokens: int,
) -> list[CodeV2ProblemData]:
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    correctness = _load_ground_truth(cache_root)
    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))

    problems: list[CodeV2ProblemData] = []
    for idx, (problem_id, run_ids) in enumerate(sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))):
        if idx % 20 == 0:
            print(f"[code-v2-preload] problem {idx + 1}/{len(groups)}", flush=True)
        run_ids = list(map(int, run_ids))
        labels = np.asarray([int(bool(correctness.get(int(run_id), False))) for run_id in run_ids], dtype=np.int32)
        run_states = []
        run_views = []
        for run_id in run_ids:
            tv = reader.get_token_view(int(run_id))
            run_states.append(prepare_code_dynamic_run_state(reader, int(run_id), token_view=tv))
            run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))
        D = engine.dense_matrix(run_views)
        min_conf_scores = np.asarray([_mean_tok_conf_from_state(state) for state in run_states], dtype=np.float64)
        copeland_scores = _compute_copeland_scores(D)
        run_summaries = [
            _precompute_run_summary(
                run_state,
                reflection_thresholds=REFLECTION_THRESHOLDS,
                reflection_lookbacks=LOOKBACKS,
                prefix_fractions=PREFIX_FRACS,
                prefix_window_tokens=int(prefix_window_tokens),
            )
            for run_state in run_states
        ]
        derived_rows = [
            _compute_derived_trace_stats(run_state, prefix_fraction=0.30)
            for run_state in run_states
        ]
        baseline_raw, _ = _build_code_raw_from_precomputed(
            run_summaries,
            reflection_threshold=float(BASELINE_PARAMS["reflection_threshold"]),
            reflection_lookback_slices=int(BASELINE_PARAMS["reflection_lookback_slices"]),
            prefix_fraction=float(BASELINE_PARAMS["prefix_fraction"]),
        )
        baseline_scores, _ = compute_code_dynamic_primary_scores_from_raw(baseline_raw)
        problems.append(
            CodeV2ProblemData(
                problem_id=str(problem_id),
                run_ids=run_ids,
                labels=labels,
                D=D,
                min_conf_scores=min_conf_scores,
                copeland_scores=copeland_scores,
                run_summaries=run_summaries,
                derived_rows=derived_rows,
                baseline_scores=np.asarray(baseline_scores, dtype=np.float64),
            )
        )
    return problems


def _evaluate_code_v2_preloaded(
    problems: list[CodeV2ProblemData],
    *,
    candidate_weights: dict[str, float],
) -> dict[str, Any]:
    compare_accs = {
        "code_baseline_v1": MetricAccumulator("code_baseline_v1", use_code_tiebreak=True),
        "code_baseline_v2_candidate": MetricAccumulator("code_baseline_v2_candidate", use_code_tiebreak=True),
        "min-confidence": MetricAccumulator("min-confidence", use_code_tiebreak=False),
        "tournament-copeland": MetricAccumulator("tournament-copeland", use_code_tiebreak=False),
    }
    loo_accs = {
        "code_baseline_v2_candidate": MetricAccumulator("code_baseline_v2_candidate", use_code_tiebreak=True),
        **{
            f"loo_without__{feat_name}": MetricAccumulator(f"loo_without__{feat_name}", use_code_tiebreak=True)
            for feat_name in CODE_V2_WEIGHT_NAMES
        },
    }
    grid_accs = {
        _grid_row_name(thr, lb, pf): MetricAccumulator(_grid_row_name(thr, lb, pf), use_code_tiebreak=True)
        for thr in REFLECTION_THRESHOLDS
        for lb in LOOKBACKS
        for pf in PREFIX_FRACS
    }

    for idx, prob in enumerate(problems):
        if idx % 20 == 0:
            print(f"[code-v2-eval] problem {idx + 1}/{len(problems)}", flush=True)
        compare_accs["code_baseline_v1"].add_problem(prob.problem_id, prob.run_ids, prob.baseline_scores, prob.labels, prob.D)
        compare_accs["min-confidence"].add_problem(prob.problem_id, prob.run_ids, prob.min_conf_scores, prob.labels, prob.D)
        compare_accs["tournament-copeland"].add_problem(prob.problem_id, prob.run_ids, prob.copeland_scores, prob.labels, prob.D)

        raw_default = _precompute_code_v2_raw(
            prob.run_summaries,
            prob.derived_rows,
            reflection_threshold=float(CANDIDATE_PARAMS["reflection_threshold"]),
            reflection_lookback_slices=int(CANDIDATE_PARAMS["reflection_lookback_slices"]),
            prefix_fraction=float(CANDIDATE_PARAMS["prefix_fraction"]),
        )
        candidate_scores, _ = compute_code_v2_primary_scores_from_raw(raw_default, weights=candidate_weights)
        compare_accs["code_baseline_v2_candidate"].add_problem(prob.problem_id, prob.run_ids, candidate_scores, prob.labels, prob.D)
        loo_accs["code_baseline_v2_candidate"].add_problem(prob.problem_id, prob.run_ids, candidate_scores, prob.labels, prob.D)
        for feat_name in CODE_V2_WEIGHT_NAMES:
            scores, _ = compute_code_v2_primary_scores_from_raw(
                raw_default,
                weights=candidate_weights,
                disabled_features=[feat_name],
            )
            loo_accs[f"loo_without__{feat_name}"].add_problem(prob.problem_id, prob.run_ids, scores, prob.labels, prob.D)

        for thr in REFLECTION_THRESHOLDS:
            for lb in LOOKBACKS:
                for pf in PREFIX_FRACS:
                    raw_grid = _precompute_code_v2_raw(
                        prob.run_summaries,
                        prob.derived_rows,
                        reflection_threshold=float(thr),
                        reflection_lookback_slices=int(lb),
                        prefix_fraction=float(pf),
                    )
                    scores, _ = compute_code_v2_primary_scores_from_raw(raw_grid, weights=candidate_weights)
                    grid_accs[_grid_row_name(thr, lb, pf)].add_problem(prob.problem_id, prob.run_ids, scores, prob.labels, prob.D)

    compare_metrics = {name: acc.finalize() for name, acc in compare_accs.items()}
    loo_metrics = {name: acc.finalize() for name, acc in loo_accs.items()}
    grid_metrics = {name: acc.finalize() for name, acc in grid_accs.items()}

    baseline_hit1 = float(compare_metrics["code_baseline_v1"]["hit@1"] or 0.0)
    guarded_grid_rows = []
    for name, metrics in grid_metrics.items():
        pairwise = metrics.get("pairwise")
        hit1 = metrics.get("hit@1")
        guard_ok = (
            pairwise is not None
            and float(pairwise) >= 0.50
            and hit1 is not None
            and float(hit1) >= baseline_hit1 - 0.01
        )
        guarded_grid_rows.append({
            "name": name,
            **_compact_metrics(metrics),
            "guard_ok": bool(guard_ok),
        })
    guarded_grid_rows.sort(
        key=lambda row: (
            not row["guard_ok"],
            -(row["selacc@10%"] or 0.0),
            -(row["pairwise"] or 0.0),
            -(row["hit@1"] or 0.0),
            row["name"],
        )
    )
    best_guarded = guarded_grid_rows[0] if guarded_grid_rows else None
    gate_passed, gate_failed = _gate_code_v2(compare_metrics)

    return {
        "compare_metrics": {name: _compact_metrics(metrics) for name, metrics in compare_metrics.items()},
        "loo_metrics": {name: _compact_metrics(metrics) for name, metrics in loo_metrics.items()},
        "guarded_grid_rows": guarded_grid_rows,
        "best_guarded": best_guarded,
        "gate": {
            "passed": gate_passed,
            "failed_thresholds": gate_failed,
        },
    }


def _evaluate_blind_shapes(
    blind_entry_map: dict[str, Any],
    *,
    candidate_weights: dict[str, float],
    distance_threads: int,
    prefix_window_tokens: int,
) -> dict[str, Any]:
    outputs = {}
    for cache_key in BLIND_CODE_KEYS:
        entry = blind_entry_map[cache_key]
        print(f"[code-v2-blind] {cache_key}", flush=True)
        cache_root = Path(entry.cache_root)
        meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
        groups = build_problem_groups(meta)
        reader = CacheReader(str(cache_root))
        engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
        selector_problem = {
            name: {}
            for name in ("code_baseline_v1", "code_baseline_v2_candidate", "min-confidence", "tournament-copeland")
        }
        sorted_groups = sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))
        for idx, (problem_id, run_ids) in enumerate(sorted_groups):
            if idx % 20 == 0:
                print(f"[code-v2-blind-eval] {cache_key} problem {idx + 1}/{len(sorted_groups)}", flush=True)
            run_ids = list(map(int, run_ids))
            run_states = []
            run_views = []
            for run_id in run_ids:
                tv = reader.get_token_view(int(run_id))
                run_states.append(prepare_code_dynamic_run_state(reader, int(run_id), token_view=tv))
                run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))
            D = engine.dense_matrix(run_views)
            min_conf_scores = np.asarray([_mean_tok_conf_from_state(state) for state in run_states], dtype=np.float64)
            copeland_scores = _compute_copeland_scores(D)
            run_summaries = [
                _precompute_run_summary(
                    run_state,
                    reflection_thresholds=[DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD],
                    reflection_lookbacks=[DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK],
                    prefix_fractions=[0.20, 0.30],
                    prefix_window_tokens=int(prefix_window_tokens),
                )
                for run_state in run_states
            ]
            derived_rows = [
                _compute_derived_trace_stats(run_state, prefix_fraction=0.30)
                for run_state in run_states
            ]
            baseline_raw, _ = _build_code_raw_from_precomputed(
                run_summaries,
                reflection_threshold=float(BASELINE_PARAMS["reflection_threshold"]),
                reflection_lookback_slices=int(BASELINE_PARAMS["reflection_lookback_slices"]),
                prefix_fraction=float(BASELINE_PARAMS["prefix_fraction"]),
            )
            baseline_scores, _ = compute_code_dynamic_primary_scores_from_raw(baseline_raw)
            candidate_raw = _precompute_code_v2_raw(
                run_summaries,
                derived_rows,
                reflection_threshold=float(CANDIDATE_PARAMS["reflection_threshold"]),
                reflection_lookback_slices=int(CANDIDATE_PARAMS["reflection_lookback_slices"]),
                prefix_fraction=float(CANDIDATE_PARAMS["prefix_fraction"]),
            )
            candidate_scores, _ = compute_code_v2_primary_scores_from_raw(
                candidate_raw,
                weights=candidate_weights,
            )
            order_map = {
                "code_baseline_v1": order_code_dynamic_group_indices(baseline_scores, D, run_ids=run_ids),
                "code_baseline_v2_candidate": order_code_dynamic_group_indices(candidate_scores, D, run_ids=run_ids),
                "min-confidence": np.asarray(sorted(range(len(run_ids)), key=lambda idx: (-float(min_conf_scores[idx]), int(run_ids[idx]))), dtype=np.int64),
                "tournament-copeland": np.asarray(sorted(range(len(run_ids)), key=lambda idx: (-float(copeland_scores[idx]), int(run_ids[idx]))), dtype=np.int64),
            }
            topk = max(1, int(math.ceil(0.10 * len(run_ids))))
            for selector_name, order in order_map.items():
                selector_problem[selector_name][str(problem_id)] = {
                    "best_run_id": int(run_ids[int(order[0])]),
                    "topk_run_ids": [int(run_ids[int(idx)]) for idx in order[:topk].tolist()],
                }

        compare = {}
        for other in ("code_baseline_v1", "min-confidence", "tournament-copeland"):
            top1 = []
            topk = []
            for problem_id in selector_problem["code_baseline_v2_candidate"]:
                base = selector_problem["code_baseline_v2_candidate"][problem_id]
                comp = selector_problem[other][problem_id]
                top1.append(int(base["best_run_id"] == comp["best_run_id"]))
                base_topk = set(base["topk_run_ids"])
                comp_topk = set(comp["topk_run_ids"])
                topk.append(len(base_topk & comp_topk) / max(len(base_topk | comp_topk), 1))
            compare[other] = {
                "top1_agreement": float(np.mean(top1)) if top1 else 0.0,
                "topk_jaccard": float(np.mean(topk)) if topk else 0.0,
            }
        outputs[cache_key] = compare
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run code_v2_candidate exhaustive evaluation and promotion gate")
    ap.add_argument("--gt-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--prefix-window-tokens", type=int, default=128)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--skip-blind", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"code_v2_candidate_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_entry_map = _load_entry_map(args.gt_cache_root)
    code_entry = gt_entry_map[CODE_CACHE_KEY]
    problems = _preload_problem_data(
        Path(code_entry.cache_root),
        distance_threads=int(args.distance_threads),
        prefix_window_tokens=int(args.prefix_window_tokens),
    )

    evaluated_candidates: list[dict[str, Any]] = []

    default_candidate = {
        "name": "default_candidate",
        "weights": {key: float(DEFAULT_CODE_V2_WEIGHTS[key]) for key in DEFAULT_CODE_V2_WEIGHTS},
    }
    print(f"[code-v2] Evaluating {default_candidate['name']} …", flush=True)
    default_result = _evaluate_code_v2_preloaded(
        problems,
        candidate_weights=default_candidate["weights"],
    )
    evaluated_candidates.append({
        **default_candidate,
        **default_result,
    })
    for candidate in _weight_search_space():
        print(f"[code-v2] Evaluating search candidate {candidate['name']} …", flush=True)
        result = _evaluate_code_v2_preloaded(
            problems,
            candidate_weights=candidate["weights"],
        )
        evaluated_candidates.append({
            **candidate,
            **result,
        })

    passing_candidates = [row for row in evaluated_candidates if row["gate"]["passed"]]
    selected_candidate = max(
        passing_candidates,
        key=lambda row: (
            float(row["compare_metrics"]["code_baseline_v2_candidate"].get("selacc@10%") or 0.0),
            float(row["compare_metrics"]["code_baseline_v2_candidate"].get("pairwise") or 0.0),
            float(row["compare_metrics"]["code_baseline_v2_candidate"].get("hit@1") or 0.0),
            row["name"],
        ),
    ) if passing_candidates else None

    best_candidate = max(
        evaluated_candidates,
        key=lambda row: (
            bool(row["gate"]["passed"]),
            float(row["compare_metrics"]["code_baseline_v2_candidate"].get("selacc@10%") or 0.0),
            float(row["compare_metrics"]["code_baseline_v2_candidate"].get("pairwise") or 0.0),
            float(row["compare_metrics"]["code_baseline_v2_candidate"].get("hit@1") or 0.0),
        ),
    ) if evaluated_candidates else None

    payload = {
        "gate_definition": {
            "selacc@10%": "candidate must exceed code_baseline_v1",
            "pairwise": "candidate must be >= 0.50",
            "hit@1": "candidate must be >= baseline - 1pp",
            "blind": "report only",
        },
        "evaluated_candidates": evaluated_candidates,
        "selected_weights": None if selected_candidate is None else selected_candidate["weights"],
        "selected_metrics": None if selected_candidate is None else selected_candidate["compare_metrics"]["code_baseline_v2_candidate"],
        "selected_gate": None if selected_candidate is None else selected_candidate["gate"],
        "best_weights": None if best_candidate is None else best_candidate["weights"],
        "best_metrics": None if best_candidate is None else best_candidate["compare_metrics"]["code_baseline_v2_candidate"],
        "best_gate": None if best_candidate is None else best_candidate["gate"],
        "selected_candidate": None if selected_candidate is None else selected_candidate["name"],
        "best_candidate": None if best_candidate is None else best_candidate["name"],
        "blind_shapes": None,
    }
    (out_dir / "code_v2_metrics_preblind.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    blind_shapes = None
    if selected_candidate is not None and not bool(args.skip_blind):
        blind_entry_map = _load_entry_map(args.blind_cache_root)
        blind_shapes = _evaluate_blind_shapes(
            blind_entry_map,
            candidate_weights=selected_candidate["weights"],
            distance_threads=int(args.distance_threads),
            prefix_window_tokens=int(args.prefix_window_tokens),
        )

    payload["blind_shapes"] = blind_shapes
    (out_dir / "code_v2_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    chosen = selected_candidate if selected_candidate is not None else best_candidate
    compare_rows = []
    loo_rows = []
    guarded_rows = []
    if chosen is not None:
        compare_rows = [
            _compact_row(name, metrics)
            for name, metrics in chosen["compare_metrics"].items()
        ]
        loo_rows = [
            _compact_row(name, metrics)
            for name, metrics in chosen["loo_metrics"].items()
        ]
        compare_rows.sort(key=lambda row: (-(row["selacc@10%"] or 0.0), -(row["pairwise"] or 0.0), row["name"]))
        loo_rows.sort(key=lambda row: (0 if row["name"] == "code_baseline_v2_candidate" else 1, -(row["selacc@10%"] or 0.0), row["name"]))
        guarded_rows = list(chosen["guarded_grid_rows"][:12])

    search_lines = [
        "| Candidate | Gate | SelAcc@10 | Hit@1 | Pairwise | Best Guarded |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in evaluated_candidates:
        cand = row["compare_metrics"]["code_baseline_v2_candidate"]
        best_guarded = row["best_guarded"]["name"] if row.get("best_guarded") else "n/a"
        search_lines.append(
            "| {name} | {gate} | {selacc} | {hit1} | {pairwise} | `{best_guarded}` |".format(
                name=row["name"],
                gate="pass" if row["gate"]["passed"] else "fail",
                selacc=_fmt_pct(cand["selacc@10%"]),
                hit1=_fmt_pct(cand["hit@1"]),
                pairwise=_fmt_pct(cand["pairwise"]),
                best_guarded=best_guarded,
            )
        )

    summary_md = [
        "# Code v2 Candidate",
        "",
        "## Search Summary",
        "",
        "\n".join(search_lines),
        "",
        f"- Selected candidate: `{selected_candidate['name']}`" if selected_candidate is not None else "- Selected candidate: none",
        f"- Best candidate: `{best_candidate['name']}`" if best_candidate is not None else "- Best candidate: none",
        "",
    ]
    if chosen is not None:
        summary_md.extend([
            "## Comparison",
            "",
            _summarize_metrics_table(compare_rows),
            "",
            "## Leave-One-Out",
            "",
            _summarize_metrics_table(loo_rows),
            "",
            "## Guarded Grid",
            "",
            _summarize_metrics_table(guarded_rows),
            "",
            f"- Best guarded config: `{chosen['best_guarded']['name']}`" if chosen.get("best_guarded") else "- Best guarded config: n/a",
            f"- Gate passed: `{chosen['gate']['passed']}`",
        ])
        if chosen["gate"]["failed_thresholds"]:
            summary_md.append(f"- Failed thresholds: `{chosen['gate']['failed_thresholds']}`")
        summary_md.extend([
            f"- Candidate weights: `{chosen['weights']}`",
            "",
        ])
    if blind_shapes is not None:
        summary_md.extend([
            "## Blind Checks",
            "",
        ])
        for cache_key, compare in blind_shapes.items():
            summary_md.append(f"- `{cache_key}`: `{compare}`")
    (out_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "selected_candidate": None if selected_candidate is None else selected_candidate["name"],
        "best_candidate": None if best_candidate is None else best_candidate["name"],
        "selected_selacc10": None if selected_candidate is None else selected_candidate["compare_metrics"]["code_baseline_v2_candidate"]["selacc@10%"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
