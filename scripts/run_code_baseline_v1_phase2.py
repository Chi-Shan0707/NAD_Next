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
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.code_dynamic_impl import (
    CODE_DYNAMIC_WEIGHT_NAMES,
    DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    compute_code_dynamic_primary_scores_from_raw,
    extract_code_dynamic_raw_from_state,
    order_code_dynamic_group_indices,
    prepare_code_dynamic_run_state,
)
from nad.core.selectors.trajectory_impl import _jaccard_sim
from nad.core.views.reader import Agg, CacheReader, CutSpec, CutType, Order, ViewSpec
from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import build_problem_groups, discover_cache_entries

DEFAULT_VIEW = ViewSpec(
    agg=Agg.MAX,
    cut=CutSpec(CutType.MASS, 1.0),
    order=Order.BY_KEY,
)
CODE_CACHE_KEY = "DS-R1/lcb_v5"
BLIND_CODE_KEYS = ("DS-R1/lcb_v5", "Qwen3-4B/lcb_v5")
COMPARE_SELECTOR_NAMES = ("min-confidence", "tournament-copeland", "code_baseline_v1")
NONCODE_SELECTOR_NAME = "tournament-copeland"


@dataclass
class ProblemResult:
    problem_id: str
    hit1: float
    pairwise: float
    local_selacc10: float
    topk: int
    best_run_id: int
    best_sample_id: int
    best_is_correct: bool | None
    topk_correct_count: int | None
    score_margin_top1_top2: float
    score_margin_top1_topk_mean: float
    topk_run_ids: list[int]
    topk_sample_ids: list[int]


class MetricAccumulator:
    def __init__(self, score_name: str, *, use_code_tiebreak: bool):
        self.score_name = str(score_name)
        self.use_code_tiebreak = bool(use_code_tiebreak)
        self.n_problems = 0
        self.hit1_total = 0.0
        self.pairwise_num = 0.0
        self.pairwise_den = 0.0
        self.all_scores: list[float] = []
        self.all_labels: list[int] = []
        self.problem_results: dict[str, ProblemResult] = {}

    def add_problem(
        self,
        problem_id: str,
        run_ids: list[int],
        scores: np.ndarray,
        labels: np.ndarray | None,
        D: np.ndarray,
    ) -> None:
        scores_arr = np.asarray(scores, dtype=np.float64)
        if self.use_code_tiebreak:
            order = order_code_dynamic_group_indices(scores_arr, D, run_ids=run_ids)
        else:
            order = np.asarray(
                sorted(range(len(run_ids)), key=lambda idx: (-float(scores_arr[idx]), int(run_ids[idx]))),
                dtype=np.int64,
            )

        topk = max(1, int(math.ceil(0.10 * len(run_ids))))
        best_idx = int(order[0]) if order.size else 0
        topk_idx = order[:topk]
        score_margin_top1_top2 = 0.0
        if order.size >= 2:
            score_margin_top1_top2 = float(scores_arr[int(order[0])] - scores_arr[int(order[1])])
        score_margin_top1_topk_mean = float(scores_arr[int(order[0])] - np.mean(scores_arr[topk_idx]))

        best_is_correct: bool | None = None
        topk_correct_count: int | None = None
        hit1 = 0.0
        local_selacc10 = 0.0
        pairwise = 0.0

        if labels is not None:
            labels_arr = np.asarray(labels, dtype=np.int32)
            hit1 = float(labels_arr[best_idx] > 0)
            best_is_correct = bool(labels_arr[best_idx] > 0)
            topk_correct_count = int(labels_arr[topk_idx].sum())
            local_selacc10 = float(labels_arr[topk_idx].mean()) if topk_idx.size else 0.0
            pos_scores = scores_arr[labels_arr > 0]
            neg_scores = scores_arr[labels_arr <= 0]
            if pos_scores.size > 0 and neg_scores.size > 0:
                wins = float((pos_scores[:, None] > neg_scores[None, :]).sum())
                total_pairs = float(pos_scores.size * neg_scores.size)
                pairwise = wins / total_pairs if total_pairs > 0 else 0.0
                self.pairwise_num += wins
                self.pairwise_den += total_pairs
            self.hit1_total += hit1
            self.all_scores.extend(scores_arr.tolist())
            self.all_labels.extend(labels_arr.tolist())

        self.problem_results[str(problem_id)] = ProblemResult(
            problem_id=str(problem_id),
            hit1=float(hit1),
            pairwise=float(pairwise),
            local_selacc10=float(local_selacc10),
            topk=int(topk),
            best_run_id=int(run_ids[best_idx]) if run_ids else 0,
            best_sample_id=int(run_ids[best_idx]) if run_ids else 0,
            best_is_correct=best_is_correct,
            topk_correct_count=topk_correct_count,
            score_margin_top1_top2=float(score_margin_top1_top2),
            score_margin_top1_topk_mean=float(score_margin_top1_topk_mean),
            topk_run_ids=[int(run_ids[int(idx)]) for idx in topk_idx.tolist()],
            topk_sample_ids=[int(run_ids[int(idx)]) for idx in topk_idx.tolist()],
        )
        self.n_problems += 1

    def finalize(self) -> dict[str, Any]:
        scores_arr = np.asarray(self.all_scores, dtype=np.float64)
        labels_arr = np.asarray(self.all_labels, dtype=np.int32)
        auroc = None
        if labels_arr.size > 0 and np.unique(labels_arr).size >= 2:
            auroc = float(roc_auc_score(labels_arr, scores_arr))
        selacc10 = 0.0
        top10_count = 0
        if labels_arr.size > 0:
            top10_count = max(1, int(math.ceil(0.10 * labels_arr.size)))
            order = np.asarray(sorted(range(labels_arr.size), key=lambda idx: (-float(scores_arr[idx]), idx)), dtype=np.int64)
            selacc10 = float(labels_arr[order[:top10_count]].mean())
        return {
            "auroc": auroc,
            "hit@1": float(self.hit1_total / self.n_problems) if self.n_problems else 0.0,
            "pairwise": float(self.pairwise_num / self.pairwise_den) if self.pairwise_den > 0 else None,
            "selacc@10%": float(selacc10),
            "n_problems": int(self.n_problems),
            "n_samples": int(labels_arr.size),
            "top10_count": int(top10_count),
            "per_problem": {
                problem_id: {
                    "hit@1": res.hit1,
                    "pairwise": res.pairwise,
                    "local_selacc10": res.local_selacc10,
                    "topk": res.topk,
                    "best_run_id": res.best_run_id,
                    "best_sample_id": res.best_sample_id,
                    "best_is_correct": res.best_is_correct,
                    "topk_correct_count": res.topk_correct_count,
                    "score_margin_top1_top2": res.score_margin_top1_top2,
                    "score_margin_top1_topk_mean": res.score_margin_top1_topk_mean,
                    "topk_run_ids": res.topk_run_ids,
                    "topk_sample_ids": res.topk_sample_ids,
                }
                for problem_id, res in self.problem_results.items()
            },
        }


def _problem_sort_key(problem_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(str(problem_id).split('-')[-1]):09d}")
    except Exception:
        return (1, str(problem_id))


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_entry_map(base_root: str | Path) -> dict[str, Any]:
    return {entry.cache_key: entry for entry in discover_cache_entries(base_root)}


def _compute_copeland_scores(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    wins = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            d_i = D[i, mask]
            d_j = D[j, mask]
            i_closer = int((d_i < d_j).sum())
            j_closer = int((d_j < d_i).sum())
            if i_closer > j_closer:
                wins[i] += 1.0
            elif j_closer > i_closer:
                wins[j] += 1.0
            else:
                wins[i] += 0.5
                wins[j] += 0.5
    return wins


def _make_empty_raw(n: int) -> dict[str, np.ndarray]:
    return {name: np.full(n, np.nan, dtype=np.float64) for name in CODE_DYNAMIC_WEIGHT_NAMES}


def _mean_tok_conf_from_state(run_state: dict[str, Any]) -> float:
    tok_conf = np.asarray(run_state.get("tok_conf", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    if tok_conf.size == 0:
        return float("-inf")
    return float(-np.mean(tok_conf))


def _compute_code_raw_for_problem(
    run_states: list[dict[str, Any]],
    *,
    reflection_threshold: float,
    reflection_lookback_slices: int,
    prefix_fraction: float,
    prefix_window_tokens: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    n = len(run_states)
    raw = _make_empty_raw(n)
    raw_rows: list[dict[str, float]] = []
    from nad.core.selectors.code_dynamic_impl import extract_code_dynamic_raw_from_state

    for idx, run_state in enumerate(run_states):
        row = extract_code_dynamic_raw_from_state(
            run_state,
            reflection_threshold=reflection_threshold,
            reflection_lookback_slices=reflection_lookback_slices,
            prefix_fraction=prefix_fraction,
            prefix_window_tokens=prefix_window_tokens,
        )
        raw_rows.append(row)
        for key in raw:
            raw[key][idx] = float(row[key])
    return raw, raw_rows


def _precompute_run_summary(
    run_state: dict[str, Any],
    *,
    reflection_thresholds: list[float],
    reflection_lookbacks: list[int],
    prefix_fractions: list[float],
    prefix_window_tokens: int,
) -> dict[str, Any]:
    base_raw = extract_code_dynamic_raw_from_state(
        run_state,
        reflection_threshold=float(DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD),
        reflection_lookback_slices=int(DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK),
        prefix_fraction=0.20,
        prefix_window_tokens=int(prefix_window_tokens),
    )
    tok_conf = np.asarray(run_state.get("tok_conf", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    slice_keysets = list(run_state.get("slice_keysets", []) or [])
    boundaries = run_state.get("boundaries")
    if boundaries is not None:
        boundaries = np.asarray(boundaries, dtype=np.int64)
    max_lookback = max(int(v) for v in reflection_lookbacks)
    prefix_best = {}
    for frac in prefix_fractions:
        prefix_raw = extract_code_dynamic_raw_from_state(
            run_state,
            reflection_threshold=float(DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD),
            reflection_lookback_slices=int(DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK),
            prefix_fraction=float(frac),
            prefix_window_tokens=int(prefix_window_tokens),
        )
        prefix_best[float(frac)] = float(prefix_raw["prefix_best_window_quality"])

    pair_sims: list[tuple[int, np.ndarray, np.ndarray]] = []
    n_slices = len(slice_keysets)
    for cur_idx in range(1, n_slices):
        start_idx = max(0, cur_idx - max_lookback)
        prev_idx = np.arange(cur_idx - 2, start_idx - 1, -1, dtype=np.int64)
        if prev_idx.size <= 0:
            pair_sims.append((cur_idx, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)))
            continue
        sims = np.asarray([
            float(_jaccard_sim(slice_keysets[cur_idx], slice_keysets[int(prev)]))
            for prev in prev_idx.tolist()
        ], dtype=np.float64)
        ages = np.asarray([int(cur_idx - int(prev)) for prev in prev_idx.tolist()], dtype=np.int64)
        pair_sims.append((cur_idx, ages, sims))

    reflection_lookup: dict[tuple[float, int], dict[str, float]] = {}
    for thr in reflection_thresholds:
        for lookback in reflection_lookbacks:
            reflection_count = 0.0
            last_event = -1
            for cur_idx, ages, sims in pair_sims:
                if sims.size <= 0:
                    continue
                mask = ages <= int(lookback)
                if mask.any() and float(np.max(sims[mask])) > float(thr):
                    reflection_count += 1.0
                    last_event = int(cur_idx)
            reflection_density = reflection_count / math.log(max(n_slices, 2)) if n_slices > 1 else 0.0
            post_reflection_recovery = 0.0
            if tok_conf.size > 0 and boundaries is not None and len(boundaries) >= 2 and last_event >= 1 and (last_event + 1) < len(boundaries):
                pre_start_slice = max(0, last_event - 1)
                pre_lo = int(boundaries[pre_start_slice])
                pre_hi = min(int(boundaries[last_event + 1]), tok_conf.size)
                post_lo = min(int(boundaries[last_event + 1]), tok_conf.size)
                if pre_hi > pre_lo and post_lo < tok_conf.size:
                    pre_mean = float(np.mean(tok_conf[pre_lo:pre_hi]))
                    post_mean = float(np.mean(tok_conf[post_lo:]))
                    post_reflection_recovery = pre_mean - post_mean
            reflection_lookup[(float(thr), int(lookback))] = {
                "reflection_density": float(reflection_density),
                "post_reflection_recovery": float(post_reflection_recovery),
            }

    return {
        "base_head_tail_gap": float(base_raw["head_tail_gap"]),
        "base_tail_variance": float(base_raw["tail_variance"]),
        "prefix_best_by_fraction": prefix_best,
        "reflection_lookup": reflection_lookup,
    }


def _build_code_raw_from_precomputed(
    summaries: list[dict[str, Any]],
    *,
    reflection_threshold: float,
    reflection_lookback_slices: int,
    prefix_fraction: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    n = len(summaries)
    raw = _make_empty_raw(n)
    rows: list[dict[str, float]] = []
    for idx, summary in enumerate(summaries):
        refl = summary["reflection_lookup"][(float(reflection_threshold), int(reflection_lookback_slices))]
        row = {
            "prefix_best_window_quality": float(summary["prefix_best_by_fraction"][float(prefix_fraction)]),
            "head_tail_gap": float(summary["base_head_tail_gap"]),
            "reflection_density": float(refl["reflection_density"]),
            "tail_variance": float(summary["base_tail_variance"]),
            "post_reflection_recovery": float(refl["post_reflection_recovery"]),
        }
        rows.append(row)
        for key in raw:
            raw[key][idx] = float(row[key])
    return raw, rows


def _compute_derived_trace_stats(run_state: dict[str, Any], *, prefix_fraction: float) -> dict[str, float]:
    tok_conf = np.asarray(run_state.get("tok_conf", np.zeros(0, dtype=np.float64)), dtype=np.float64)
    n = int(tok_conf.size)
    if n <= 0:
        return {
            "post_prefix_settle_score": 0.0,
            "post_prefix_tail_worst_window": 0.0,
            "last_block_instability_score": 0.0,
            "last_block_worst_window": 0.0,
        }
    prefix_end = max(1, min(n, int(math.ceil(float(prefix_fraction) * n))))
    tail_len = max(1, int(math.ceil(0.20 * n)))
    post_prefix = tok_conf[prefix_end:]
    tail = tok_conf[-tail_len:]
    mid = post_prefix if post_prefix.size > 0 else tail
    post_prefix_tail_worst_window = float(np.min(tail)) if tail.size > 0 else 0.0
    post_prefix_settle_score = float(np.mean(mid) - np.mean(tail)) if tail.size > 0 else 0.0
    last_block_worst_window = float(np.min(tail)) if tail.size > 0 else 0.0
    last_block_instability_score = float(np.var(tail) + max(0.0, np.mean(tail) - last_block_worst_window))
    return {
        "post_prefix_settle_score": post_prefix_settle_score,
        "post_prefix_tail_worst_window": post_prefix_tail_worst_window,
        "last_block_instability_score": last_block_instability_score,
        "last_block_worst_window": last_block_worst_window,
    }


def _config_name(prefix: str, **kwargs: Any) -> str:
    parts = [prefix]
    for key, val in kwargs.items():
        if isinstance(val, float):
            parts.append(f"{key}{val:.2f}".replace(".", "p"))
        else:
            parts.append(f"{key}{val}")
    return "__".join(parts)


def _summarize_metrics_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Name | AUROC | Hit@1 | Pairwise | SelAcc@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {auroc} | {hit1} | {pairwise} | {selacc} |".format(
                name=row["name"],
                auroc=_fmt_pct(row.get("auroc")),
                hit1=_fmt_pct(row.get("hit@1")),
                pairwise=_fmt_pct(row.get("pairwise")),
                selacc=_fmt_pct(row.get("selacc@10%")),
            )
        )
    return "\n".join(lines)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _evaluate_code_phase2(
    cache_root: Path,
    *,
    distance_threads: int,
    prefix_window_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    reflection_thresholds = [0.20, 0.25, 0.30, 0.35]
    lookbacks = [8, 16, 24]
    prefix_fracs = [0.10, 0.20, 0.30]
    baseline_params = {
        "reflection_threshold": DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
        "reflection_lookback_slices": DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
        "prefix_fraction": 0.20,
        "prefix_window_tokens": int(prefix_window_tokens),
    }

    baseline_name = "code_baseline_v1"
    loo_configs = [
        {"name": baseline_name, "disabled_features": []},
        *[
            {
                "name": f"loo_without__{feat_name}",
                "disabled_features": [feat_name],
            }
            for feat_name in CODE_DYNAMIC_WEIGHT_NAMES
        ],
    ]
    grid_configs = [
        {
            "name": _config_name(
                "grid",
                thr=thr,
                lb=lb,
                pf=pf,
            ),
            "reflection_threshold": float(thr),
            "reflection_lookback_slices": int(lb),
            "prefix_fraction": float(pf),
            "prefix_window_tokens": int(prefix_window_tokens),
        }
        for thr in reflection_thresholds
        for lb in lookbacks
        for pf in prefix_fracs
    ]

    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    correctness = _load_ground_truth(cache_root)
    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))

    loo_accs = {cfg["name"]: MetricAccumulator(cfg["name"], use_code_tiebreak=True) for cfg in loo_configs}
    grid_accs = {cfg["name"]: MetricAccumulator(cfg["name"], use_code_tiebreak=True) for cfg in grid_configs}
    compare_accs = {
        "min-confidence": MetricAccumulator("min-confidence", use_code_tiebreak=False),
        "tournament-copeland": MetricAccumulator("tournament-copeland", use_code_tiebreak=False),
        baseline_name: loo_accs[baseline_name],
    }
    disagreement_features: dict[str, dict[str, Any]] = {}

    for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
        if compare_accs[baseline_name].n_problems % 20 == 0:
            print(f"[code-phase2] problem {compare_accs[baseline_name].n_problems + 1}/{len(groups)}", flush=True)
        run_ids = list(map(int, run_ids))
        run_states: list[dict[str, Any]] = []
        run_views = []
        labels = np.asarray([int(bool(correctness.get(int(run_id), False))) for run_id in run_ids], dtype=np.int32)
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
                reflection_thresholds=reflection_thresholds,
                reflection_lookbacks=lookbacks,
                prefix_fractions=prefix_fracs,
                prefix_window_tokens=int(prefix_window_tokens),
            )
            for run_state in run_states
        ]

        compare_accs["min-confidence"].add_problem(str(problem_id), run_ids, min_conf_scores, labels, D)
        compare_accs["tournament-copeland"].add_problem(str(problem_id), run_ids, copeland_scores, labels, D)

        base_raw, base_rows = _build_code_raw_from_precomputed(
            run_summaries,
            reflection_threshold=float(baseline_params["reflection_threshold"]),
            reflection_lookback_slices=int(baseline_params["reflection_lookback_slices"]),
            prefix_fraction=float(baseline_params["prefix_fraction"]),
        )
        for cfg in loo_configs:
            scores, _ = compute_code_dynamic_primary_scores_from_raw(
                base_raw,
                disabled_features=cfg["disabled_features"],
            )
            loo_accs[cfg["name"]].add_problem(str(problem_id), run_ids, scores, labels, D)

        for cfg in grid_configs:
            raw, _ = _build_code_raw_from_precomputed(
                run_summaries,
                reflection_threshold=float(cfg["reflection_threshold"]),
                reflection_lookback_slices=int(cfg["reflection_lookback_slices"]),
                prefix_fraction=float(cfg["prefix_fraction"]),
            )
            scores, _ = compute_code_dynamic_primary_scores_from_raw(raw)
            grid_accs[cfg["name"]].add_problem(str(problem_id), run_ids, scores, labels, D)

        base_scores, _ = compute_code_dynamic_primary_scores_from_raw(base_raw)
        best_idx = int(order_code_dynamic_group_indices(base_scores, D, run_ids=run_ids)[0])
        derived = _compute_derived_trace_stats(run_states[best_idx], prefix_fraction=float(baseline_params["prefix_fraction"]))
        disagreement_features[str(problem_id)] = {
            "selected_run_id": int(run_ids[best_idx]),
            "selected_is_correct": bool(labels[best_idx] > 0),
            "prefix_best_window_quality": float(base_rows[best_idx]["prefix_best_window_quality"]),
            "head_tail_gap": float(base_rows[best_idx]["head_tail_gap"]),
            "tail_variance": float(base_rows[best_idx]["tail_variance"]),
            "post_reflection_recovery": float(base_rows[best_idx]["post_reflection_recovery"]),
            **derived,
        }

    loo_metrics = {name: acc.finalize() for name, acc in loo_accs.items()}
    grid_metrics = {name: acc.finalize() for name, acc in grid_accs.items()}
    compare_metrics = {name: acc.finalize() for name, acc in compare_accs.items()}

    baseline_problem = compare_metrics[baseline_name]["per_problem"]
    mc_problem = compare_metrics["min-confidence"]["per_problem"]
    tc_problem = compare_metrics["tournament-copeland"]["per_problem"]

    win_rows = []
    loss_rows = []
    for problem_id, row in baseline_problem.items():
        best_other_selacc = max(float(mc_problem[problem_id]["local_selacc10"]), float(tc_problem[problem_id]["local_selacc10"]))
        best_other_pair = max(float(mc_problem[problem_id]["pairwise"]), float(tc_problem[problem_id]["pairwise"]))
        best_other_hit1 = max(float(mc_problem[problem_id]["hit@1"]), float(tc_problem[problem_id]["hit@1"]))
        gap_selacc = float(row["local_selacc10"]) - best_other_selacc
        gap_pair = float(row["pairwise"]) - best_other_pair
        gap_hit1 = float(row["hit@1"]) - best_other_hit1
        margin = gap_selacc + 0.25 * gap_pair + 0.05 * gap_hit1
        payload = {
            "problem_id": problem_id,
            "margin": float(margin),
            "gap_selacc10": float(gap_selacc),
            "gap_pairwise": float(gap_pair),
            "gap_hit1": float(gap_hit1),
            "code_baseline_v1": row,
            "min-confidence": mc_problem[problem_id],
            "tournament-copeland": tc_problem[problem_id],
            "selected_features": disagreement_features[problem_id],
        }
        if gap_selacc > 0.0 or (gap_selacc >= 0.0 and gap_pair > 0.0):
            win_rows.append(payload)
        if gap_selacc < 0.0 or (gap_selacc <= 0.0 and gap_pair < 0.0):
            loss_rows.append(payload)

    win_rows.sort(key=lambda row: (-row["margin"], -row["gap_selacc10"], -row["gap_pairwise"], row["problem_id"]))
    loss_rows.sort(key=lambda row: (row["margin"], row["gap_selacc10"], row["gap_pairwise"], row["problem_id"]))
    win_rows = win_rows[:50]
    loss_rows = loss_rows[:50]

    def _bucket_mean(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return float(np.mean([float(row["selected_features"][key]) for row in rows]))

    settle_sep = abs(_bucket_mean(win_rows, "post_prefix_settle_score") - _bucket_mean(loss_rows, "post_prefix_settle_score"))
    instability_sep = abs(_bucket_mean(win_rows, "last_block_instability_score") - _bucket_mean(loss_rows, "last_block_instability_score"))
    candidate_family = "post-prefix settle shape" if settle_sep >= instability_sep else "last-block instability"
    disagreement = {
        "wins": win_rows,
        "losses": loss_rows,
        "candidate_feature_family": candidate_family,
        "candidate_signal_strength": {
            "post_prefix_settle_shape": float(settle_sep),
            "last_block_instability": float(instability_sep),
        },
    }
    return loo_metrics, grid_metrics, compare_metrics, disagreement


def _evaluate_transfer_gt(
    gt_entry_map: dict[str, Any],
    *,
    distance_threads: int,
    prefix_window_tokens: int,
) -> dict[str, Any]:
    selector_rows = []
    domain_table = []
    code_baseline_row = None

    for cache_key, entry in sorted(gt_entry_map.items()):
        print(f"[transfer-gt] {cache_key}", flush=True)
        cache_root = Path(entry.cache_root)
        correctness = _load_ground_truth(cache_root)
        meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
        groups = build_problem_groups(meta)
        reader = CacheReader(str(cache_root))
        engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
        accs = {
            "code_baseline_v1": MetricAccumulator("code_baseline_v1", use_code_tiebreak=True),
            "min-confidence": MetricAccumulator("min-confidence", use_code_tiebreak=False),
            "tournament-copeland": MetricAccumulator("tournament-copeland", use_code_tiebreak=False),
        }
        for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
            run_ids = list(map(int, run_ids))
            run_states = []
            run_views = []
            labels = np.asarray([int(bool(correctness.get(int(run_id), False))) for run_id in run_ids], dtype=np.int32)
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
                    prefix_fractions=[0.20],
                    prefix_window_tokens=int(prefix_window_tokens),
                )
                for run_state in run_states
            ]
            raw, _ = _build_code_raw_from_precomputed(
                run_summaries,
                reflection_threshold=DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
                reflection_lookback_slices=DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
                prefix_fraction=0.20,
            )
            code_scores, _ = compute_code_dynamic_primary_scores_from_raw(raw)
            accs["code_baseline_v1"].add_problem(str(problem_id), run_ids, code_scores, labels, D)
            accs["min-confidence"].add_problem(str(problem_id), run_ids, min_conf_scores, labels, D)
            accs["tournament-copeland"].add_problem(str(problem_id), run_ids, copeland_scores, labels, D)

        finalized = {name: acc.finalize() for name, acc in accs.items()}
        dataset_name = str(getattr(entry, "dataset_name", cache_key.split("/", 1)[-1]))
        domain = "code" if dataset_name in {"livecodebench_v5", "lcb_v5"} else "noncode"
        for name, metrics in finalized.items():
            row = {
                "cache_key": cache_key,
                "dataset": dataset_name,
                "domain": domain,
                "selector": name,
                "auroc": metrics["auroc"],
                "hit@1": metrics["hit@1"],
                "pairwise": metrics["pairwise"],
                "selacc@10%": metrics["selacc@10%"],
            }
            selector_rows.append(row)
            if cache_key == CODE_CACHE_KEY and name == "code_baseline_v1":
                code_baseline_row = row

    def _domain_mean(selector: str, domain: str, metric: str) -> float | None:
        values = [float(row[metric]) for row in selector_rows if row["selector"] == selector and row["domain"] == domain and row[metric] is not None]
        if not values:
            return None
        return float(np.mean(values))

    for selector in ("code_baseline_v1", NONCODE_SELECTOR_NAME):
        domain_table.append({
            "selector": selector,
            "code_selacc@10%": _domain_mean(selector, "code", "selacc@10%"),
            "code_pairwise": _domain_mean(selector, "code", "pairwise"),
            "noncode_hit@1": _domain_mean(selector, "noncode", "hit@1"),
            "noncode_selacc@10%": _domain_mean(selector, "noncode", "selacc@10%"),
        })

    return {
        "rows": selector_rows,
        "transfer_2x2": domain_table,
        "primary_code_row": code_baseline_row,
    }


def _evaluate_blind_code_shapes(
    blind_entry_map: dict[str, Any],
    *,
    distance_threads: int,
    prefix_window_tokens: int,
    gt_code_metrics: dict[str, Any],
) -> dict[str, Any]:
    outputs = {}
    gt_code_problem = gt_code_metrics["code_baseline_v1"]["per_problem"]
    gt_selected_feature_means = {}
    feature_keys = [
        "prefix_best_window_quality",
        "head_tail_gap",
        "tail_variance",
        "post_reflection_recovery",
        "post_prefix_settle_score",
        "last_block_instability_score",
    ]
    for feat_key in feature_keys:
        vals = [float(row["selected_features"][feat_key]) for row in gt_code_metrics["disagreement"]["wins"] + gt_code_metrics["disagreement"]["losses"] if feat_key in row["selected_features"]]
        if vals:
            gt_selected_feature_means[feat_key] = float(np.mean(vals))

    for cache_key in BLIND_CODE_KEYS:
        print(f"[blind-code] {cache_key}", flush=True)
        entry = blind_entry_map[cache_key]
        cache_root = Path(entry.cache_root)
        meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
        groups = build_problem_groups(meta)
        reader = CacheReader(str(cache_root))
        engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
        selector_problem = {name: {} for name in COMPARE_SELECTOR_NAMES}
        feature_rows = []
        for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
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
                    prefix_fractions=[0.20],
                    prefix_window_tokens=int(prefix_window_tokens),
                )
                for run_state in run_states
            ]
            raw, raw_rows = _build_code_raw_from_precomputed(
                run_summaries,
                reflection_threshold=DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
                reflection_lookback_slices=DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
                prefix_fraction=0.20,
            )
            code_scores, _ = compute_code_dynamic_primary_scores_from_raw(raw)
            order_map = {
                "code_baseline_v1": order_code_dynamic_group_indices(code_scores, D, run_ids=run_ids),
                "min-confidence": np.asarray(sorted(range(len(run_ids)), key=lambda idx: (-float(min_conf_scores[idx]), int(run_ids[idx]))), dtype=np.int64),
                "tournament-copeland": np.asarray(sorted(range(len(run_ids)), key=lambda idx: (-float(copeland_scores[idx]), int(run_ids[idx]))), dtype=np.int64),
            }
            topk = max(1, int(math.ceil(0.10 * len(run_ids))))
            for selector_name, order in order_map.items():
                selector_problem[selector_name][str(problem_id)] = {
                    "best_run_id": int(run_ids[int(order[0])]),
                    "topk_run_ids": [int(run_ids[int(idx)]) for idx in order[:topk].tolist()],
                    "score_margin_top1_top2": float((code_scores if selector_name == "code_baseline_v1" else min_conf_scores if selector_name == "min-confidence" else copeland_scores)[int(order[0])] - (code_scores if selector_name == "code_baseline_v1" else min_conf_scores if selector_name == "min-confidence" else copeland_scores)[int(order[1])] if len(order) >= 2 else 0.0),
                }
            code_best_idx = int(order_map["code_baseline_v1"][0])
            derived = _compute_derived_trace_stats(run_states[code_best_idx], prefix_fraction=0.20)
            feature_rows.append({
                "problem_id": str(problem_id),
                "selected_run_id": int(run_ids[code_best_idx]),
                "prefix_best_window_quality": float(raw_rows[code_best_idx]["prefix_best_window_quality"]),
                "head_tail_gap": float(raw_rows[code_best_idx]["head_tail_gap"]),
                "tail_variance": float(raw_rows[code_best_idx]["tail_variance"]),
                "post_reflection_recovery": float(raw_rows[code_best_idx]["post_reflection_recovery"]),
                **derived,
            })

        top1_agreement = {}
        topk_jaccard = {}
        for other in ("min-confidence", "tournament-copeland"):
            agree = []
            jacc = []
            margin_gap = []
            for problem_id in selector_problem["code_baseline_v1"]:
                base = selector_problem["code_baseline_v1"][problem_id]
                comp = selector_problem[other][problem_id]
                agree.append(int(base["best_run_id"] == comp["best_run_id"]))
                base_topk = set(base["topk_run_ids"])
                comp_topk = set(comp["topk_run_ids"])
                jacc.append(len(base_topk & comp_topk) / max(len(base_topk | comp_topk), 1))
                margin_gap.append(float(base["score_margin_top1_top2"] - comp["score_margin_top1_top2"]))
            top1_agreement[other] = float(np.mean(agree)) if agree else 0.0
            topk_jaccard[other] = {
                "mean_jaccard": float(np.mean(jacc)) if jacc else 0.0,
                "mean_margin_gap": float(np.mean(margin_gap)) if margin_gap else 0.0,
            }

        feature_means = {
            key: float(np.mean([row[key] for row in feature_rows])) if feature_rows else 0.0
            for key in feature_keys
        }
        feature_shift = {
            key: float(feature_means[key] - gt_selected_feature_means.get(key, 0.0))
            for key in feature_keys
        }
        outputs[cache_key] = {
            "top1_agreement_rate": top1_agreement,
            "topk_overlap": topk_jaccard,
            "feature_mean": feature_means,
            "feature_shift_vs_gt_selected": feature_shift,
        }
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run code_baseline_v1 phase-2 experiments")
    ap.add_argument("--gt-cache-root", default="MUI_HUB/cache", help="Ground-truth cache root")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test", help="Blind cache root")
    ap.add_argument("--distance-threads", type=int, default=8, help="Distance computation threads")
    ap.add_argument("--prefix-window-tokens", type=int, default=128)
    ap.add_argument("--out-dir", default="", help="Optional output directory; defaults to result/code_baseline_v1_phase2_<timestamp>")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"code_baseline_v1_phase2_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_entry_map = _load_entry_map(args.gt_cache_root)
    blind_entry_map = _load_entry_map(args.blind_cache_root)

    code_entry = gt_entry_map[CODE_CACHE_KEY]
    print(f"[phase2] code cache = {code_entry.cache_root}", flush=True)
    loo_metrics, grid_metrics, compare_metrics, disagreement = _evaluate_code_phase2(
        Path(code_entry.cache_root),
        distance_threads=int(args.distance_threads),
        prefix_window_tokens=int(args.prefix_window_tokens),
    )
    gt_phase_metrics = {
        **compare_metrics,
        "disagreement": disagreement,
    }

    guarded_grid_rows = []
    baseline_pairwise = float(compare_metrics["code_baseline_v1"]["pairwise"] or 0.0)
    baseline_hit1 = float(compare_metrics["code_baseline_v1"]["hit@1"] or 0.0)
    for name, metrics in grid_metrics.items():
        pairwise = metrics.get("pairwise")
        hit1 = metrics.get("hit@1")
        guard_ok = (
            pairwise is not None
            and float(pairwise) >= 0.50
            and hit1 is not None
            and float(hit1) >= baseline_hit1 - 0.01
        )
        row = {"name": name, **{k: metrics[k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}, "guard_ok": guard_ok}
        guarded_grid_rows.append(row)
    guarded_grid_rows.sort(key=lambda row: (not row["guard_ok"], -(row["selacc@10%"] or 0.0), -(row["pairwise"] or 0.0), -(row["hit@1"] or 0.0), row["name"]))
    best_guarded = next((row for row in guarded_grid_rows if row["guard_ok"]), guarded_grid_rows[0] if guarded_grid_rows else None)

    transfer_gt = _evaluate_transfer_gt(
        gt_entry_map,
        distance_threads=int(args.distance_threads),
        prefix_window_tokens=int(args.prefix_window_tokens),
    )
    blind_shapes = _evaluate_blind_code_shapes(
        blind_entry_map,
        distance_threads=int(args.distance_threads),
        prefix_window_tokens=int(args.prefix_window_tokens),
        gt_code_metrics=gt_phase_metrics,
    )

    loo_rows = [{"name": name, **{k: metrics[k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}} for name, metrics in loo_metrics.items()]
    loo_rows.sort(key=lambda row: (0 if row["name"] == "code_baseline_v1" else 1, -(row["selacc@10%"] or 0.0), row["name"]))

    (out_dir / "loo_metrics.json").write_text(json.dumps(loo_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "grid_metrics.json").write_text(json.dumps({
        "grid_metrics": grid_metrics,
        "guarded_rows": guarded_grid_rows,
        "best_guarded": best_guarded,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "disagreement_cases.json").write_text(json.dumps(disagreement, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "transfer_gate.json").write_text(json.dumps({
        "gt_transfer": transfer_gt,
        "blind_shapes": blind_shapes,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    loo_md = [
        "# Code Baseline v1 Phase 2 — Leave-One-Out",
        "",
        _summarize_metrics_table(loo_rows),
        "",
        "Frozen baseline remains `code_baseline_v1`.",
    ]
    (out_dir / "loo_summary.md").write_text("\n".join(loo_md) + "\n", encoding="utf-8")

    grid_md = [
        "# Code Baseline v1 Phase 2 — Small Grid",
        "",
        _summarize_metrics_table(guarded_grid_rows[:12]),
        "",
        f"Best guarded config: `{best_guarded['name']}`" if best_guarded else "No guarded config found.",
        f"- Baseline pairwise: {_fmt_pct(baseline_pairwise)}",
        f"- Baseline Hit@1: {_fmt_pct(baseline_hit1)}",
        "- Guardrail: `Pairwise >= 50%` and `Hit@1` drop <= `1pp` vs baseline",
    ]
    (out_dir / "grid_summary.md").write_text("\n".join(grid_md) + "\n", encoding="utf-8")

    disagreement_md = [
        "# Code Baseline v1 Phase 2 — Disagreement Mining",
        "",
        f"- Candidate feature family: `{disagreement['candidate_feature_family']}`",
        f"- Win bucket size: `{len(disagreement['wins'])}`",
        f"- Loss bucket size: `{len(disagreement['losses'])}`",
        f"- Post-prefix settle separation: `{disagreement['candidate_signal_strength']['post_prefix_settle_shape']:.4f}`",
        f"- Last-block instability separation: `{disagreement['candidate_signal_strength']['last_block_instability']:.4f}`",
        "",
        "Win/loss cases are saved in `disagreement_cases.json`.",
    ]
    (out_dir / "disagreement_summary.md").write_text("\n".join(disagreement_md) + "\n", encoding="utf-8")

    transfer_md = [
        "# Code Baseline v1 Phase 2 — Transfer Gate",
        "",
        "## GT 2×2 summary",
        "",
        "| Selector | Code SelAcc@10 | Code Pairwise | Noncode Hit@1 | Noncode SelAcc@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in transfer_gt["transfer_2x2"]:
        transfer_md.append(
            "| {selector} | {code_selacc} | {code_pairwise} | {noncode_hit1} | {noncode_selacc} |".format(
                selector=row["selector"],
                code_selacc=_fmt_pct(row["code_selacc@10%"]),
                code_pairwise=_fmt_pct(row["code_pairwise"]),
                noncode_hit1=_fmt_pct(row["noncode_hit@1"]),
                noncode_selacc=_fmt_pct(row["noncode_selacc@10%"]),
            )
        )
    transfer_md.extend([
        "",
        "## Blind coding shape checks",
        "",
        "Blind DS/Qwen outputs are reported as overlap / ranking-shape / feature-shift only.",
    ])
    for cache_key, payload in blind_shapes.items():
        transfer_md.append("")
        transfer_md.append(f"- `{cache_key}` top1 agreement: `{payload['top1_agreement_rate']}`")
        transfer_md.append(f"- `{cache_key}` topk overlap: `{payload['topk_overlap']}`")
        transfer_md.append(f"- `{cache_key}` feature shift: `{payload['feature_shift_vs_gt_selected']}`")
    (out_dir / "transfer_gate.md").write_text("\n".join(transfer_md) + "\n", encoding="utf-8")

    summary = {
        "out_dir": str(out_dir),
        "best_guarded_grid": best_guarded,
        "candidate_feature_family": disagreement["candidate_feature_family"],
        "transfer_2x2": transfer_gt["transfer_2x2"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
