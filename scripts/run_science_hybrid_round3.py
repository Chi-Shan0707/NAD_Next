#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.code_dynamic_impl import order_code_dynamic_group_indices, prepare_code_dynamic_run_state
from nad.core.selectors.code_v2_impl import compute_code_v2_primary_scores_from_raw
from nad.core.selectors.gpqa_pairwise_impl import (
    GPQAPairwiseScorer,
    build_gpqa_pairwise_dominance_feature,
    build_gpqa_pairwise_features_configurable,
    build_gpqa_pairwise_margin_feature,
    build_pairwise_training_pairs,
    extract_gpqa_pairwise_raw,
)
from nad.core.selectors.science_dynamic_impl import DEFAULT_SCIENCE_DYNAMIC_WEIGHTS, compute_science_dynamic_primary_scores_from_raw
from nad.core.selectors.science_hybrid_impl import (
    ScienceHybridConfig,
    compute_pairwise_backend_scores_from_matrix,
    compute_pairwise_probability_matrix,
    compute_science_hybrid_decision,
    default_gpqa_pairwise_model_path,
)
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import discover_cache_entries, load_model, score_cache_entry
from scripts.run_code_baseline_v1_phase2 import (
    DEFAULT_VIEW,
    MetricAccumulator,
    _build_code_raw_from_precomputed,
    _compute_copeland_scores,
    _compute_derived_trace_stats,
    _fmt_pct,
    _load_entry_map,
    _precompute_run_summary,
    _problem_sort_key,
)
from scripts.run_gpqa_pairwise_round1 import _ProblemData, _extract_all_problems
from scripts.run_gpqa_pairwise_round2 import _variant_configs, _variant_feature_matrix

EXTREME12_MANIFEST = REPO_ROOT / "submission/BestofN/extreme12/manifests/extreme12_baseline12_pointwise_export_manifest.json"
EXTREME12_TEST_ANALYSIS_DOC = REPO_ROOT / "docs/EXTREME12_TEST_ANALYSIS.md"
CODE_V2_EXHAUSTIVE_JSON = REPO_ROOT / "result/code_v2_candidate_20260406_exhaustive/code_v2_metrics.json"
OVERALL_DS_CACHE_KEYS = (
    "DS-R1/aime24",
    "DS-R1/aime25",
    "DS-R1/brumo25",
    "DS-R1/gpqa",
    "DS-R1/hmmt25",
    "DS-R1/lcb_v5",
)
BLIND_GPQAS = ("DS-R1/gpqa", "Qwen3-4B/gpqa")
PAIRWISE_BACKEND_GRID = (
    {"name": "mean", "backend": "mean", "temperature": 1.0},
    {"name": "softmax_mean_t0p75", "backend": "softmax_mean", "temperature": 0.75},
    {"name": "softmax_mean_t0p50", "backend": "softmax_mean", "temperature": 0.50},
    {"name": "win_count", "backend": "win_count", "temperature": 1.0},
    {"name": "copeland_margin", "backend": "copeland_margin", "temperature": 1.0},
)
BLEND_ALPHAS = (0.25, 0.50, 0.75)
SHORTLIST_KS = (2, 3, 4, 5)
OVERRIDE_MS = (0.00, 0.02, 0.04, 0.06)


@dataclass(frozen=True)
class ProblemScoreRecord:
    cache_key: str
    problem_id: str
    sample_ids: list[int]
    labels: np.ndarray
    scores: np.ndarray


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _order_to_rank_scores(order: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.float64)
    if int(n) <= 0:
        return out
    if int(n) == 1:
        out[int(order[0])] = 1.0
        return out
    values = np.linspace(1.0, 0.0, num=int(n), dtype=np.float64)
    for rank_pos, group_idx in enumerate(np.asarray(order, dtype=np.int64).tolist()):
        out[int(group_idx)] = float(values[int(rank_pos)])
    return out


def _average_rank(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return np.zeros(arr.size, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    ranks = np.zeros(arr.size, dtype=np.float64)
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and np.isclose(float(sorted_vals[end]), float(sorted_vals[start]), atol=1e-12, rtol=0.0):
            end += 1
        avg_rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size <= 1 or y_arr.size <= 1:
        return None
    x_center = x_arr - float(np.mean(x_arr))
    y_center = y_arr - float(np.mean(y_arr))
    x_std = float(np.sqrt(np.mean(x_center * x_center)))
    y_std = float(np.sqrt(np.mean(y_center * y_center)))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return None
    return float(np.mean((x_center / x_std) * (y_center / y_std)))


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return _pearson(_average_rank(x), _average_rank(y))


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in ("auroc", "hit@1", "pairwise", "selacc@10%", "n_problems", "n_samples", "top10_count")
    }


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


def _candidate_name(config: ScienceHybridConfig) -> str:
    if config.family == "margin_fallback":
        return f"familyA__tau{config.tau:.4f}__k{int(config.k)}".replace(".", "p")
    if config.family == "shortlist_blend":
        return f"familyB__k{int(config.k)}__a{config.alpha:.2f}".replace(".", "p")
    if config.family == "hard_override":
        return f"familyC__tau{config.tau:.4f}__m{config.m:.2f}".replace(".", "p")
    raise ValueError(f"Unknown family: {config.family}")


def _build_pairwise_backend_acc(
    name: str,
    all_problems: list[_ProblemData],
    prob_matrices: list[np.ndarray],
    *,
    backend: str,
    temperature: float,
) -> dict[str, Any]:
    acc = MetricAccumulator(name, use_code_tiebreak=True)
    for prob, prob_matrix in zip(all_problems, prob_matrices):
        scores = compute_pairwise_backend_scores_from_matrix(
            prob_matrix,
            backend=backend,
            temperature=temperature,
        )
        acc.add_problem(prob.problem_id, prob.run_ids, scores, prob.labels, prob.D)
    return acc.finalize()


def _fit_final_variant_orders(
    all_problems: list[_ProblemData],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    variant_X = [_variant_feature_matrix(prob, cfg) for prob in all_problems]
    pair_X_list: list[np.ndarray] = []
    pair_y_list: list[np.ndarray] = []
    for prob, X_prob in zip(all_problems, variant_X):
        X_pairs, y_pairs = build_pairwise_training_pairs(X_prob, prob.labels)
        if X_pairs.shape[0] > 0:
            pair_X_list.append(X_pairs)
            pair_y_list.append(y_pairs)

    scorer = GPQAPairwiseScorer(
        C=float(cfg["C"]),
        include_margin=bool(cfg["include_margin"]),
        include_dominance=bool(cfg["include_dominance"]),
    )
    scorer.fit(np.concatenate(pair_X_list, axis=0), np.concatenate(pair_y_list, axis=0))

    out: dict[str, np.ndarray] = {}
    for prob, X_prob in zip(all_problems, variant_X):
        scores = scorer.score_group(X_prob)
        order = order_code_dynamic_group_indices(scores, prob.D, run_ids=prob.run_ids)
        out[str(prob.problem_id)] = np.asarray(order, dtype=np.int64)
    return out


def _problem_records_from_prob_score_arrays(
    cache_key: str,
    all_problems: list[_ProblemData],
    score_by_problem: dict[str, np.ndarray],
) -> list[ProblemScoreRecord]:
    records: list[ProblemScoreRecord] = []
    for prob in all_problems:
        scores = np.asarray(score_by_problem[str(prob.problem_id)], dtype=np.float64)
        records.append(
            ProblemScoreRecord(
                cache_key=str(cache_key),
                problem_id=str(prob.problem_id),
                sample_ids=list(map(int, prob.run_ids)),
                labels=np.asarray(prob.labels, dtype=np.int32),
                scores=scores,
            )
        )
    return records


def _problem_records_from_cache_scores(cache_scores, correctness: dict[int, bool]) -> list[ProblemScoreRecord]:
    records: list[ProblemScoreRecord] = []
    for problem_id, item in sorted(cache_scores.problem_scores.items(), key=lambda kv: _problem_sort_key(kv[0])):
        sample_ids = list(map(int, item["sample_ids"]))
        labels = np.asarray([int(bool(correctness[int(sample_id)])) for sample_id in sample_ids], dtype=np.int32)
        scores = np.asarray(item["score_best"], dtype=np.float64)
        records.append(
            ProblemScoreRecord(
                cache_key=str(cache_scores.entry.cache_key),
                problem_id=str(problem_id),
                sample_ids=sample_ids,
                labels=labels,
                scores=scores,
            )
        )
    return records


def _evaluate_problem_records(records: list[ProblemScoreRecord]) -> dict[str, Any]:
    all_scores: list[float] = []
    all_labels: list[int] = []
    hit1_total = 0.0
    hit3_total = 0.0
    pairwise_num = 0.0
    pairwise_den = 0.0
    avg_best_correct_rank_total = 0.0
    n_problems = 0

    for record in records:
        scores = np.asarray(record.scores, dtype=np.float64)
        labels = np.asarray(record.labels, dtype=np.int32)
        if scores.size <= 0:
            continue
        order = np.argsort(-scores, kind="mergesort")
        hit1_total += float(labels[order[0]] > 0)
        hit3_total += float(np.any(labels[order[: min(3, order.size)]] > 0))
        pos = np.where(labels[order] > 0)[0]
        best_correct_rank = int(pos[0] + 1) if pos.size > 0 else int(order.size + 1)
        avg_best_correct_rank_total += float(best_correct_rank)

        pos_scores = scores[labels > 0]
        neg_scores = scores[labels <= 0]
        if pos_scores.size > 0 and neg_scores.size > 0:
            diff = pos_scores[:, None] - neg_scores[None, :]
            pairwise_num += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
            pairwise_den += float(diff.size)

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        n_problems += 1

    scores_arr = np.asarray(all_scores, dtype=np.float64)
    labels_arr = np.asarray(all_labels, dtype=np.int32)
    auroc = None
    if labels_arr.size > 0 and np.unique(labels_arr).size >= 2:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(labels_arr, scores_arr))

    selacc10 = 0.0
    top10_count = 0
    if labels_arr.size > 0:
        top10_count = max(1, int(math.ceil(0.10 * labels_arr.size)))
        order = np.argsort(-scores_arr, kind="mergesort")
        selacc10 = float(labels_arr[order[:top10_count]].mean())

    return {
        "auroc": auroc,
        "hit@1": float(hit1_total / n_problems) if n_problems else 0.0,
        "hit@3": float(hit3_total / n_problems) if n_problems else 0.0,
        "pairwise": float(pairwise_num / pairwise_den) if pairwise_den > 0 else None,
        "selacc@10%": float(selacc10),
        "avg_rank_proxy": float(avg_best_correct_rank_total / n_problems) if n_problems else None,
        "n_problems": int(n_problems),
        "n_samples": int(labels_arr.size),
        "top10_count": int(top10_count),
    }


def _mean_cache_metrics(records: list[ProblemScoreRecord]) -> dict[str, Any]:
    grouped: dict[str, list[ProblemScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[str(record.cache_key)].append(record)
    per_cache = {cache_key: _evaluate_problem_records(cache_records) for cache_key, cache_records in sorted(grouped.items())}
    out: dict[str, Any] = {"per_cache": per_cache}
    for key in ("auroc", "hit@1", "hit@3", "pairwise", "selacc@10%", "avg_rank_proxy"):
        values = [per_cache[cache_key].get(key) for cache_key in per_cache]
        finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        out[key] = None if not finite else float(np.mean(finite))
    return out


def _system_bundle(records: list[ProblemScoreRecord]) -> dict[str, Any]:
    return {
        "sample_weighted": _evaluate_problem_records(records),
        "equal_cache_mean": _mean_cache_metrics(records),
    }


def _load_extreme12_test_metrics(doc_path: Path) -> dict[str, dict[str, Any]]:
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    metrics: dict[str, dict[str, Any]] = {}
    in_table = False
    for line in lines:
        if line.startswith("| Cache | AUROC | Hit@1 | Hit@3 | SelAcc@10% | Pairwise | Samples |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.strip():
            break
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 7:
            continue
        cache_key = str(parts[0])
        if not cache_key.startswith("DS-R1/"):
            continue
        metrics[cache_key] = {
            "auroc": float(parts[1]),
            "hit@1": float(parts[2]),
            "hit@3": float(parts[3]),
            "selacc@10%": float(parts[4]),
            "pairwise": float(parts[5]),
            "n_samples": int(parts[6].replace(",", "")),
        }
    if not metrics:
        raise RuntimeError(f"Failed to parse DS-R1 metrics from {doc_path}")
    return metrics


def _load_code_v2_proxy_metrics(
    result_path: Path,
    *,
    fallback_hit3: float,
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_metrics")
    if not isinstance(selected, dict):
        raise RuntimeError(f"Failed to load selected code_v2 metrics from {result_path}")
    metrics = dict(selected)
    metrics["hit@3"] = float(fallback_hit3)
    metrics["source"] = str(result_path)
    metrics["selected_candidate"] = payload.get("selected_candidate")
    return metrics


def _problem_counts_from_entry_map(entry_map: dict[str, Any], cache_keys: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for cache_key in cache_keys:
        entry = entry_map[cache_key]
        meta = json.loads((Path(entry.cache_root) / "meta.json").read_text(encoding="utf-8"))
        problem_ids = {str(sample["problem_id"]) for sample in meta["samples"]}
        out[cache_key] = int(len(problem_ids))
    return out


def _combine_cache_metric_proxy(cache_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not cache_metrics:
        return {"sample_weighted": {}, "equal_cache_mean": {}, "per_cache": {}}

    total_problem_weight = sum(int(item["n_problems"]) for item in cache_metrics.values())
    total_sample_weight = sum(int(item["n_samples"]) for item in cache_metrics.values())
    total_top10_weight = sum(int(item.get("top10_count", max(1, int(math.ceil(0.10 * int(item["n_samples"])))))) for item in cache_metrics.values())

    sample_weighted = {
        "hit@1": float(sum(float(item["hit@1"]) * int(item["n_problems"]) for item in cache_metrics.values()) / max(total_problem_weight, 1)),
        "hit@3": float(sum(float(item["hit@3"]) * int(item["n_problems"]) for item in cache_metrics.values()) / max(total_problem_weight, 1)),
        "pairwise": float(sum(float(item["pairwise"]) * int(item["n_samples"]) for item in cache_metrics.values()) / max(total_sample_weight, 1)),
        "selacc@10%": float(
            sum(
                float(item["selacc@10%"]) * int(item.get("top10_count", max(1, int(math.ceil(0.10 * int(item["n_samples"]))))))
                for item in cache_metrics.values()
            )
            / max(total_top10_weight, 1)
        ),
        "n_problems": int(total_problem_weight),
        "n_samples": int(total_sample_weight),
        "top10_count": int(total_top10_weight),
    }
    equal_cache_mean = {
        key: float(np.mean([float(item[key]) for item in cache_metrics.values()]))
        for key in ("hit@1", "hit@3", "pairwise", "selacc@10%")
    }
    return {
        "sample_weighted": sample_weighted,
        "equal_cache_mean": equal_cache_mean,
        "per_cache": cache_metrics,
    }


def _build_code_v2_records(cache_root: Path, *, distance_threads: int, prefix_window_tokens: int) -> list[ProblemScoreRecord]:
    correctness = _load_ground_truth(cache_root)
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for sample_id, sample in enumerate(meta["samples"]):
        groups[str(sample["problem_id"])].append(int(sample_id))

    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
    records: list[ProblemScoreRecord] = []

    for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
        run_ids = list(map(int, run_ids))
        run_states = []
        run_views = []
        for run_id in run_ids:
            tv = reader.get_token_view(int(run_id))
            run_states.append(prepare_code_dynamic_run_state(reader, int(run_id), token_view=tv))
            run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))
        D = engine.dense_matrix(run_views)
        run_summaries = [
            _precompute_run_summary(
                run_state,
                reflection_thresholds=[0.30],
                reflection_lookbacks=[16],
                prefix_fractions=[0.30],
                prefix_window_tokens=int(prefix_window_tokens),
            )
            for run_state in run_states
        ]
        derived_rows = [
            _compute_derived_trace_stats(run_state, prefix_fraction=0.30)
            for run_state in run_states
        ]
        raw, _ = _build_code_raw_from_precomputed(
            run_summaries,
            reflection_threshold=0.30,
            reflection_lookback_slices=16,
            prefix_fraction=0.30,
        )
        raw["last_block_instability"] = np.asarray(
            [float(row["last_block_instability_score"]) for row in derived_rows],
            dtype=np.float64,
        )
        scores, _ = compute_code_v2_primary_scores_from_raw(raw)
        order = order_code_dynamic_group_indices(scores, D, run_ids=run_ids)
        rank_scores = _order_to_rank_scores(order, len(run_ids))
        labels = np.asarray([int(bool(correctness[int(run_id)])) for run_id in run_ids], dtype=np.int32)
        records.append(
            ProblemScoreRecord(
                cache_key="DS-R1/lcb_v5",
                problem_id=str(problem_id),
                sample_ids=run_ids,
                labels=labels,
                scores=rank_scores,
            )
        )
    return records


def _round2_sanity(all_problems: list[_ProblemData]) -> dict[str, Any]:
    recency_corr_rows = []
    for prob in all_problems:
        recency_r = np.asarray(prob.X[:, 4], dtype=np.float64)
        margin_r = build_gpqa_pairwise_margin_feature(np.asarray(prob.sci_raw["recency_conf_mean"], dtype=np.float64))
        dominance_r = build_gpqa_pairwise_dominance_feature(np.asarray(prob.sci_raw["recency_conf_mean"], dtype=np.float64))
        recency_corr_rows.append(
            {
                "problem_id": str(prob.problem_id),
                "margin_pearson": _pearson(recency_r, margin_r),
                "margin_spearman": _spearman(recency_r, margin_r),
                "dominance_pearson": _pearson(recency_r, dominance_r),
                "dominance_spearman": _spearman(recency_r, dominance_r),
            }
        )

    def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        if not vals:
            return None
        return float(np.mean(vals))

    base_cfg = {
        "name": "round1_mean",
        "include_margin": False,
        "include_dominance": False,
        "C": 1.0,
    }
    base_orders = _fit_final_variant_orders(all_problems, base_cfg)
    variant_consistency = []
    for cfg in _variant_configs():
        variant_orders = _fit_final_variant_orders(all_problems, cfg)
        top1 = []
        exact = []
        for prob in all_problems:
            problem_id = str(prob.problem_id)
            ref = np.asarray(base_orders[problem_id], dtype=np.int64)
            cur = np.asarray(variant_orders[problem_id], dtype=np.int64)
            top1.append(int(int(ref[0]) == int(cur[0])))
            exact.append(int(np.array_equal(ref, cur)))
        variant_consistency.append(
            {
                "name": str(cfg["name"]),
                "top1_agreement_vs_round1_mean": float(np.mean(top1)) if top1 else 0.0,
                "exact_order_match_vs_round1_mean": float(np.mean(exact)) if exact else 0.0,
            }
        )

    return {
        "recency_feature_correlation": {
            "mean_margin_pearson": _mean_key(recency_corr_rows, "margin_pearson"),
            "mean_margin_spearman": _mean_key(recency_corr_rows, "margin_spearman"),
            "mean_dominance_pearson": _mean_key(recency_corr_rows, "dominance_pearson"),
            "mean_dominance_spearman": _mean_key(recency_corr_rows, "dominance_spearman"),
            "per_problem": recency_corr_rows,
        },
        "variant_order_consistency": variant_consistency,
    }


def _lopo_prob_matrices(all_problems: list[_ProblemData]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    n_total = len(all_problems)
    for held_out_idx, held_out in enumerate(all_problems):
        if held_out_idx % 20 == 0:
            print(f"[science-round3:lopo] fold {held_out_idx + 1}/{n_total}", flush=True)
        pair_X_list: list[np.ndarray] = []
        pair_y_list: list[np.ndarray] = []
        for idx, prob in enumerate(all_problems):
            if idx == held_out_idx:
                continue
            X_pairs, y_pairs = build_pairwise_training_pairs(prob.X, prob.labels)
            if X_pairs.shape[0] > 0:
                pair_X_list.append(X_pairs)
                pair_y_list.append(y_pairs)
        scorer = GPQAPairwiseScorer()
        scorer.fit(np.concatenate(pair_X_list, axis=0), np.concatenate(pair_y_list, axis=0))
        out.append(compute_pairwise_probability_matrix(scorer, held_out.X))
    return out


def _disagreement_sanity(
    all_problems: list[_ProblemData],
    baseline_scores_by_problem: dict[str, np.ndarray],
    baseline_gate_scores_by_problem: dict[str, np.ndarray],
    prob_matrices: list[np.ndarray],
) -> dict[str, Any]:
    same = []
    baseline_correct = []
    pairwise_correct = []
    disagree_baseline_correct = []
    disagree_pairwise_correct = []
    agree_gaps = []
    disagree_gaps = []

    for prob, prob_matrix in zip(all_problems, prob_matrices):
        baseline_scores = np.asarray(baseline_scores_by_problem[str(prob.problem_id)], dtype=np.float64)
        baseline_gate_scores = np.asarray(baseline_gate_scores_by_problem[str(prob.problem_id)], dtype=np.float64)
        baseline_order = order_code_dynamic_group_indices(baseline_scores, prob.D, run_ids=prob.run_ids)
        baseline_best = int(baseline_order[0])
        baseline_gap = (
            float(baseline_gate_scores[int(baseline_order[0])] - baseline_gate_scores[int(baseline_order[1])])
            if baseline_order.size > 1
            else 0.0
        )
        pairwise_scores = compute_pairwise_backend_scores_from_matrix(prob_matrix, backend="mean", temperature=1.0)
        pairwise_order = order_code_dynamic_group_indices(pairwise_scores, prob.D, run_ids=prob.run_ids)
        pairwise_best = int(pairwise_order[0])
        base_ok = int(prob.labels[baseline_best] > 0)
        pair_ok = int(prob.labels[pairwise_best] > 0)
        baseline_correct.append(base_ok)
        pairwise_correct.append(pair_ok)
        is_same = int(baseline_best == pairwise_best)
        same.append(is_same)
        if is_same:
            agree_gaps.append(baseline_gap)
        else:
            disagree_gaps.append(baseline_gap)
            disagree_baseline_correct.append(base_ok)
            disagree_pairwise_correct.append(pair_ok)

    def _summ(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None}
        arr = np.asarray(values, dtype=np.float64)
        return {"mean": float(np.mean(arr)), "median": float(np.median(arr))}

    return {
        "top1_agreement": float(np.mean(same)) if same else 0.0,
        "baseline_top1_correct_rate": float(np.mean(baseline_correct)) if baseline_correct else 0.0,
        "pairwise_top1_correct_rate": float(np.mean(pairwise_correct)) if pairwise_correct else 0.0,
        "disagreement_rate": 1.0 - (float(np.mean(same)) if same else 0.0),
        "disagreement_baseline_correct_rate": float(np.mean(disagree_baseline_correct)) if disagree_baseline_correct else None,
        "disagreement_pairwise_correct_rate": float(np.mean(disagree_pairwise_correct)) if disagree_pairwise_correct else None,
        "baseline_gap_when_agree": _summ(agree_gaps),
        "baseline_gap_when_disagree": _summ(disagree_gaps),
    }


def _tau_grid_from_baseline_gaps(
    all_problems: list[_ProblemData],
    baseline_scores_by_problem: dict[str, np.ndarray],
    baseline_gate_scores_by_problem: dict[str, np.ndarray],
) -> list[float]:
    gaps = []
    for prob in all_problems:
        scores = np.asarray(baseline_scores_by_problem[str(prob.problem_id)], dtype=np.float64)
        gate_scores = np.asarray(baseline_gate_scores_by_problem[str(prob.problem_id)], dtype=np.float64)
        order = order_code_dynamic_group_indices(scores, prob.D, run_ids=prob.run_ids)
        gap = float(gate_scores[int(order[0])] - gate_scores[int(order[1])]) if order.size > 1 else 0.0
        gaps.append(gap)
    quantiles = np.quantile(np.asarray(gaps, dtype=np.float64), [0.20, 0.35, 0.50, 0.65, 0.80])
    return [float(x) for x in sorted({round(float(q), 12) for q in quantiles.tolist()})]


def _hybrid_candidate_configs(backend_name: str, temperature: float, tau_grid: list[float]) -> list[ScienceHybridConfig]:
    configs: list[ScienceHybridConfig] = []
    for tau in tau_grid:
        for k in SHORTLIST_KS:
            configs.append(ScienceHybridConfig(family="margin_fallback", backend=backend_name, tau=float(tau), k=int(k), alpha=0.50, m=0.0, temperature=float(temperature)))
    for k in SHORTLIST_KS:
        for alpha in BLEND_ALPHAS:
            configs.append(ScienceHybridConfig(family="shortlist_blend", backend=backend_name, tau=0.0, k=int(k), alpha=float(alpha), m=0.0, temperature=float(temperature)))
    for tau in tau_grid:
        for margin in OVERRIDE_MS:
            configs.append(ScienceHybridConfig(family="hard_override", backend=backend_name, tau=float(tau), k=3, alpha=0.50, m=float(margin), temperature=float(temperature)))
    return configs


def _system_delta(candidate_bundle: dict[str, Any], base_bundle: dict[str, Any]) -> dict[str, Any]:
    main = candidate_bundle["sample_weighted"]
    base_main = base_bundle["sample_weighted"]
    eq = candidate_bundle["equal_cache_mean"]
    base_eq = base_bundle["equal_cache_mean"]
    return {
        "sample_weighted": {
            "hit@1": float(_to_float(main.get("hit@1")) - _to_float(base_main.get("hit@1"))),
            "hit@3": float(_to_float(main.get("hit@3")) - _to_float(base_main.get("hit@3"))),
            "pairwise": float(_to_float(main.get("pairwise")) - _to_float(base_main.get("pairwise"))),
            "selacc@10%": float(_to_float(main.get("selacc@10%")) - _to_float(base_main.get("selacc@10%"))),
            "avg_rank_proxy": float(_to_float(main.get("avg_rank_proxy")) - _to_float(base_main.get("avg_rank_proxy"))),
        },
        "equal_cache_mean": {
            "hit@1": float(_to_float(eq.get("hit@1")) - _to_float(base_eq.get("hit@1"))),
            "hit@3": float(_to_float(eq.get("hit@3")) - _to_float(base_eq.get("hit@3"))),
            "pairwise": float(_to_float(eq.get("pairwise")) - _to_float(base_eq.get("pairwise"))),
            "selacc@10%": float(_to_float(eq.get("selacc@10%")) - _to_float(base_eq.get("selacc@10%"))),
            "avg_rank_proxy": float(_to_float(eq.get("avg_rank_proxy")) - _to_float(base_eq.get("avg_rank_proxy"))),
        },
    }


def _comprehensive_gate(
    candidate_bundle: dict[str, Any],
    current_bundle: dict[str, Any],
    *,
    delta_override: dict[str, Any] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    delta = delta_override if delta_override is not None else _system_delta(candidate_bundle, current_bundle)
    main = delta["sample_weighted"]
    eq = delta["equal_cache_mean"]
    failed: list[str] = []
    main_hit = float(main["hit@1"])
    main_sel = float(main["selacc@10%"])
    main_avg_rank = float(main["avg_rank_proxy"])
    if not (((main_hit > 0.0) and (main_sel >= -0.001)) or ((main_sel > 0.0) and (main_hit >= -0.001))):
        failed.append("sample_weighted Hit@1/SelAcc@10 did not produce a guarded improvement")
    if main_avg_rank > 1e-12:
        failed.append("sample_weighted AvgRank proxy got worse")
    if float(eq["hit@1"]) < 0.0 and float(eq["selacc@10%"]) < 0.0:
        failed.append("equal-cache mean Hit@1 and SelAcc@10 both regressed")
    return (len(failed) == 0, failed, delta)


def _science_gate(candidate_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failed: list[str] = []
    if float(_to_float(candidate_metrics.get("hit@1"))) < float(_to_float(baseline_metrics.get("hit@1"))):
        failed.append("Hit@1 below science_baseline_v1")
    if float(_to_float(candidate_metrics.get("selacc@10%"))) < float(_to_float(baseline_metrics.get("selacc@10%"))):
        failed.append("SelAcc@10 below science_baseline_v1")
    return (len(failed) == 0, failed)


def _choose_backend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mean_row = next(row for row in rows if row["name"] == "mean")
    best = max(
        rows,
        key=lambda row: (
            float(_to_float(row["metrics"].get("hit@1"))),
            float(_to_float(row["metrics"].get("selacc@10%"))),
            float(_to_float(row["metrics"].get("pairwise"))),
            row["name"],
        ),
    )
    best_tuple = (
        float(_to_float(best["metrics"].get("hit@1"))),
        float(_to_float(best["metrics"].get("selacc@10%"))),
        float(_to_float(best["metrics"].get("pairwise"))),
    )
    mean_tuple = (
        float(_to_float(mean_row["metrics"].get("hit@1"))),
        float(_to_float(mean_row["metrics"].get("selacc@10%"))),
        float(_to_float(mean_row["metrics"].get("pairwise"))),
    )
    if best_tuple > mean_tuple:
        return best
    return mean_row


def _build_blind_shape_report(
    blind_entry_map: dict[str, Any],
    *,
    config: ScienceHybridConfig,
    distance_threads: int,
    model_path: Path,
) -> dict[str, Any]:
    scorer = GPQAPairwiseScorer.load(model_path)
    outputs = {}
    for cache_key in BLIND_GPQAS:
        entry = blind_entry_map[cache_key]
        cache_root = Path(entry.cache_root)
        meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
        groups = defaultdict(list)
        for sample_id, sample in enumerate(meta["samples"]):
            groups[str(sample["problem_id"])].append(int(sample_id))
        reader = CacheReader(str(cache_root))
        engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
        rows = {"science_baseline_v1": {}, "gpqa_pairwise_round1": {}, "science_hybrid_round3": {}}
        trigger_flags = []
        for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
            run_ids = list(map(int, run_ids))
            views = [reader.get_run_view(int(run_id), DEFAULT_VIEW) for run_id in run_ids]
            D = engine.dense_matrix(views)
            ctx = SelectorContext(cache=reader, problem_id=str(problem_id), run_ids=run_ids, views=views)
            raw = extract_gpqa_pairwise_raw(ctx)
            baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(
                {
                    "prefix_conf_mean": raw["prefix_conf_mean"],
                    "recency_conf_mean": raw["recency_conf_mean"],
                    "late_worst_window": raw["late_worst_window"],
                    "late_recovery": raw["late_recovery"],
                },
                weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
            )
            X = build_gpqa_pairwise_features_configurable(
                raw,
                include_margin=bool(getattr(scorer, "include_margin", False)),
                include_dominance=bool(getattr(scorer, "include_dominance", False)),
            )
            prob_matrix = compute_pairwise_probability_matrix(scorer, X)
            pairwise_scores = compute_pairwise_backend_scores_from_matrix(prob_matrix, backend="mean", temperature=1.0)
            decision = compute_science_hybrid_decision(
                baseline_scores,
                prob_matrix,
                D,
                run_ids=run_ids,
                config=config,
            )
            trigger_flags.append(int(decision.triggered))
            topk = max(1, int(math.ceil(0.10 * len(run_ids))))
            baseline_order = order_code_dynamic_group_indices(baseline_scores, D, run_ids=run_ids)
            pairwise_order = order_code_dynamic_group_indices(pairwise_scores, D, run_ids=run_ids)
            for name, order in (
                ("science_baseline_v1", baseline_order),
                ("gpqa_pairwise_round1", pairwise_order),
                ("science_hybrid_round3", decision.hybrid_order),
            ):
                rows[name][str(problem_id)] = {
                    "best_run_id": int(run_ids[int(order[0])]),
                    "topk_run_ids": [int(run_ids[int(idx)]) for idx in order[:topk].tolist()],
                }
        compare = {}
        for other in ("science_baseline_v1", "gpqa_pairwise_round1"):
            top1 = []
            topk = []
            for problem_id in rows["science_hybrid_round3"]:
                base = rows["science_hybrid_round3"][problem_id]
                comp = rows[other][problem_id]
                top1.append(int(base["best_run_id"] == comp["best_run_id"]))
                base_topk = set(base["topk_run_ids"])
                comp_topk = set(comp["topk_run_ids"])
                topk.append(len(base_topk & comp_topk) / max(len(base_topk | comp_topk), 1))
            compare[other] = {
                "top1_agreement": float(np.mean(top1)) if top1 else 0.0,
                "topk_jaccard": float(np.mean(topk)) if topk else 0.0,
            }
        outputs[cache_key] = {
            "compare": compare,
            "trigger_rate": float(np.mean(trigger_flags)) if trigger_flags else 0.0,
        }
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run science round-3 hybrid GPQA + comprehensive proxy evaluation")
    ap.add_argument("--gpqa-cache-root", default="")
    ap.add_argument("--overall-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--distance-threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefix-window-tokens", type=int, default=128)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--skip-blind-shapes", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"science_hybrid_round3_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_entry_map = _load_entry_map(args.overall_cache_root)
    gpqa_cache_root = Path(args.gpqa_cache_root) if args.gpqa_cache_root else Path(overall_entry_map["DS-R1/gpqa"].cache_root)

    print("[science-round3] Extracting GPQA problems …", flush=True)
    all_problems = _extract_all_problems(
        gpqa_cache_root,
        distance_threads=int(args.distance_threads),
        workers=int(args.workers),
    )

    baseline_acc = MetricAccumulator("science_baseline_v1", use_code_tiebreak=True)
    copeland_acc = MetricAccumulator("tournament-copeland", use_code_tiebreak=True)
    baseline_scores_by_problem: dict[str, np.ndarray] = {}
    baseline_gate_scores_by_problem: dict[str, np.ndarray] = {}
    baseline_records: list[ProblemScoreRecord] = []
    pairwise_round1_records: list[ProblemScoreRecord] = []
    for prob in all_problems:
        baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(
            prob.sci_raw,
            weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
        )
        baseline_scores = np.asarray(baseline_scores, dtype=np.float64)
        baseline_scores_by_problem[str(prob.problem_id)] = baseline_scores
        baseline_gate_scores_by_problem[str(prob.problem_id)] = np.asarray(prob.sci_raw["recency_conf_mean"], dtype=np.float64)
        copeland_scores = _compute_copeland_scores(prob.D)
        baseline_acc.add_problem(prob.problem_id, prob.run_ids, baseline_scores, prob.labels, prob.D)
        copeland_acc.add_problem(prob.problem_id, prob.run_ids, copeland_scores, prob.labels, prob.D)
        baseline_records.append(
            ProblemScoreRecord(
                cache_key="DS-R1/gpqa",
                problem_id=str(prob.problem_id),
                sample_ids=list(map(int, prob.run_ids)),
                labels=np.asarray(prob.labels, dtype=np.int32),
                scores=_order_to_rank_scores(order_code_dynamic_group_indices(baseline_scores, prob.D, run_ids=prob.run_ids), len(prob.run_ids)),
            )
        )

    print("[science-round3] Running LOPO pairwise probability extraction …", flush=True)
    prob_matrices = _lopo_prob_matrices(all_problems)

    round2_sanity = _round2_sanity(all_problems)
    disagreement_sanity = _disagreement_sanity(
        all_problems,
        baseline_scores_by_problem,
        baseline_gate_scores_by_problem,
        prob_matrices,
    )

    backend_rows = []
    for backend_cfg in PAIRWISE_BACKEND_GRID:
        metrics = _build_pairwise_backend_acc(
            backend_cfg["name"],
            all_problems,
            prob_matrices,
            backend=str(backend_cfg["backend"]),
            temperature=float(backend_cfg["temperature"]),
        )
        backend_rows.append({**backend_cfg, "metrics": _compact_metrics(metrics)})
    chosen_backend = _choose_backend(backend_rows)

    for prob, prob_matrix in zip(all_problems, prob_matrices):
        mean_scores = compute_pairwise_backend_scores_from_matrix(prob_matrix, backend="mean", temperature=1.0)
        pairwise_round1_records.append(
            ProblemScoreRecord(
                cache_key="DS-R1/gpqa",
                problem_id=str(prob.problem_id),
                sample_ids=list(map(int, prob.run_ids)),
                labels=np.asarray(prob.labels, dtype=np.int32),
                scores=np.asarray(mean_scores, dtype=np.float64),
            )
        )

    tau_grid = _tau_grid_from_baseline_gaps(
        all_problems,
        baseline_scores_by_problem,
        baseline_gate_scores_by_problem,
    )
    candidate_configs = _hybrid_candidate_configs(str(chosen_backend["backend"]), float(chosen_backend["temperature"]), tau_grid)

    candidate_rows: list[dict[str, Any]] = []
    candidate_gpqa_records: dict[str, list[ProblemScoreRecord]] = {}
    gpqa_metrics_baseline = baseline_acc.finalize()
    gpqa_metrics_copeland = copeland_acc.finalize()
    gpqa_metrics_pairwise_round1 = _evaluate_problem_records(pairwise_round1_records)

    for config in candidate_configs:
        name = _candidate_name(config)
        acc = MetricAccumulator(name, use_code_tiebreak=True)
        records: list[ProblemScoreRecord] = []
        triggered = 0
        overridden = 0
        top1_changed = 0
        for prob, prob_matrix in zip(all_problems, prob_matrices):
            decision = compute_science_hybrid_decision(
                baseline_scores_by_problem[str(prob.problem_id)],
                prob_matrix,
                prob.D,
                run_ids=prob.run_ids,
                baseline_gate_scores=baseline_gate_scores_by_problem[str(prob.problem_id)],
                config=config,
            )
            acc.add_problem(prob.problem_id, prob.run_ids, decision.hybrid_scores, prob.labels, prob.D)
            records.append(
                ProblemScoreRecord(
                    cache_key="DS-R1/gpqa",
                    problem_id=str(prob.problem_id),
                    sample_ids=list(map(int, prob.run_ids)),
                    labels=np.asarray(prob.labels, dtype=np.int32),
                    scores=np.asarray(decision.hybrid_scores, dtype=np.float64),
                )
            )
            triggered += int(decision.triggered)
            overridden += int(decision.overridden)
            top1_changed += int(int(decision.baseline_order[0]) != int(decision.hybrid_order[0]))
        metrics = acc.finalize()
        gpqa_proxy_metrics = _evaluate_problem_records(records)
        passed_science_gate, science_failed = _science_gate(metrics, gpqa_metrics_baseline)
        candidate_rows.append(
            {
                "name": name,
                "config": config.as_dict(),
                "metrics": _compact_metrics(metrics),
                "gpqa_proxy_metrics": gpqa_proxy_metrics,
                "trigger_rate": float(triggered / len(all_problems)),
                "override_rate": float(overridden / len(all_problems)),
                "top1_change_rate": float(top1_changed / len(all_problems)),
                "science_gate_passed": bool(passed_science_gate),
                "science_gate_failed": science_failed,
            }
        )
        candidate_gpqa_records[name] = records

    print("[science-round3] Building transparent cache-level comprehensive proxy …", flush=True)
    base_doc_metrics = _load_extreme12_test_metrics(EXTREME12_TEST_ANALYSIS_DOC)
    problem_counts = _problem_counts_from_entry_map(overall_entry_map, OVERALL_DS_CACHE_KEYS)
    for cache_key in OVERALL_DS_CACHE_KEYS:
        base_doc_metrics[cache_key]["n_problems"] = int(problem_counts[cache_key])
        base_doc_metrics[cache_key]["top10_count"] = max(1, int(math.ceil(0.10 * int(base_doc_metrics[cache_key]["n_samples"]))))

    print("[science-round3] Loading code_v2 proxy metrics for DS-R1/lcb_v5 …", flush=True)
    code_v2_proxy_metrics = _load_code_v2_proxy_metrics(
        CODE_V2_EXHAUSTIVE_JSON,
        fallback_hit3=float(base_doc_metrics["DS-R1/lcb_v5"]["hit@3"]),
    )
    science_baseline_proxy_metrics = _evaluate_problem_records(baseline_records)

    base_cache_metrics = {cache_key: dict(base_doc_metrics[cache_key]) for cache_key in OVERALL_DS_CACHE_KEYS}
    code_v2_cache_metrics = {cache_key: dict(base_doc_metrics[cache_key]) for cache_key in OVERALL_DS_CACHE_KEYS}
    code_v2_cache_metrics["DS-R1/lcb_v5"] = dict(code_v2_proxy_metrics)
    current_cache_metrics = {cache_key: dict(base_doc_metrics[cache_key]) for cache_key in OVERALL_DS_CACHE_KEYS}
    current_cache_metrics["DS-R1/lcb_v5"] = dict(code_v2_proxy_metrics)
    current_cache_metrics["DS-R1/gpqa"] = dict(science_baseline_proxy_metrics)

    system_metrics = {
        "extreme12_base": _combine_cache_metric_proxy(base_cache_metrics),
        "extreme12_plus_code_v2": _combine_cache_metric_proxy(code_v2_cache_metrics),
        "extreme12_plus_code_v2_plus_science_baseline_v1": _combine_cache_metric_proxy(current_cache_metrics),
    }
    current_system = system_metrics["extreme12_plus_code_v2_plus_science_baseline_v1"]
    total_problem_count = sum(problem_counts.values())
    gpqa_problem_count = int(problem_counts["DS-R1/gpqa"])
    n_system_caches = len(OVERALL_DS_CACHE_KEYS)
    current_gpqa_avg_rank = float(science_baseline_proxy_metrics["avg_rank_proxy"])

    for row in candidate_rows:
        cache_metrics = {cache_key: dict(base_doc_metrics[cache_key]) for cache_key in OVERALL_DS_CACHE_KEYS}
        cache_metrics["DS-R1/lcb_v5"] = dict(code_v2_proxy_metrics)
        cache_metrics["DS-R1/gpqa"] = dict(row["gpqa_proxy_metrics"])
        bundle = _combine_cache_metric_proxy(cache_metrics)
        gpqa_avg_rank_delta = float(_to_float(row["gpqa_proxy_metrics"]["avg_rank_proxy"]) - current_gpqa_avg_rank)
        delta = _system_delta(bundle, current_system)
        delta["sample_weighted"]["avg_rank_proxy"] = float(gpqa_avg_rank_delta * gpqa_problem_count / max(total_problem_count, 1))
        delta["equal_cache_mean"]["avg_rank_proxy"] = float(gpqa_avg_rank_delta / max(n_system_caches, 1))
        gate_passed, gate_failed, delta = _comprehensive_gate(bundle, current_system, delta_override=delta)
        row["comprehensive_metrics"] = bundle
        row["comprehensive_delta_vs_current"] = delta
        row["comprehensive_gate_passed"] = bool(gate_passed)
        row["comprehensive_gate_failed"] = gate_failed

    gpqa_summary_rows = [
        {"name": "science_baseline_v1", **_compact_metrics(gpqa_metrics_baseline)},
        {"name": "tournament-copeland", **_compact_metrics(gpqa_metrics_copeland)},
        {"name": "gpqa_pairwise_round1", **_compact_metrics(gpqa_metrics_pairwise_round1)},
        *[
            {"name": row["name"], **row["metrics"]}
            for row in candidate_rows
        ],
    ]

    promote_full_rows = [
        row for row in candidate_rows if row["science_gate_passed"] and row["comprehensive_gate_passed"]
    ]
    promote_science_only_rows = [
        row for row in candidate_rows if row["science_gate_passed"] and not row["comprehensive_gate_passed"]
    ]
    closest_candidate = max(
        candidate_rows,
        key=lambda row: (
            bool(row["comprehensive_gate_passed"]),
            bool(row["science_gate_passed"]),
            float(_to_float(row["comprehensive_delta_vs_current"]["sample_weighted"]["selacc@10%"])),
            float(_to_float(row["comprehensive_delta_vs_current"]["sample_weighted"]["hit@1"])),
            -float(_to_float(row["comprehensive_delta_vs_current"]["sample_weighted"]["avg_rank_proxy"])),
            float(_to_float(row["metrics"]["hit@1"])),
            float(_to_float(row["metrics"]["selacc@10%"])),
        ),
    )
    selected_candidate = None
    final_decision = "No-Promote"
    if promote_full_rows:
        selected_candidate = max(
            promote_full_rows,
            key=lambda row: (
                float(_to_float(row["comprehensive_metrics"]["sample_weighted"]["hit@1"])),
                float(_to_float(row["comprehensive_metrics"]["sample_weighted"]["selacc@10%"])),
                -float(_to_float(row["comprehensive_delta_vs_current"]["sample_weighted"]["avg_rank_proxy"])),
                float(_to_float(row["metrics"]["hit@1"])),
                row["name"],
            ),
        )
        final_decision = "Promote for full system"
    elif promote_science_only_rows:
        selected_candidate = max(
            promote_science_only_rows,
            key=lambda row: (
                float(_to_float(row["metrics"]["hit@1"])),
                float(_to_float(row["metrics"]["selacc@10%"])),
                float(_to_float(row["metrics"]["pairwise"])),
                row["name"],
            ),
        )
        final_decision = "Promote for science only"
    else:
        selected_candidate = closest_candidate

    blind_shapes = None
    if not args.skip_blind_shapes:
        print("[science-round3] Computing blind-shape report for selected candidate …", flush=True)
        blind_entry_map = _load_entry_map(args.blind_cache_root)
        blind_shapes = _build_blind_shape_report(
            blind_entry_map,
            config=ScienceHybridConfig(**selected_candidate["config"]),
            distance_threads=max(1, int(args.distance_threads)),
            model_path=default_gpqa_pairwise_model_path(),
        )

    payload = {
        "status_summary": {
            "science_baseline_v1_is_frozen_baseline": True,
            "gpqa_pairwise_round2_is_no_promote": True,
            "code_v2_is_promoted_default": True,
            "round3_priority_is_hybrid_rule_not_new_feature_family": True,
        },
        "inputs": {
            "gpqa_cache_root": str(gpqa_cache_root),
            "overall_cache_root": str(args.overall_cache_root),
            "blind_cache_root": str(args.blind_cache_root),
            "distance_threads": int(args.distance_threads),
            "workers": int(args.workers),
        },
        "comprehensive_proxy_definition": {
            "fixed_ds_r1_math_and_base_slices_from": str(EXTREME12_TEST_ANALYSIS_DOC),
            "code_v2_lcb_v5_proxy_from": str(CODE_V2_EXHAUSTIVE_JSON),
            "code_v2_hit3_fallback_from": str(EXTREME12_TEST_ANALYSIS_DOC),
            "mutable_science_slice": "DS-R1/gpqa",
            "sample_weighted": {
                "hit@1": "weighted by n_problems",
                "hit@3": "weighted by n_problems",
                "pairwise": "weighted by n_samples",
                "selacc@10%": "weighted by top10_count",
                "avg_rank_proxy": "delta-only from mutable GPQA slice",
            },
            "equal_cache_mean": "simple mean over DS-R1 caches",
        },
        "sanity_checks": {
            "round2_collinearity": round2_sanity,
            "baseline_pairwise_disagreement": disagreement_sanity,
        },
        "pairwise_backend_scan": {
            "rows": backend_rows,
            "selected": chosen_backend,
        },
        "gpqa_metrics": {
            "science_baseline_v1": _compact_metrics(gpqa_metrics_baseline),
            "tournament-copeland": _compact_metrics(gpqa_metrics_copeland),
            "gpqa_pairwise_round1": _compact_metrics(gpqa_metrics_pairwise_round1),
            "hybrid_candidates": candidate_rows,
        },
        "system_metrics": {
            "extreme12_base": system_metrics["extreme12_base"],
            "extreme12_plus_code_v2": system_metrics["extreme12_plus_code_v2"],
            "extreme12_plus_code_v2_plus_science_baseline_v1": current_system,
        },
        "selected_candidate": selected_candidate,
        "closest_candidate": closest_candidate,
        "decision": {
            "science_gate": {
                "passed_candidates": [row["name"] for row in candidate_rows if row["science_gate_passed"]],
            },
            "comprehensive_gate": {
                "passed_candidates": [row["name"] for row in candidate_rows if row["comprehensive_gate_passed"]],
            },
            "final_decision": final_decision,
        },
        "blind_shapes": blind_shapes,
    }
    (out_dir / "science_hybrid_round3.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "# Science Hybrid Round 3",
        "",
        "## GPQA Summary",
        "",
        _summarize_metrics_table(gpqa_summary_rows),
        "",
        "## Pairwise Backend Scan",
        "",
        _summarize_metrics_table(
            [
                {"name": row["name"], **row["metrics"]}
                for row in backend_rows
            ]
        ),
        "",
        f"- Selected pairwise backend: `{chosen_backend['name']}`",
        "",
        "## Sanity Checks",
        "",
        f"- Round-2 margin mean Spearman vs recency: `{_fmt_pct(round2_sanity['recency_feature_correlation']['mean_margin_spearman'])}`",
        f"- Round-2 dominance mean Spearman vs recency: `{_fmt_pct(round2_sanity['recency_feature_correlation']['mean_dominance_spearman'])}`",
        f"- Baseline/pairwise top1 agreement: `{_fmt_pct(disagreement_sanity['top1_agreement'])}`",
        f"- Disagreement baseline correct rate: `{_fmt_pct(disagreement_sanity['disagreement_baseline_correct_rate'])}`",
        f"- Disagreement pairwise correct rate: `{_fmt_pct(disagreement_sanity['disagreement_pairwise_correct_rate'])}`",
        "",
        "## Decision",
        "",
        f"- Final decision: `{final_decision}`",
        f"- Selected candidate: `{selected_candidate['name']}`",
        f"- Closest candidate: `{closest_candidate['name']}`",
    ]
    if blind_shapes is not None:
        summary_lines.extend([
            "",
            "## Blind Shapes",
            "",
        ])
        for cache_key, item in blind_shapes.items():
            summary_lines.append(f"- `{cache_key}`: `{item}`")
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "selected_candidate": selected_candidate["name"],
        "closest_candidate": closest_candidate["name"],
        "final_decision": final_decision,
        "selected_backend": chosen_backend["name"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
