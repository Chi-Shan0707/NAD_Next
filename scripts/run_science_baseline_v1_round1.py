#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.code_dynamic_impl import order_code_dynamic_group_indices
from nad.core.selectors.science_dynamic_impl import (
    DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    DEFAULT_SCIENCE_PREFIX_FRACTION,
    DEFAULT_SCIENCE_RECENCY_EXP,
    DEFAULT_SCIENCE_TAIL_FRACTION,
    DEFAULT_SCIENCE_WINDOW_TOKENS,
    SCIENCE_DYNAMIC_CANDIDATE_WEIGHTS,
    SCIENCE_DYNAMIC_WEIGHT_NAMES,
    compute_science_dynamic_primary_scores_from_raw,
    extract_science_dynamic_raw_from_state,
    prepare_science_dynamic_run_state,
)
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import build_problem_groups
from scripts.run_code_baseline_v1_phase2 import (
    DEFAULT_VIEW,
    MetricAccumulator,
    _compute_copeland_scores,
    _fmt_pct,
    _load_entry_map,
    _problem_sort_key,
)

GT_SCIENCE_KEY = "DS-R1/gpqa"
BLIND_SCIENCE_KEYS = ("DS-R1/gpqa", "Qwen3-4B/gpqa")
SCIENCE_COMPARE_NAMES = (
    "science_prefix_control",
    "science_recency_control",
    "science_baseline_v1",
    "science_candidate_round1",
    "tournament-copeland",
)


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _precompute_science_raw(
    run_states: list[dict[str, Any]],
    *,
    prefix_fraction: float,
    tail_fraction: float,
    recency_exp: float,
    window_tokens: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    n = len(run_states)
    raw = {
        "prefix_conf_mean": np.full(n, np.nan, dtype=np.float64),
        "recency_conf_mean": np.full(n, np.nan, dtype=np.float64),
        "late_worst_window": np.full(n, np.nan, dtype=np.float64),
        "late_recovery": np.full(n, np.nan, dtype=np.float64),
    }
    rows: list[dict[str, float]] = []
    for idx, run_state in enumerate(run_states):
        row = extract_science_dynamic_raw_from_state(
            run_state,
            prefix_fraction=prefix_fraction,
            tail_fraction=tail_fraction,
            recency_exp=recency_exp,
            window_tokens=window_tokens,
        )
        rows.append(row)
        for key in raw:
            raw[key][idx] = float(row[key])
    return raw, rows


def _raw_control_scores(raw: dict[str, np.ndarray], key: str) -> np.ndarray:
    values = np.asarray(raw[key], dtype=np.float64)
    ranked = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(ranked)
    if not valid.any() or valid.sum() == 1:
        return np.full(ranked.shape, 0.5, dtype=np.float64)
    filled = np.where(valid, ranked, float(np.median(ranked[valid])))
    order = np.argsort(np.argsort(filled))
    return np.asarray(order / max(len(order) - 1, 1), dtype=np.float64)


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


def _evaluate_science_gt(
    cache_root: Path,
    *,
    distance_threads: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    correctness = _load_ground_truth(cache_root)
    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))

    raw_params = {
        "prefix_fraction": DEFAULT_SCIENCE_PREFIX_FRACTION,
        "tail_fraction": DEFAULT_SCIENCE_TAIL_FRACTION,
        "recency_exp": DEFAULT_SCIENCE_RECENCY_EXP,
        "window_tokens": DEFAULT_SCIENCE_WINDOW_TOKENS,
    }
    compare_accs = {
        name: MetricAccumulator(name, use_code_tiebreak=True)
        for name in SCIENCE_COMPARE_NAMES
    }
    loo_accs = {
        "science_candidate_round1": MetricAccumulator("science_candidate_round1", use_code_tiebreak=True),
        **{
            f"loo_without__{feat_name}": MetricAccumulator(f"loo_without__{feat_name}", use_code_tiebreak=True)
            for feat_name in SCIENCE_DYNAMIC_WEIGHT_NAMES
        },
    }

    for idx, (problem_id, run_ids) in enumerate(sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))):
        if idx % 20 == 0:
            print(f"[science-gt] problem {idx + 1}/{len(groups)}", flush=True)
        run_ids = list(map(int, run_ids))
        run_states = []
        run_views = []
        labels = np.asarray([int(bool(correctness.get(int(run_id), False))) for run_id in run_ids], dtype=np.int32)
        for run_id in run_ids:
            tv = reader.get_token_view(int(run_id))
            run_states.append(prepare_science_dynamic_run_state(reader, int(run_id), token_view=tv))
            run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))
        D = engine.dense_matrix(run_views)
        copeland_scores = _compute_copeland_scores(D)
        raw, _ = _precompute_science_raw(run_states, **raw_params)

        prefix_scores = _raw_control_scores(raw, "prefix_conf_mean")
        recency_scores = _raw_control_scores(raw, "recency_conf_mean")
        baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(
            raw,
            weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
        )
        candidate_scores, _ = compute_science_dynamic_primary_scores_from_raw(
            raw,
            weights=SCIENCE_DYNAMIC_CANDIDATE_WEIGHTS,
        )

        compare_accs["science_prefix_control"].add_problem(str(problem_id), run_ids, prefix_scores, labels, D)
        compare_accs["science_recency_control"].add_problem(str(problem_id), run_ids, recency_scores, labels, D)
        compare_accs["science_baseline_v1"].add_problem(str(problem_id), run_ids, baseline_scores, labels, D)
        compare_accs["science_candidate_round1"].add_problem(str(problem_id), run_ids, candidate_scores, labels, D)
        compare_accs["tournament-copeland"].add_problem(str(problem_id), run_ids, copeland_scores, labels, D)

        loo_accs["science_candidate_round1"].add_problem(str(problem_id), run_ids, candidate_scores, labels, D)
        for feat_name in SCIENCE_DYNAMIC_WEIGHT_NAMES:
            scores, _ = compute_science_dynamic_primary_scores_from_raw(
                raw,
                weights=SCIENCE_DYNAMIC_CANDIDATE_WEIGHTS,
                disabled_features=[feat_name],
            )
            loo_accs[f"loo_without__{feat_name}"].add_problem(str(problem_id), run_ids, scores, labels, D)

    compare_metrics = {name: acc.finalize() for name, acc in compare_accs.items()}
    loo_metrics = {name: acc.finalize() for name, acc in loo_accs.items()}
    return compare_metrics, loo_metrics


def _evaluate_science_transfer_gt(
    gt_entry_map: dict[str, Any],
    *,
    distance_threads: int,
) -> dict[str, Any]:
    selector_rows = []
    for cache_key, entry in sorted(gt_entry_map.items()):
        print(f"[science-transfer] {cache_key}", flush=True)
        cache_root = Path(entry.cache_root)
        correctness = _load_ground_truth(cache_root)
        meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
        groups = build_problem_groups(meta)
        reader = CacheReader(str(cache_root))
        engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
        accs = {
            "science_baseline_v1": MetricAccumulator("science_baseline_v1", use_code_tiebreak=True),
            "science_candidate_round1": MetricAccumulator("science_candidate_round1", use_code_tiebreak=True),
            "tournament-copeland": MetricAccumulator("tournament-copeland", use_code_tiebreak=True),
        }
        for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
            run_ids = list(map(int, run_ids))
            run_states = []
            run_views = []
            labels = np.asarray([int(bool(correctness.get(int(run_id), False))) for run_id in run_ids], dtype=np.int32)
            for run_id in run_ids:
                tv = reader.get_token_view(int(run_id))
                run_states.append(prepare_science_dynamic_run_state(reader, int(run_id), token_view=tv))
                run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))
            D = engine.dense_matrix(run_views)
            copeland_scores = _compute_copeland_scores(D)
            raw, _ = _precompute_science_raw(
                run_states,
                prefix_fraction=DEFAULT_SCIENCE_PREFIX_FRACTION,
                tail_fraction=DEFAULT_SCIENCE_TAIL_FRACTION,
                recency_exp=DEFAULT_SCIENCE_RECENCY_EXP,
                window_tokens=DEFAULT_SCIENCE_WINDOW_TOKENS,
            )
            baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(
                raw,
                weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
            )
            candidate_scores, _ = compute_science_dynamic_primary_scores_from_raw(
                raw,
                weights=SCIENCE_DYNAMIC_CANDIDATE_WEIGHTS,
            )
            accs["science_baseline_v1"].add_problem(str(problem_id), run_ids, baseline_scores, labels, D)
            accs["science_candidate_round1"].add_problem(str(problem_id), run_ids, candidate_scores, labels, D)
            accs["tournament-copeland"].add_problem(str(problem_id), run_ids, copeland_scores, labels, D)

        finalized = {name: acc.finalize() for name, acc in accs.items()}
        dataset_name = str(getattr(entry, "dataset_name", cache_key.split("/", 1)[-1]))
        domain = "science" if dataset_name == "gpqa" else "non_science"
        for name, metrics in finalized.items():
            selector_rows.append({
                "cache_key": cache_key,
                "dataset": dataset_name,
                "domain": domain,
                "selector": name,
                "auroc": metrics["auroc"],
                "hit@1": metrics["hit@1"],
                "pairwise": metrics["pairwise"],
                "selacc@10%": metrics["selacc@10%"],
            })

    def _domain_mean(selector: str, domain: str, metric: str) -> float | None:
        values = [
            float(row[metric])
            for row in selector_rows
            if row["selector"] == selector and row["domain"] == domain and row[metric] is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    transfer_2x2 = []
    for selector in ("science_baseline_v1", "science_candidate_round1", "tournament-copeland"):
        transfer_2x2.append({
            "selector": selector,
            "science_hit@1": _domain_mean(selector, "science", "hit@1"),
            "science_pairwise": _domain_mean(selector, "science", "pairwise"),
            "science_selacc@10%": _domain_mean(selector, "science", "selacc@10%"),
            "non_science_selacc@10%": _domain_mean(selector, "non_science", "selacc@10%"),
        })

    return {
        "rows": selector_rows,
        "transfer_2x2": transfer_2x2,
    }


def _evaluate_blind_science_shapes(
    blind_entry_map: dict[str, Any],
    *,
    distance_threads: int,
) -> dict[str, Any]:
    outputs = {}
    for cache_key in BLIND_SCIENCE_KEYS:
        entry = blind_entry_map[cache_key]
        print(f"[science-blind] {cache_key}", flush=True)
        cache_root = Path(entry.cache_root)
        meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
        groups = build_problem_groups(meta)
        reader = CacheReader(str(cache_root))
        engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
        selector_problem = {name: {} for name in ("science_baseline_v1", "science_candidate_round1", "tournament-copeland")}
        feature_rows = []
        for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
            run_ids = list(map(int, run_ids))
            run_states = []
            run_views = []
            for run_id in run_ids:
                tv = reader.get_token_view(int(run_id))
                run_states.append(prepare_science_dynamic_run_state(reader, int(run_id), token_view=tv))
                run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))
            D = engine.dense_matrix(run_views)
            copeland_scores = _compute_copeland_scores(D)
            raw, raw_rows = _precompute_science_raw(
                run_states,
                prefix_fraction=DEFAULT_SCIENCE_PREFIX_FRACTION,
                tail_fraction=DEFAULT_SCIENCE_TAIL_FRACTION,
                recency_exp=DEFAULT_SCIENCE_RECENCY_EXP,
                window_tokens=DEFAULT_SCIENCE_WINDOW_TOKENS,
            )
            baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(raw, weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS)
            candidate_scores, _ = compute_science_dynamic_primary_scores_from_raw(raw, weights=SCIENCE_DYNAMIC_CANDIDATE_WEIGHTS)
            order_map = {
                "science_baseline_v1": order_code_dynamic_group_indices(baseline_scores, D, run_ids=run_ids),
                "science_candidate_round1": order_code_dynamic_group_indices(candidate_scores, D, run_ids=run_ids),
                "tournament-copeland": order_code_dynamic_group_indices(copeland_scores, D, run_ids=run_ids),
            }
            topk = max(1, int(math.ceil(0.10 * len(run_ids))))
            for selector_name, order in order_map.items():
                selector_problem[selector_name][str(problem_id)] = {
                    "best_run_id": int(run_ids[int(order[0])]),
                    "topk_run_ids": [int(run_ids[int(idx)]) for idx in order[:topk].tolist()],
                }
            best_idx = int(order_map["science_baseline_v1"][0])
            feature_rows.append({
                "problem_id": str(problem_id),
                "prefix_conf_mean": float(raw_rows[best_idx]["prefix_conf_mean"]),
                "recency_conf_mean": float(raw_rows[best_idx]["recency_conf_mean"]),
                "late_worst_window": float(raw_rows[best_idx]["late_worst_window"]),
                "late_recovery": float(raw_rows[best_idx]["late_recovery"]),
            })

        compare = {}
        for other in ("science_candidate_round1", "tournament-copeland"):
            top1 = []
            topk = []
            for problem_id in selector_problem["science_baseline_v1"]:
                base = selector_problem["science_baseline_v1"][problem_id]
                comp = selector_problem[other][problem_id]
                top1.append(int(base["best_run_id"] == comp["best_run_id"]))
                base_topk = set(base["topk_run_ids"])
                comp_topk = set(comp["topk_run_ids"])
                topk.append(len(base_topk & comp_topk) / max(len(base_topk | comp_topk), 1))
            compare[other] = {
                "top1_agreement": float(np.mean(top1)) if top1 else 0.0,
                "topk_jaccard": float(np.mean(topk)) if topk else 0.0,
            }
        feature_means = {
            key: float(np.mean([row[key] for row in feature_rows])) if feature_rows else 0.0
            for key in ("prefix_conf_mean", "recency_conf_mean", "late_worst_window", "late_recovery")
        }
        outputs[cache_key] = {
            "compare": compare,
            "feature_means": feature_means,
        }
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run science_baseline_v1 round-1 evaluation")
    ap.add_argument("--gt-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"science_baseline_v1_round1_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_entry_map = _load_entry_map(args.gt_cache_root)
    blind_entry_map = _load_entry_map(args.blind_cache_root)

    science_entry = gt_entry_map[GT_SCIENCE_KEY]
    compare_metrics, loo_metrics = _evaluate_science_gt(
        Path(science_entry.cache_root),
        distance_threads=int(args.distance_threads),
    )
    transfer_gt = _evaluate_science_transfer_gt(
        gt_entry_map,
        distance_threads=int(args.distance_threads),
    )
    blind_shapes = _evaluate_blind_science_shapes(
        blind_entry_map,
        distance_threads=int(args.distance_threads),
    )

    baseline = compare_metrics["science_baseline_v1"]
    candidate = compare_metrics["science_candidate_round1"]
    promote_candidate = bool(
        float(candidate["selacc@10%"]) >= float(baseline["selacc@10%"]) + 0.01
        and float(candidate["pairwise"] or 0.0) >= float(baseline["pairwise"] or 0.0) - 0.005
        and float(candidate["hit@1"] or 0.0) >= float(baseline["hit@1"] or 0.0) - 0.01
    )

    compare_rows = [{"name": name, **{k: metrics[k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}} for name, metrics in compare_metrics.items()]
    loo_rows = [{"name": name, **{k: metrics[k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}} for name, metrics in loo_metrics.items()]
    compare_rows.sort(key=lambda row: (-(row["selacc@10%"] or 0.0), -(row["pairwise"] or 0.0), row["name"]))
    loo_rows.sort(key=lambda row: (0 if row["name"] == "science_candidate_round1" else 1, -(row["selacc@10%"] or 0.0), row["name"]))

    payload = {
        "compare_metrics": compare_metrics,
        "loo_metrics": loo_metrics,
        "transfer_gt": transfer_gt,
        "blind_shapes": blind_shapes,
        "gate": {
            "promote_candidate": promote_candidate,
            "baseline_selector": "science_baseline_v1",
            "candidate_selector": "science_candidate_round1",
        },
    }
    (out_dir / "science_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = [
        "# Science Baseline v1 Round 1",
        "",
        "## Comparison",
        "",
        _summarize_metrics_table(compare_rows),
        "",
        "## Candidate Leave-One-Out",
        "",
        _summarize_metrics_table(loo_rows),
        "",
        "## Transfer Gate",
        "",
        "| Selector | Science Hit@1 | Science Pairwise | Science SelAcc@10 | Non-science SelAcc@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in transfer_gt["transfer_2x2"]:
        summary_md.append(
            "| {selector} | {science_hit1} | {science_pairwise} | {science_selacc} | {non_science_selacc} |".format(
                selector=row["selector"],
                science_hit1=_fmt_pct(row["science_hit@1"]),
                science_pairwise=_fmt_pct(row["science_pairwise"]),
                science_selacc=_fmt_pct(row["science_selacc@10%"]),
                non_science_selacc=_fmt_pct(row["non_science_selacc@10%"]),
            )
        )
    summary_md.extend([
        "",
        "## Blind GPQA Checks",
        "",
        f"- Promote candidate: `{promote_candidate}`",
    ])
    for cache_key, item in blind_shapes.items():
        summary_md.append(f"- `{cache_key}`: `{item['compare']}`")
    (out_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "promote_candidate": promote_candidate,
        "baseline_selacc10": baseline["selacc@10%"],
        "candidate_selacc10": candidate["selacc@10%"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
