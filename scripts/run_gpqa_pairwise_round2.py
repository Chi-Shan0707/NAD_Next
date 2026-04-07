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
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.code_dynamic_impl import order_code_dynamic_group_indices
from nad.core.selectors.gpqa_pairwise_impl import (
    GPQAPairwiseScorer,
    build_gpqa_pairwise_dominance_feature,
    build_gpqa_pairwise_features_configurable,
    build_gpqa_pairwise_margin_feature,
    extract_gpqa_pairwise_raw,
    get_gpqa_pairwise_feature_names,
)
from nad.core.selectors.science_dynamic_impl import (
    DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    compute_science_dynamic_primary_scores_from_raw,
)
from nad.core.views.reader import CacheReader
from nad.ops.bestofn_extreme8 import build_problem_groups
from scripts.run_code_baseline_v1_phase2 import (
    DEFAULT_VIEW,
    MetricAccumulator,
    _compute_copeland_scores,
    _fmt_pct,
    _load_entry_map,
    _problem_sort_key,
)
from scripts.run_gpqa_pairwise_round1 import (
    _ProblemData,
    _check_gate,
    _extract_all_problems,
    _summarize_table,
)

BLIND_SCIENCE_KEYS = ("DS-R1/gpqa", "Qwen3-4B/gpqa")
DEFAULT_MODEL_OUT = REPO_ROOT / "models" / "ml_selectors" / "gpqa_pairwise_v1.pkl"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _variant_configs() -> list[dict[str, Any]]:
    return [
        {
            "name": "margin",
            "include_margin": True,
            "include_dominance": False,
            "C": 1.0,
            "stage": "core",
        },
        {
            "name": "dominance",
            "include_margin": False,
            "include_dominance": True,
            "C": 1.0,
            "stage": "core",
        },
        {
            "name": "stronger_regularization",
            "include_margin": False,
            "include_dominance": False,
            "C": 0.1,
            "stage": "core",
        },
        {
            "name": "margin_reg",
            "include_margin": True,
            "include_dominance": False,
            "C": 0.1,
            "stage": "combo",
        },
        {
            "name": "dominance_reg",
            "include_margin": False,
            "include_dominance": True,
            "C": 0.1,
            "stage": "combo",
        },
        {
            "name": "margin_dominance",
            "include_margin": True,
            "include_dominance": True,
            "C": 1.0,
            "stage": "combo",
        },
        {
            "name": "margin_dominance_reg",
            "include_margin": True,
            "include_dominance": True,
            "C": 0.1,
            "stage": "combo",
        },
    ]


def _variant_feature_matrix(prob: _ProblemData, cfg: dict[str, Any]) -> np.ndarray:
    cols = [np.asarray(prob.X, dtype=np.float64)]
    recency = np.asarray(prob.sci_raw["recency_conf_mean"], dtype=np.float64)
    if bool(cfg["include_margin"]):
        cols.append(build_gpqa_pairwise_margin_feature(recency)[:, None])
    if bool(cfg["include_dominance"]):
        cols.append(build_gpqa_pairwise_dominance_feature(recency)[:, None])
    if len(cols) == 1:
        return cols[0]
    return np.concatenate(cols, axis=1)


def _evaluate_variant(
    all_problems: list[_ProblemData],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[np.ndarray]]:
    variant_X = [_variant_feature_matrix(prob, cfg) for prob in all_problems]
    acc = MetricAccumulator(cfg["name"], use_code_tiebreak=True)
    n_total = len(all_problems)

    for held_out_idx, held_out in enumerate(all_problems):
        if held_out_idx % 20 == 0:
            print(f"[gpqa-round2:{cfg['name']}] fold {held_out_idx + 1}/{n_total}", flush=True)
        pair_X_list: list[np.ndarray] = []
        pair_y_list: list[np.ndarray] = []
        for idx, prob in enumerate(all_problems):
            if idx == held_out_idx:
                continue
            X_prob = variant_X[idx]
            correct_idx = np.where(prob.labels > 0)[0]
            wrong_idx = np.where(prob.labels <= 0)[0]
            if correct_idx.size == 0 or wrong_idx.size == 0:
                continue
            X_correct = X_prob[correct_idx]
            X_wrong = X_prob[wrong_idx]
            n_pairs = correct_idx.size * wrong_idx.size
            diffs_pos = (X_correct[:, None, :] - X_wrong[None, :, :]).reshape(n_pairs, X_prob.shape[1])
            diffs_neg = (X_wrong[None, :, :] - X_correct[:, None, :]).reshape(n_pairs, X_prob.shape[1])
            pair_X_list.append(np.concatenate([diffs_pos, diffs_neg], axis=0))
            pair_y_list.append(np.concatenate([
                np.ones(n_pairs, dtype=np.int32),
                np.zeros(n_pairs, dtype=np.int32),
            ], axis=0))

        if not pair_X_list:
            scores = np.zeros(len(held_out.run_ids), dtype=np.float64)
        else:
            scorer = GPQAPairwiseScorer(
                C=float(cfg["C"]),
                include_margin=bool(cfg["include_margin"]),
                include_dominance=bool(cfg["include_dominance"]),
            )
            scorer.fit(
                np.concatenate(pair_X_list, axis=0),
                np.concatenate(pair_y_list, axis=0),
            )
            scores = scorer.score_group(variant_X[held_out_idx])
        acc.add_problem(held_out.problem_id, held_out.run_ids, scores, held_out.labels, held_out.D)

    return acc.finalize(), variant_X


def _baseline_rows(all_problems: list[_ProblemData]) -> dict[str, dict[str, Any]]:
    acc_baseline = MetricAccumulator("science_baseline_v1", use_code_tiebreak=True)
    acc_copeland = MetricAccumulator("tournament-copeland", use_code_tiebreak=True)
    for prob in all_problems:
        baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(
            prob.sci_raw,
            weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
        )
        copeland_scores = _compute_copeland_scores(prob.D)
        acc_baseline.add_problem(prob.problem_id, prob.run_ids, baseline_scores, prob.labels, prob.D)
        acc_copeland.add_problem(prob.problem_id, prob.run_ids, copeland_scores, prob.labels, prob.D)
    return {
        "science_baseline_v1": acc_baseline.finalize(),
        "tournament-copeland": acc_copeland.finalize(),
    }


def _train_final_model(
    all_problems: list[_ProblemData],
    cfg: dict[str, Any],
    *,
    model_out: Path,
) -> None:
    variant_X = [_variant_feature_matrix(prob, cfg) for prob in all_problems]
    pair_X_list: list[np.ndarray] = []
    pair_y_list: list[np.ndarray] = []
    for idx, prob in enumerate(all_problems):
        X_prob = variant_X[idx]
        correct_idx = np.where(prob.labels > 0)[0]
        wrong_idx = np.where(prob.labels <= 0)[0]
        if correct_idx.size == 0 or wrong_idx.size == 0:
            continue
        X_correct = X_prob[correct_idx]
        X_wrong = X_prob[wrong_idx]
        n_pairs = correct_idx.size * wrong_idx.size
        diffs_pos = (X_correct[:, None, :] - X_wrong[None, :, :]).reshape(n_pairs, X_prob.shape[1])
        diffs_neg = (X_wrong[None, :, :] - X_correct[:, None, :]).reshape(n_pairs, X_prob.shape[1])
        pair_X_list.append(np.concatenate([diffs_pos, diffs_neg], axis=0))
        pair_y_list.append(np.concatenate([
            np.ones(n_pairs, dtype=np.int32),
            np.zeros(n_pairs, dtype=np.int32),
        ], axis=0))

    if not pair_X_list:
        raise RuntimeError("No GPQA pairwise training pairs found for final model.")

    scorer = GPQAPairwiseScorer(
        C=float(cfg["C"]),
        include_margin=bool(cfg["include_margin"]),
        include_dominance=bool(cfg["include_dominance"]),
    )
    scorer.fit(
        np.concatenate(pair_X_list, axis=0),
        np.concatenate(pair_y_list, axis=0),
    )
    scorer.save(model_out)


def _evaluate_blind_shapes(
    blind_entry_map: dict[str, Any],
    *,
    model_path: Path,
    distance_threads: int,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    scorer = GPQAPairwiseScorer.load(model_path)
    for cache_key in BLIND_SCIENCE_KEYS:
        entry = blind_entry_map[cache_key]
        print(f"[gpqa-round2-blind] {cache_key}", flush=True)
        cache_root = Path(entry.cache_root)
        meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
        groups = build_problem_groups(meta)
        reader = CacheReader(str(cache_root))
        engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))
        selector_problem = {
            name: {}
            for name in ("science_baseline_v1", "gpqa_pairwise_v1", "tournament-copeland")
        }
        for problem_id, run_ids in sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0])):
            run_ids = list(map(int, run_ids))
            run_views = [reader.get_run_view(int(run_id), DEFAULT_VIEW) for run_id in run_ids]
            D = engine.dense_matrix(run_views)
            context = SelectorContext(
                cache=reader,
                problem_id=str(problem_id),
                run_ids=run_ids,
                views=run_views,
                pos_window=None,
            )
            raw = extract_gpqa_pairwise_raw(context)
            X = build_gpqa_pairwise_features_configurable(
                raw,
                include_margin=bool(getattr(scorer, "include_margin", False)),
                include_dominance=bool(getattr(scorer, "include_dominance", False)),
            )
            pairwise_scores = scorer.score_group(X)
            baseline_scores, _ = compute_science_dynamic_primary_scores_from_raw(
                {
                    "prefix_conf_mean": raw["prefix_conf_mean"],
                    "recency_conf_mean": raw["recency_conf_mean"],
                    "late_worst_window": raw["late_worst_window"],
                    "late_recovery": raw["late_recovery"],
                },
                weights=DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
            )
            copeland_scores = _compute_copeland_scores(D)
            order_map = {
                "science_baseline_v1": order_code_dynamic_group_indices(baseline_scores, D, run_ids=run_ids),
                "gpqa_pairwise_v1": order_code_dynamic_group_indices(pairwise_scores, D, run_ids=run_ids),
                "tournament-copeland": order_code_dynamic_group_indices(copeland_scores, D, run_ids=run_ids),
            }
            topk = max(1, int(math.ceil(0.10 * len(run_ids))))
            for selector_name, order in order_map.items():
                selector_problem[selector_name][str(problem_id)] = {
                    "best_run_id": int(run_ids[int(order[0])]),
                    "topk_run_ids": [int(run_ids[int(idx)]) for idx in order[:topk].tolist()],
                }

        compare = {}
        for other in ("science_baseline_v1", "tournament-copeland"):
            top1 = []
            topk = []
            for problem_id, base in selector_problem["gpqa_pairwise_v1"].items():
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
    ap = argparse.ArgumentParser(description="Run GPQA pairwise round-2 ablations and promotion gate")
    ap.add_argument("--cache-root", required=True, help="Path to GPQA GT cache directory")
    ap.add_argument("--blind-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_test")
    ap.add_argument("--distance-threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--model-out", default="", help=f"Promoted model path (default: {DEFAULT_MODEL_OUT})")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"gpqa_pairwise_round2_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_out = Path(args.model_out) if args.model_out else DEFAULT_MODEL_OUT

    print("[gpqa-round2] Extracting GT features once …", flush=True)
    all_problems = _extract_all_problems(
        Path(args.cache_root),
        distance_threads=int(args.distance_threads),
        workers=int(args.workers),
    )
    baseline_metrics = _baseline_rows(all_problems)

    variant_rows: list[dict[str, Any]] = []
    passed_variant: dict[str, Any] | None = None
    for cfg in _variant_configs():
        metrics, _ = _evaluate_variant(all_problems, cfg)
        passed, failed = _check_gate(metrics)
        row = {
            **cfg,
            "feature_names": get_gpqa_pairwise_feature_names(
                include_margin=bool(cfg["include_margin"]),
                include_dominance=bool(cfg["include_dominance"]),
            ),
            "metrics": metrics,
            "gate_passed": bool(passed),
            "failed_thresholds": failed,
        }
        variant_rows.append(row)
        print()
        print(json.dumps({
            "variant": cfg["name"],
            "metrics": {k: metrics[k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")},
            "gate_passed": passed,
        }, indent=2, ensure_ascii=False))
        if passed and passed_variant is None:
            passed_variant = row
            break

    best_variant = max(
        variant_rows,
        key=lambda row: (
            bool(row["gate_passed"]),
            float(row["metrics"].get("selacc@10%") or 0.0),
            float(row["metrics"].get("pairwise") or 0.0),
            float(row["metrics"].get("hit@1") or 0.0),
        ),
    ) if variant_rows else None

    blind_shapes = None
    if passed_variant is not None:
        print(f"[gpqa-round2] Training promoted model: {passed_variant['name']}", flush=True)
        _train_final_model(all_problems, passed_variant, model_out=model_out)
        blind_entry_map = _load_entry_map(args.blind_cache_root)
        blind_shapes = _evaluate_blind_shapes(
            blind_entry_map,
            model_path=model_out,
            distance_threads=int(args.distance_threads),
        )

    payload = {
        "baseline_metrics": baseline_metrics,
        "variants": variant_rows,
        "passed_variant": None if passed_variant is None else passed_variant["name"],
        "best_variant": None if best_variant is None else best_variant["name"],
        "model_path": str(model_out) if passed_variant is not None else None,
        "blind_shapes": blind_shapes,
    }
    (out_dir / "gpqa_pairwise_round2.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_rows = [
        {"name": "science_baseline_v1", **{k: baseline_metrics["science_baseline_v1"][k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}},
        {"name": "tournament-copeland", **{k: baseline_metrics["tournament-copeland"][k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}},
        *[
            {"name": row["name"], **{k: row["metrics"][k] for k in ("auroc", "hit@1", "pairwise", "selacc@10%")}}
            for row in variant_rows
        ],
    ]
    summary_lines = [
        "# GPQA Pairwise Round 2",
        "",
        "## Metrics",
        "",
        _summarize_table(summary_rows),
        "",
        "## Gate",
        "",
        f"- Passed variant: `{passed_variant['name']}`" if passed_variant is not None else "- Passed variant: none",
        f"- Best variant: `{best_variant['name']}`" if best_variant is not None else "- Best variant: none",
        f"- Model path: `{model_out}`" if passed_variant is not None else "- Model path: n/a",
        "",
        "## Variant Details",
        "",
    ]
    for row in variant_rows:
        summary_lines.extend([
            f"### `{row['name']}`",
            f"- Features: `{row['feature_names']}`",
            f"- Regularization C: `{row['C']}`",
            f"- Gate passed: `{row['gate_passed']}`",
        ])
        if row["failed_thresholds"]:
            summary_lines.append(f"- Failed thresholds: `{row['failed_thresholds']}`")
        summary_lines.append("")
    if blind_shapes is not None:
        summary_lines.extend([
            "## Blind GPQA Checks",
            "",
        ])
        for cache_key, compare in blind_shapes.items():
            summary_lines.append(f"- `{cache_key}`: `{compare}`")
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "passed_variant": None if passed_variant is None else passed_variant["name"],
        "best_variant": None if best_variant is None else best_variant["name"],
        "best_selacc10": None if best_variant is None else best_variant["metrics"].get("selacc@10%"),
        "model_path": str(model_out) if passed_variant is not None else None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
