#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.deepsets_core import DeepSetsConfig
from nad.core.selectors.math_deepsets_impl import MathDeepSetsScorer
from scripts.run_code_baseline_v1_phase2 import _fmt_pct, _load_entry_map
from scripts.run_math_svm_sweep import (
    MATH_CACHE_PROFILES,
    _MathProblemData,
    _baseline_metrics,
    _evaluate_variant,
    _extract_all_problems,
    _prepare_family_views,
)
from scripts.run_science_hybrid_round3 import (
    CODE_V2_EXHAUSTIVE_JSON,
    EXTREME12_TEST_ANALYSIS_DOC,
    OVERALL_DS_CACHE_KEYS,
    ProblemScoreRecord,
    _combine_cache_metric_proxy,
    _comprehensive_gate,
    _evaluate_problem_records,
    _load_code_v2_proxy_metrics,
    _load_extreme12_test_metrics,
    _problem_counts_from_entry_map,
    _system_delta,
)

DEFAULT_CURRENT_SCIENCE_JSON_GLOB = "result/science_hybrid_round3_*/science_hybrid_round3.json"
MODEL_DIR = REPO_ROOT / "models" / "ml_selectors"
DEFAULT_WINNER_MODEL_OUT = MODEL_DIR / "math_deepsets_round1.pkl"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _latest_current_science_json() -> Path:
    matches = sorted(glob.glob(str(REPO_ROOT / DEFAULT_CURRENT_SCIENCE_JSON_GLOB)))
    if not matches:
        raise FileNotFoundError("No science_hybrid_round3.json payload found under result/")
    return Path(matches[-1])


def _load_current_science_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_candidate")
    if not isinstance(selected, dict):
        raise RuntimeError(f"Invalid science hybrid payload at {path}")
    return payload


def _result_rows_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Selector | AUROC | Hit@1 | Pairwise | SelAcc@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {auroc} | {hit1} | {pairwise} | {selacc} |".format(
                name=str(row["name"]),
                auroc=_fmt_pct(row.get("auroc")),
                hit1=_fmt_pct(row.get("hit@1")),
                pairwise=_fmt_pct(row.get("pairwise")),
                selacc=_fmt_pct(row.get("selacc@10%")),
            )
        )
    return "\n".join(lines)


def _system_rows_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| System | Hit@1 | Hit@3 | Pairwise | SelAcc@10 | AvgRank proxy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {hit1} | {hit3} | {pairwise} | {selacc} | {avg_rank} |".format(
                name=str(row["name"]),
                hit1=_fmt_pct(row.get("hit@1")),
                hit3=_fmt_pct(row.get("hit@3")),
                pairwise=_fmt_pct(row.get("pairwise")),
                selacc=_fmt_pct(row.get("selacc@10%")),
                avg_rank="n/a" if row.get("avg_rank_proxy") is None else f"{float(row['avg_rank_proxy']):.4f}",
            )
        )
    return "\n".join(lines)


def _variant_name(config: DeepSetsConfig) -> str:
    base = f"math_deepsets_round1_{config.pooling}"
    if float(config.pairwise_aux_weight) > 0.0:
        return f"{base}_pairaux{config.pairwise_aux_weight:.2f}".replace(".", "p")
    return base


def _math_gate(candidate_metrics: dict[str, Any], knn_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failed: list[str] = []
    if float(candidate_metrics.get("hit@1") or 0.0) < float(knn_metrics.get("hit@1") or 0.0):
        failed.append("Hit@1 below knn-medoid")
    if (
        float(candidate_metrics.get("selacc@10%") or 0.0) < float(knn_metrics.get("selacc@10%") or 0.0)
        and float(candidate_metrics.get("pairwise") or 0.0) < float(knn_metrics.get("pairwise") or 0.0)
    ):
        failed.append("Both SelAcc@10 and Pairwise below knn-medoid")
    return (len(failed) == 0, failed)


def _run_variant_lopo(
    problems: list[_MathProblemData],
    *,
    config: DeepSetsConfig,
    torch_threads: int,
) -> tuple[dict[str, Any], list[ProblemScoreRecord], dict[str, Any], dict[str, Any]]:
    variant_name = _variant_name(config)
    from scripts.run_code_baseline_v1_phase2 import MetricAccumulator

    acc = MetricAccumulator(variant_name, use_code_tiebreak=False)
    n_total = len(problems)
    X_all = np.stack([np.asarray(prob.X_all, dtype=np.float32) for prob in problems], axis=0)
    y_all = np.stack([np.asarray(prob.labels, dtype=np.float32) for prob in problems], axis=0)
    records: list[ProblemScoreRecord] = []
    fold_losses: list[float] = []

    for held_out_idx, held_out in enumerate(problems):
        if held_out_idx % 20 == 0:
            print(f"[math-deepsets:{variant_name}] fold {held_out_idx + 1}/{n_total}", flush=True)
        if held_out_idx == 0:
            X_train = X_all[1:]
            y_train = y_all[1:]
        elif held_out_idx == n_total - 1:
            X_train = X_all[:-1]
            y_train = y_all[:-1]
        else:
            X_train = np.concatenate([X_all[:held_out_idx], X_all[held_out_idx + 1 :]], axis=0)
            y_train = np.concatenate([y_all[:held_out_idx], y_all[held_out_idx + 1 :]], axis=0)

        scorer = MathDeepSetsScorer(config=config)
        scorer.fit_problem_batches(X_train, y_train, torch_threads=torch_threads)
        fold_losses.append(float(scorer.training_summary.get("train_loss", 0.0)))
        scores = scorer.score_group(X_all[held_out_idx])
        acc.add_problem(held_out.problem_id, held_out.run_ids, scores, held_out.labels, held_out.D)
        records.append(
            ProblemScoreRecord(
                cache_key=f"DS-R1/{held_out.dataset}",
                problem_id=str(held_out.problem_id),
                sample_ids=list(map(int, held_out.run_ids)),
                labels=np.asarray(held_out.labels, dtype=np.int32),
                scores=np.asarray(scores, dtype=np.float64),
            )
        )

    metrics = acc.finalize()
    metrics["mean_train_loss"] = float(np.mean(fold_losses)) if fold_losses else None
    proxy_metrics = _evaluate_problem_records(records)
    train_summary = {
        "mean_train_loss": float(np.mean(fold_losses)) if fold_losses else None,
        "n_groups": float(len(problems)),
        "n_runs_per_group": float(X_all.shape[1]) if X_all.ndim == 3 else None,
        "pairwise_aux_weight": float(config.pairwise_aux_weight),
    }
    return metrics, records, proxy_metrics, train_summary


def _train_final_variant_model(
    problems: list[_MathProblemData],
    *,
    config: DeepSetsConfig,
    torch_threads: int,
    out_path: Path,
) -> dict[str, Any]:
    scorer = MathDeepSetsScorer(config=config)
    X_groups = np.stack([np.asarray(prob.X_all, dtype=np.float32) for prob in problems], axis=0)
    y_groups = np.stack([np.asarray(prob.labels, dtype=np.float32) for prob in problems], axis=0)
    scorer.fit_problem_batches(X_groups, y_groups, torch_threads=torch_threads)
    scorer.save(out_path)
    return dict(scorer.training_summary)


def _pick_best_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            bool(row["math_gate_passed"]),
            bool(row["system_gate_passed"]),
            float(row["system_delta"]["sample_weighted"]["hit@1"]),
            float(row["system_delta"]["sample_weighted"]["selacc@10%"]),
            float(row["metrics"].get("hit@1") or 0.0),
            float(row["metrics"].get("selacc@10%") or 0.0),
            float(row["metrics"].get("pairwise") or 0.0),
            row["name"],
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run math DeepSets round-1 evaluation and system proxy patching")
    ap.add_argument("--profile", default="main", choices=sorted(MATH_CACHE_PROFILES.keys()))
    ap.add_argument("--current-science-json", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=max(1, min(8, (os.cpu_count() or 8))))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pairwise-aux-weight", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"math_deepsets_round1_{args.profile}_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    current_science_json = Path(args.current_science_json) if args.current_science_json else _latest_current_science_json()
    current_science_payload = _load_current_science_payload(current_science_json)
    current_science_candidate = dict(current_science_payload["selected_candidate"])
    current_system_bundle = dict(current_science_candidate["comprehensive_metrics"])

    print(f"[math-deepsets] Extracting math problems for profile={args.profile} …", flush=True)
    problems = _extract_all_problems(args.profile, distance_threads=int(args.distance_threads))
    baseline_metrics = _baseline_metrics(problems)
    family_views = _prepare_family_views(problems)

    runwise_cfg = {
        "name": "runwise__all_aug__squared_hinge__C0p10__bias__balanced",
        "model_family": "runwise",
        "feature_family": "all_aug",
        "loss": "squared_hinge",
        "C": 0.10,
        "fit_intercept": True,
        "class_weight": "balanced",
        "dual": "auto",
        "max_iter": 100000,
        "tol": 1e-4,
    }
    ranksvm_cfg = {
        "name": "ranksvm__no_logs__squared_hinge__C0p10__bias__mean_margin",
        "model_family": "ranksvm",
        "feature_family": "no_logs",
        "loss": "squared_hinge",
        "C": 0.10,
        "fit_intercept": True,
        "backend": "mean_margin",
        "dual": "auto",
        "max_iter": 100000,
        "tol": 1e-4,
    }
    runwise_metrics, runwise_aux = _evaluate_variant(problems, family_views, runwise_cfg)
    ranksvm_metrics, ranksvm_aux = _evaluate_variant(problems, family_views, ranksvm_cfg)

    overall_entry_map = _load_entry_map("MUI_HUB/cache")
    base_doc_metrics = _load_extreme12_test_metrics(EXTREME12_TEST_ANALYSIS_DOC)
    problem_counts = _problem_counts_from_entry_map(overall_entry_map, OVERALL_DS_CACHE_KEYS)
    for cache_key in OVERALL_DS_CACHE_KEYS:
        base_doc_metrics[cache_key]["n_problems"] = int(problem_counts[cache_key])
        base_doc_metrics[cache_key]["top10_count"] = max(1, int(math.ceil(0.10 * int(base_doc_metrics[cache_key]["n_samples"]))))

    current_code_v2_metrics = _load_code_v2_proxy_metrics(
        CODE_V2_EXHAUSTIVE_JSON,
        fallback_hit3=float(base_doc_metrics["DS-R1/lcb_v5"]["hit@3"]),
    )

    variant_configs = [
        DeepSetsConfig(
            pooling="mean",
            hidden_dim=16,
            embed_dim=8,
            head_hidden_dim=8,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            pairwise_aux_weight=0.0,
            seed=int(args.seed),
        ).validate(),
        DeepSetsConfig(
            pooling="max",
            hidden_dim=16,
            embed_dim=8,
            head_hidden_dim=8,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            pairwise_aux_weight=0.0,
            seed=int(args.seed),
        ).validate(),
    ]
    if float(args.pairwise_aux_weight) > 0.0:
        variant_configs.append(
            DeepSetsConfig(
                pooling="max",
                hidden_dim=16,
                embed_dim=8,
                head_hidden_dim=8,
                epochs=int(args.epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                pairwise_aux_weight=float(args.pairwise_aux_weight),
                seed=int(args.seed),
            ).validate()
        )

    variant_rows: list[dict[str, Any]] = []
    mutable_math_keys = [f"DS-R1/{dataset}" for dataset, _ in MATH_CACHE_PROFILES[str(args.profile)]]
    for config in variant_configs:
        metrics, records, proxy_metrics, fold_summary = _run_variant_lopo(
            problems,
            config=config,
            torch_threads=int(args.torch_threads),
        )
        per_cache = {}
        for cache_key in mutable_math_keys:
            cache_records = [record for record in records if record.cache_key == cache_key]
            if cache_records:
                per_cache[cache_key] = _evaluate_problem_records(cache_records)

        cache_metrics = {cache_key: dict(base_doc_metrics[cache_key]) for cache_key in OVERALL_DS_CACHE_KEYS}
        cache_metrics["DS-R1/gpqa"] = dict(current_science_candidate["gpqa_proxy_metrics"])
        cache_metrics["DS-R1/lcb_v5"] = dict(current_code_v2_metrics)
        for cache_key, cache_payload in per_cache.items():
            cache_metrics[cache_key] = dict(cache_payload)
        candidate_bundle = _combine_cache_metric_proxy(cache_metrics)
        system_delta = _system_delta(candidate_bundle, current_system_bundle)
        system_delta["sample_weighted"]["avg_rank_proxy"] = 0.0
        system_delta["equal_cache_mean"]["avg_rank_proxy"] = 0.0
        system_gate_passed, system_gate_failed, system_delta = _comprehensive_gate(
            candidate_bundle,
            current_system_bundle,
            delta_override=system_delta,
        )
        math_gate_passed, math_gate_failed = _math_gate(metrics, baseline_metrics["knn-medoid"]["metrics"])

        variant_name = _variant_name(config)
        model_out = MODEL_DIR / f"{variant_name}.pkl"
        final_train_summary = _train_final_variant_model(
            problems,
            config=config,
            torch_threads=int(args.torch_threads),
            out_path=model_out,
        )
        final_train_summary.update(fold_summary)

        variant_rows.append(
            {
                "name": variant_name,
                "config": config.as_dict(),
                "metrics": metrics,
                "math_proxy_metrics": proxy_metrics,
                "per_cache_metrics": per_cache,
                "math_gate_passed": bool(math_gate_passed),
                "math_gate_failed": math_gate_failed,
                "system_gate_passed": bool(system_gate_passed),
                "system_gate_failed": system_gate_failed,
                "system_bundle": candidate_bundle,
                "system_delta": system_delta,
                "model_path": str(model_out),
                "final_train_summary": final_train_summary,
            }
        )

    selected_variant = _pick_best_variant(variant_rows)
    shutil.copyfile(selected_variant["model_path"], DEFAULT_WINNER_MODEL_OUT)

    promote = bool(selected_variant["math_gate_passed"] and selected_variant["system_gate_passed"])
    decision = "Promote" if promote else "No-Promote"

    payload = {
        "status_summary": {
            "code_v2_is_promoted_default": True,
            "science_hybrid_round3_is_promoted_science_patch": True,
            "math_currently_uses_generic_mainline": True,
            "new_research_line_includes_minimal_deepsets": True,
        },
        "inputs": {
            "profile": str(args.profile),
            "current_science_json": str(current_science_json),
            "distance_threads": int(args.distance_threads),
            "torch_threads": int(args.torch_threads),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        },
        "compare_rows": {
            "medoid": baseline_metrics["medoid"]["metrics"],
            "knn-medoid": baseline_metrics["knn-medoid"]["metrics"],
            "deepconf": baseline_metrics["deepconf"]["metrics"],
            "tournament-copeland": baseline_metrics["tournament-copeland"]["metrics"],
            "math_svm_runwise_recipe": {
                "metrics": runwise_metrics,
                "aux": runwise_aux,
                "config": runwise_cfg,
            },
            "math_svm_ranksvm_recipe": {
                "metrics": ranksvm_metrics,
                "aux": ranksvm_aux,
                "config": ranksvm_cfg,
            },
            "math_deepsets_variants": variant_rows,
            "selected_variant": selected_variant,
        },
        "current_system": {
            "math_slice_name": "generic_extreme12_math",
            "bundle": current_system_bundle,
        },
        "patched_system": {
            "math_slice_name": str(selected_variant["name"]),
            "math_slice_metrics": dict(selected_variant["metrics"]),
            "bundle": dict(selected_variant["system_bundle"]),
            "delta_vs_current": dict(selected_variant["system_delta"]),
        },
        "decision": {
            "promote": promote,
            "label": decision,
            "math_gate_passed": bool(selected_variant["math_gate_passed"]),
            "math_gate_failed": list(selected_variant["math_gate_failed"]),
            "system_gate_passed": bool(selected_variant["system_gate_passed"]),
            "system_gate_failed": list(selected_variant["system_gate_failed"]),
        },
    }

    (out_dir / "math_deepsets_round1.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    compare_rows = [
        {"name": "medoid", **baseline_metrics["medoid"]["metrics"]},
        {"name": "knn-medoid", **baseline_metrics["knn-medoid"]["metrics"]},
        {"name": runwise_cfg["name"], **runwise_metrics},
        {"name": ranksvm_cfg["name"], **ranksvm_metrics},
        *[
            {"name": row["name"], **dict(row["metrics"])}
            for row in variant_rows
        ],
    ]
    sample_rows = [
        {"name": "current = generic math + code_v2 + science_hybrid_round3", **dict(current_system_bundle["sample_weighted"])},
        {"name": f"patched = {selected_variant['name']} + code_v2 + science_hybrid_round3", **dict(selected_variant["system_bundle"]["sample_weighted"])},
    ]
    equal_rows = [
        {"name": "current = generic math + code_v2 + science_hybrid_round3", **dict(current_system_bundle["equal_cache_mean"])},
        {"name": f"patched = {selected_variant['name']} + code_v2 + science_hybrid_round3", **dict(selected_variant["system_bundle"]["equal_cache_mean"])},
    ]
    summary_lines = [
        "# Math DeepSets Round 1",
        "",
        f"## Math Single-Domain (`profile={args.profile}`)",
        "",
        _result_rows_markdown(compare_rows),
        "",
        f"- Selected DeepSets variant: `{selected_variant['name']}`",
        f"- Decision: `{decision}`",
        "",
        "## Current vs Patched System Proxy",
        "",
        "### Sample-weighted",
        "",
        _system_rows_markdown(sample_rows),
        "",
        "### Equal-cache-mean",
        "",
        _system_rows_markdown(equal_rows),
        "",
        "### Gate Read",
        "",
        f"- Math gate passed: `{selected_variant['math_gate_passed']}`",
        f"- Math gate failures: `{selected_variant['math_gate_failed']}`",
        f"- System gate passed: `{selected_variant['system_gate_passed']}`",
        f"- System gate failures: `{selected_variant['system_gate_failed']}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "selected_variant": selected_variant["name"],
                "decision": decision,
                "promote": promote,
                "winner_model_path": str(DEFAULT_WINNER_MODEL_OUT),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
