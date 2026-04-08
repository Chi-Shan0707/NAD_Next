#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.code_deepsets_impl import CodeDeepSetsScorer
from nad.core.selectors.code_dynamic_impl import (
    compute_code_dynamic_primary_scores_from_raw,
    extract_code_dynamic_raw_from_state,
    prepare_code_dynamic_run_state,
)
from nad.core.selectors.code_v2_impl import (
    DEFAULT_CODE_V2_WEIGHTS,
    build_code_v2_rank_features_from_raw,
    extract_code_v2_raw_from_state,
)
from nad.core.selectors.deepsets_core import DeepSetsConfig
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import build_problem_groups
from scripts.run_code_baseline_v1_phase2 import (
    CODE_CACHE_KEY,
    DEFAULT_VIEW,
    MetricAccumulator,
    _fmt_pct,
    _load_entry_map,
    _problem_sort_key,
)
from scripts.run_code_v2_candidate import (
    BASELINE_PARAMS,
    CANDIDATE_PARAMS,
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
DEFAULT_WINNER_MODEL_OUT = MODEL_DIR / "code_deepsets_round1.pkl"


@dataclass
class CodeDeepSetsProblemData:
    problem_id: str
    run_ids: list[int]
    labels: np.ndarray
    D: np.ndarray
    baseline_scores: np.ndarray
    code_v2_scores: np.ndarray
    code_deepsets_feat: np.ndarray


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


def _coding_gate(candidate_metrics: dict[str, Any], current_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failed: list[str] = []
    if float(candidate_metrics.get("selacc@10%") or 0.0) < float(current_metrics.get("selacc@10%") or 0.0):
        failed.append("SelAcc@10 below current code_v2")
    if float(candidate_metrics.get("pairwise") or 0.0) < 0.50:
        failed.append("Pairwise below 50%")
    if float(candidate_metrics.get("hit@1") or 0.0) < float(current_metrics.get("hit@1") or 0.0) - 0.01:
        failed.append("Hit@1 below current code_v2 guardrail")
    return (len(failed) == 0, failed)


def _variant_name(config: DeepSetsConfig) -> str:
    base = f"code_deepsets_round1_{config.pooling}"
    if float(config.pairwise_aux_weight) > 0.0:
        return f"{base}_pairaux{config.pairwise_aux_weight:.2f}".replace(".", "p")
    return base


def _preload_code_deepsets_problems(
    cache_root: Path,
    *,
    distance_threads: int,
    prefix_window_tokens: int,
    max_problems: int = 0,
) -> list[CodeDeepSetsProblemData]:
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    correctness = _load_ground_truth(cache_root)
    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))

    problems: list[CodeDeepSetsProblemData] = []
    for idx, (problem_id, run_ids) in enumerate(sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))):
        if int(max_problems) > 0 and len(problems) >= int(max_problems):
            break
        if idx % 20 == 0:
            print(f"[code-deepsets-preload] problem {idx + 1}/{len(groups)}", flush=True)
        run_ids = list(map(int, run_ids))
        labels = np.asarray([int(bool(correctness.get(int(run_id), False))) for run_id in run_ids], dtype=np.int32)
        run_states = []
        run_views = []
        baseline_raw = {
            "prefix_best_window_quality": np.full(len(run_ids), np.nan, dtype=np.float64),
            "head_tail_gap": np.full(len(run_ids), np.nan, dtype=np.float64),
            "reflection_density": np.full(len(run_ids), np.nan, dtype=np.float64),
            "tail_variance": np.full(len(run_ids), np.nan, dtype=np.float64),
            "post_reflection_recovery": np.full(len(run_ids), np.nan, dtype=np.float64),
        }
        code_v2_raw = {
            "prefix_best_window_quality": np.full(len(run_ids), np.nan, dtype=np.float64),
            "head_tail_gap": np.full(len(run_ids), np.nan, dtype=np.float64),
            "tail_variance": np.full(len(run_ids), np.nan, dtype=np.float64),
            "post_reflection_recovery": np.full(len(run_ids), np.nan, dtype=np.float64),
            "last_block_instability": np.full(len(run_ids), np.nan, dtype=np.float64),
        }
        for ridx, run_id in enumerate(run_ids):
            tv = reader.get_token_view(int(run_id))
            run_state = prepare_code_dynamic_run_state(reader, int(run_id), token_view=tv)
            run_states.append(run_state)
            run_views.append(reader.get_run_view(int(run_id), DEFAULT_VIEW))

            base_row = extract_code_dynamic_raw_from_state(
                run_state,
                reflection_threshold=float(BASELINE_PARAMS["reflection_threshold"]),
                reflection_lookback_slices=int(BASELINE_PARAMS["reflection_lookback_slices"]),
                prefix_fraction=float(BASELINE_PARAMS["prefix_fraction"]),
                prefix_window_tokens=int(prefix_window_tokens),
            )
            for key in baseline_raw:
                baseline_raw[key][ridx] = float(base_row[key])

            code_row = extract_code_v2_raw_from_state(
                run_state,
                reflection_threshold=float(CANDIDATE_PARAMS["reflection_threshold"]),
                reflection_lookback_slices=int(CANDIDATE_PARAMS["reflection_lookback_slices"]),
                prefix_fraction=float(CANDIDATE_PARAMS["prefix_fraction"]),
                prefix_window_tokens=int(prefix_window_tokens),
            )
            for key in code_v2_raw:
                code_v2_raw[key][ridx] = float(code_row[key])

        D = engine.dense_matrix(run_views)
        baseline_scores, _ = compute_code_dynamic_primary_scores_from_raw(baseline_raw)
        code_feat = build_code_v2_rank_features_from_raw(code_v2_raw)
        code_v2_scores = np.dot(
            code_feat,
            np.asarray([DEFAULT_CODE_V2_WEIGHTS[k] for k in DEFAULT_CODE_V2_WEIGHTS], dtype=np.float64),
        )
        problems.append(
            CodeDeepSetsProblemData(
                problem_id=str(problem_id),
                run_ids=run_ids,
                labels=labels,
                D=D,
                baseline_scores=np.asarray(baseline_scores, dtype=np.float64),
                code_v2_scores=np.asarray(code_v2_scores, dtype=np.float64),
                code_deepsets_feat=np.asarray(code_feat, dtype=np.float32),
            )
        )
    return problems


def _problem_feature_matrix(prob: CodeDeepSetsProblemData) -> np.ndarray:
    return np.asarray(prob.code_deepsets_feat, dtype=np.float32)


def _run_variant_lopo(
    problems: list[CodeDeepSetsProblemData],
    *,
    config: DeepSetsConfig,
    torch_threads: int,
) -> tuple[dict[str, Any], list[ProblemScoreRecord], dict[str, Any], dict[str, Any]]:
    variant_name = _variant_name(config)
    acc = MetricAccumulator(variant_name, use_code_tiebreak=True)
    n_total = len(problems)
    X_all = np.stack([_problem_feature_matrix(prob) for prob in problems], axis=0)
    y_all = np.stack([np.asarray(prob.labels, dtype=np.float32) for prob in problems], axis=0)
    records: list[ProblemScoreRecord] = []
    fold_losses: list[float] = []

    for held_out_idx, held_out in enumerate(problems):
        if held_out_idx % 20 == 0:
            print(f"[code-deepsets:{variant_name}] fold {held_out_idx + 1}/{n_total}", flush=True)
        if held_out_idx == 0:
            X_train = X_all[1:]
            y_train = y_all[1:]
        elif held_out_idx == n_total - 1:
            X_train = X_all[:-1]
            y_train = y_all[:-1]
        else:
            X_train = np.concatenate([X_all[:held_out_idx], X_all[held_out_idx + 1 :]], axis=0)
            y_train = np.concatenate([y_all[:held_out_idx], y_all[held_out_idx + 1 :]], axis=0)

        scorer = CodeDeepSetsScorer(config=config)
        scorer.fit_problem_batches(X_train, y_train, torch_threads=torch_threads)
        fold_losses.append(float(scorer.training_summary.get("train_loss", 0.0)))
        scores = scorer.score_group(X_all[held_out_idx])
        acc.add_problem(held_out.problem_id, held_out.run_ids, scores, held_out.labels, held_out.D)
        records.append(
            ProblemScoreRecord(
                cache_key="DS-R1/lcb_v5",
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
    problems: list[CodeDeepSetsProblemData],
    *,
    config: DeepSetsConfig,
    torch_threads: int,
    out_path: Path,
) -> dict[str, Any]:
    scorer = CodeDeepSetsScorer(config=config)
    X_groups = np.stack([_problem_feature_matrix(prob) for prob in problems], axis=0)
    y_groups = np.stack([np.asarray(prob.labels, dtype=np.float32) for prob in problems], axis=0)
    scorer.fit_problem_batches(X_groups, y_groups, torch_threads=torch_threads)
    scorer.save(out_path)
    return dict(scorer.training_summary)


def _pick_best_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            bool(row["coding_gate_passed"]),
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
    ap = argparse.ArgumentParser(description="Run coding DeepSets round-1 evaluation and system proxy patching")
    ap.add_argument("--gt-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--cache-root", default="")
    ap.add_argument("--current-science-json", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--prefix-window-tokens", type=int, default=128)
    ap.add_argument("--torch-threads", type=int, default=max(1, min(8, (os.cpu_count() or 8))))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pairwise-aux-weight", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"code_deepsets_round1_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    current_science_json = Path(args.current_science_json) if args.current_science_json else _latest_current_science_json()
    current_science_payload = _load_current_science_payload(current_science_json)
    current_science_candidate = dict(current_science_payload["selected_candidate"])
    current_system_bundle = dict(current_science_candidate["comprehensive_metrics"])

    gt_entry_map = _load_entry_map(args.gt_cache_root)
    cache_root = Path(args.cache_root) if args.cache_root else Path(gt_entry_map[CODE_CACHE_KEY].cache_root)

    print("[code-deepsets] Preloading coding problems …", flush=True)
    problems = _preload_code_deepsets_problems(
        cache_root,
        distance_threads=int(args.distance_threads),
        prefix_window_tokens=int(args.prefix_window_tokens),
    )

    compare_accs = {
        "code_baseline_v1": MetricAccumulator("code_baseline_v1", use_code_tiebreak=True),
        "code_v2": MetricAccumulator("code_v2", use_code_tiebreak=True),
    }
    for prob in problems:
        compare_accs["code_baseline_v1"].add_problem(prob.problem_id, prob.run_ids, prob.baseline_scores, prob.labels, prob.D)
        compare_accs["code_v2"].add_problem(prob.problem_id, prob.run_ids, prob.code_v2_scores, prob.labels, prob.D)

    baseline_compare_rows = {name: acc.finalize() for name, acc in compare_accs.items()}

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
    for config in variant_configs:
        metrics, records, proxy_metrics, fold_summary = _run_variant_lopo(
            problems,
            config=config,
            torch_threads=int(args.torch_threads),
        )
        cache_metrics = {cache_key: dict(base_doc_metrics[cache_key]) for cache_key in OVERALL_DS_CACHE_KEYS}
        cache_metrics["DS-R1/gpqa"] = dict(current_science_candidate["gpqa_proxy_metrics"])
        cache_metrics["DS-R1/lcb_v5"] = dict(proxy_metrics)
        candidate_bundle = _combine_cache_metric_proxy(cache_metrics)
        system_delta = _system_delta(candidate_bundle, current_system_bundle)
        system_delta["sample_weighted"]["avg_rank_proxy"] = 0.0
        system_delta["equal_cache_mean"]["avg_rank_proxy"] = 0.0
        system_gate_passed, system_gate_failed, system_delta = _comprehensive_gate(
            candidate_bundle,
            current_system_bundle,
            delta_override=system_delta,
        )
        coding_gate_passed, coding_gate_failed = _coding_gate(metrics, baseline_compare_rows["code_v2"])

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
                "coding_proxy_metrics": proxy_metrics,
                "coding_gate_passed": bool(coding_gate_passed),
                "coding_gate_failed": coding_gate_failed,
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

    promote = bool(selected_variant["coding_gate_passed"] and selected_variant["system_gate_passed"])
    decision = "Promote" if promote else "No-Promote"

    payload = {
        "status_summary": {
            "code_v2_is_promoted_default": True,
            "science_hybrid_round3_is_promoted_science_patch": True,
            "new_research_line_includes_minimal_deepsets": True,
        },
        "inputs": {
            "cache_root": str(cache_root),
            "current_science_json": str(current_science_json),
            "distance_threads": int(args.distance_threads),
            "prefix_window_tokens": int(args.prefix_window_tokens),
            "torch_threads": int(args.torch_threads),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        },
        "compare_rows": {
            "code_baseline_v1": baseline_compare_rows["code_baseline_v1"],
            "code_v2": baseline_compare_rows["code_v2"],
            "code_deepsets_variants": variant_rows,
            "selected_variant": selected_variant,
        },
        "current_system": {
            "coding_slice_name": "code_v2",
            "coding_slice_metrics": current_code_v2_metrics,
            "bundle": current_system_bundle,
        },
        "patched_system": {
            "coding_slice_name": str(selected_variant["name"]),
            "coding_slice_metrics": dict(selected_variant["metrics"]),
            "bundle": dict(selected_variant["system_bundle"]),
            "delta_vs_current": dict(selected_variant["system_delta"]),
        },
        "decision": {
            "promote": promote,
            "label": decision,
            "coding_gate_passed": bool(selected_variant["coding_gate_passed"]),
            "coding_gate_failed": list(selected_variant["coding_gate_failed"]),
            "system_gate_passed": bool(selected_variant["system_gate_passed"]),
            "system_gate_failed": list(selected_variant["system_gate_failed"]),
        },
    }

    (out_dir / "code_deepsets_round1.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    compare_rows = [
        {"name": "code_baseline_v1", **baseline_compare_rows["code_baseline_v1"]},
        {"name": "code_v2", **baseline_compare_rows["code_v2"]},
        *[
            {"name": row["name"], **dict(row["metrics"])}
            for row in variant_rows
        ],
    ]
    sample_rows = [
        {"name": "current = code_v2 + science_hybrid_round3", **dict(current_system_bundle["sample_weighted"])},
        {"name": f"patched = {selected_variant['name']} + science_hybrid_round3", **dict(selected_variant["system_bundle"]["sample_weighted"])},
    ]
    equal_rows = [
        {"name": "current = code_v2 + science_hybrid_round3", **dict(current_system_bundle["equal_cache_mean"])},
        {"name": f"patched = {selected_variant['name']} + science_hybrid_round3", **dict(selected_variant["system_bundle"]["equal_cache_mean"])},
    ]
    summary_lines = [
        "# Code DeepSets Round 1",
        "",
        "## Coding Single-Domain",
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
        f"- Coding gate passed: `{selected_variant['coding_gate_passed']}`",
        f"- Coding gate failures: `{selected_variant['coding_gate_failed']}`",
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
