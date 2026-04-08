#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.selectors.gpqa_deepsets_impl import GPQADeepSetsConfig, GPQADeepSetsScorer
from scripts.run_code_baseline_v1_phase2 import MetricAccumulator, _fmt_pct, _load_entry_map
from scripts.run_gpqa_pairwise_round1 import (
    _DEFAULT_DISTANCE_THREADS,
    _DEFAULT_WORKERS,
    _ProblemData,
    _extract_all_problems,
)
from scripts.run_science_hybrid_round3 import (
    CODE_V2_EXHAUSTIVE_JSON,
    EXTREME12_TEST_ANALYSIS_DOC,
    OVERALL_DS_CACHE_KEYS,
    ProblemScoreRecord,
    _combine_cache_metric_proxy,
    _compact_metrics,
    _comprehensive_gate,
    _evaluate_problem_records,
    _load_code_v2_proxy_metrics,
    _load_extreme12_test_metrics,
    _problem_counts_from_entry_map,
    _system_delta,
)

DEFAULT_CACHE_ROOT = REPO_ROOT / "MUI_HUB" / "cache" / "DeepSeek-R1-0528-Qwen3-8B" / "gpqa" / "cache_neuron_output_1_act_no_rms_20251126_111853"
DEFAULT_CURRENT_SCIENCE_JSON_GLOB = "result/science_hybrid_round3_*/science_hybrid_round3.json"
MODEL_DIR = REPO_ROOT / "models" / "ml_selectors"
DEFAULT_WINNER_MODEL_OUT = MODEL_DIR / "gpqa_deepsets_round1.pkl"


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


def _problem_records_for_scores(
    all_problems: list[_ProblemData],
    *,
    cache_key: str,
    scores_by_problem: dict[str, np.ndarray],
) -> list[ProblemScoreRecord]:
    records: list[ProblemScoreRecord] = []
    for prob in all_problems:
        records.append(
            ProblemScoreRecord(
                cache_key=str(cache_key),
                problem_id=str(prob.problem_id),
                sample_ids=list(map(int, prob.run_ids)),
                labels=np.asarray(prob.labels, dtype=np.int32),
                scores=np.asarray(scores_by_problem[str(prob.problem_id)], dtype=np.float64),
            )
        )
    return records


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


def _delta_rows_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Patch | sample-weighted Hit@1 | sample-weighted SelAcc@10 | sample-weighted AvgRank proxy | equal-cache Hit@1 | equal-cache SelAcc@10 | Actual leaderboard delta |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {main_hit} | {main_sel} | {main_rank} | {eq_hit} | {eq_sel} | {actual} |".format(
                name=str(row["name"]),
                main_hit=_fmt_pct(row["sample_weighted"]["hit@1"]),
                main_sel=_fmt_pct(row["sample_weighted"]["selacc@10%"]),
                main_rank=f"{float(row['sample_weighted']['avg_rank_proxy']):+.4f}",
                eq_hit=_fmt_pct(row["equal_cache_mean"]["hit@1"]),
                eq_sel=_fmt_pct(row["equal_cache_mean"]["selacc@10%"]),
                actual=str(row["actual_leaderboard_delta"]),
            )
        )
    return "\n".join(lines)


def _science_gate(candidate_metrics: dict[str, Any], current_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failed: list[str] = []
    if float(candidate_metrics.get("hit@1") or 0.0) < float(current_metrics.get("hit@1") or 0.0):
        failed.append("Hit@1 below current science_hybrid_round3")
    if float(candidate_metrics.get("selacc@10%") or 0.0) < float(current_metrics.get("selacc@10%") or 0.0):
        failed.append("SelAcc@10 below current science_hybrid_round3")
    return (len(failed) == 0, failed)


def _train_final_variant_model(
    all_problems: list[_ProblemData],
    *,
    config: GPQADeepSetsConfig,
    torch_threads: int,
    out_path: Path,
) -> dict[str, Any]:
    scorer = GPQADeepSetsScorer(config=config)
    X_groups = np.stack([np.asarray(prob.X, dtype=np.float32) for prob in all_problems], axis=0)
    y_groups = np.stack([np.asarray(prob.labels, dtype=np.float32) for prob in all_problems], axis=0)
    scorer.fit_problem_batches(X_groups, y_groups, torch_threads=torch_threads)
    scorer.save(out_path)
    return dict(scorer.training_summary)


def _run_variant_lopo(
    all_problems: list[_ProblemData],
    *,
    config: GPQADeepSetsConfig,
    torch_threads: int,
) -> tuple[dict[str, Any], list[ProblemScoreRecord], dict[str, Any]]:
    acc = MetricAccumulator(f"gpqa_deepsets_round1_{config.pooling}", use_code_tiebreak=True)
    n_total = len(all_problems)
    X_all = np.stack([np.asarray(prob.X, dtype=np.float32) for prob in all_problems], axis=0)
    y_all = np.stack([np.asarray(prob.labels, dtype=np.float32) for prob in all_problems], axis=0)
    records: list[ProblemScoreRecord] = []
    fold_losses: list[float] = []

    for held_out_idx, held_out in enumerate(all_problems):
        if held_out_idx % 20 == 0:
            print(f"[gpqa-deepsets:{config.pooling}] fold {held_out_idx + 1}/{n_total}", flush=True)
        if held_out_idx == 0:
            X_train = X_all[1:]
            y_train = y_all[1:]
        elif held_out_idx == n_total - 1:
            X_train = X_all[:-1]
            y_train = y_all[:-1]
        else:
            X_train = np.concatenate([X_all[:held_out_idx], X_all[held_out_idx + 1 :]], axis=0)
            y_train = np.concatenate([y_all[:held_out_idx], y_all[held_out_idx + 1 :]], axis=0)

        scorer = GPQADeepSetsScorer(config=config)
        scorer.fit_problem_batches(X_train, y_train, torch_threads=torch_threads)
        fold_losses.append(float(scorer.training_summary.get("train_loss", 0.0)))
        scores = scorer.score_group(np.asarray(held_out.X, dtype=np.float32))
        acc.add_problem(held_out.problem_id, held_out.run_ids, scores, held_out.labels, held_out.D)
        records.append(
            ProblemScoreRecord(
                cache_key="DS-R1/gpqa",
                problem_id=str(held_out.problem_id),
                sample_ids=list(map(int, held_out.run_ids)),
                labels=np.asarray(held_out.labels, dtype=np.int32),
                scores=np.asarray(scores, dtype=np.float64),
            )
        )

    metrics = acc.finalize()
    metrics["mean_train_loss"] = float(np.mean(fold_losses)) if fold_losses else None
    proxy_metrics = _evaluate_problem_records(records)
    return metrics, records, proxy_metrics


def _pick_best_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            bool(row["science_gate_passed"]),
            bool(row["system_gate_passed"]),
            float(row["system_delta"]["sample_weighted"]["hit@1"]),
            float(row["system_delta"]["sample_weighted"]["selacc@10%"]),
            -float(row["system_delta"]["sample_weighted"]["avg_rank_proxy"]),
            float(row["metrics"].get("hit@1") or 0.0),
            float(row["metrics"].get("selacc@10%") or 0.0),
            float(row["metrics"].get("pairwise") or 0.0),
            row["name"],
        ),
    )


def _variant_name(config: GPQADeepSetsConfig) -> str:
    base = f"gpqa_deepsets_round1_{config.pooling}"
    if float(config.pairwise_aux_weight) > 0.0:
        return f"{base}_pairaux{config.pairwise_aux_weight:.2f}".replace(".", "p")
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GPQA DeepSets round-1 LOPO evaluation and system proxy patching")
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--overall-cache-root", default="MUI_HUB/cache")
    ap.add_argument("--current-science-json", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--distance-threads", type=int, default=_DEFAULT_DISTANCE_THREADS)
    ap.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    ap.add_argument("--torch-threads", type=int, default=max(1, min(8, (os.cpu_count() or 8))))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pairwise-aux-weight", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "result" / f"gpqa_deepsets_round1_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    current_science_json = Path(args.current_science_json) if args.current_science_json else _latest_current_science_json()
    current_science_payload = _load_current_science_payload(current_science_json)
    current_science_candidate = dict(current_science_payload["selected_candidate"])
    current_science_metrics = dict(current_science_candidate["metrics"])
    current_system_bundle = dict(current_science_candidate["comprehensive_metrics"])

    print("[gpqa-deepsets] Extracting GPQA problems …", flush=True)
    all_problems = _extract_all_problems(
        cache_root,
        distance_threads=int(args.distance_threads),
        workers=int(args.workers),
    )

    overall_entry_map = _load_entry_map(args.overall_cache_root)
    base_doc_metrics = _load_extreme12_test_metrics(EXTREME12_TEST_ANALYSIS_DOC)
    problem_counts = _problem_counts_from_entry_map(overall_entry_map, OVERALL_DS_CACHE_KEYS)
    for cache_key in OVERALL_DS_CACHE_KEYS:
        base_doc_metrics[cache_key]["n_problems"] = int(problem_counts[cache_key])
        base_doc_metrics[cache_key]["top10_count"] = max(1, int(math.ceil(0.10 * int(base_doc_metrics[cache_key]["n_samples"]))))

    code_v2_proxy_metrics = _load_code_v2_proxy_metrics(
        CODE_V2_EXHAUSTIVE_JSON,
        fallback_hit3=float(base_doc_metrics["DS-R1/lcb_v5"]["hit@3"]),
    )

    variant_rows: list[dict[str, Any]] = []
    variant_configs = [
        GPQADeepSetsConfig(
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
        GPQADeepSetsConfig(
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
            GPQADeepSetsConfig(
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
    for config in variant_configs:
        metrics, records, proxy_metrics = _run_variant_lopo(
            all_problems,
            config=config,
            torch_threads=int(args.torch_threads),
        )
        cache_metrics = {cache_key: dict(base_doc_metrics[cache_key]) for cache_key in OVERALL_DS_CACHE_KEYS}
        cache_metrics["DS-R1/lcb_v5"] = dict(code_v2_proxy_metrics)
        cache_metrics["DS-R1/gpqa"] = dict(proxy_metrics)
        candidate_bundle = _combine_cache_metric_proxy(cache_metrics)
        gpqa_avg_rank_delta = float((proxy_metrics.get("avg_rank_proxy") or 0.0) - (current_science_candidate["gpqa_proxy_metrics"].get("avg_rank_proxy") or 0.0))
        system_delta = _system_delta(candidate_bundle, current_system_bundle)
        total_problem_count = sum(problem_counts.values())
        n_system_caches = len(OVERALL_DS_CACHE_KEYS)
        gpqa_problem_count = int(problem_counts["DS-R1/gpqa"])
        system_delta["sample_weighted"]["avg_rank_proxy"] = float(gpqa_avg_rank_delta * gpqa_problem_count / max(total_problem_count, 1))
        system_delta["equal_cache_mean"]["avg_rank_proxy"] = float(gpqa_avg_rank_delta / max(n_system_caches, 1))
        system_gate_passed, system_gate_failed, system_delta = _comprehensive_gate(
            candidate_bundle,
            current_system_bundle,
            delta_override=system_delta,
        )
        science_gate_passed, science_gate_failed = _science_gate(metrics, current_science_metrics)

        variant_name = _variant_name(config)
        model_out = MODEL_DIR / f"{variant_name}.pkl"
        final_train_summary = _train_final_variant_model(
            all_problems,
            config=config,
            torch_threads=int(args.torch_threads),
            out_path=model_out,
        )

        variant_rows.append(
            {
                "name": variant_name,
                "pooling": config.pooling,
                "config": config.as_dict(),
                "metrics": _compact_metrics(metrics),
                "gpqa_proxy_metrics": proxy_metrics,
                "science_gate_passed": bool(science_gate_passed),
                "science_gate_failed": science_gate_failed,
                "system_gate_passed": bool(system_gate_passed),
                "system_gate_failed": system_gate_failed,
                "system_bundle": candidate_bundle,
                "system_delta": system_delta,
                "model_path": str(model_out),
                "final_train_summary": final_train_summary,
            }
        )

    selected_variant = _pick_best_variant(variant_rows)
    winner_model_path = Path(selected_variant["model_path"])
    DEFAULT_WINNER_MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    winner_bytes = winner_model_path.read_bytes()
    DEFAULT_WINNER_MODEL_OUT.write_bytes(winner_bytes)

    selected_metrics = dict(selected_variant["metrics"])
    selected_bundle = dict(selected_variant["system_bundle"])
    selected_delta = dict(selected_variant["system_delta"])
    promote = bool(selected_variant["science_gate_passed"] and selected_variant["system_gate_passed"])
    decision = "Promote" if promote else "No-Promote"

    alignment_rows = []
    code_v2_bundle = dict(current_science_payload["system_metrics"]["extreme12_plus_code_v2"])
    baseline_bundle = dict(current_science_payload["system_metrics"]["extreme12_base"])
    science_baseline_bundle = dict(current_science_payload["system_metrics"]["extreme12_plus_code_v2_plus_science_baseline_v1"])
    alignment_rows.append(
        {
            "name": "code_v2",
            **_system_delta(code_v2_bundle, baseline_bundle),
            "actual_leaderboard_delta": "unavailable in repo",
        }
    )
    alignment_rows.append(
        {
            "name": "science_baseline_v1",
            **_system_delta(science_baseline_bundle, code_v2_bundle),
            "actual_leaderboard_delta": "unavailable in repo",
        }
    )
    alignment_rows.append(
        {
            "name": "science_hybrid_round3",
            **dict(current_science_candidate["comprehensive_delta_vs_current"]),
            "actual_leaderboard_delta": "unavailable in repo",
        }
    )
    alignment_rows.append(
        {
            "name": "gpqa_deepsets_round1",
            **selected_delta,
            "actual_leaderboard_delta": "unavailable in repo",
        }
    )

    payload = {
        "status_summary": {
            "code_v2_is_promoted_default": True,
            "science_baseline_v1_is_frozen_baseline": True,
            "gpqa_pairwise_round2_is_no_promote": True,
            "science_hybrid_round3_has_narrow_full_system_promote": True,
            "graph_heavy_and_new_monotonic_recency_families_are_paused": True,
            "new_research_line_is_minimal_full_group_contextual_model": True,
        },
        "inputs": {
            "cache_root": str(cache_root),
            "overall_cache_root": str(args.overall_cache_root),
            "current_science_json": str(current_science_json),
            "distance_threads": int(args.distance_threads),
            "workers": int(args.workers),
            "torch_threads": int(args.torch_threads),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        },
        "compare_rows": {
            "science_baseline_v1": dict(current_science_payload["gpqa_metrics"]["science_baseline_v1"]),
            "gpqa_pairwise_round1": dict(current_science_payload["gpqa_metrics"]["gpqa_pairwise_round1"]),
            "science_hybrid_round3": dict(current_science_candidate["metrics"]),
            "gpqa_deepsets_variants": variant_rows,
            "selected_variant": selected_variant,
        },
        "current_system": {
            "science_slice_name": str(current_science_candidate["name"]),
            "science_slice_metrics": dict(current_science_candidate["metrics"]),
            "bundle": current_system_bundle,
        },
        "patched_system": {
            "science_slice_name": str(selected_variant["name"]),
            "science_slice_metrics": dict(selected_metrics),
            "bundle": selected_bundle,
            "delta_vs_current": selected_delta,
        },
        "alignment_analysis": {
            "rows": alignment_rows,
            "note": "Actual leaderboard/public/private score deltas are unavailable in repo; proxy deltas are reported under one unified current proxy definition.",
        },
        "decision": {
            "promote": promote,
            "label": decision,
            "science_gate_passed": bool(selected_variant["science_gate_passed"]),
            "science_gate_failed": list(selected_variant["science_gate_failed"]),
            "system_gate_passed": bool(selected_variant["system_gate_passed"]),
            "system_gate_failed": list(selected_variant["system_gate_failed"]),
        },
    }

    (out_dir / "gpqa_deepsets_round1.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    gpqa_rows = [
        {"name": "science_baseline_v1", **dict(current_science_payload["gpqa_metrics"]["science_baseline_v1"])},
        {"name": "gpqa_pairwise_round1", **dict(current_science_payload["gpqa_metrics"]["gpqa_pairwise_round1"])},
        {"name": "science_hybrid_round3", **dict(current_science_candidate["metrics"])},
        *[
            {"name": row["name"], **dict(row["metrics"])}
            for row in variant_rows
        ],
    ]
    sample_rows = [
        {"name": "current = code_v2 + science_hybrid_round3", **dict(current_system_bundle["sample_weighted"])},
        {"name": f"patched = code_v2 + {selected_variant['name']}", **dict(selected_bundle["sample_weighted"])},
    ]
    equal_rows = [
        {"name": "current = code_v2 + science_hybrid_round3", **dict(current_system_bundle["equal_cache_mean"])},
        {"name": f"patched = code_v2 + {selected_variant['name']}", **dict(selected_bundle["equal_cache_mean"])},
    ]
    summary_lines = [
        "# GPQA DeepSets Round 1",
        "",
        "## GPQA Single-Domain",
        "",
        _result_rows_markdown(gpqa_rows),
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
        "### Patched delta vs current",
        "",
        _delta_rows_markdown(
            [
                {
                    "name": selected_variant["name"],
                    **selected_delta,
                    "actual_leaderboard_delta": "unavailable in repo",
                }
            ]
        ),
        "",
        "## Offline–Online Alignment",
        "",
        _delta_rows_markdown(alignment_rows),
        "",
        "- Read: sample-weighted `Hit@1` plus `avg_rank_proxy` is the closest thing to the repo's current promote-sensitive direction.",
        "- Warning: the repo still does not contain a trustworthy leaderboard delta table, so proxy/leaderboard alignment remains unverified.",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "selected_variant": selected_variant["name"],
        "decision": decision,
        "promote": promote,
        "winner_model_path": str(DEFAULT_WINNER_MODEL_OUT),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
