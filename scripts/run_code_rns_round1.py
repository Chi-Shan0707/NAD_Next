#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.core.distance.engine import DistanceEngine, DistanceSpec
from nad.core.selectors.base import SelectorContext
from nad.core.selectors.code_rns_impl import (
    CodeRNSConfig,
    CodeRNSScorer,
    extract_code_rns_context_payload,
)
from nad.core.views.reader import CacheReader
from nad.ops.accuracy import _load_ground_truth
from nad.ops.bestofn_extreme8 import build_problem_groups, discover_cache_entries
from scripts.run_bestofn_score_recovery_20260408 import (
    DEFAULT_SCIENCE_JSON_GLOB,
    _display_path,
    _load_current_system_bundle,
)
from scripts.run_code_baseline_v1_phase2 import (
    CODE_CACHE_KEY,
    DEFAULT_VIEW,
    MetricAccumulator,
    _load_entry_map,
    _problem_sort_key,
)
from scripts.run_science_hybrid_round3 import (
    CODE_V2_EXHAUSTIVE_JSON,
    ProblemScoreRecord,
    _combine_cache_metric_proxy,
    _evaluate_problem_records,
    _system_delta,
)

DEFAULT_OUT_ROOT = REPO_ROOT / "result"
DEFAULT_MODEL_OUT = REPO_ROOT / "models" / "ml_selectors" / "code_rns_round1.pkl"
DEFAULT_GT_CACHE_ROOT = REPO_ROOT / "MUI_HUB" / "cache"
DEFAULT_BLIND_CACHE_ROOT = Path("/home/jovyan/public-ro/MUI_HUB/cache_test")
BLIND_QWEN_CODE_KEY = "Qwen3-4B/lcb_v5"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    use_rns: bool
    shortlist_size: int
    knn_k: int
    lambda_weight: float
    use_cf_negatives: bool

    def config(self) -> CodeRNSConfig:
        return CodeRNSConfig(
            shortlist_size=int(self.shortlist_size),
            knn_k=int(self.knn_k),
            lambda_weight=float(self.lambda_weight),
            use_cf_negatives=bool(self.use_cf_negatives),
        ).validate()


@dataclass
class CodeRNSProblemData:
    problem_id: str
    run_ids: list[int]
    labels: np.ndarray
    D: np.ndarray
    code_v2_scores: np.ndarray
    X_rank: np.ndarray
    nuisance: np.ndarray


@dataclass
class BlindCodeProblemData:
    problem_id: str
    run_ids: list[int]
    D: np.ndarray
    code_v2_scores: np.ndarray
    X_rank: np.ndarray


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec("code_v2_baseline", False, 0, 0, 0.0, False),
    CandidateSpec("rns_hard_only__top10__knn5__lam0p15", True, 10, 5, 0.15, False),
    CandidateSpec("rns_hard_cf__top10__knn5__lam0p15", True, 10, 5, 0.15, True),
    CandidateSpec("rns_hard_cf__top10__knn5__lam0p25", True, 10, 5, 0.25, True),
    CandidateSpec("rns_hard_cf__top10__knn3__lam0p15", True, 10, 3, 0.15, True),
    CandidateSpec("rns_hard_cf__top10__knn8__lam0p15", True, 10, 8, 0.15, True),
    CandidateSpec("rns_hard_cf__top5__knn5__lam0p15", True, 5, 5, 0.15, True),
    CandidateSpec("rns_hard_cf__top5__knn5__lam0p25", True, 5, 5, 0.25, True),
)


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _latest_current_science_json() -> Path:
    matches = sorted(glob.glob(str(REPO_ROOT / DEFAULT_SCIENCE_JSON_GLOB)))
    if not matches:
        raise FileNotFoundError("No science_hybrid_round3.json payload found under result/")
    return Path(matches[-1])


def _build_proxy_bundle(
    *,
    current_bundle: dict[str, Any],
    current_cache_metrics: dict[str, dict[str, Any]],
    candidate_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_cache_metrics = {cache_key: dict(metrics) for cache_key, metrics in current_cache_metrics.items()}
    full_cache_metrics[CODE_CACHE_KEY] = dict(candidate_metrics)
    full_bundle = _combine_cache_metric_proxy(full_cache_metrics)
    return full_bundle, _system_delta(full_bundle, current_bundle)


def _coding_rns_gate(candidate_metrics: dict[str, Any], current_metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failed: list[str] = []
    if float(candidate_metrics.get("selacc@10%") or 0.0) < float(current_metrics.get("selacc@10%") or 0.0):
        failed.append("SelAcc@10 below current code_v2")
    if float(candidate_metrics.get("pairwise") or 0.0) < 0.50:
        failed.append("Pairwise below 50%")
    if float(candidate_metrics.get("hit@1") or 0.0) < float(current_metrics.get("hit@1") or 0.0) - 0.0025:
        failed.append("Hit@1 below current code_v2 minus 0.25pp guardrail")
    return (len(failed) == 0, failed)


def _row_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    blind = dict(row.get("blind_qwen_risk") or {})
    return (
        bool(row.get("system_gate_passed")),
        bool(row.get("coding_gate_passed")),
        float(row["system_delta"]["sample_weighted"]["hit@1"]),
        float(row["system_delta"]["sample_weighted"]["selacc@10%"]),
        float(row["metrics"].get("selacc@10%") or 0.0),
        float(row["metrics"].get("hit@1") or 0.0),
        -float(blind.get("flip_rate") or 0.0),
        str(row["name"]),
    )


def _preload_code_problems(
    cache_root: Path,
    *,
    distance_threads: int,
    max_problems: int = 0,
) -> list[CodeRNSProblemData]:
    meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    correctness = _load_ground_truth(cache_root)
    reader = CacheReader(str(cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))

    problems: list[CodeRNSProblemData] = []
    for idx, (problem_id, run_ids) in enumerate(sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))):
        if int(max_problems) > 0 and len(problems) >= int(max_problems):
            break
        if idx % 20 == 0:
            print(f"[code-rns-preload] problem {idx + 1}/{len(groups)}", flush=True)
        run_ids = list(map(int, run_ids))
        if len(run_ids) < 2:
            continue
        views = [reader.get_run_view(int(run_id), DEFAULT_VIEW) for run_id in run_ids]
        D = engine.dense_matrix(views)
        ctx = SelectorContext(
            cache=reader,
            problem_id=str(problem_id),
            run_ids=run_ids,
            views=views,
        )
        payload = extract_code_rns_context_payload(ctx)
        labels = np.asarray([int(bool(correctness.get(int(run_id), False))) for run_id in run_ids], dtype=np.int32)
        problems.append(
            CodeRNSProblemData(
                problem_id=str(problem_id),
                run_ids=run_ids,
                labels=labels,
                D=np.asarray(D, dtype=np.float64),
                code_v2_scores=np.asarray(payload["baseline_scores"], dtype=np.float64),
                X_rank=np.asarray(payload["features"], dtype=np.float64),
                nuisance=np.asarray(payload["nuisance"], dtype=np.float64),
            )
        )
    return problems


def _preload_blind_qwen_problems(
    cache_root: Path,
    *,
    distance_threads: int,
    max_problems: int = 0,
) -> list[BlindCodeProblemData]:
    entry_map = {entry.cache_key: entry for entry in discover_cache_entries(cache_root)}
    entry = entry_map.get(BLIND_QWEN_CODE_KEY)
    if entry is None:
        raise FileNotFoundError(f"Missing blind code cache key under {_display_path(cache_root)}: {BLIND_QWEN_CODE_KEY}")
    meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
    groups = build_problem_groups(meta)
    reader = CacheReader(str(entry.cache_root))
    engine = DistanceEngine(DistanceSpec("ja", num_threads=int(distance_threads)))

    problems: list[BlindCodeProblemData] = []
    for idx, (problem_id, run_ids) in enumerate(sorted(groups.items(), key=lambda kv: _problem_sort_key(kv[0]))):
        if int(max_problems) > 0 and len(problems) >= int(max_problems):
            break
        if idx % 20 == 0:
            print(f"[code-rns-blind] problem {idx + 1}/{len(groups)}", flush=True)
        run_ids = list(map(int, run_ids))
        if len(run_ids) < 2:
            continue
        views = [reader.get_run_view(int(run_id), DEFAULT_VIEW) for run_id in run_ids]
        D = engine.dense_matrix(views)
        ctx = SelectorContext(
            cache=reader,
            problem_id=str(problem_id),
            run_ids=run_ids,
            views=views,
        )
        payload = extract_code_rns_context_payload(ctx)
        problems.append(
            BlindCodeProblemData(
                problem_id=str(problem_id),
                run_ids=run_ids,
                D=np.asarray(D, dtype=np.float64),
                code_v2_scores=np.asarray(payload["baseline_scores"], dtype=np.float64),
                X_rank=np.asarray(payload["features"], dtype=np.float64),
            )
        )
    return problems


def _records_proxy_metrics(records: list[ProblemScoreRecord]) -> dict[str, Any]:
    return _evaluate_problem_records(records)


def _evaluate_baseline_candidate(
    problems: list[CodeRNSProblemData],
    *,
    current_bundle: dict[str, Any],
    current_cache_metrics: dict[str, dict[str, Any]],
    current_code_metrics: dict[str, Any],
    blind_total_problems: int,
) -> dict[str, Any]:
    acc = MetricAccumulator("code_v2_baseline", use_code_tiebreak=True)
    records: list[ProblemScoreRecord] = []
    for prob in problems:
        scores = np.asarray(prob.code_v2_scores, dtype=np.float64)
        acc.add_problem(prob.problem_id, prob.run_ids, scores, prob.labels, prob.D)
        records.append(
            ProblemScoreRecord(
                cache_key=CODE_CACHE_KEY,
                problem_id=str(prob.problem_id),
                sample_ids=list(map(int, prob.run_ids)),
                labels=np.asarray(prob.labels, dtype=np.int32),
                scores=scores,
            )
        )
    metrics = acc.finalize()
    proxy_metrics = _records_proxy_metrics(records)
    full_bundle, system_delta = _build_proxy_bundle(
        current_bundle=current_bundle,
        current_cache_metrics=current_cache_metrics,
        candidate_metrics=proxy_metrics,
    )
    return {
        "name": "code_v2_baseline",
        "family": "baseline",
        "config": {},
        "metrics": metrics,
        "coding_proxy_metrics": proxy_metrics,
        "coding_gate_passed": True,
        "coding_gate_failed": [],
        "system_gate_passed": True,
        "system_gate_failed": [],
        "full_system_proxy": full_bundle,
        "system_delta": system_delta,
        "blind_qwen_risk": {
            "changed_problems": 0,
            "total_problems": int(blind_total_problems),
            "flip_rate": 0.0,
        },
        "mean_anchor_counts": None,
    }


def _evaluate_rns_candidate(
    candidate: CandidateSpec,
    problems: list[CodeRNSProblemData],
    *,
    current_bundle: dict[str, Any],
    current_cache_metrics: dict[str, dict[str, Any]],
    current_code_metrics: dict[str, Any],
) -> dict[str, Any]:
    acc = MetricAccumulator(candidate.name, use_code_tiebreak=True)
    records: list[ProblemScoreRecord] = []
    anchor_summaries: list[dict[str, Any]] = []

    for held_out_idx, held_out in enumerate(problems):
        if held_out_idx % 20 == 0:
            print(f"[code-rns:{candidate.name}] fold {held_out_idx + 1}/{len(problems)}", flush=True)
        train_X = np.concatenate([prob.X_rank for idx, prob in enumerate(problems) if idx != held_out_idx], axis=0)
        train_y = np.concatenate([prob.labels for idx, prob in enumerate(problems) if idx != held_out_idx], axis=0)
        train_nuisance = np.concatenate([prob.nuisance for idx, prob in enumerate(problems) if idx != held_out_idx], axis=0)
        scorer = CodeRNSScorer(config=candidate.config())
        scorer.fit_anchor_bank(train_X, train_y, nuisance=train_nuisance)
        scores = scorer.decision_from_group_features(
            held_out.X_rank,
            held_out.code_v2_scores,
            held_out.D,
            run_ids=held_out.run_ids,
        ).final_scores
        acc.add_problem(held_out.problem_id, held_out.run_ids, scores, held_out.labels, held_out.D)
        records.append(
            ProblemScoreRecord(
                cache_key=CODE_CACHE_KEY,
                problem_id=str(held_out.problem_id),
                sample_ids=list(map(int, held_out.run_ids)),
                labels=np.asarray(held_out.labels, dtype=np.int32),
                scores=np.asarray(scores, dtype=np.float64),
            )
        )
        anchor_summaries.append(dict(scorer.bundle.training_summary if scorer.bundle is not None else {}))

    metrics = acc.finalize()
    proxy_metrics = _records_proxy_metrics(records)
    full_bundle, system_delta = _build_proxy_bundle(
        current_bundle=current_bundle,
        current_cache_metrics=current_cache_metrics,
        candidate_metrics=proxy_metrics,
    )
    coding_gate_passed, coding_gate_failed = _coding_rns_gate(metrics, current_code_metrics)
    system_gate_passed = bool(
        float(system_delta["sample_weighted"]["hit@1"]) >= 0.0
        and float(system_delta["sample_weighted"]["selacc@10%"]) >= -1e-9
    )
    system_gate_failed = [] if system_gate_passed else ["Full-system proxy regressed vs current code_v2 + science_hybrid_round3 stack"]

    mean_anchor_counts = None
    if anchor_summaries:
        mean_anchor_counts = {
            "n_positive_anchors": float(np.mean([float(row.get("n_positive_anchors", 0.0)) for row in anchor_summaries])),
            "n_hard_negative_anchors": float(np.mean([float(row.get("n_hard_negative_anchors", 0.0)) for row in anchor_summaries])),
            "n_cf_negative_anchors": float(np.mean([float(row.get("n_cf_negative_anchors", 0.0)) for row in anchor_summaries])),
        }

    return {
        "name": candidate.name,
        "family": "code_rns",
        "config": candidate.config().as_dict(),
        "metrics": metrics,
        "coding_proxy_metrics": proxy_metrics,
        "coding_gate_passed": bool(coding_gate_passed),
        "coding_gate_failed": coding_gate_failed,
        "system_gate_passed": bool(system_gate_passed),
        "system_gate_failed": system_gate_failed,
        "full_system_proxy": full_bundle,
        "system_delta": system_delta,
        "blind_qwen_risk": None,
        "mean_anchor_counts": mean_anchor_counts,
    }


def _fit_full_candidate_scorer(candidate: CandidateSpec, problems: list[CodeRNSProblemData]) -> CodeRNSScorer | None:
    if not candidate.use_rns:
        return None
    scorer = CodeRNSScorer(config=candidate.config())
    scorer.fit_anchor_bank(
        np.concatenate([prob.X_rank for prob in problems], axis=0),
        np.concatenate([prob.labels for prob in problems], axis=0),
        nuisance=np.concatenate([prob.nuisance for prob in problems], axis=0),
    )
    return scorer


def _blind_qwen_flip_risk(
    scorer: CodeRNSScorer | None,
    blind_problems: list[BlindCodeProblemData],
) -> dict[str, Any]:
    total = int(len(blind_problems))
    if scorer is None:
        return {"changed_problems": 0, "total_problems": total, "flip_rate": 0.0}
    changed = 0
    for prob in blind_problems:
        decision = scorer.decision_from_group_features(
            prob.X_rank,
            prob.code_v2_scores,
            prob.D,
            run_ids=prob.run_ids,
        )
        baseline_run_id = int(prob.run_ids[int(decision.baseline_order[0])]) if decision.baseline_order.size else -1
        final_run_id = int(prob.run_ids[int(decision.final_order[0])]) if decision.final_order.size else -1
        if baseline_run_id != final_run_id:
            changed += 1
    flip_rate = (float(changed) / float(total)) if total > 0 else 0.0
    return {
        "changed_problems": int(changed),
        "total_problems": int(total),
        "flip_rate": float(flip_rate),
    }


def _train_and_save_best_candidate(
    candidate_row: dict[str, Any],
    spec_by_name: dict[str, CandidateSpec],
    problems: list[CodeRNSProblemData],
    *,
    model_out: Path,
) -> str | None:
    candidate = spec_by_name.get(str(candidate_row["name"]))
    if candidate is None or not candidate.use_rns:
        return None
    scorer = _fit_full_candidate_scorer(candidate, problems)
    if scorer is None:
        return None
    if scorer.bundle is not None:
        scorer.bundle.training_summary["selected_candidate_name"] = str(candidate_row["name"])
        scorer.bundle.training_summary["selected_candidate_system_gate_passed"] = bool(candidate_row.get("system_gate_passed"))
        scorer.bundle.training_summary["selected_candidate_coding_gate_passed"] = bool(candidate_row.get("coding_gate_passed"))
    scorer.save(model_out)
    return _display_path(model_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Coding RNS round-1: shortlist rerank on top of code_v2")
    ap.add_argument("--gt-cache-root", default=str(DEFAULT_GT_CACHE_ROOT))
    ap.add_argument("--cache-root", default="")
    ap.add_argument("--blind-cache-root", default=str(DEFAULT_BLIND_CACHE_ROOT))
    ap.add_argument("--science-json", default="")
    ap.add_argument("--code-v2-json", default=str(CODE_V2_EXHAUSTIVE_JSON))
    ap.add_argument("--distance-threads", type=int, default=8)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--blind-max-problems", type=int, default=0)
    ap.add_argument("--skip-save-model", action="store_true")
    args = ap.parse_args()

    gt_cache_root = Path(args.gt_cache_root)
    blind_cache_root = Path(args.blind_cache_root)
    science_json = Path(args.science_json) if args.science_json else _latest_current_science_json()
    code_v2_json = Path(args.code_v2_json)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / f"code_rns_round1_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    current_bundle, current_cache_metrics, _ = _load_current_system_bundle(
        ds_cache_root=gt_cache_root,
        science_json=science_json,
        code_v2_json=code_v2_json,
    )
    current_code_metrics = dict(current_cache_metrics[CODE_CACHE_KEY])
    gt_entry_map = _load_entry_map(str(gt_cache_root))
    cache_root = Path(args.cache_root) if args.cache_root else Path(gt_entry_map[CODE_CACHE_KEY].cache_root)

    print("[code-rns] preloading DS-R1/lcb_v5 problems …", flush=True)
    problems = _preload_code_problems(
        cache_root,
        distance_threads=int(args.distance_threads),
        max_problems=int(args.max_problems),
    )
    print("[code-rns] preloading blind Qwen LCB problems …", flush=True)
    blind_max_problems = int(args.blind_max_problems) if int(args.blind_max_problems) > 0 else int(args.max_problems)
    blind_qwen_problems = _preload_blind_qwen_problems(
        blind_cache_root,
        distance_threads=int(args.distance_threads),
        max_problems=int(blind_max_problems),
    )

    rows: list[dict[str, Any]] = [
        _evaluate_baseline_candidate(
            problems,
            current_bundle=current_bundle,
            current_cache_metrics=current_cache_metrics,
            current_code_metrics=current_code_metrics,
            blind_total_problems=len(blind_qwen_problems),
        )
    ]
    spec_by_name = {spec.name: spec for spec in CANDIDATES}

    for candidate in CANDIDATES[1:]:
        rows.append(
            _evaluate_rns_candidate(
                candidate,
                problems,
                current_bundle=current_bundle,
                current_cache_metrics=current_cache_metrics,
                current_code_metrics=current_code_metrics,
            )
        )

    for row in rows:
        candidate = spec_by_name.get(str(row["name"]))
        scorer = _fit_full_candidate_scorer(candidate, problems) if candidate is not None else None
        row["blind_qwen_risk"] = _blind_qwen_flip_risk(scorer, blind_qwen_problems)

    sorted_rows = sorted(rows, key=_row_rank_key, reverse=True)
    promote_rows = [
        row
        for row in sorted_rows
        if str(row.get("family")) == "code_rns" and bool(row.get("coding_gate_passed")) and bool(row.get("system_gate_passed"))
    ]
    recommended_candidate = promote_rows[0] if promote_rows else rows[0]
    best_nonbaseline = max([row for row in rows if str(row.get("family")) == "code_rns"], key=_row_rank_key)

    saved_model_path = None
    if not bool(args.skip_save_model):
        saved_model_path = _train_and_save_best_candidate(
            best_nonbaseline,
            spec_by_name,
            problems,
            model_out=DEFAULT_MODEL_OUT,
        )

    payload = {
        "metadata": {
            "title": "Coding RNS Round 1",
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "code_v2_is_promoted_default": True,
            "science_hybrid_round3_is_promoted_patch": True,
            "goal": "conservative coding shortlist calibration using code_v2 feature space only",
            "no_new_model_training": True,
            "no_deepsets_or_attention_changes": True,
        },
        "protocol": {
            "gt_cache_root": _display_path(gt_cache_root),
            "coding_cache_root": _display_path(cache_root),
            "blind_cache_root": _display_path(blind_cache_root),
            "science_json": _display_path(science_json),
            "code_v2_json": _display_path(code_v2_json),
            "distance_threads": int(args.distance_threads),
            "max_problems": int(args.max_problems),
            "blind_max_problems": int(blind_max_problems),
            "candidate_count": int(len(CANDIDATES)),
        },
        "current_code_metrics": current_code_metrics,
        "current_system_proxy": current_bundle,
        "candidate_rows": rows,
        "sorted_candidates": [row["name"] for row in sorted_rows],
        "recommended_candidate": recommended_candidate,
        "best_nonbaseline_candidate": best_nonbaseline,
        "saved_model_path": saved_model_path,
    }

    out_path = out_dir / "code_rns_round1.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {_display_path(out_path)}", flush=True)
    if saved_model_path:
        print(f"[saved] {saved_model_path}", flush=True)
    print(f"[recommended] {recommended_candidate['name']}", flush=True)


if __name__ == "__main__":
    main()
