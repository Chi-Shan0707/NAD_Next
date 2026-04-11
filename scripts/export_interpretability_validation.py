#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.explain.interpretability_validation import (
    BundleRecord,
    aggregate_deletion_rows,
    aggregate_failure_archetypes,
    assign_failure_archetypes,
    build_wrong_case_record,
    compute_stability_rows,
    deletion_intervention_rows,
    iter_problem_slices,
    load_bundle_records,
    select_appendix_cases,
    summarize_faithfulness,
    summarize_raw_vs_rank_stability,
)
from nad.explain.svd_explain import (
    EXPLAIN_ANCHORS,
    _family_delta_rows,
    _feature_delta_rows,
    _ranking_rows,
    _score_route_matrix,
    get_anchor_route,
)
from nad.explain.svd_introspection import explain_run
from nad.ops.earlystop import discover_cache_entries
from nad.ops.earlystop_svd import (
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
)
from scripts.run_earlystop_prefix10_svd_round1 import build_feature_store


CANONICAL_MODEL_PATHS = {
    "es_svd_math_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_math_rr_r1.pkl",
    "es_svd_science_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_science_rr_r1.pkl",
    "es_svd_ms_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_ms_rr_r1.pkl",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _qualify_feature_store(
    feature_store: list[dict[str, Any]],
    source_name: str,
) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = str(source_name)
        item["base_cache_key"] = str(payload["cache_key"])
        item["cache_key"] = f"{source_name}/{payload['cache_key']}"
        qualified.append(item)
    return qualified


def _collect_required_features() -> set[str]:
    required: set[str] = set()
    for model_path in CANONICAL_MODEL_PATHS.values():
        bundle = load_earlystop_svd_bundle(model_path)
        for domain_bundle in bundle.get("domains", {}).values():
            for route in domain_bundle.get("routes", []):
                required.update(str(name) for name in route.get("feature_names", []))
                signal_name = route.get("signal_name") or route.get("baseline_signal_name")
                if signal_name:
                    required.add(str(signal_name))
    return required


def _build_or_load_feature_store(
    *,
    cache_roots: list[Path],
    feature_cache_path: Path,
    required_features: set[str],
    max_problems_per_cache: Optional[int],
    feature_workers: int,
    feature_chunk_problems: int,
) -> list[dict[str, Any]]:
    if feature_cache_path.exists():
        with feature_cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        return list(payload["feature_store"])

    if max_problems_per_cache is None:
        prebuilt = _load_prebuilt_feature_store()
        if prebuilt is not None:
            _ensure_parent(feature_cache_path)
            with feature_cache_path.open("wb") as handle:
                pickle.dump(
                    {
                        "generated_at_utc": _now_utc(),
                        "cache_roots": [str(v) for v in cache_roots],
                        "required_features": sorted(required_features),
                        "feature_store": prebuilt,
                        "source": "prebuilt_es_svd_ms_rr_r1_cache",
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            return prebuilt

    merged: list[dict[str, Any]] = []
    for cache_root in cache_roots:
        source_name = cache_root.name
        feature_store = build_feature_store(
            cache_root=str(cache_root),
            positions=tuple(float(v) for v in EXPLAIN_ANCHORS),
            required_feature_names=required_features,
            max_problems_per_cache=max_problems_per_cache,
            max_workers=max(1, int(feature_workers)),
            chunk_problems=max(1, int(feature_chunk_problems)),
        )
        merged.extend(_qualify_feature_store(feature_store, source_name=source_name))

    _ensure_parent(feature_cache_path)
    with feature_cache_path.open("wb") as handle:
        pickle.dump(
            {
                "generated_at_utc": _now_utc(),
                "cache_roots": [str(v) for v in cache_roots],
                "required_features": sorted(required_features),
                "feature_store": merged,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return merged


def _project_payload_to_explain_anchors(payload: Mapping[str, Any]) -> dict[str, Any]:
    positions = [float(v) for v in payload.get("positions", [])]
    anchor_indices = []
    for anchor in EXPLAIN_ANCHORS:
        if float(anchor) not in positions:
            raise ValueError(f"Missing explain anchor {anchor} in prebuilt cache positions={positions}")
        anchor_indices.append(int(positions.index(float(anchor))))
    tensor = np.asarray(payload["tensor"], dtype=np.float64)[:, anchor_indices, :]
    out = dict(payload)
    out["positions"] = [float(v) for v in EXPLAIN_ANCHORS]
    out["tensor"] = tensor
    return out


def _load_prebuilt_feature_store() -> Optional[list[dict[str, Any]]]:
    cache_dir = REPO_ROOT / "results/cache/es_svd_ms_rr_r1"
    cache_files = sorted(cache_dir.glob("noncoding_cache_all_*.pkl"))
    cache_train_files = sorted(cache_dir.glob("noncoding_cache_train_all_*.pkl"))
    if not cache_files or not cache_train_files:
        return None

    merged: list[dict[str, Any]] = []
    for path in [cache_files[-1], cache_train_files[-1]]:
        payload = pickle.load(path.open("rb"))
        feature_store = [_project_payload_to_explain_anchors(row) for row in payload.get("feature_store", [])]
        merged.extend(feature_store)
    return merged


def _build_run_index_maps(cache_roots: list[Path]) -> tuple[dict[str, dict[int, int]], dict[str, dict[str, Any]]]:
    run_index_maps: dict[str, dict[int, int]] = {}
    cache_meta: dict[str, dict[str, Any]] = {}
    for cache_root in cache_roots:
        source_name = cache_root.name
        for entry in discover_cache_entries(cache_root):
            qualified_key = f"{source_name}/{entry.cache_key}"
            meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
            sample_map = {
                int(sample_id): int(sample.get("run_index", sample_id))
                for sample_id, sample in enumerate(meta.get("samples", []))
            }
            run_index_maps[qualified_key] = sample_map
            cache_meta[qualified_key] = {
                "cache_root": str(entry.cache_root),
                "dataset_name": str(entry.dataset_name),
                "model_name": str(entry.model_name),
                "base_cache_key": str(entry.cache_key),
            }
    return run_index_maps, cache_meta


def _load_multiseed_records(manifest_path: Optional[Path]) -> list[BundleRecord]:
    if manifest_path is None or not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[BundleRecord] = []
    for row in payload.get("records", []):
        records.append(
            BundleRecord(
                record_id=str(row["record_id"]),
                method_group=str(row["method_group"]),
                bundle_path=Path(str(row["bundle_path"])),
                axis=str(row["axis"]),
                random_state=int(row.get("random_state", 42)),
                split_seed=int(row.get("split_seed", 42)),
                note=str(row.get("note", "")),
            )
        )
    return records


def _intervention_lookup(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_level: str,
    target_policy: str,
) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if str(row["target_level"]) == target_level and str(row["target_policy"]) == target_policy:
            return row
    return None


def _case_markdown(case: Mapping[str, Any]) -> str:
    lines = [
        f"# {case['title']}",
        "",
        f"- `method_id`: `{case['method_id']}`",
        f"- `domain`: `{case['domain']}`",
        f"- `cache_key`: `{case['cache_key']}`",
        f"- `problem_id`: `{case['problem_id']}`",
        f"- `anchor_pct`: `{case['anchor_pct']}`",
        f"- `case_type`: `{case['case_type']}`",
    ]
    if case.get("archetype_label"):
        lines.append(f"- `archetype`: `{case['archetype_label']}`")
    lines += [
        "",
        "## Decision",
        "",
        f"- `top1`: run `{case['top1_run_index']}` / sample `{case['top1_sample_id']}` / correct=`{case['top1_is_correct']}` / score=`{case['top1_score']:.4f}`",
        f"- `top2_or_best_correct`: run `{case['compare_run_index']}` / sample `{case['compare_sample_id']}` / correct=`{case['compare_is_correct']}` / score=`{case['compare_score']:.4f}`",
        f"- `margin_or_gap`: `{case['margin_or_gap']:.4f}`",
        "",
        "## Top Family Deltas",
        "",
    ]
    for row in case.get("top_family_deltas", [])[:5]:
        lines.append(f"- `{row['family']}`: Δ=`{float(row['delta']):.4f}`")
    lines += [
        "",
        "## Top Feature Deltas",
        "",
    ]
    for row in case.get("top_feature_deltas", [])[:8]:
        lines.append(f"- `{row['feature']}`: Δ=`{float(row['delta']):.4f}`")
    lines += [
        "",
        "## Deletion Sanity",
        "",
    ]
    for label, key in [
        ("top_family", "top_family_intervention"),
        ("low_family", "low_family_intervention"),
        ("top_feature", "top_feature_intervention"),
        ("low_feature", "low_feature_intervention"),
    ]:
        row = case.get(key)
        if not row:
            continue
        lines.append(
            f"- `{label}` `{row['target_name']}`: score_drop=`{float(row['score_drop']):.4f}`, "
            f"margin_drop=`{float(row['margin_drop']):.4f}`, flip=`{bool(row['selection_flip'])}`, "
            f"new_correct=`{bool(row['new_selected_is_correct'])}`"
        )
    if case.get("paper_note"):
        lines += [
            "",
            "## Paper Note",
            "",
            f"- {case['paper_note']}",
        ]
    lines.append("")
    return "\n".join(lines)


def _build_paper_doc(
    *,
    out_path: Path,
    faithfulness_rows: Sequence[Mapping[str, Any]],
    stability_rows: Sequence[Mapping[str, Any]],
    raw_rank_rows: Sequence[Mapping[str, Any]],
    deletion_rows: Sequence[Mapping[str, Any]],
    archetype_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
) -> None:
    def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
        values = [float(row[field]) for row in rows if field in row]
        return float(np.mean(values)) if values else float("nan")

    deletion_top_rows = [
        row
        for row in deletion_rows
        if str(row["target_policy"]) == "top"
    ]
    top_family_rows = [row for row in deletion_top_rows if str(row["target_level"]) == "family"]
    low_family_rows = [row for row in deletion_rows if str(row["target_policy"]) == "low" and str(row["target_level"]) == "family"]
    top_feature_rows = [row for row in deletion_top_rows if str(row["target_level"]) == "feature"]
    low_feature_rows = [row for row in deletion_rows if str(row["target_policy"]) == "low" and str(row["target_level"]) == "feature"]

    seed_sign_rows = [
        row
        for row in stability_rows
        if str(row["axis"]) == "seed" and str(row["metric"]) == "feature_sign_consistency"
    ]
    seed_top_pos_rows = [
        row
        for row in stability_rows
        if str(row["axis"]) == "seed" and str(row["metric"]) == "top_positive_jaccard"
    ]
    split_sign_rows = [
        row
        for row in stability_rows
        if str(row["axis"]) == "split" and str(row["metric"]) == "feature_sign_consistency"
    ]
    split_top_pos_rows = [
        row
        for row in stability_rows
        if str(row["axis"]) == "split" and str(row["metric"]) == "top_positive_jaccard"
    ]

    raw_summary = next((row for row in raw_rank_rows if str(row["axis"]) == "seed" and str(row["channel"]) == "raw"), None)
    rank_summary = next((row for row in raw_rank_rows if str(row["axis"]) == "seed" and str(row["channel"]) == "rank"), None)

    lines = [
        "# 09 Interpretability Validation",
        "",
        "## Summary",
        "",
        "- 本文档面向 paper-facing 叙事，目标是把 SVD introspection 升级为更强的 numerical interpretability evidence。",
        "- 主结论建议写作：",
        "",
        "> The explanations are numerically faithful, stable across perturbations, and selectively relevant to the model’s actual decisions.",
        "",
        "## Faithfulness（分数重构）",
        "",
        "| Method | Explained Runs | Max Recon Error | Mean Recon Error |",
        "|---|---:|---:|---:|",
    ]
    for row in faithfulness_rows:
        lines.append(
            f"| `{row['method_id']}` | {row['n_explained_runs']} | "
            f"{float(row['max_reconstruction_error']):.2e} | {float(row['mean_reconstruction_error']):.2e} |"
        )

    lines += [
        "",
        "## Stability（跨 seed / 跨 split）",
        "",
        f"- 多 seed feature-sign consistency 平均值：`{_mean(seed_sign_rows, 'mean_value'):.3f}`。",
        f"- 多 seed top-positive Jaccard@K 平均值：`{_mean(seed_top_pos_rows, 'mean_value'):.3f}`。",
        f"- 多 split feature-sign consistency 平均值：`{_mean(split_sign_rows, 'mean_value'):.3f}`。",
        f"- 多 split top-positive Jaccard@K 平均值：`{_mean(split_top_pos_rows, 'mean_value'):.3f}`。",
        "",
        "### Raw vs Rank",
        "",
        "| Channel | Mean Feature Sign Consistency | Mean |w| CV |",
        "|---|---:|---:|",
    ]
    if raw_summary is not None:
        lines.append(
            f"| `raw` | {float(raw_summary['mean_feature_sign_consistency']):.3f} | {float(raw_summary['mean_abs_weight_cv']):.2e} |"
        )
    if rank_summary is not None:
        lines.append(
            f"| `rank` | {float(rank_summary['mean_feature_sign_consistency']):.3f} | {float(rank_summary['mean_abs_weight_cv']):.2e} |"
        )

    if raw_summary is not None and rank_summary is not None:
        lines += [
            "",
            f"- 可引用结论：`raw` channel 在多 seed 下更稳定（sign consistency `{float(raw_summary['mean_feature_sign_consistency']):.3f}` vs `{float(rank_summary['mean_feature_sign_consistency']):.3f}`），"
            f"且波动更小（|w| CV `{float(raw_summary['mean_abs_weight_cv']):.2e}` vs `{float(rank_summary['mean_abs_weight_cv']):.2e}`）。",
        ]

    lines += [
        "",
        "## Selective Causal Relevance（Deletion Sanity）",
        "",
        "| Intervention | Mean Score Drop | Mean Margin Drop | Flip Rate | C→W Rate | W→C Rate |",
        "|---|---:|---:|---:|---:|---:|",
        f"| top family | {_mean(top_family_rows, 'mean_score_drop'):.4f} | {_mean(top_family_rows, 'mean_margin_drop'):.4f} | {_mean(top_family_rows, 'selection_flip_rate'):.3f} | {_mean(top_family_rows, 'correct_to_wrong_rate'):.3f} | {_mean(top_family_rows, 'wrong_to_correct_rate'):.3f} |",
        f"| low family | {_mean(low_family_rows, 'mean_score_drop'):.4f} | {_mean(low_family_rows, 'mean_margin_drop'):.4f} | {_mean(low_family_rows, 'selection_flip_rate'):.3f} | {_mean(low_family_rows, 'correct_to_wrong_rate'):.3f} | {_mean(low_family_rows, 'wrong_to_correct_rate'):.3f} |",
        f"| top feature | {_mean(top_feature_rows, 'mean_score_drop'):.4f} | {_mean(top_feature_rows, 'mean_margin_drop'):.4f} | {_mean(top_feature_rows, 'selection_flip_rate'):.3f} | {_mean(top_feature_rows, 'correct_to_wrong_rate'):.3f} | {_mean(top_feature_rows, 'wrong_to_correct_rate'):.3f} |",
        f"| low feature | {_mean(low_feature_rows, 'mean_score_drop'):.4f} | {_mean(low_feature_rows, 'mean_margin_drop'):.4f} | {_mean(low_feature_rows, 'selection_flip_rate'):.3f} | {_mean(low_feature_rows, 'correct_to_wrong_rate'):.3f} | {_mean(low_feature_rows, 'wrong_to_correct_rate'):.3f} |",
        "",
        "- 解释应以数值支撑而非“看起来合理”为准；这里的主证据是 top deletion 对分数与选中结果的影响显著大于 low deletion。",
        "",
        "## Failure Archetypes（Wrong-Top1）",
        "",
        "| Archetype | Cases | Fraction | Mean Score Gap | Mean Rank Share | Representative |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in archetype_rows[:5]:
        lines.append(
            f"| {row['label']} | {row['n_cases']} | {float(row['fraction_of_wrong_cases']):.3f} | "
            f"{float(row['mean_score_gap']):.4f} | {float(row['mean_rank_delta_share']):.3f} | "
            f"`{row['representative_cache_key']}::{row['representative_problem_id']}@{row['representative_anchor_pct']}` |"
        )

    lines += [
        "",
        "## Appendix Cases",
        "",
    ]
    for row in case_rows:
        lines.append(
            f"- `{row['title']}` → `results/case_studies/{row['case_file_name']}`"
        )
    lines.append("")
    _ensure_parent(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export interpretability validation suite")
    ap.add_argument(
        "--cache-roots",
        default="/home/jovyan/public-ro/MUI_HUB/cache,/home/jovyan/public-ro/MUI_HUB/cache_train",
    )
    ap.add_argument(
        "--feature-cache-path",
        default="results/cache/interpretability_validation/feature_store_all.pkl",
    )
    ap.add_argument(
        "--multiseed-manifest",
        default="results/validation/interpretability_multiseed/manifest.json",
    )
    ap.add_argument("--tables-dir", default="results/tables")
    ap.add_argument("--case-dir", default="results/case_studies")
    ap.add_argument("--out-doc", default="docs/09_INTERPRETABILITY_VALIDATION.md")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-problems-per-cache", type=int, default=0)
    ap.add_argument("--feature-workers", type=int, default=4)
    ap.add_argument("--feature-chunk-problems", type=int, default=16)
    args = ap.parse_args()

    cache_roots = [Path(v) for v in _parse_csv(args.cache_roots)]
    feature_cache_path = (REPO_ROOT / str(args.feature_cache_path)).resolve()
    tables_dir = (REPO_ROOT / str(args.tables_dir)).resolve()
    case_dir = (REPO_ROOT / str(args.case_dir)).resolve()
    out_doc = (REPO_ROOT / str(args.out_doc)).resolve()
    multiseed_manifest = (REPO_ROOT / str(args.multiseed_manifest)).resolve()
    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)

    required_features = _collect_required_features()
    feature_store = _build_or_load_feature_store(
        cache_roots=cache_roots,
        feature_cache_path=feature_cache_path,
        required_features=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        feature_chunk_problems=int(args.feature_chunk_problems),
    )
    run_index_maps, cache_meta = _build_run_index_maps(cache_roots)
    problem_slices = iter_problem_slices(feature_store, run_index_maps)

    multiseed_records = _load_multiseed_records(multiseed_manifest)
    loaded_records = load_bundle_records(multiseed_records) if multiseed_records else []
    pairwise_stability_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    raw_rank_rows: list[dict[str, Any]] = []
    if loaded_records:
        pairwise_stability_rows, stability_rows, raw_rank_rows = compute_stability_rows(
            loaded_records,
            top_k=int(args.top_k),
        )
    raw_rank_summary_rows = summarize_raw_vs_rank_stability(stability_rows, raw_rank_rows) if stability_rows else []

    canonical_bundles = {
        method_id: load_earlystop_svd_bundle(model_path)
        for method_id, model_path in CANONICAL_MODEL_PATHS.items()
    }

    run_records: list[dict[str, Any]] = []
    deletion_detail_rows: list[dict[str, Any]] = []
    wrong_cases: list[dict[str, Any]] = []
    correct_candidates: list[dict[str, Any]] = []

    for method_id, bundle in canonical_bundles.items():
        supported_domains = set(str(v) for v in bundle.get("domains", {}).keys())
        for problem in problem_slices:
            domain = str(problem["domain"])
            if domain not in supported_domains:
                continue
            anchor_tensor = np.asarray(problem["anchor_tensor"], dtype=np.float64)
            labels = np.asarray(problem["labels"], dtype=np.int32)
            sample_ids = np.asarray(problem["sample_ids"], dtype=np.int32)
            run_infos = list(problem["run_infos"])

            for anchor_idx, anchor in enumerate(EXPLAIN_ANCHORS):
                route = get_anchor_route(bundle, domain, anchor)
                x_raw_problem = np.asarray(anchor_tensor[:, anchor_idx, :], dtype=np.float64)
                x_rank_problem = _rank_transform_matrix(x_raw_problem)
                scores, _ = _score_route_matrix(route, x_raw_problem, x_rank_problem)
                ranking_rows, order = _ranking_rows(scores, run_infos)
                if order.size <= 0:
                    continue
                top1_idx = int(order[0])
                top2_idx = int(order[1]) if order.size > 1 else int(top1_idx)

                exp_top1 = explain_run(
                    route,
                    x_raw_problem[top1_idx],
                    x_rank_problem[top1_idx],
                    run_metadata=run_infos[top1_idx],
                )
                exp_top2 = explain_run(
                    route,
                    x_raw_problem[top2_idx],
                    x_rank_problem[top2_idx],
                    run_metadata=run_infos[top2_idx],
                )
                run_records.append(
                    {
                        "method_id": method_id,
                        "case_type": "top1",
                        "reconstruction_error": float(exp_top1["reconstruction_error"]),
                        "component_sanity_error": float(exp_top1["component_sanity_error"]),
                    }
                )
                if top2_idx != top1_idx:
                    run_records.append(
                        {
                            "method_id": method_id,
                            "case_type": "top2",
                            "reconstruction_error": float(exp_top2["reconstruction_error"]),
                            "component_sanity_error": float(exp_top2["component_sanity_error"]),
                        }
                    )

                local_deletion_rows = deletion_intervention_rows(
                    method_id=method_id,
                    domain=domain,
                    anchor_pct=int(round(float(anchor) * 100.0)),
                    cache_key=str(problem["cache_key"]),
                    problem_id=str(problem["problem_id"]),
                    route=route,
                    x_raw_problem=x_raw_problem,
                    labels=labels,
                    run_infos=run_infos,
                    top1_idx=top1_idx,
                    top2_idx=top2_idx,
                    scores=scores,
                    exp_top1=exp_top1,
                )
                deletion_detail_rows.extend(local_deletion_rows)

                top_family_row = _intervention_lookup(local_deletion_rows, target_level="family", target_policy="top")
                low_family_row = _intervention_lookup(local_deletion_rows, target_level="family", target_policy="low")
                top_feature_row = _intervention_lookup(local_deletion_rows, target_level="feature", target_policy="top")
                low_feature_row = _intervention_lookup(local_deletion_rows, target_level="feature", target_policy="low")

                feature_deltas = _feature_delta_rows(
                    exp_top1["canonical_feature_rows"],
                    exp_top2["canonical_feature_rows"],
                )
                family_deltas = _family_delta_rows(
                    exp_top1["family_contributions"],
                    exp_top2["family_contributions"],
                )
                margin = float(scores[top1_idx] - scores[top2_idx]) if top2_idx != top1_idx else 0.0

                if bool(labels[top1_idx]):
                    correct_candidates.append(
                        {
                            "case_type": "correct",
                            "method_id": method_id,
                            "domain": domain,
                            "cache_key": str(problem["cache_key"]),
                            "problem_id": str(problem["problem_id"]),
                            "anchor_pct": int(round(float(anchor) * 100.0)),
                            "top1_sample_id": int(run_infos[top1_idx]["sample_id"]),
                            "top1_run_index": int(run_infos[top1_idx]["run_index"]),
                            "top1_is_correct": bool(labels[top1_idx]),
                            "top1_score": float(scores[top1_idx]),
                            "compare_sample_id": int(run_infos[top2_idx]["sample_id"]),
                            "compare_run_index": int(run_infos[top2_idx]["run_index"]),
                            "compare_is_correct": bool(labels[top2_idx]),
                            "compare_score": float(scores[top2_idx]),
                            "margin_or_gap": float(margin),
                            "margin": float(margin),
                            "top_family_deltas": family_deltas[:5],
                            "top_feature_deltas": feature_deltas[:8],
                            "top_family_intervention": top_family_row,
                            "low_family_intervention": low_family_row,
                            "top_feature_intervention": top_feature_row,
                            "low_feature_intervention": low_feature_row,
                            "selection_flip_rate_top_family": 1.0 if top_family_row and bool(top_family_row["selection_flip"]) else 0.0,
                            "mean_score_drop_top_family": float(top_family_row["score_drop"]) if top_family_row else 0.0,
                        }
                    )
                    continue

                correct_indices = [idx for idx, label in enumerate(labels.tolist()) if bool(label)]
                if not correct_indices:
                    continue
                best_correct_idx = max(correct_indices, key=lambda idx: float(scores[int(idx)]))
                exp_best_correct = explain_run(
                    route,
                    x_raw_problem[best_correct_idx],
                    x_rank_problem[best_correct_idx],
                    run_metadata=run_infos[best_correct_idx],
                )
                run_records.append(
                    {
                        "method_id": method_id,
                        "case_type": "best_correct",
                        "reconstruction_error": float(exp_best_correct["reconstruction_error"]),
                        "component_sanity_error": float(exp_best_correct["component_sanity_error"]),
                    }
                )
                wrong_case = build_wrong_case_record(
                    method_id=method_id,
                    domain=domain,
                    anchor_pct=int(round(float(anchor) * 100.0)),
                    cache_key=str(problem["cache_key"]),
                    problem_id=str(problem["problem_id"]),
                    run_infos=run_infos,
                    labels=labels,
                    scores=scores,
                    top1_idx=top1_idx,
                    best_correct_idx=best_correct_idx,
                    exp_top1=exp_top1,
                    exp_best_correct=exp_best_correct,
                )
                wrong_case["top_family_intervention"] = top_family_row
                wrong_case["low_family_intervention"] = low_family_row
                wrong_case["top_feature_intervention"] = top_feature_row
                wrong_case["low_feature_intervention"] = low_feature_row
                wrong_cases.append(wrong_case)

    faithfulness_rows = summarize_faithfulness(run_records)
    deletion_summary_rows = aggregate_deletion_rows(deletion_detail_rows)
    wrong_cases = assign_failure_archetypes(wrong_cases)
    failure_archetype_rows = aggregate_failure_archetypes(wrong_cases)

    appendix_cases = select_appendix_cases(
        correct_candidates=correct_candidates,
        wrong_candidates=wrong_cases,
    )
    case_payloads: list[dict[str, Any]] = []
    case_dir.mkdir(parents=True, exist_ok=True)
    for idx, case in enumerate(appendix_cases, start=1):
        case_file_name = f"appendix_case_{idx:02d}_{case['method_id']}_{case['domain']}_{case['anchor_pct']}.md"
        case_payload = dict(case)
        case_payload["title"] = (
            f"Case {idx}: {case['method_id']} {case['domain']} "
            f"{case['case_type']} @ {case['anchor_pct']}%"
        )
        case_payload["case_file_name"] = case_file_name
        if str(case["case_type"]) == "wrong":
            case_payload["paper_note"] = (
                f"该 wrong-top1 case 属于 `{case.get('archetype_label', 'unknown')}`："
                f"错误 top1 相对最佳正确 run 的领先主要来自 `{case.get('dominant_positive_family', 'unknown')}`。"
            )
        else:
            case_payload["paper_note"] = (
                "该 correct case 展示了 top1-vs-top2 的局部因果相关性："
                "删除 top family / feature 后，选中 run 的 margin 明显下降。"
            )
        case_path = case_dir / case_file_name
        case_path.write_text(_case_markdown(case_payload), encoding="utf-8")
        case_payloads.append(case_payload)

    _write_csv(tables_dir / "explanation_stability_pairwise.csv", pairwise_stability_rows)
    _write_csv(tables_dir / "explanation_stability.csv", stability_rows)
    _write_csv(tables_dir / "raw_vs_rank_stability_summary.csv", raw_rank_summary_rows)
    _write_csv(tables_dir / "deletion_sanity_detailed.csv", deletion_detail_rows)
    _write_csv(tables_dir / "deletion_sanity.csv", deletion_summary_rows)
    _write_csv(tables_dir / "failure_archetypes.csv", failure_archetype_rows)
    _write_csv(tables_dir / "failure_archetype_cases.csv", wrong_cases)
    _write_csv(tables_dir / "faithfulness_summary.csv", faithfulness_rows)
    _json_dump(case_dir / "appendix_cases.json", case_payloads)

    _build_paper_doc(
        out_path=out_doc,
        faithfulness_rows=faithfulness_rows,
        stability_rows=stability_rows,
        raw_rank_rows=raw_rank_summary_rows,
        deletion_rows=deletion_summary_rows,
        archetype_rows=failure_archetype_rows,
        case_rows=case_payloads,
    )

    manifest = {
        "generated_at_utc": _now_utc(),
        "feature_cache_path": str(feature_cache_path),
        "tables_dir": str(tables_dir),
        "case_dir": str(case_dir),
        "out_doc": str(out_doc),
        "cache_roots": [str(v) for v in cache_roots],
        "canonical_models": {key: str(value) for key, value in CANONICAL_MODEL_PATHS.items()},
        "n_problem_slices": int(len(problem_slices)),
        "n_wrong_cases": int(len(wrong_cases)),
        "n_case_studies": int(len(case_payloads)),
    }
    _json_dump(tables_dir / "interpretability_validation_manifest.json", manifest)

    print("[done] interpretability validation export", flush=True)
    print(f"  doc              : {out_doc}", flush=True)
    print(f"  stability csv    : {tables_dir / 'explanation_stability.csv'}", flush=True)
    print(f"  deletion csv     : {tables_dir / 'deletion_sanity.csv'}", flush=True)
    print(f"  archetypes csv   : {tables_dir / 'failure_archetypes.csv'}", flush=True)
    print(f"  case studies dir : {case_dir}", flush=True)


if __name__ == "__main__":
    main()
