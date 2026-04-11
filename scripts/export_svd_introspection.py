#!/usr/bin/env python3
"""
Export SVD introspection artifacts for paper-quality analysis.

Usage:
    python scripts/export_svd_introspection.py \
      --methods es_svd_math_rr_r1,es_svd_science_rr_r1,es_svd_ms_rr_r1 \
      --cache-roots MUI_HUB/cache MUI_HUB/cache_train \
      --anchors 10,40,70,100 \
      --out-root results/interpretability \
      --max-problems 0

Output per method:
    results/interpretability/<method_id>/
        route_inventory.json
        effective_weights.csv
        component_table.csv
        family_summary.csv
        failure_modes.csv
        stability_report.csv
        problem_top1_vs_top2.jsonl
        run_contributions/anchor{pct}/<cache_key>.jsonl
        sanity_checks.json
    docs/SVD_INTROSPECTION_RESULTS_<method_id>.md
    docs/SVD_FAILURE_MODES_<method_id>.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from nad.explain.svd_explain import (
    EXPLAIN_ANCHORS,
    aggregate_failure_modes,
    collect_bundle_required_features,
    summarize_wrong_top1_case,
)
from nad.explain.svd_introspection import (
    build_family_weight_rows,
    build_problem_rank_matrix,
    build_route_inventory,
    compute_weight_stability,
    expand_feature_names_for_representation,
    explain_problem_decision,
    explain_run,
    export_failure_modes_csv,
    generate_failure_modes_doc,
    generate_introspection_results_doc,
    iter_routes,
    recover_effective_weights,
    extract_problem_raw_matrix,
)
from nad.explain.svd_explain import feature_family
from nad.ops.accuracy import load_correctness_map
from nad.ops.earlystop import _problem_sort_key, build_problem_groups, discover_cache_entries
from nad.ops.earlystop_svd import get_domain, load_earlystop_svd_bundle
from nad.core.views.reader import CacheReader


CANONICAL_SVD_MODEL_PATHS = {
    "es_svd_math_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_math_rr_r1.pkl",
    "es_svd_science_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_science_rr_r1.pkl",
    "es_svd_ms_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_ms_rr_r1.pkl",
    "es_svd_coding_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_coding_rr_r1.pkl",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _normalize_anchor(raw: str | float | int) -> float:
    value = float(raw)
    if value > 1.0:
        value = value / 100.0
    for anchor in EXPLAIN_ANCHORS:
        if abs(float(anchor) - value) < 1e-9:
            return float(anchor)
    raise ValueError(f"Unsupported anchor: {raw}")


def _safe_name(text: str) -> str:
    return str(text).replace("/", "__").replace(" ", "_")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl_append(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _clear_path(path: Path) -> None:
    if path.is_file():
        path.unlink()
        return
    if path.is_dir():
        for child in sorted(path.iterdir()):
            _clear_path(child)
        path.rmdir()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _problem_run_infos(
    meta: dict[str, Any],
    sample_ids: list[int],
    correctness_map: dict[int, bool],
) -> list[dict[str, Any]]:
    rows = []
    samples = meta.get("samples", [])
    for sample_id in sample_ids:
        sample = samples[int(sample_id)]
        run_index = int(sample.get("run_index", sample_id))
        rows.append(
            {
                "sample_id": int(sample_id),
                "run_index": int(run_index),
                "is_correct": bool(correctness_map.get(int(sample_id), False)),
            }
        )
    rows.sort(key=lambda row: int(row["run_index"]))
    return rows


def _filter_entries(entries: list[Any], method_id: str) -> list[Any]:
    bundle = load_earlystop_svd_bundle(CANONICAL_SVD_MODEL_PATHS[method_id])
    supported_domains = set(str(v) for v in bundle.get("domains", {}).keys())
    return [e for e in entries if get_domain(e.dataset_name) in supported_domains]


# ---------------------------------------------------------------------------
# Effective weights → CSV
# ---------------------------------------------------------------------------


def _build_effective_weights_rows(
    bundle: dict[str, Any], method_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain, position, route in iter_routes(bundle):
        eff = recover_effective_weights(route)
        anchor_pct = int(round(float(position) * 100.0))
        w_orig = eff["w_orig"]
        rep_names = eff["rep_feature_names"]
        for j, fname in enumerate(rep_names):
            base_feat = fname.split("::")[0]
            channel = fname.split("::")[-1] if "::" in fname else "raw"
            rows.append(
                {
                    "method_id": method_id,
                    "domain": domain,
                    "anchor_pct": anchor_pct,
                    "rep_feature": fname,
                    "feature": base_feat,
                    "channel": channel,
                    "w_orig": float(w_orig[j]),
                    "family": feature_family(base_feat),
                }
            )
    return rows


def _build_component_table_rows(
    bundle: dict[str, Any], method_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain, position, route in iter_routes(bundle):
        eff = recover_effective_weights(route)
        anchor_pct = int(round(float(position) * 100.0))
        for comp in eff["aligned_components"]:
            rows.append(
                {
                    "method_id": method_id,
                    "domain": domain,
                    "anchor_pct": anchor_pct,
                    "comp_k": int(comp["comp_k"]),
                    "alpha_eff_aligned": float(comp["alpha_eff_aligned"]),
                    "abs_importance": float(comp["alpha_eff_aligned"]),
                    "top_positive_features": ";".join(comp["top_positive_rep_features"]),
                    "top_negative_features": ";".join(comp["top_negative_rep_features"]),
                    "dominant_family": str(comp["dominant_family"]),
                    "family_purity": float(comp["family_purity"]),
                    "auto_label": str(comp["auto_label"]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Main per-method export
# ---------------------------------------------------------------------------


def _export_method(
    method_id: str,
    bundle: dict[str, Any],
    entries: list[Any],
    anchors: list[float],
    out_root: Path,
    docs_root: Path,
    max_problems: int | None,
    required_features_union: set[str],
    problem_tensor_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    method_dir = out_root / method_id
    if method_dir.exists():
        _clear_path(method_dir)
    method_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 2: Route inventory
    # ------------------------------------------------------------------ #
    route_inventory = build_route_inventory(bundle, method_id)
    _json_dump(method_dir / "route_inventory.json", route_inventory)

    # ------------------------------------------------------------------ #
    # Step 3: Effective weights + component table
    # ------------------------------------------------------------------ #
    ew_rows = _build_effective_weights_rows(bundle, method_id)
    _write_csv(method_dir / "effective_weights.csv", ew_rows)

    ct_rows = _build_component_table_rows(bundle, method_id)
    _write_csv(method_dir / "component_table.csv", ct_rows)

    # ------------------------------------------------------------------ #
    # Step 4: Family weight rows (seed; mean_abs_contribution filled later)
    # ------------------------------------------------------------------ #
    # key: (domain, anchor_pct, family, channel)
    family_row_accumulator: dict[tuple, dict[str, Any]] = {}
    for domain, position, route in iter_routes(bundle):
        for row in build_family_weight_rows(route, domain, position):
            key = (str(row["domain"]), int(row["anchor_pct"]), str(row["family"]), str(row["channel"]))
            family_row_accumulator[key] = row.copy()
            family_row_accumulator[key]["method_id"] = method_id
            family_row_accumulator[key]["abs_contrib_sum"] = 0.0
            family_row_accumulator[key]["abs_contrib_count"] = 0
            family_row_accumulator[key]["delta_sum"] = 0.0
            family_row_accumulator[key]["delta_count"] = 0

    # ------------------------------------------------------------------ #
    # Step 5: Discover caches and iterate problems
    # ------------------------------------------------------------------ #
    filtered_entries = _filter_entries(entries, method_id)
    required_features = collect_bundle_required_features(bundle)

    problem_lines_path = method_dir / "problem_top1_vs_top2.jsonl"
    wrong_cases_all: list[dict[str, Any]] = []
    wrong_cases_by_anchor: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sanity_values_all: list[float] = []
    sanity_values_by_anchor: dict[int, list[float]] = defaultdict(list)
    manifest_entries: list[dict[str, Any]] = []

    total_problems = 0
    total_problem_anchors = 0
    total_runs = 0

    for entry in filtered_entries:
        reader = CacheReader(str(entry.cache_root))
        meta = json.loads((entry.cache_root / "meta.json").read_text(encoding="utf-8"))
        correctness_map = load_correctness_map(str(entry.cache_root))
        groups = build_problem_groups(meta)
        ordered_problem_ids = sorted(groups.keys(), key=_problem_sort_key)
        if max_problems is not None:
            ordered_problem_ids = ordered_problem_ids[: int(max_problems)]

        domain = get_domain(entry.dataset_name)
        cache_sanity_values: list[float] = []
        cache_problem_count = 0
        cache_pa_count = 0
        cache_run_count = 0

        for problem_id in ordered_problem_ids:
            total_problems += 1
            cache_problem_count += 1

            cache_key = (str(entry.cache_root), str(problem_id))
            cached = problem_tensor_cache.get(cache_key)
            if cached is None:
                sample_ids = [int(v) for v in groups[str(problem_id)]]
                run_infos = _problem_run_infos(meta, sample_ids, correctness_map)
                anchor_tensor = extract_problem_raw_matrix(
                    reader,
                    [int(row["sample_id"]) for row in run_infos],
                    anchors,
                    required_features_union,
                )
                cached = {"run_infos": run_infos, "anchor_tensor": anchor_tensor}
                problem_tensor_cache[cache_key] = cached

            run_infos = cached["run_infos"]
            anchor_tensor = cached["anchor_tensor"]  # (n_runs, n_anchors, n_features)

            labels = [bool(r.get("is_correct", False)) for r in run_infos]
            sample_id_list = [int(r["sample_id"]) for r in run_infos]

            for anchor_idx, anchor in enumerate(anchors):
                anchor_pct = int(round(float(anchor) * 100.0))

                # Get route for this domain + anchor
                try:
                    from nad.explain.svd_explain import get_anchor_route
                    route = get_anchor_route(bundle, domain, anchor)
                except (KeyError, ValueError):
                    continue

                x_raw = anchor_tensor[:, anchor_idx, :].astype(np.float64)
                x_rank = build_problem_rank_matrix(x_raw)

                # ---- Problem-level decision ----
                decision = explain_problem_decision(
                    route,
                    x_raw,
                    x_rank,
                    sample_id_list,
                    labels,
                )

                problem_record = {
                    "method_id": method_id,
                    "cache_key": str(entry.cache_key),
                    "cache_root": str(entry.cache_root),
                    "dataset_name": str(entry.dataset_name),
                    "model_name": str(entry.model_name),
                    "problem_id": str(problem_id),
                    "domain": domain,
                    "anchor": float(anchor),
                    "anchor_pct": anchor_pct,
                    "top1": decision["top1"],
                    "top2": decision["top2"],
                    "margin": decision["margin"],
                    "why_top1": decision["why_selected"],
                    "whether_top1_is_correct": decision["whether_top1_is_correct"],
                    "top_feature_deltas": decision["delta_feature_contrib"][:12],
                    "top_family_deltas": decision["delta_family_contrib"][:8],
                    "sanity_check": decision["sanity_check"],
                }
                _jsonl_append(problem_lines_path, problem_record)

                # Track wrong-top1 cases (compatible with aggregate_failure_modes format)
                if not decision["whether_top1_is_correct"]:
                    top1_exp = decision["run_explanations"].get("top1", {})
                    top2_exp = decision["run_explanations"].get("top2", {})
                    wrong_case = {
                        "problem_id": str(problem_id),
                        "cache_key": str(entry.cache_key),
                        "method_id": method_id,
                        "domain": domain,
                        "anchor": float(anchor),
                        "anchor_pct": anchor_pct,
                        "top_feature_deltas": decision["delta_feature_contrib"][:12],
                        "top_family_deltas": decision["delta_family_contrib"][:8],
                    }
                    wrong_cases_all.append(wrong_case)
                    wrong_cases_by_anchor[anchor_pct].append(wrong_case)

                # ---- Run-level explanations ----
                run_path = method_dir / "run_contributions" / f"anchor{anchor_pct:03d}" / f"{_safe_name(entry.cache_key)}.jsonl"
                run_sanity_errors: list[float] = []

                for run_idx, run_info in enumerate(run_infos):
                    run_exp = explain_run(
                        route,
                        x_raw[run_idx],
                        x_rank[run_idx],
                        run_metadata=run_info,
                    )
                    run_sanity_errors.append(run_exp["reconstruction_error"])

                    run_record = {
                        "method_id": method_id,
                        "cache_key": str(entry.cache_key),
                        "cache_root": str(entry.cache_root),
                        "dataset_name": str(entry.dataset_name),
                        "model_name": str(entry.model_name),
                        "problem_id": str(problem_id),
                        "domain": domain,
                        "anchor": float(anchor),
                        "anchor_pct": anchor_pct,
                        "run": run_info,
                        "exact_score": run_exp["exact_score"],
                        "intercept": run_exp["intercept"],
                        "feature_contributions": run_exp["canonical_feature_rows"],
                        "family_contributions": run_exp["family_contributions"],
                        "component_contributions": run_exp["component_contributions"],
                        "reconstruction_error": run_exp["reconstruction_error"],
                        "human_summary": run_exp["human_summary"],
                    }
                    _jsonl_append(run_path, run_record)

                    # Fill family_row_accumulator with abs contribution data
                    for fam_row in run_exp["family_contributions"]:
                        fam = str(fam_row["family"])
                        contrib = abs(float(fam_row["total_contribution"]))
                        for ch in ["raw+rank", "raw", "rank"]:
                            key = (domain, anchor_pct, fam, ch)
                            if key in family_row_accumulator:
                                family_row_accumulator[key]["abs_contrib_sum"] += contrib
                                family_row_accumulator[key]["abs_contrib_count"] += 1

                # Fill discriminative_delta (from delta_family_contrib for top1)
                for fam_delta in decision["delta_family_contrib"]:
                    fam = str(fam_delta["family"])
                    delta = abs(float(fam_delta["delta"]))
                    for ch in ["raw+rank", "raw", "rank"]:
                        key = (domain, anchor_pct, fam, ch)
                        if key in family_row_accumulator:
                            family_row_accumulator[key]["delta_sum"] += delta
                            family_row_accumulator[key]["delta_count"] += 1

                error_max = float(max(run_sanity_errors)) if run_sanity_errors else 0.0
                error_mean = float(np.mean(run_sanity_errors)) if run_sanity_errors else 0.0
                sanity_values_all.extend(run_sanity_errors)
                sanity_values_by_anchor[anchor_pct].extend(run_sanity_errors)
                cache_sanity_values.extend(run_sanity_errors)

                total_problem_anchors += 1
                total_runs += len(run_infos)
                cache_pa_count += 1
                cache_run_count += len(run_infos)

        manifest_entries.append(
            {
                "cache_key": str(entry.cache_key),
                "cache_root": str(entry.cache_root),
                "dataset_name": str(entry.dataset_name),
                "model_name": str(entry.model_name),
                "domain": domain,
                "problem_count": cache_problem_count,
                "problem_anchor_count": cache_pa_count,
                "run_count": cache_run_count,
                "max_reconstruction_error": float(max(cache_sanity_values)) if cache_sanity_values else 0.0,
                "mean_reconstruction_error": float(np.mean(cache_sanity_values)) if cache_sanity_values else 0.0,
            }
        )

    # ------------------------------------------------------------------ #
    # Step 7: Write family_summary.csv
    # ------------------------------------------------------------------ #
    final_family_rows: list[dict[str, Any]] = []
    for key, acc in family_row_accumulator.items():
        n_contrib = int(acc["abs_contrib_count"])
        n_delta = int(acc["delta_count"])
        row = {
            "method_id": method_id,
            "domain": acc["domain"],
            "anchor_pct": acc["anchor_pct"],
            "family": acc["family"],
            "channel": acc["channel"],
            "signed_weight_sum": float(acc["signed_weight_sum"]),
            "abs_weight_sum": float(acc["abs_weight_sum"]),
            "mean_abs_contribution": (
                float(acc["abs_contrib_sum"]) / n_contrib if n_contrib > 0 else float("nan")
            ),
            "discriminative_delta": (
                float(acc["delta_sum"]) / n_delta if n_delta > 0 else float("nan")
            ),
        }
        final_family_rows.append(row)

    _write_csv(method_dir / "family_summary.csv", final_family_rows)

    # ------------------------------------------------------------------ #
    # Step 8: Failure modes
    # ------------------------------------------------------------------ #
    failure_summary = {
        "generated_at_utc": _now_utc(),
        "method_id": method_id,
        "total_wrong_top1_cases": len(wrong_cases_all),
        **aggregate_failure_modes(wrong_cases_all),
        "by_anchor": {
            str(ap): aggregate_failure_modes(cases)
            for ap, cases in sorted(wrong_cases_by_anchor.items())
        },
    }
    export_failure_modes_csv(failure_summary, method_dir / "failure_modes.csv")
    generate_failure_modes_doc(failure_summary, method_id, route_inventory, docs_root / f"SVD_FAILURE_MODES_{method_id}.md")

    # ------------------------------------------------------------------ #
    # Step 9: Stability
    # ------------------------------------------------------------------ #
    stability_rows = compute_weight_stability(bundle, method_id)
    _write_csv(method_dir / "stability_report.csv", stability_rows)

    # ------------------------------------------------------------------ #
    # Step 10: Introspection results doc
    # ------------------------------------------------------------------ #
    generate_introspection_results_doc(
        method_id=method_id,
        route_inventory=route_inventory,
        family_summary=final_family_rows,
        failure_summary=failure_summary,
        stability_rows=stability_rows,
        out_path=docs_root / f"SVD_INTROSPECTION_RESULTS_{method_id}.md",
    )

    # ------------------------------------------------------------------ #
    # Sanity checks JSON
    # ------------------------------------------------------------------ #
    sanity_summary = {
        "generated_at_utc": _now_utc(),
        "method_id": method_id,
        "problem_count": total_problems,
        "problem_anchor_count": total_problem_anchors,
        "overall": {
            "run_count": int(len(sanity_values_all)),
            "max_abs_error": float(max(sanity_values_all)) if sanity_values_all else 0.0,
            "mean_abs_error": float(np.mean(sanity_values_all)) if sanity_values_all else 0.0,
        },
        "by_anchor": {
            str(ap): {
                "run_count": int(len(vals)),
                "max_abs_error": float(max(vals)) if vals else 0.0,
                "mean_abs_error": float(np.mean(vals)) if vals else 0.0,
            }
            for ap, vals in sorted(sanity_values_by_anchor.items())
        },
        "per_cache": manifest_entries,
    }
    _json_dump(method_dir / "sanity_checks.json", sanity_summary)

    manifest = {
        "generated_at_utc": _now_utc(),
        "method_id": method_id,
        "model_path": str(CANONICAL_SVD_MODEL_PATHS.get(method_id, "unknown")),
        "anchors": [int(round(float(a) * 100.0)) for a in anchors],
        "max_problems_per_cache": None if max_problems is None else int(max_problems),
        "problem_count": total_problems,
        "problem_anchor_count": total_problem_anchors,
        "run_count": total_runs,
        "artifacts": {
            "route_inventory": "route_inventory.json",
            "effective_weights": "effective_weights.csv",
            "component_table": "component_table.csv",
            "family_summary": "family_summary.csv",
            "failure_modes": "failure_modes.csv",
            "stability_report": "stability_report.csv",
            "problem_top1_vs_top2": "problem_top1_vs_top2.jsonl",
            "run_contributions": "run_contributions/",
            "sanity_checks": "sanity_checks.json",
        },
    }
    _json_dump(method_dir / "manifest.json", manifest)

    return sanity_summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SVD introspection artifacts")
    ap.add_argument(
        "--methods",
        default="es_svd_math_rr_r1,es_svd_science_rr_r1,es_svd_ms_rr_r1",
        help="Comma-separated canonical method IDs",
    )
    ap.add_argument(
        "--cache-roots",
        default="/home/jovyan/public-ro/MUI_HUB/cache,/home/jovyan/public-ro/MUI_HUB/cache_train",
        help="Comma-separated cache roots",
    )
    ap.add_argument(
        "--anchors",
        default="10,40,70,100",
        help="Comma-separated anchors (10/40/70/100 or 0.1/0.4/0.7/1.0)",
    )
    ap.add_argument("--out-root", default="results/interpretability")
    ap.add_argument("--max-problems", type=int, default=0, help="Cap per cache; 0=all")
    args = ap.parse_args()

    method_ids = _parse_csv(args.methods)
    cache_roots = [Path(v) for v in _parse_csv(args.cache_roots)]
    anchors = [_normalize_anchor(v) for v in _parse_csv(args.anchors)]
    out_root = (REPO_ROOT / str(args.out_root)).resolve()
    docs_root = (REPO_ROOT / "docs").resolve()
    docs_root.mkdir(parents=True, exist_ok=True)
    max_problems = None if int(args.max_problems) <= 0 else int(args.max_problems)

    # Discover all cache entries
    entries: list[Any] = []
    for cache_root in cache_roots:
        entries.extend(discover_cache_entries(cache_root))

    # Pre-load bundles and collect required features
    bundle_by_method: dict[str, dict[str, Any]] = {}
    for method_id in method_ids:
        if method_id not in CANONICAL_SVD_MODEL_PATHS:
            raise ValueError(f"Unknown method: {method_id}. Add to CANONICAL_SVD_MODEL_PATHS.")
        bundle_by_method[method_id] = load_earlystop_svd_bundle(CANONICAL_SVD_MODEL_PATHS[method_id])

    required_features_union: set[str] = set()
    for bundle in bundle_by_method.values():
        required_features_union.update(collect_bundle_required_features(bundle))

    # Shared problem tensor cache (avoid re-extracting for multiple methods)
    problem_tensor_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for method_id in method_ids:
        print(f"[start] method={method_id}")
        bundle = bundle_by_method[method_id]
        sanity = _export_method(
            method_id=method_id,
            bundle=bundle,
            entries=entries,
            anchors=anchors,
            out_root=out_root,
            docs_root=docs_root,
            max_problems=max_problems,
            required_features_union=required_features_union,
            problem_tensor_cache=problem_tensor_cache,
        )
        overall = sanity["overall"]
        print(
            f"[done] method={method_id} "
            f"problems={sanity['problem_count']} "
            f"runs={overall['run_count']} "
            f"max_err={overall['max_abs_error']:.2e}"
        )


if __name__ == "__main__":
    main()
