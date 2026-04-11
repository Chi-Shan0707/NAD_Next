#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from nad.explain.svd_explain import (
    EXPLAIN_ANCHORS,
    aggregate_failure_modes,
    build_problem_anchor_tensor,
    collect_bundle_required_features,
    explain_problem_from_anchor_tensor,
    model_summary_from_bundle,
    summarize_wrong_top1_case,
)
from nad.ops.accuracy import load_correctness_map
from nad.ops.earlystop import _problem_sort_key, build_problem_groups, discover_cache_entries
from nad.ops.earlystop_svd import get_domain, load_earlystop_svd_bundle
from nad.core.views.reader import CacheReader


CANONICAL_SVD_MODEL_PATHS = {
    "es_svd_math_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_math_rr_r1.pkl",
    "es_svd_science_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_science_rr_r1.pkl",
    "es_svd_ms_rr_r1": REPO_ROOT / "models/ml_selectors/es_svd_ms_rr_r1.pkl",
}


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


def _problem_run_infos(meta: dict[str, Any], sample_ids: list[int], correctness_map: dict[int, bool]) -> list[dict[str, Any]]:
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
    out = []
    for entry in entries:
        if get_domain(entry.dataset_name) not in supported_domains:
            continue
        out.append(entry)
    return out


def _build_model_summary(bundle: dict[str, Any], method_id: str, anchors: list[float]) -> dict[str, Any]:
    payload = {
        "generated_at_utc": _now_utc(),
        "method_id": str(method_id),
        "anchors": [int(round(float(v) * 100.0)) for v in anchors],
        "domains": {},
    }
    for domain in sorted(bundle.get("domains", {}).keys()):
        payload["domains"][domain] = {}
        for anchor in anchors:
            payload["domains"][domain][str(int(round(float(anchor) * 100.0)))] = model_summary_from_bundle(
                bundle,
                method_id=method_id,
                domain=domain,
                anchor=anchor,
                top_k=8,
            )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Export canonical SVD interpretability artifacts")
    ap.add_argument(
        "--methods",
        default="es_svd_math_rr_r1,es_svd_science_rr_r1,es_svd_ms_rr_r1",
        help="Comma-separated canonical method ids",
    )
    ap.add_argument(
        "--cache-roots",
        default="/home/jovyan/public-ro/MUI_HUB/cache,/home/jovyan/public-ro/MUI_HUB/cache_train",
        help="Comma-separated cache roots to scan",
    )
    ap.add_argument(
        "--anchors",
        default="10,40,70,100",
        help="Comma-separated anchors, accepts 10/40/70/100 or 0.1/0.4/0.7/1.0",
    )
    ap.add_argument("--out-root", default="results/interpretability")
    ap.add_argument("--max-problems", type=int, default=0, help="Optional cap per cache; 0 means all")
    args = ap.parse_args()

    method_ids = _parse_csv(args.methods)
    cache_roots = [Path(v) for v in _parse_csv(args.cache_roots)]
    anchors = [_normalize_anchor(v) for v in _parse_csv(args.anchors)]
    if [float(v) for v in anchors] != [float(v) for v in EXPLAIN_ANCHORS]:
        raise ValueError("This exporter currently requires anchors=10,40,70,100")
    out_root = (REPO_ROOT / str(args.out_root)).resolve()
    max_problems = None if int(args.max_problems) <= 0 else int(args.max_problems)

    entries = []
    for cache_root in cache_roots:
        entries.extend(discover_cache_entries(cache_root))

    bundle_by_method = {
        method_id: load_earlystop_svd_bundle(CANONICAL_SVD_MODEL_PATHS[method_id])
        for method_id in method_ids
    }
    required_by_method = {
        method_id: collect_bundle_required_features(bundle_by_method[method_id])
        for method_id in method_ids
    }
    required_features_union = set()
    for values in required_by_method.values():
        required_features_union.update(values)

    problem_tensor_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for method_id in method_ids:
        if method_id not in CANONICAL_SVD_MODEL_PATHS:
            raise ValueError(f"Unsupported canonical SVD method: {method_id}")

        bundle = bundle_by_method[method_id]
        required_features = required_by_method[method_id]
        method_dir = out_root / method_id
        if method_dir.exists():
            _clear_path(method_dir)
        method_dir.mkdir(parents=True, exist_ok=True)

        model_summary = _build_model_summary(bundle, method_id, anchors)
        _json_dump(method_dir / "model_summary.json", model_summary)

        problem_lines_path = method_dir / "problem_top1_vs_top2.jsonl"
        wrong_lines_path = method_dir / "wrong_top1_cases.jsonl"
        run_dir = method_dir / "run_contributions"
        run_dir.mkdir(parents=True, exist_ok=True)

        filtered_entries = _filter_entries(entries, method_id)
        wrong_cases_all: list[dict[str, Any]] = []
        wrong_cases_by_anchor: dict[int, list[dict[str, Any]]] = defaultdict(list)
        sanity_values_all: list[float] = []
        sanity_values_by_anchor: dict[int, list[float]] = defaultdict(list)
        manifest_entries = []
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

            cache_problem_count = 0
            cache_problem_anchor_count = 0
            cache_run_count = 0
            cache_sanity_values: list[float] = []

            for problem_id in ordered_problem_ids:
                total_problems += 1
                cache_problem_count += 1
                cache_key = (str(entry.cache_root), str(problem_id))
                cached = problem_tensor_cache.get(cache_key)
                if cached is None:
                    sample_ids = [int(v) for v in groups[str(problem_id)]]
                    run_infos = _problem_run_infos(meta, sample_ids, correctness_map)
                    anchor_tensor = build_problem_anchor_tensor(
                        reader,
                        [int(row["sample_id"]) for row in run_infos],
                        positions=anchors,
                        required_features=required_features_union,
                    )
                    cached = {
                        "run_infos": run_infos,
                        "anchor_tensor": anchor_tensor,
                    }
                    problem_tensor_cache[cache_key] = cached
                run_infos = cached["run_infos"]
                anchor_tensor = cached["anchor_tensor"]

                for anchor_idx, anchor in enumerate(anchors):
                    anchor_pct = int(round(float(anchor) * 100.0))
                    payload = explain_problem_from_anchor_tensor(
                        bundle,
                        method_id=method_id,
                        domain=get_domain(entry.dataset_name),
                        anchor=anchor,
                        anchor_tensor=anchor_tensor,
                        run_infos=run_infos,
                        problem_id=str(problem_id),
                        cache_key=str(entry.cache_key),
                    )
                    payload["cache_root"] = str(entry.cache_root)
                    payload["dataset_name"] = str(entry.dataset_name)
                    payload["model_name"] = str(entry.model_name)

                    problem_record = {
                        "method_id": method_id,
                        "cache_key": str(entry.cache_key),
                        "cache_root": str(entry.cache_root),
                        "dataset_name": str(entry.dataset_name),
                        "model_name": str(entry.model_name),
                        "problem_id": str(problem_id),
                        "domain": str(payload["domain"]),
                        "anchor": float(anchor),
                        "anchor_pct": int(payload["anchor_pct"]),
                        "route_meta": payload["route_meta"],
                        "top1": payload["top_runs"][0] if payload["top_runs"] else None,
                        "top2": payload["top_runs"][1] if len(payload["top_runs"]) > 1 else None,
                        "margin": None
                        if len(payload["top_runs"]) < 2
                        else float(payload["top_runs"][0]["score"] - payload["top_runs"][1]["score"]),
                        "why_top1": payload["why_selected"],
                        "top_feature_deltas": payload["top_feature_deltas"][:12],
                        "top_family_deltas": payload["top_family_deltas"][:8],
                        "sanity_check": payload["sanity_check"],
                    }
                    _jsonl_append(problem_lines_path, problem_record)

                    wrong_case = summarize_wrong_top1_case(payload)
                    if wrong_case is not None:
                        _jsonl_append(wrong_lines_path, wrong_case)
                        wrong_cases_all.append(wrong_case)
                        wrong_cases_by_anchor[anchor_pct].append(wrong_case)

                    run_path = run_dir / f"anchor{anchor_pct:03d}" / f"{_safe_name(entry.cache_key)}.jsonl"
                    for run_payload in payload["run_explanations"]:
                        _jsonl_append(
                            run_path,
                            {
                                "method_id": method_id,
                                "cache_key": str(entry.cache_key),
                                "cache_root": str(entry.cache_root),
                                "dataset_name": str(entry.dataset_name),
                                "model_name": str(entry.model_name),
                                "problem_id": str(problem_id),
                                "domain": str(payload["domain"]),
                                "anchor": float(anchor),
                                "anchor_pct": int(payload["anchor_pct"]),
                                "route_meta": payload["route_meta"],
                                **run_payload,
                            },
                        )

                    error_values = [float(row["reconstruction_error"]) for row in payload["run_explanations"]]
                    sanity_values_all.extend(error_values)
                    sanity_values_by_anchor[anchor_pct].extend(error_values)
                    cache_sanity_values.extend(error_values)
                    total_problem_anchors += 1
                    total_runs += int(len(payload["run_explanations"]))
                    cache_problem_anchor_count += 1
                    cache_run_count += int(len(payload["run_explanations"]))

            manifest_entries.append(
                {
                    "cache_key": str(entry.cache_key),
                    "cache_root": str(entry.cache_root),
                    "dataset_name": str(entry.dataset_name),
                    "model_name": str(entry.model_name),
                    "domain": str(get_domain(entry.dataset_name)),
                    "problem_count": int(cache_problem_count),
                    "problem_anchor_count": int(cache_problem_anchor_count),
                    "run_count": int(cache_run_count),
                    "max_reconstruction_error": float(max(cache_sanity_values) if cache_sanity_values else 0.0),
                    "mean_reconstruction_error": float(np.mean(cache_sanity_values) if cache_sanity_values else 0.0),
                }
            )

        failure_summary = {
            "generated_at_utc": _now_utc(),
            "method_id": method_id,
            "overall": aggregate_failure_modes(wrong_cases_all),
            "by_anchor": {
                str(anchor_pct): aggregate_failure_modes(cases)
                for anchor_pct, cases in sorted(wrong_cases_by_anchor.items())
            },
        }
        _json_dump(method_dir / "failure_mode_summary.json", failure_summary)

        sanity_summary = {
            "generated_at_utc": _now_utc(),
            "method_id": method_id,
            "overall": {
                "run_count": int(len(sanity_values_all)),
                "max_abs_error": float(max(sanity_values_all) if sanity_values_all else 0.0),
                "mean_abs_error": float(np.mean(sanity_values_all) if sanity_values_all else 0.0),
            },
            "by_anchor": {
                str(anchor_pct): {
                    "run_count": int(len(values)),
                    "max_abs_error": float(max(values) if values else 0.0),
                    "mean_abs_error": float(np.mean(values) if values else 0.0),
                }
                for anchor_pct, values in sorted(sanity_values_by_anchor.items())
            },
            "per_cache": manifest_entries,
        }
        _json_dump(method_dir / "sanity_checks.json", sanity_summary)

        manifest = {
            "generated_at_utc": _now_utc(),
            "method_id": method_id,
            "model_path": str(CANONICAL_SVD_MODEL_PATHS[method_id].relative_to(REPO_ROOT)),
            "anchors": [int(round(float(v) * 100.0)) for v in anchors],
            "max_problems_per_cache": None if max_problems is None else int(max_problems),
            "cache_roots": [str(path) for path in cache_roots],
            "required_features": sorted(required_features),
            "problem_count": int(total_problems),
            "problem_anchor_count": int(total_problem_anchors),
            "run_count": int(total_runs),
            "model_summary_path": "model_summary.json",
            "problem_top1_vs_top2_path": "problem_top1_vs_top2.jsonl",
            "wrong_top1_cases_path": "wrong_top1_cases.jsonl",
            "failure_mode_summary_path": "failure_mode_summary.json",
            "sanity_checks_path": "sanity_checks.json",
            "run_contributions_dir": "run_contributions",
            "caches": manifest_entries,
        }
        _json_dump(method_dir / "manifest.json", manifest)

        print(
            f"[done] method={method_id} problems={total_problems} anchors={total_problem_anchors} runs={total_runs} "
            f"max_err={sanity_summary['overall']['max_abs_error']:.8f}"
        )


if __name__ == "__main__":
    main()
