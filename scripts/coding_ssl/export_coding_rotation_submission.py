#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS, validate_earlystop_payload, write_earlystop_payload
from scripts.coding_ssl.train_coding_rotation_adapter import (
    BUNDLE_TO_FEATURES,
    FEATURE_TO_INDEX,
    RotationSpec,
    _anchor_index,
    _build_pairwise_diffs,
    _build_payload_matrix,
    _covariance_torch,
    _find_payload,
    _fit_svd_basis,
    _load_feature_store,
    _rotation_matrix_torch,
    _row_from_result,
    _score_linear,
    _z_transform,
)


DEFAULT_RESULTS_CSV = "results/tables/coding_rotation_adapter.csv"
DEFAULT_SOURCE_FEATURE_STORE = "results/cache/es_svd_ms_rr_r1/cache_all_547b9060debe139e.pkl"
DEFAULT_BLIND_FEATURE_STORE = (
    "results/cache/export_earlystop_svd_submission_strongfeat_20260410/"
    "feature_store_all_ref030_18a73b5e30f1a00d.pkl"
)
DEFAULT_BASE_SUBMISSION = "submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json"
DEFAULT_OUT_SUBMISSION = "submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_late70100.json"
DEFAULT_OUT_MANIFEST = "results/scans/earlystop/coding_rotation_adapter_submission_manifest.json"
DEFAULT_METHOD_NAME = "es_svd_ms_rr_r1__coding_rotation_adapter_late70100"
DEFAULT_SOURCE_CACHE_KEY = "cache/DS-R1/lcb_v5"
DEFAULT_TARGET_CACHE_KEYS = ("DS-R1/lcb_v5", "Qwen3-4B/lcb_v5")
DEFAULT_TRAIN_TARGET_POOL = ("Qwen3-4B/lcb_v5",)

LATE_SLOT_POLICY = {
    6: 70,   # 70%
    7: 70,   # 80% -> 70-anchor route
    8: 70,   # 90% -> 70-anchor route
    9: 100,  # 100%
}


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    return value_f if np.isfinite(value_f) else None


def _metric_key(value: Any) -> float:
    value_f = _safe_float(value)
    return float("-inf") if value_f is None else float(value_f)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                item[key] = None
                continue
            try:
                if key in {"seed", "anchor_pct", "rank", "fit__best_step", "fit__eval_every", "fit__steps"}:
                    item[key] = int(value)
                elif key in {"bundle", "target_pool", "method", "head", "class_weight", "target_label_status"}:
                    item[key] = value
                else:
                    item[key] = float(value)
            except Exception:
                item[key] = value
        out.append(item)
    return out


def _select_anchor_row(
    rows: list[dict[str, Any]],
    anchor_pct: int,
    target_corr_min: float,
    top1_agreement_min: float,
    bundle: Optional[str] = None,
    head: Optional[str] = None,
) -> dict[str, Any]:
    learned = [
        row for row in rows
        if str(row.get("method")) == "learned_rotation"
        and int(row.get("anchor_pct")) == int(anchor_pct)
    ]
    if not learned:
        raise ValueError(f"No learned_rotation rows found for anchor={anchor_pct}")

    if bundle:
        learned = [row for row in learned if str(row.get("bundle")) == str(bundle)]
        if not learned:
            raise ValueError(f"No learned_rotation rows found for anchor={anchor_pct} bundle={bundle}")

    if head:
        learned = [row for row in learned if str(row.get("head")) == str(head)]
        if not learned:
            raise ValueError(f"No learned_rotation rows found for anchor={anchor_pct} head={head}")

    filtered = [
        row for row in learned
        if _metric_key(row.get("target_score_corr_vs_no_rotation")) >= float(target_corr_min)
        and _metric_key(row.get("target_top1_agreement_vs_no_rotation")) >= float(top1_agreement_min)
    ]
    candidates = filtered if filtered else learned
    best = max(
        candidates,
        key=lambda row: (
            _metric_key(row.get("source_val_pairwise")),
            _metric_key(row.get("source_val_hit1")),
            _metric_key(row.get("source_val_auroc")),
            _metric_key(row.get("target_score_corr_vs_no_rotation")),
            -_metric_key(row.get("rotation_norm")),
        ),
    )
    return dict(best)


def _fit_full_head(source_x: np.ndarray, source_y: np.ndarray, source_groups: np.ndarray, row: dict[str, Any], seed: int) -> dict[str, Any]:
    head_kind = str(row["head"])
    rank = int(row["rank"])
    scaler, svd = _fit_svd_basis(source_x, rank=rank, seed=seed)
    z_train = _z_transform(source_x, scaler, svd)

    if head_kind == "pointwise":
        clf = LogisticRegression(
            C=float(row["c_value"]),
            class_weight=None if str(row["class_weight"]) == "none" else "balanced",
            max_iter=2000,
            random_state=int(seed),
        )
        clf.fit(z_train, np.asarray(source_y, dtype=np.int32))
        return {
            "head": "pointwise",
            "rank": int(z_train.shape[1]),
            "scaler": scaler,
            "svd": svd,
            "weight": np.asarray(clf.coef_[0], dtype=np.float64),
            "intercept": float(clf.intercept_[0]),
        }

    x_pairs, y_pairs = _build_pairwise_diffs(z_train, source_y, source_groups)
    if x_pairs.shape[0] <= 0 or np.unique(y_pairs).shape[0] < 2:
        raise ValueError(f"Cannot fit pairwise head for anchor row={row}")
    clf = LogisticRegression(
        C=float(row["c_value"]),
        class_weight=None,
        max_iter=2000,
        random_state=int(seed),
    )
    clf.fit(x_pairs, y_pairs)
    return {
        "head": "pairwise",
        "rank": int(z_train.shape[1]),
        "scaler": scaler,
        "svd": svd,
        "weight": np.asarray(clf.coef_[0], dtype=np.float64),
        "intercept": float(clf.intercept_[0]),
    }


def _fit_rotation_fixed_steps(
    *,
    head: dict[str, Any],
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_groups: np.ndarray,
    target_x_pool: np.ndarray,
    align_weight: float,
    identity_weight: float,
    lr: float,
    steps: int,
    seed: int,
    torch_threads: int,
) -> np.ndarray:
    torch.set_num_threads(max(1, int(torch_threads)))
    torch.manual_seed(int(seed))

    z_source = _z_transform(source_x, head["scaler"], head["svd"])
    z_target = _z_transform(target_x_pool, head["scaler"], head["svd"])

    rank = int(head["rank"])
    if int(steps) <= 0:
        return np.eye(rank, dtype=np.float64)

    z_src_t = torch.as_tensor(z_source, dtype=torch.float64)
    y_src_t = torch.as_tensor(np.asarray(source_y, dtype=np.float64), dtype=torch.float64)
    z_tgt_t = torch.as_tensor(z_target, dtype=torch.float64)
    w_t = torch.as_tensor(np.asarray(head["weight"], dtype=np.float64), dtype=torch.float64)
    b_t = torch.tensor(float(head["intercept"]), dtype=torch.float64)

    src_mean_t = z_src_t.mean(dim=0)
    src_cov_t = _covariance_torch(z_src_t)

    raw_param = torch.zeros((rank, rank), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw_param], lr=float(lr))

    pos_idx = neg_idx = None
    if str(head["head"]) == "pairwise":
        from scripts.coding_ssl.train_coding_rotation_adapter import _pair_indices_for_source

        pos_arr, neg_arr = _pair_indices_for_source(source_y, source_groups)
        pos_idx = torch.as_tensor(pos_arr, dtype=torch.long)
        neg_idx = torch.as_tensor(neg_arr, dtype=torch.long)

    eye = torch.eye(rank, dtype=torch.float64)
    for _step in range(1, int(steps) + 1):
        rotation_t = _rotation_matrix_torch(raw_param)
        z_src_rot = z_src_t @ rotation_t.T

        if str(head["head"]) == "pointwise":
            logits = z_src_rot @ w_t + b_t
            sup_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_src_t)
        else:
            if pos_idx is None or neg_idx is None or pos_idx.numel() <= 0:
                sup_loss = torch.tensor(0.0, dtype=torch.float64)
            else:
                pair_scores = z_src_rot @ w_t
                margins = pair_scores[pos_idx] - pair_scores[neg_idx]
                sup_loss = torch.nn.functional.softplus(-margins).mean()

        z_tgt_rot = z_tgt_t @ rotation_t.T
        mean_loss = torch.mean((z_tgt_rot.mean(dim=0) - src_mean_t) ** 2)
        cov_loss = torch.mean((_covariance_torch(z_tgt_rot) - src_cov_t) ** 2)
        align_loss = mean_loss + cov_loss
        identity_loss = torch.mean((rotation_t - eye) ** 2)
        orth_loss = torch.mean((rotation_t.T @ rotation_t - eye) ** 2)

        total_loss = sup_loss + float(align_weight) * align_loss + float(identity_weight) * identity_loss + 0.10 * orth_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    rotation = _rotation_matrix_torch(raw_param).detach().cpu().numpy().astype(np.float64, copy=False)
    return rotation


def _score_blind_payload_for_anchor(payload: dict[str, Any], *, feature_names: tuple[str, ...], head: dict[str, Any], rotation: np.ndarray, position_index: int) -> dict[str, dict[str, float]]:
    matrix = _build_payload_matrix(
        payload,
        anchor_pct=int(round(float(EARLY_STOP_POSITIONS[int(position_index)]) * 100.0)),
        feature_names=feature_names,
    )
    z = _z_transform(matrix["x"], head["scaler"], head["svd"]) @ np.asarray(rotation, dtype=np.float64).T
    scores = _score_linear(z, head["weight"], head["intercept"])

    problem_scores: dict[str, dict[str, float]] = {}
    offsets = [int(v) for v in payload["problem_offsets"]]
    sample_ids = np.asarray(payload["sample_ids"], dtype=np.int64)
    problem_ids = [str(v) for v in payload["problem_ids"]]
    for problem_idx, problem_id in enumerate(problem_ids):
        start = offsets[problem_idx]
        end = offsets[problem_idx + 1]
        run_scores = {}
        for local_idx in range(start, end):
            run_scores[str(int(sample_ids[local_idx]))] = float(scores[local_idx])
        problem_scores[str(problem_id)] = run_scores
    return problem_scores


def _rotation_norm(rotation: np.ndarray) -> float:
    rot = np.asarray(rotation, dtype=np.float64)
    return float(np.linalg.norm(rot - np.eye(rot.shape[0], dtype=np.float64), ord="fro"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a coding-rotation EarlyStop submission patch")
    ap.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV)
    ap.add_argument("--source-feature-store", default=DEFAULT_SOURCE_FEATURE_STORE)
    ap.add_argument("--blind-feature-store", default=DEFAULT_BLIND_FEATURE_STORE)
    ap.add_argument("--base-submission", default=DEFAULT_BASE_SUBMISSION)
    ap.add_argument("--source-cache-key", default=DEFAULT_SOURCE_CACHE_KEY)
    ap.add_argument("--patch-cache-keys", default=",".join(DEFAULT_TARGET_CACHE_KEYS))
    ap.add_argument("--train-target-pool", default=",".join(DEFAULT_TRAIN_TARGET_POOL))
    ap.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    ap.add_argument("--out-submission", default=DEFAULT_OUT_SUBMISSION)
    ap.add_argument("--out-manifest", default=DEFAULT_OUT_MANIFEST)
    ap.add_argument("--target-corr-min", type=float, default=0.80)
    ap.add_argument("--target-top1-agreement-min", type=float, default=0.10)
    ap.add_argument("--anchor70-bundle", default=None)
    ap.add_argument("--anchor70-head", default=None)
    ap.add_argument("--anchor100-bundle", default=None)
    ap.add_argument("--anchor100-head", default=None)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()

    results_csv = (REPO_ROOT / str(args.results_csv)).resolve()
    source_feature_store = (REPO_ROOT / str(args.source_feature_store)).resolve()
    blind_feature_store = (REPO_ROOT / str(args.blind_feature_store)).resolve()
    base_submission_path = (REPO_ROOT / str(args.base_submission)).resolve()
    out_submission = (REPO_ROOT / str(args.out_submission)).resolve()
    out_manifest = (REPO_ROOT / str(args.out_manifest)).resolve()

    rows = _load_rows(results_csv)
    row70 = _select_anchor_row(
        rows,
        70,
        args.target_corr_min,
        args.target_top1_agreement_min,
        bundle=args.anchor70_bundle,
        head=args.anchor70_head,
    )
    row100 = _select_anchor_row(
        rows,
        100,
        args.target_corr_min,
        args.target_top1_agreement_min,
        bundle=args.anchor100_bundle,
        head=args.anchor100_head,
    )
    chosen_rows = {70: row70, 100: row100}

    print(
        "[select] anchor70 bundle={bundle} head={head} rank={rank} pairwise={pairwise:.4f} target_corr={corr:.4f}".format(
            bundle=row70["bundle"], head=row70["head"], rank=int(row70["rank"]),
            pairwise=float(row70["source_val_pairwise"]), corr=float(row70["target_score_corr_vs_no_rotation"]),
        ),
        flush=True,
    )
    print(
        "[select] anchor100 bundle={bundle} head={head} rank={rank} pairwise={pairwise:.4f} target_corr={corr:.4f}".format(
            bundle=row100["bundle"], head=row100["head"], rank=int(row100["rank"]),
            pairwise=float(row100["source_val_pairwise"]), corr=float(row100["target_score_corr_vs_no_rotation"]),
        ),
        flush=True,
    )

    source_store = _load_feature_store(source_feature_store)
    blind_store = _load_feature_store(blind_feature_store)
    source_payload = _find_payload(source_store, args.source_cache_key)
    blind_payloads = {str(payload["cache_key"]): payload for payload in blind_store}

    patch_cache_keys = _parse_csv(args.patch_cache_keys)
    train_target_pool_keys = _parse_csv(args.train_target_pool)
    missing_patch = [key for key in patch_cache_keys if key not in blind_payloads]
    if missing_patch:
        raise ValueError(f"Missing blind payloads for patch cache keys: {missing_patch}")
    missing_target_pool = [key for key in train_target_pool_keys if key not in blind_payloads]
    if missing_target_pool:
        raise ValueError(f"Missing blind payloads for train target pool: {missing_target_pool}")

    fitted_by_anchor: dict[int, dict[str, Any]] = {}
    for anchor_pct, row in chosen_rows.items():
        feature_names = BUNDLE_TO_FEATURES[str(row["bundle"])]
        source_matrix = _build_payload_matrix(source_payload, anchor_pct=anchor_pct, feature_names=feature_names)
        head = _fit_full_head(
            source_matrix["x"],
            source_matrix["labels"],
            source_matrix["group_keys"],
            row,
            seed=int(args.random_state),
        )
        target_pool_x_parts = []
        for cache_key in train_target_pool_keys:
            payload = blind_payloads[cache_key]
            target_matrix = _build_payload_matrix(payload, anchor_pct=anchor_pct, feature_names=feature_names)
            target_pool_x_parts.append(np.asarray(target_matrix["x"], dtype=np.float64))
        target_pool_x = np.concatenate(target_pool_x_parts, axis=0).astype(np.float64, copy=False)

        best_step = int(row.get("fit__best_step") or 0)
        if best_step <= 0:
            rotation = np.eye(int(head["rank"]), dtype=np.float64)
        else:
            rotation = _fit_rotation_fixed_steps(
                head=head,
                source_x=source_matrix["x"],
                source_y=source_matrix["labels"],
                source_groups=source_matrix["group_keys"],
                target_x_pool=target_pool_x,
                align_weight=float(row.get("fit__align_weight") or 0.0),
                identity_weight=float(row.get("fit__identity_weight") or 0.0),
                lr=float(row.get("fit__lr") or 0.05),
                steps=best_step,
                seed=int(args.random_state),
                torch_threads=int(args.torch_threads),
            )
        fitted_by_anchor[anchor_pct] = {
            "row": row,
            "feature_names": feature_names,
            "head": head,
            "rotation": rotation,
            "rotation_norm_refit": _rotation_norm(rotation),
        }

    base_payload = json.loads(base_submission_path.read_text(encoding="utf-8"))
    validate_earlystop_payload(base_payload)
    patched_payload = copy.deepcopy(base_payload)
    patched_payload["method_name"] = str(args.method_name)

    delta_summary: dict[str, Any] = {}
    for cache_key in patch_cache_keys:
        base_cache_scores = base_payload["scores"][cache_key]
        patched_cache_scores = copy.deepcopy(base_cache_scores)
        payload = blind_payloads[cache_key]

        for position_index, anchor_pct in LATE_SLOT_POLICY.items():
            fitted = fitted_by_anchor[anchor_pct]
            problem_scores = _score_blind_payload_for_anchor(
                payload,
                feature_names=fitted["feature_names"],
                head=fitted["head"],
                rotation=fitted["rotation"],
                position_index=position_index,
            )
            for problem_id, sample_map in problem_scores.items():
                for sample_id, score in sample_map.items():
                    patched_cache_scores[problem_id][sample_id][int(position_index)] = float(score)

        patched_payload["scores"][cache_key] = patched_cache_scores

        base_vals: list[float] = []
        new_vals: list[float] = []
        for problem_id, sample_map in patched_cache_scores.items():
            for sample_id, new_score_list in sample_map.items():
                base_score_list = base_cache_scores[problem_id][sample_id]
                for pos_idx in sorted(LATE_SLOT_POLICY.keys()):
                    base_vals.append(float(base_score_list[pos_idx]))
                    new_vals.append(float(new_score_list[pos_idx]))
        base_arr = np.asarray(base_vals, dtype=np.float64)
        new_arr = np.asarray(new_vals, dtype=np.float64)
        delta_summary[cache_key] = {
            "late_slot_count": int(new_arr.size),
            "mean_abs_delta": float(np.mean(np.abs(new_arr - base_arr))) if new_arr.size else 0.0,
            "score_corr": float(np.corrcoef(base_arr, new_arr)[0, 1]) if new_arr.size > 1 else float("nan"),
            "new_mean": float(np.mean(new_arr)) if new_arr.size else float("nan"),
            "new_std": float(np.std(new_arr)) if new_arr.size else float("nan"),
        }

    validation = validate_earlystop_payload(patched_payload)
    out_submission.parent.mkdir(parents=True, exist_ok=True)
    write_earlystop_payload(patched_payload, out_submission)

    manifest = {
        "created_at_utc": _now_utc(),
        "base_submission": _display_path(base_submission_path),
        "results_csv": _display_path(results_csv),
        "source_feature_store": _display_path(source_feature_store),
        "blind_feature_store": _display_path(blind_feature_store),
        "method_name": str(args.method_name),
        "out_submission": _display_path(out_submission),
        "selection_overrides": {
            "70": {"bundle": args.anchor70_bundle, "head": args.anchor70_head},
            "100": {"bundle": args.anchor100_bundle, "head": args.anchor100_head},
        },
        "patch_cache_keys": list(patch_cache_keys),
        "train_target_pool": list(train_target_pool_keys),
        "late_slot_policy": {str(k): int(v) for k, v in LATE_SLOT_POLICY.items()},
        "selected_rows": {
            str(anchor): {
                key: value
                for key, value in fitted_by_anchor[anchor]["row"].items()
                if key not in {"qwen_pairwise", "qwen_hit1", "qwen_auroc"}
            }
            for anchor in sorted(fitted_by_anchor.keys())
        },
        "refit_rotation_norms": {
            str(anchor): float(fitted_by_anchor[anchor]["rotation_norm_refit"])
            for anchor in sorted(fitted_by_anchor.keys())
        },
        "late_delta_summary": delta_summary,
        "validation": validation,
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[done] submission={_display_path(out_submission)}", flush=True)
    print(f"[done] manifest={_display_path(out_manifest)}", flush=True)
    print(f"[done] validation={validation}", flush=True)


if __name__ == "__main__":
    main()
