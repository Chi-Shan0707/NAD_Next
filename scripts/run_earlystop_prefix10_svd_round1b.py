#!/usr/bin/env python3
"""Run EarlyStop prefix10 SVD round1b with expanded train split + holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop import EARLY_STOP_POSITIONS, _problem_sort_key
from nad.ops.earlystop_svd import (
    FULL_FEATURE_NAMES,
    _rank_transform_matrix,
    load_earlystop_svd_bundle,
    save_earlystop_svd_bundle,
)
from nad.ops.earlystop_svm import load_earlystop_svm_bundle
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    CONTROL_POSITIONS,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    EXTRACTION_POSITION_INDEX,
    EXTRACTION_POSITIONS,
    OFFICIAL_SLOT_TO_ANCHOR,
    PREFIX_SAFE_FEATURE_FAMILY_MAP,
    REPO_STATUS_LINES,
    SEARCH_C_VALUES,
    SEARCH_CLASS_WEIGHT,
    SEARCH_FAMILIES,
    SEARCH_RANKS,
    SEARCH_REPRESENTATIONS,
    SEARCH_WHITEN,
    _display_path,
    _fmt_earliest,
    _fmt_pct,
    _now_utc,
    _pct_label,
    build_anchor_bundle,
    build_feature_store,
    choose_best_candidate,
    collect_required_features_for_eval,
    evaluate_method_from_feature_store,
    evaluate_single_position_route_from_feature_store,
    make_bridge_score_fn,
    make_svd_bundle_score_fn,
    make_tok_conf_score_fn,
    summarise_route,
    train_routes_for_scope,
)


def _stable_hash(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _qualify_feature_store(feature_store: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    qualified: list[dict[str, Any]] = []
    for payload in feature_store:
        item = dict(payload)
        item["source_name"] = str(source_name)
        item["base_cache_key"] = str(payload["cache_key"])
        item["cache_key"] = f"{source_name}/{payload['cache_key']}"
        qualified.append(item)
    return qualified


def _subset_payload_by_problem_ids(
    payload: dict[str, Any],
    selected_problem_ids: set[str],
) -> Optional[dict[str, Any]]:
    if not selected_problem_ids:
        return None

    tensor_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    rank_group_parts: list[np.ndarray] = []
    cv_group_parts: list[np.ndarray] = []
    problem_ids: list[str] = []
    problem_offsets = [0]

    offsets = [int(v) for v in payload["problem_offsets"]]
    all_problem_ids = [str(v) for v in payload["problem_ids"]]
    total_samples = 0

    for problem_idx, problem_id in enumerate(all_problem_ids):
        if problem_id not in selected_problem_ids:
            continue
        start = offsets[problem_idx]
        end = offsets[problem_idx + 1]
        width = max(0, end - start)
        if width <= 0:
            continue

        tensor_parts.append(np.asarray(payload["tensor"][start:end], dtype=np.float64))
        label_parts.append(np.asarray(payload["labels"][start:end], dtype=np.int32))
        sample_parts.append(np.asarray(payload["sample_ids"][start:end], dtype=np.int32))
        rank_group_parts.append(
            np.asarray([f"{payload['cache_key']}::{problem_id}"] * width, dtype=object)
        )
        cv_group_parts.append(
            np.asarray([f"{payload['dataset_name']}::{problem_id}"] * width, dtype=object)
        )
        problem_ids.append(str(problem_id))
        total_samples += int(width)
        problem_offsets.append(problem_offsets[-1] + int(width))

    if not tensor_parts:
        return None

    return {
        "cache_key": str(payload["cache_key"]),
        "base_cache_key": str(payload["base_cache_key"]),
        "source_name": str(payload["source_name"]),
        "dataset_name": str(payload["dataset_name"]),
        "domain": str(payload["domain"]),
        "positions": list(payload["positions"]),
        "tensor": np.concatenate(tensor_parts, axis=0).astype(np.float64, copy=False),
        "labels": np.concatenate(label_parts).astype(np.int32, copy=False),
        "sample_ids": np.concatenate(sample_parts).astype(np.int32, copy=False),
        "group_keys": np.concatenate(rank_group_parts).astype(object, copy=False),
        "cv_group_keys": np.concatenate(cv_group_parts).astype(object, copy=False),
        "problem_ids": problem_ids,
        "problem_offsets": problem_offsets,
        "samples": int(total_samples),
    }


def _annotate_full_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _subset_payload_by_problem_ids(payload, {str(v) for v in payload["problem_ids"]}) or {
        "cache_key": str(payload["cache_key"]),
        "base_cache_key": str(payload["base_cache_key"]),
        "source_name": str(payload["source_name"]),
        "dataset_name": str(payload["dataset_name"]),
        "domain": str(payload["domain"]),
        "positions": list(payload["positions"]),
        "tensor": np.zeros((0, len(EXTRACTION_POSITIONS), len(FULL_FEATURE_NAMES)), dtype=np.float64),
        "labels": np.zeros((0,), dtype=np.int32),
        "sample_ids": np.zeros((0,), dtype=np.int32),
        "group_keys": np.asarray([], dtype=object),
        "cv_group_keys": np.asarray([], dtype=object),
        "problem_ids": [],
        "problem_offsets": [0],
        "samples": 0,
    }


def _build_holdout_problem_map(
    feature_store: list[dict[str, Any]],
    *,
    holdout_split: float,
    split_seed: int,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    datasets: dict[str, set[str]] = {}
    for payload in feature_store:
        if payload["domain"] not in {"math", "science"}:
            continue
        datasets.setdefault(str(payload["dataset_name"]), set()).update(str(v) for v in payload["problem_ids"])

    holdout_map: dict[str, set[str]] = {}
    summary: dict[str, Any] = {}
    for dataset_name in sorted(datasets.keys()):
        ordered_problem_ids = sorted(datasets[dataset_name], key=_problem_sort_key)
        if len(ordered_problem_ids) < 2:
            holdout_ids: set[str] = set()
        else:
            rng = np.random.RandomState(int(split_seed) + _stable_hash(dataset_name))
            order = rng.permutation(len(ordered_problem_ids))
            n_holdout = int(round(len(ordered_problem_ids) * float(holdout_split)))
            n_holdout = max(1, n_holdout)
            n_holdout = min(len(ordered_problem_ids) - 1, n_holdout)
            holdout_ids = {ordered_problem_ids[int(idx)] for idx in order[:n_holdout].tolist()}
        holdout_map[dataset_name] = holdout_ids
        summary[dataset_name] = {
            "total_unique_problem_ids": int(len(ordered_problem_ids)),
            "holdout_unique_problem_ids": int(len(holdout_ids)),
            "train_unique_problem_ids": int(len(ordered_problem_ids) - len(holdout_ids)),
            "holdout_problem_ids": sorted(holdout_ids, key=_problem_sort_key),
        }
    return holdout_map, summary


def _split_feature_store(
    feature_store: list[dict[str, Any]],
    *,
    holdout_problem_map: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    full_payloads: list[dict[str, Any]] = []
    train_payloads: list[dict[str, Any]] = []
    holdout_payloads: list[dict[str, Any]] = []

    for payload in feature_store:
        full_payload = _annotate_full_payload(payload)
        if full_payload["samples"] > 0:
            full_payloads.append(full_payload)

        if payload["domain"] == "coding":
            train_payload = _subset_payload_by_problem_ids(payload, {str(v) for v in payload["problem_ids"]})
            if train_payload is not None and train_payload["samples"] > 0:
                train_payloads.append(train_payload)
            continue

        holdout_ids = set(holdout_problem_map.get(str(payload["dataset_name"]), set()))
        train_ids = {str(v) for v in payload["problem_ids"] if str(v) not in holdout_ids}

        train_payload = _subset_payload_by_problem_ids(payload, train_ids)
        if train_payload is not None and train_payload["samples"] > 0:
            train_payloads.append(train_payload)

        holdout_payload = _subset_payload_by_problem_ids(payload, holdout_ids)
        if holdout_payload is not None and holdout_payload["samples"] > 0:
            holdout_payloads.append(holdout_payload)

    return train_payloads, holdout_payloads, full_payloads


def _build_scope_training_tables_with_dual_groups(
    feature_store: list[dict[str, Any]],
    positions: tuple[float, ...],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    rows: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    labels: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    rank_groups: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }
    cv_groups: dict[str, dict[int, list[np.ndarray]]] = {
        "global": {idx: [] for idx in range(len(positions))},
        "noncoding": {idx: [] for idx in range(len(positions))},
    }

    position_indices = [EXTRACTION_POSITION_INDEX[float(position)] for position in positions]
    for payload in feature_store:
        tensor = payload["tensor"]
        if tensor.shape[0] == 0:
            continue
        y = payload["labels"]
        local_rank_groups = payload["group_keys"]
        local_cv_groups = payload["cv_group_keys"]
        for local_pos_idx, src_pos_idx in enumerate(position_indices):
            x_raw = tensor[:, src_pos_idx, :]
            rows["global"][local_pos_idx].append(x_raw)
            labels["global"][local_pos_idx].append(y)
            rank_groups["global"][local_pos_idx].append(local_rank_groups)
            cv_groups["global"][local_pos_idx].append(local_cv_groups)
            if payload["domain"] in {"math", "science"}:
                rows["noncoding"][local_pos_idx].append(x_raw)
                labels["noncoding"][local_pos_idx].append(y)
                rank_groups["noncoding"][local_pos_idx].append(local_rank_groups)
                cv_groups["noncoding"][local_pos_idx].append(local_cv_groups)

    out: dict[str, dict[int, dict[str, np.ndarray]]] = {"global": {}, "noncoding": {}}
    for scope in ("global", "noncoding"):
        for pos_idx in range(len(positions)):
            if rows[scope][pos_idx]:
                x_raw = np.vstack(rows[scope][pos_idx]).astype(np.float64, copy=False)
                y = np.concatenate(labels[scope][pos_idx]).astype(np.int32, copy=False)
                groups_rank = np.concatenate(rank_groups[scope][pos_idx]).astype(object, copy=False)
                groups_cv = np.concatenate(cv_groups[scope][pos_idx]).astype(object, copy=False)
            else:
                x_raw = np.zeros((0, len(FULL_FEATURE_NAMES)), dtype=np.float64)
                y = np.zeros((0,), dtype=np.int32)
                groups_rank = np.asarray([], dtype=object)
                groups_cv = np.asarray([], dtype=object)

            x_rank = np.zeros_like(x_raw)
            if x_raw.shape[0] > 0:
                by_rank_group: dict[Any, list[int]] = {}
                for row_idx, group_key in enumerate(groups_rank.tolist()):
                    by_rank_group.setdefault(group_key, []).append(row_idx)
                for group_rows in by_rank_group.values():
                    x_rank[group_rows] = _rank_transform_matrix(x_raw[group_rows])

            out[scope][pos_idx] = {
                "x_raw": x_raw,
                "x_rank": x_rank,
                "y": y,
                "groups": groups_cv,
            }
    return out


def _summarise_feature_store(feature_store: list[dict[str, Any]]) -> dict[str, Any]:
    per_cache = []
    total_samples = 0
    total_problems = 0
    for payload in sorted(feature_store, key=lambda item: str(item["cache_key"])):
        problems = int(len(payload["problem_ids"]))
        samples = int(payload["samples"])
        total_samples += samples
        total_problems += problems
        per_cache.append({
            "cache_key": str(payload["cache_key"]),
            "source_name": str(payload["source_name"]),
            "dataset_name": str(payload["dataset_name"]),
            "domain": str(payload["domain"]),
            "n_problems": problems,
            "n_samples": samples,
        })
    return {
        "num_caches": int(len(per_cache)),
        "total_problems": int(total_problems),
        "total_samples": int(total_samples),
        "per_cache": per_cache,
    }


def _render_control_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Checkpoint | AUROC | SelAcc@10 | StopAcc |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        agg = row["aggregate"]
        lines.append(
            "| {position} | {auroc} | {selacc} | {stop} |".format(
                position=_pct_label(row["position"]),
                auroc=_fmt_pct(agg["auroc"]),
                selacc=_fmt_pct(agg["selacc@10%"]),
                stop=_fmt_pct(agg["stop_acc"]),
            )
        )
    lines.append("")
    return lines


def _render_method_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        agg = row["aggregate"]
        lines.append(
            "| {name} | {auc_auroc} | {auc_selacc} | {earliest} | {auroc100} | {stop100} |".format(
                name=row["method_name"],
                auc_auroc=_fmt_pct(agg["auc_of_auroc"]),
                auc_selacc=_fmt_pct(agg["auc_of_selacc"]),
                earliest=_fmt_earliest(agg.get("earliest_gt_0.6")),
                auroc100=_fmt_pct(agg["auroc@100%"]),
                stop100=_fmt_pct(agg["stop_acc@100%"]),
            )
        )
    lines.append("")
    return lines


def _render_cache_table(title: str, method_eval: dict[str, Any]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_eval["by_cache"]:
        lines.append(
            "| {cache_key} | {auc_auroc} | {auc_selacc} | {earliest} | {auroc100} | {stop100} |".format(
                cache_key=row["cache_key"],
                auc_auroc=_fmt_pct(row["auc_of_auroc"]),
                auc_selacc=_fmt_pct(row["auc_of_selacc"]),
                earliest=_fmt_earliest(row["earliest_gt_0.6"]),
                auroc100=_fmt_pct(row["auroc@100%"]),
                stop100=_fmt_pct(row["stop_acc@100%"]),
            )
        )
    lines.append("")
    return lines


def _holdout_decision(best_new: dict[str, Any], baseline_v1: dict[str, Any], allow_export: bool) -> dict[str, Any]:
    cand = best_new["aggregate"]
    base = baseline_v1["aggregate"]
    beats = (
        float(cand["auc_of_selacc"]) > float(base["auc_of_selacc"])
        and float(cand["auc_of_auroc"]) >= float(base["auc_of_auroc"])
        and float(cand["auroc@100%"]) >= float(base["auroc@100%"])
        and float(cand["stop_acc@100%"]) >= float(base["stop_acc@100%"])
    )
    if beats:
        reason = "holdout split strict dominance over earlystop_svd_lowrank_lr_v1"
    else:
        failures = []
        if float(cand["auc_of_selacc"]) <= float(base["auc_of_selacc"]):
            failures.append("AUC of SelAcc 未超过 v1")
        if float(cand["auc_of_auroc"]) < float(base["auc_of_auroc"]):
            failures.append("AUC of AUROC 低于 v1")
        if float(cand["auroc@100%"]) < float(base["auroc@100%"]):
            failures.append("AUROC@100% 低于 v1")
        if float(cand["stop_acc@100%"]) < float(base["stop_acc@100%"]):
            failures.append("Stop Acc@100% 低于 v1")
        reason = "；".join(failures) if failures else "未满足严格胜出条件"
    if beats and not allow_export:
        reason += "；本轮先不自动导出，保留为下一次 blind 提交候选"
    return {
        "winner_method_name": str(best_new["method_name"]),
        "holdout_beats_v1": bool(beats),
        "export_recommended": bool(beats and allow_export),
        "reason": str(reason),
    }


def _write_results_doc(path: Path, summary: dict[str, Any], changed_files: list[str]) -> None:
    holdout_best_name = summary["decision"]["winner_method_name"]
    holdout_best = summary["holdout_eval"]["candidates"][holdout_best_name]
    baseline_v1_holdout = summary["holdout_eval"]["baselines"]["earlystop_svd_lowrank_lr_v1"]
    train_best = summary["train_eval"]["candidates"][holdout_best_name]

    holdout_global_controls = summary["holdout_eval"]["controls"]["global"]
    holdout_noncoding_controls = summary["holdout_eval"]["controls"]["noncoding"]
    global_best_control = max(holdout_global_controls, key=lambda row: float(row["aggregate"]["auroc"]))
    noncoding_best_control = max(holdout_noncoding_controls, key=lambda row: float(row["aggregate"]["auroc"]))

    lines: list[str] = [
        "# EARLYSTOP PREFIX10 / ANCHOR4 SVD ROUND1B RESULTS (2026-04-09)",
        "",
        "## 开头确认",
        "",
    ]
    for line in REPO_STATUS_LINES:
        lines.append(f"- {line}")

    lines.extend([
        "",
        "## 1. 我确认的当前 repo 状态",
        "",
        "- repo 已有 `earlystop_svd_lowrank_lr_v1` 的完整导出链路；本轮仍只做 Early-Stop。",
        "- round1 的离线提升来自 `MUI_HUB/cache` 训练侧 / 同源 direct-eval，没有把 `cache_train` 纳入正式协议。",
        "- round1b 改成：扩大训练集，并对 non-coding 按 `dataset + problem_id` 做确定性 holdout，再继续筛选与训练。",
        "",
        "## 2. 训练 / 自测 / holdout 口径",
        "",
        f"- `主 labeled root`：`{summary['protocol']['main_cache_root']}`。",
        f"- `额外 labeled root`：`{summary['protocol']['extra_cache_root']}`。",
        f"- `holdout 规则`：对 non-coding（math + science）按 `dataset + problem_id` 做确定性 `{100 - int(round(100 * float(summary['protocol']['holdout_split'])))} / {int(round(100 * float(summary['protocol']['holdout_split'])))} train/holdout split`；同一题若同时出现在 `cache` 与 `cache_train`，会被分到同一侧，避免跨 root 泄露。",
        "- `coding`：`cache_train` 没有 `lcb_v5`，因此 coding 仍只来自 `MUI_HUB/cache`，并继续保留 `earlystop_svd_lowrank_lr_v1` fallback。",
        f"- `train-side direct-eval`：在 split-train 上打分，只用于看训练侧增益；本轮 train slice 共 `{summary['protocol']['train_store']['total_problems']}` 个 problem-slices / `{summary['protocol']['train_store']['total_samples']}` 个 samples。",
        f"- `holdout self-test`：在 split-holdout 上打分，这是本轮主选择口径；holdout slice 共 `{summary['protocol']['holdout_store']['total_problems']}` 个 problem-slices / `{summary['protocol']['holdout_store']['total_samples']}` 个 samples。",
        "",
    ])
    if int(summary["config"]["max_problems_per_cache"]) > 0:
        lines.extend([
            f"- `capped screening`：本轮每个 cache 最多使用 `{summary['config']['max_problems_per_cache']}` 个 problems，用于先把正确协议下的筛选跑通；不是 full-data 终版。",
            "",
        ])
    lines.extend([
        "### non-coding split 摘要",
        "",
        "| Dataset | Unique Problems | Train | Holdout |",
        "|---|---:|---:|---:|",
    ])
    for dataset_name, row in sorted(summary["protocol"]["holdout_problem_summary"].items()):
        lines.append(
            f"| {dataset_name} | {row['total_unique_problem_ids']} | {row['train_unique_problem_ids']} | {row['holdout_unique_problem_ids']} |"
        )

    lines.extend([
        "",
        "## 3. prefix-10 特征定义",
        "",
        "### 保留特征",
        "",
        "- `token_only`: `tok_conf_*`、`tok_gini_*`、`tok_neg_entropy_*`、`tok_selfcert_*`、`tok_logprob_*` 与对应 token availability flags。",
        "- `token_plus_traj`: 上述 token 特征 + `traj_continuity`、`traj_reflection_count`、`traj_novelty`、`traj_max_reflection`、`traj_late_convergence` + `has_rows_bank`。",
        "- `all`: `token_plus_traj` + `nc_mean`、`nc_slope` + 全部 availability flags。",
        "",
        "### 删除特征",
        "",
        "- `self_similarity`：会把更后段信息泄露进 prefix 视角；round1b 继续删除。",
        "",
        "## 4. 5/10/15/20 checkpoint 对照（holdout）",
        "",
    ])
    lines.extend(_render_control_table("全域统一单 checkpoint / holdout", holdout_global_controls))
    lines.extend(_render_control_table("非 coding 单 checkpoint / holdout", holdout_noncoding_controls))
    lines.extend([
        "- 本轮决策以 holdout 为准，不再用 train-side direct-eval 决定是否推进。",
        f"- holdout control 结果里，全域最好是 `{_pct_label(global_best_control['position'])}`，非 coding 最好是 `{_pct_label(noncoding_best_control['position'])}`。",
        "",
        "## 5. 与 `earlystop_svd_lowrank_lr_v1` 的对比（holdout）",
        "",
    ])

    holdout_compare_rows = [
        summary["holdout_eval"]["baselines"]["tok_conf_prefix_mean_v1"],
        summary["holdout_eval"]["baselines"]["earlystop_from_bestofn_svm_bridge_v1"],
        baseline_v1_holdout,
        summary["holdout_eval"]["candidates"]["global_anchor4"],
        summary["holdout_eval"]["candidates"]["noncoding_anchor4_coding_v1"],
    ]
    lines.extend(_render_method_table("holdout 总体对比", holdout_compare_rows))
    lines.extend(_render_cache_table("holdout best candidate per-cache", holdout_best))

    lines.extend([
        "## 6. 训练侧 direct-eval（只作诊断，不作决策）",
        "",
    ])
    train_compare_rows = [
        summary["train_eval"]["baselines"]["tok_conf_prefix_mean_v1"],
        summary["train_eval"]["baselines"]["earlystop_from_bestofn_svm_bridge_v1"],
        summary["train_eval"]["baselines"]["earlystop_svd_lowrank_lr_v1"],
        summary["train_eval"]["candidates"]["global_anchor4"],
        summary["train_eval"]["candidates"]["noncoding_anchor4_coding_v1"],
    ]
    lines.extend(_render_method_table("train-side 总体对比", train_compare_rows))
    lines.extend([
        f"- holdout winner 在 train-side 的方法名仍是 `{holdout_best_name}`。",
        "- 这里的提升仅用于判断是否存在明显过拟合；最终是否推进，只看 holdout 表。",
        "",
        "## 7. 是否导出新 submission",
        "",
        f"- `holdout winner`：`{summary['decision']['winner_method_name']}`。",
        f"- `holdout 是否严格胜出 v1`：`{'YES' if summary['decision']['holdout_beats_v1'] else 'NO'}`。",
        f"- `本轮是否导出`：`{'YES' if summary['decision']['export_recommended'] else 'NO'}`。",
        f"- 判定理由：{summary['decision']['reason']}。",
        "",
        "## 8. 如果没有胜出，失败原因是什么",
        "",
        f"- {'不适用：holdout 已严格胜出 v1。' if summary['decision']['holdout_beats_v1'] else summary['decision']['reason'] + '.'}",
        "",
        "## 9. 改了哪些文件",
        "",
    ])
    for file_path in changed_files:
        lines.append(f"- `{file_path}`")

    lines.extend([
        "",
        "## 10. 如何复跑",
        "",
        "```bash",
        "bash cookbook/00_setup/verify.sh",
        "python3 scripts/run_earlystop_prefix10_svd_round1b.py \\",
        "  --main-cache-root MUI_HUB/cache \\",
        "  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train",
        "```",
        "",
        "### full-fit candidate",
        "",
        f"- 保存路径：`{summary['fullfit']['saved_model']}`。",
        f"- full-fit 采用的 winner family：`{summary['fullfit']['winner_method_name']}`。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run EarlyStop prefix10 SVD round1b with expanded train + holdout split")
    ap.add_argument("--main-cache-root", default="MUI_HUB/cache", help="Primary labeled cache root")
    ap.add_argument("--extra-cache-root", default="/home/jovyan/public-ro/MUI_HUB/cache_train", help="Extra labeled cache root")
    ap.add_argument("--out-model", default="models/ml_selectors/earlystop_prefix10_svd_round1b.pkl")
    ap.add_argument("--out-summary", default="results/scans/earlystop/earlystop_prefix10_svd_round1b_summary.json")
    ap.add_argument("--out-eval", default="results/scans/earlystop/earlystop_prefix10_svd_round1b_eval.json")
    ap.add_argument("--out-doc", default="docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_RESULTS_20260409.md")
    ap.add_argument("--holdout-split", type=float, default=0.20)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--export-if-holdout-win", action="store_true", help="Export only if holdout strictly beats v1")
    args = ap.parse_args()

    main_cache_root = args.main_cache_root
    if not Path(main_cache_root).is_absolute():
        main_cache_root = str((REPO_ROOT / main_cache_root).resolve())
    extra_cache_root = str(Path(args.extra_cache_root).resolve())

    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)

    v1_bundle = load_earlystop_svd_bundle(REPO_ROOT / "models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl")
    bridge_bundle = load_earlystop_svm_bundle(REPO_ROOT / "models/ml_selectors/bestofn_svm_bridge_v1.pkl")
    required_features = collect_required_features_for_eval(
        v1_bundle=v1_bundle,
        bridge_bundle=bridge_bundle,
    )

    print(f"[round1b] building feature store for main root={main_cache_root}")
    main_store = _qualify_feature_store(
        build_feature_store(
            cache_root=main_cache_root,
            positions=EXTRACTION_POSITIONS,
            required_feature_names=required_features,
            max_problems_per_cache=max_problems_per_cache,
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
        ),
        source_name="cache",
    )
    print(f"[round1b] building feature store for extra root={extra_cache_root}")
    extra_store = _qualify_feature_store(
        build_feature_store(
            cache_root=extra_cache_root,
            positions=EXTRACTION_POSITIONS,
            required_feature_names=required_features,
            max_problems_per_cache=max_problems_per_cache,
            max_workers=int(args.workers),
            chunk_problems=int(args.feature_chunk_problems),
        ),
        source_name="cache_train",
    )

    combined_store = main_store + extra_store
    holdout_problem_map, holdout_problem_summary = _build_holdout_problem_map(
        combined_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_store, holdout_store, full_store = _split_feature_store(
        combined_store,
        holdout_problem_map=holdout_problem_map,
    )

    holdout_store = [payload for payload in holdout_store if payload["domain"] in {"math", "science"}]

    positions_needed = tuple(sorted(set(CONTROL_POSITIONS + ANCHOR_POSITIONS)))
    train_tables = _build_scope_training_tables_with_dual_groups(train_store, positions=positions_needed)
    full_tables = _build_scope_training_tables_with_dual_groups(full_store, positions=positions_needed)

    anchor_indices = tuple(positions_needed.index(p) for p in ANCHOR_POSITIONS)
    control_indices = tuple(positions_needed.index(p) for p in CONTROL_POSITIONS)

    splitfit_global_anchor_routes = train_routes_for_scope(
        tables=[train_tables["global"][i] for i in anchor_indices],
        positions=ANCHOR_POSITIONS,
        scope_name="global_splitfit",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )
    splitfit_noncoding_anchor_routes = train_routes_for_scope(
        tables=[train_tables["noncoding"][i] for i in anchor_indices],
        positions=ANCHOR_POSITIONS,
        scope_name="noncoding_splitfit",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )
    splitfit_global_control_routes = train_routes_for_scope(
        tables=[train_tables["global"][i] for i in control_indices],
        positions=CONTROL_POSITIONS,
        scope_name="global_control_splitfit",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )
    splitfit_noncoding_control_routes = train_routes_for_scope(
        tables=[train_tables["noncoding"][i] for i in control_indices],
        positions=CONTROL_POSITIONS,
        scope_name="noncoding_control_splitfit",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )

    splitfit_global_bundle = build_anchor_bundle(
        bundle_version="earlystop_prefix10_svd_round1b_global_anchor4_splitfit",
        anchor_routes=splitfit_global_anchor_routes,
    )
    splitfit_noncoding_bundle = build_anchor_bundle(
        bundle_version="earlystop_prefix10_svd_round1b_noncoding_anchor4_coding_v1_splitfit",
        anchor_routes=splitfit_noncoding_anchor_routes,
        coding_routes=v1_bundle["domains"]["coding"]["routes"],
    )

    print("[round1b] evaluating split-fit routes on holdout")
    holdout_baselines = {
        "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
            method_name="tok_conf_prefix_mean_v1",
            feature_store=holdout_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_tok_conf_score_fn(),
        ),
        "earlystop_from_bestofn_svm_bridge_v1": evaluate_method_from_feature_store(
            method_name="earlystop_from_bestofn_svm_bridge_v1",
            feature_store=holdout_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_bridge_score_fn(bridge_bundle),
        ),
        "earlystop_svd_lowrank_lr_v1": evaluate_method_from_feature_store(
            method_name="earlystop_svd_lowrank_lr_v1",
            feature_store=holdout_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(v1_bundle),
        ),
    }
    holdout_candidates = {
        "global_anchor4": evaluate_method_from_feature_store(
            method_name="global_anchor4",
            feature_store=holdout_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(splitfit_global_bundle),
        ),
        "noncoding_anchor4_coding_v1": evaluate_method_from_feature_store(
            method_name="noncoding_anchor4_coding_v1",
            feature_store=holdout_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(splitfit_noncoding_bundle),
        ),
    }
    holdout_global_controls = [
        {
            "position": float(position),
            **evaluate_single_position_route_from_feature_store(
                feature_store=holdout_store,
                position=float(position),
                route_resolver=lambda _domain, p=float(position): splitfit_global_control_routes[p],
            ),
        }
        for position in CONTROL_POSITIONS
    ]
    holdout_noncoding_controls = [
        {
            "position": float(position),
            **evaluate_single_position_route_from_feature_store(
                feature_store=holdout_store,
                position=float(position),
                route_resolver=lambda _domain, p=float(position): splitfit_noncoding_control_routes[p],
            ),
        }
        for position in CONTROL_POSITIONS
    ]

    print("[round1b] evaluating split-fit routes on train-side")
    train_baselines = {
        "tok_conf_prefix_mean_v1": evaluate_method_from_feature_store(
            method_name="tok_conf_prefix_mean_v1",
            feature_store=train_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_tok_conf_score_fn(),
        ),
        "earlystop_from_bestofn_svm_bridge_v1": evaluate_method_from_feature_store(
            method_name="earlystop_from_bestofn_svm_bridge_v1",
            feature_store=train_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_bridge_score_fn(bridge_bundle),
        ),
        "earlystop_svd_lowrank_lr_v1": evaluate_method_from_feature_store(
            method_name="earlystop_svd_lowrank_lr_v1",
            feature_store=train_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(v1_bundle),
        ),
    }
    train_candidates = {
        "global_anchor4": evaluate_method_from_feature_store(
            method_name="global_anchor4",
            feature_store=train_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(splitfit_global_bundle),
        ),
        "noncoding_anchor4_coding_v1": evaluate_method_from_feature_store(
            method_name="noncoding_anchor4_coding_v1",
            feature_store=train_store,
            position_values=tuple(float(p) for p in EARLY_STOP_POSITIONS),
            score_fn=make_svd_bundle_score_fn(splitfit_noncoding_bundle),
        ),
    }

    holdout_best = choose_best_candidate(list(holdout_candidates.values()))
    decision = _holdout_decision(
        best_new=holdout_best,
        baseline_v1=holdout_baselines["earlystop_svd_lowrank_lr_v1"],
        allow_export=bool(args.export_if_holdout_win),
    )

    print("[round1b] refitting winner family on full expanded data")
    fullfit_global_anchor_routes = train_routes_for_scope(
        tables=[full_tables["global"][i] for i in anchor_indices],
        positions=ANCHOR_POSITIONS,
        scope_name="global_fullfit",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )
    fullfit_noncoding_anchor_routes = train_routes_for_scope(
        tables=[full_tables["noncoding"][i] for i in anchor_indices],
        positions=ANCHOR_POSITIONS,
        scope_name="noncoding_fullfit",
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        max_workers=int(args.workers),
    )

    fullfit_global_bundle = build_anchor_bundle(
        bundle_version="earlystop_prefix10_svd_round1b_global_anchor4_fullfit",
        anchor_routes=fullfit_global_anchor_routes,
    )
    fullfit_noncoding_bundle = build_anchor_bundle(
        bundle_version="earlystop_prefix10_svd_round1b_noncoding_anchor4_coding_v1_fullfit",
        anchor_routes=fullfit_noncoding_anchor_routes,
        coding_routes=v1_bundle["domains"]["coding"]["routes"],
    )

    winner_method_name = str(decision["winner_method_name"])
    best_bundle = fullfit_global_bundle if winner_method_name == "global_anchor4" else fullfit_noncoding_bundle
    out_model = REPO_ROOT / args.out_model
    save_earlystop_svd_bundle(best_bundle, out_model)

    summary = {
        "created_at_utc": _now_utc(),
        "repo_status": REPO_STATUS_LINES,
        "protocol": {
            "main_cache_root": str(main_cache_root),
            "extra_cache_root": str(extra_cache_root),
            "holdout_split": float(args.holdout_split),
            "split_seed": int(args.split_seed),
            "train_store": _summarise_feature_store(train_store),
            "holdout_store": _summarise_feature_store(holdout_store),
            "full_store": _summarise_feature_store(full_store),
            "holdout_problem_summary": holdout_problem_summary,
        },
        "config": {
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "workers": int(args.workers),
            "feature_chunk_problems": int(args.feature_chunk_problems),
            "max_problems_per_cache": 0 if max_problems_per_cache is None else int(max_problems_per_cache),
            "control_positions": list(CONTROL_POSITIONS),
            "anchor_positions": list(ANCHOR_POSITIONS),
            "official_slot_to_anchor": {str(k): float(v) for k, v in OFFICIAL_SLOT_TO_ANCHOR.items()},
            "families": list(SEARCH_FAMILIES),
            "representations": list(SEARCH_REPRESENTATIONS),
            "svd_dims": list(SEARCH_RANKS),
            "c_values": list(SEARCH_C_VALUES),
            "class_weight": list(SEARCH_CLASS_WEIGHT),
            "whiten": [bool(v) for v in SEARCH_WHITEN],
        },
        "feature_definition": {
            "kept_features": list(PREFIX_SAFE_FEATURE_FAMILY_MAP["all"]),
            "dropped_features": ["self_similarity"],
            "family_map": PREFIX_SAFE_FEATURE_FAMILY_MAP,
        },
        "train_eval": {
            "baselines": train_baselines,
            "candidates": train_candidates,
        },
        "holdout_eval": {
            "baselines": holdout_baselines,
            "candidates": holdout_candidates,
            "controls": {
                "global": holdout_global_controls,
                "noncoding": holdout_noncoding_controls,
            },
        },
        "splitfit_route_summary": {
            "global_anchor4": {
                _pct_label(position): summarise_route(route)
                for position, route in splitfit_global_anchor_routes.items()
            },
            "noncoding_anchor4_coding_v1": {
                "noncoding_anchor_routes": {
                    _pct_label(position): summarise_route(route)
                    for position, route in splitfit_noncoding_anchor_routes.items()
                },
                "coding_routes_source": "earlystop_svd_lowrank_lr_v1",
            },
        },
        "fullfit": {
            "winner_method_name": winner_method_name,
            "saved_model": _display_path(out_model),
            "bundle_version": str(best_bundle["bundle_version"]),
            "global_anchor4": {
                _pct_label(position): summarise_route(route)
                for position, route in fullfit_global_anchor_routes.items()
            },
            "noncoding_anchor4_coding_v1": {
                "noncoding_anchor_routes": {
                    _pct_label(position): summarise_route(route)
                    for position, route in fullfit_noncoding_anchor_routes.items()
                },
                "coding_routes_source": "earlystop_svd_lowrank_lr_v1",
            },
        },
        "decision": decision,
    }

    out_summary = REPO_ROOT / args.out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_eval = REPO_ROOT / args.out_eval
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    out_eval.write_text(
        json.dumps(
            {
                "train_eval": summary["train_eval"],
                "holdout_eval": summary["holdout_eval"],
                "decision": summary["decision"],
                "fullfit": {
                    "winner_method_name": summary["fullfit"]["winner_method_name"],
                    "saved_model": summary["fullfit"]["saved_model"],
                    "bundle_version": summary["fullfit"]["bundle_version"],
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    changed_files = [
        "docs/EARLYSTOP_PREFIX10_SVD_ROUND1B_PLAN_20260409.md",
        str(args.out_doc),
        "scripts/run_earlystop_prefix10_svd_round1b.py",
        _display_path(out_model),
    ]
    _write_results_doc(
        path=REPO_ROOT / args.out_doc,
        summary=summary,
        changed_files=changed_files,
    )

    print("\nRound1b complete")
    print(f"  Saved full-fit bundle : {out_model}")
    print(f"  Summary JSON          : {out_summary}")
    print(f"  Eval JSON             : {out_eval}")
    print(f"  Results doc           : {REPO_ROOT / args.out_doc}")


if __name__ == "__main__":
    main()
