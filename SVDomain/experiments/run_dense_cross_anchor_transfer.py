#!/usr/bin/env python3
"""Run dense 10-anchor cross-anchor frozen-basis transfer for canonical es_svd_ms_rr_r2."""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from nad.ops.earlystop import EARLY_STOP_POSITIONS
from nad.ops.earlystop_svd import load_earlystop_svd_bundle
from run_cross_anchor_transfer import (
    CONDITION_ORDER,
    DEFAULT_FEATURE_CHUNK_PROBLEMS,
    DOMAIN_ORDER,
    EXTRACTION_POSITIONS,
    METRIC_ORDER,
    _collect_task_representation,
    _cv_compare_metrics,
    _fmt_pct,
    _fmt_signed_pct_points,
    _format_float,
    _load_or_build_noncoding_feature_store,
    _now_utc,
    _resolve_path,
    _route_feature_key,
    _verdict,
    _write_csv,
)


DEFAULT_BUNDLE_PATH = "models/ml_selectors/es_svd_ms_rr_r2.pkl"
DEFAULT_FEATURE_CACHE_DIR = "results/cache/dense_cross_anchor_transfer"
DEFAULT_OUT_MATRIX = "results/tables/dense_cross_anchor_transfer_matrix.csv"
DEFAULT_OUT_DELTAS = "results/tables/dense_cross_anchor_transfer_deltas.csv"
DEFAULT_OUT_SUMMARY = "results/tables/dense_cross_anchor_transfer_summary.csv"
DEFAULT_OUT_NOTE = "docs/17_DENSE_CROSS_ANCHOR_TRANSFER.md"
DEFAULT_PAPER_OUT_MATRIX = "SVDomain/results/tables/dense_cross_anchor_transfer_matrix.csv"
DEFAULT_PAPER_OUT_DELTAS = "SVDomain/results/tables/dense_cross_anchor_transfer_deltas.csv"
DEFAULT_PAPER_OUT_SUMMARY = "SVDomain/results/tables/dense_cross_anchor_transfer_summary.csv"
DENSE_ANCHOR_PCTS = tuple(int(round(float(v) * 100.0)) for v in EARLY_STOP_POSITIONS)
NEAR_GAP_MAX = 20
FAR_GAP_MIN = 50


def default_feature_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(8, cpu))


def default_suite_workers() -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(12, cpu))


def _parse_anchor_pcts(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value not in DENSE_ANCHOR_PCTS:
            raise ValueError(
                f"Unsupported anchor {value}; expected subset of {list(DENSE_ANCHOR_PCTS)}"
            )
        values.append(int(value))
    if not values:
        raise ValueError("Need at least one anchor")
    return tuple(sorted(dict.fromkeys(values)))


def _finite_float(value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value_f if np.isfinite(value_f) else float("nan")


def _aggregate_delta_rows(
    rows: list[dict[str, Any]],
    *,
    summary_type: str,
    domain: str,
    metric: str,
    source_anchor_pct: Any = "",
    target_anchor_pct: Any = "",
    gap_pct: Any = "",
    direction: str = "",
    pair_label: str = "",
) -> Optional[dict[str, Any]]:
    if not rows:
        return None

    def _mean_of(key: str) -> float:
        values = [_finite_float(row.get(key)) for row in rows]
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    return {
        "summary_type": str(summary_type),
        "domain": str(domain),
        "metric": str(metric),
        "source_anchor_pct": source_anchor_pct,
        "target_anchor_pct": target_anchor_pct,
        "gap_pct": gap_pct,
        "direction": str(direction),
        "pair_count": int(len(rows)),
        "pair_label": str(pair_label),
        "frozen_basis": _mean_of("frozen_basis"),
        "task_specific": _mean_of("task_specific"),
        "no_svd": _mean_of("no_svd"),
        "delta_fb_minus_ts": _mean_of("delta_fb_minus_ts"),
        "delta_fb_minus_nosvd": _mean_of("delta_fb_minus_nosvd"),
    }


def build_summary_rows(
    delta_rows: list[dict[str, Any]],
    *,
    anchors: tuple[int, ...],
    domains: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchor_set = {int(v) for v in anchors}

    for domain in domains:
        for metric in METRIC_ORDER:
            metric_rows = [
                row
                for row in delta_rows
                if str(row["domain"]) == str(domain) and str(row["metric"]) == str(metric)
            ]
            if not metric_rows:
                continue

            diagonal_rows = [
                row
                for row in metric_rows
                if int(row["source_anchor_pct"]) == int(row["target_anchor_pct"])
            ]
            offdiag_rows = [
                row
                for row in metric_rows
                if int(row["source_anchor_pct"]) != int(row["target_anchor_pct"])
            ]
            near_rows = [
                row
                for row in offdiag_rows
                if abs(int(row["target_anchor_pct"]) - int(row["source_anchor_pct"])) <= NEAR_GAP_MAX
            ]
            far_rows = [
                row
                for row in offdiag_rows
                if abs(int(row["target_anchor_pct"]) - int(row["source_anchor_pct"])) >= FAR_GAP_MIN
            ]
            forward_rows = [
                row
                for row in offdiag_rows
                if int(row["target_anchor_pct"]) > int(row["source_anchor_pct"])
            ]
            backward_rows = [
                row
                for row in offdiag_rows
                if int(row["target_anchor_pct"]) < int(row["source_anchor_pct"])
            ]

            for summary_type, subset, direction in (
                ("diagonal_mean", diagonal_rows, "diagonal"),
                ("offdiag_all_mean", offdiag_rows, "combined"),
                ("near_offdiag_mean", near_rows, "combined"),
                ("far_offdiag_mean", far_rows, "combined"),
                ("forward_all_mean", forward_rows, "forward"),
                ("backward_all_mean", backward_rows, "backward"),
            ):
                aggregate = _aggregate_delta_rows(
                    subset,
                    summary_type=summary_type,
                    domain=domain,
                    metric=metric,
                    direction=direction,
                )
                if aggregate is not None:
                    rows.append(aggregate)

            gap_values = sorted(
                {
                    abs(int(row["target_anchor_pct"]) - int(row["source_anchor_pct"]))
                    for row in offdiag_rows
                }
            )
            for gap_pct in gap_values:
                gap_rows = [
                    row
                    for row in offdiag_rows
                    if abs(int(row["target_anchor_pct"]) - int(row["source_anchor_pct"])) == int(gap_pct)
                ]
                gap_forward_rows = [
                    row
                    for row in offdiag_rows
                    if int(row["target_anchor_pct"]) - int(row["source_anchor_pct"]) == int(gap_pct)
                ]
                gap_backward_rows = [
                    row
                    for row in offdiag_rows
                    if int(row["source_anchor_pct"]) - int(row["target_anchor_pct"]) == int(gap_pct)
                ]
                for summary_type, subset, direction in (
                    ("gap_mean", gap_rows, "combined"),
                    ("forward_gap_mean", gap_forward_rows, "forward"),
                    ("backward_gap_mean", gap_backward_rows, "backward"),
                ):
                    aggregate = _aggregate_delta_rows(
                        subset,
                        summary_type=summary_type,
                        domain=domain,
                        metric=metric,
                        gap_pct=int(gap_pct),
                        direction=direction,
                    )
                    if aggregate is not None:
                        rows.append(aggregate)

            for source_anchor_pct in anchors:
                subset = [
                    row
                    for row in offdiag_rows
                    if int(row["source_anchor_pct"]) == int(source_anchor_pct)
                    and int(row["target_anchor_pct"]) in anchor_set
                ]
                aggregate = _aggregate_delta_rows(
                    subset,
                    summary_type="source_anchor_mean",
                    domain=domain,
                    metric=metric,
                    source_anchor_pct=int(source_anchor_pct),
                    direction="source",
                )
                if aggregate is not None:
                    rows.append(aggregate)

            for target_anchor_pct in anchors:
                subset = [
                    row
                    for row in offdiag_rows
                    if int(row["target_anchor_pct"]) == int(target_anchor_pct)
                    and int(row["source_anchor_pct"]) in anchor_set
                ]
                aggregate = _aggregate_delta_rows(
                    subset,
                    summary_type="target_anchor_mean",
                    domain=domain,
                    metric=metric,
                    target_anchor_pct=int(target_anchor_pct),
                    direction="target",
                )
                if aggregate is not None:
                    rows.append(aggregate)

    summary_order = {
        "diagonal_mean": 0,
        "offdiag_all_mean": 1,
        "near_offdiag_mean": 2,
        "far_offdiag_mean": 3,
        "forward_all_mean": 4,
        "backward_all_mean": 5,
        "gap_mean": 6,
        "forward_gap_mean": 7,
        "backward_gap_mean": 8,
        "source_anchor_mean": 9,
        "target_anchor_mean": 10,
    }
    metric_order = {name: idx for idx, name in enumerate(METRIC_ORDER)}
    domain_order = {name: idx for idx, name in enumerate(DOMAIN_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            summary_order.get(str(row["summary_type"]), 999),
            domain_order.get(str(row["domain"]), 999),
            metric_order.get(str(row["metric"]), 999),
            999 if row["gap_pct"] in {"", None} else int(row["gap_pct"]),
            999 if row["source_anchor_pct"] in {"", None} else int(row["source_anchor_pct"]),
            999 if row["target_anchor_pct"] in {"", None} else int(row["target_anchor_pct"]),
        ),
    )


def _find_summary(
    summary_rows: list[dict[str, Any]],
    *,
    summary_type: str,
    domain: str,
    metric: str = "auroc",
    gap_pct: Optional[int] = None,
    source_anchor_pct: Optional[int] = None,
    target_anchor_pct: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    for row in summary_rows:
        if str(row["summary_type"]) != str(summary_type):
            continue
        if str(row["domain"]) != str(domain) or str(row["metric"]) != str(metric):
            continue
        if gap_pct is not None and int(row["gap_pct"]) != int(gap_pct):
            continue
        if source_anchor_pct is not None and int(row["source_anchor_pct"]) != int(source_anchor_pct):
            continue
        if target_anchor_pct is not None and int(row["target_anchor_pct"]) != int(target_anchor_pct):
            continue
        return row
    return None


def _rank_summary(
    summary_rows: list[dict[str, Any]],
    *,
    summary_type: str,
    domain: str,
    metric: str = "auroc",
    anchor_field: str,
) -> list[tuple[int, float]]:
    ranked: list[tuple[int, float]] = []
    for row in summary_rows:
        if str(row["summary_type"]) != str(summary_type):
            continue
        if str(row["domain"]) != str(domain) or str(row["metric"]) != str(metric):
            continue
        anchor_value = row.get(anchor_field, "")
        if anchor_value in {"", None}:
            continue
        delta = _finite_float(row.get("delta_fb_minus_ts"))
        if np.isfinite(delta):
            ranked.append((int(anchor_value), float(delta)))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _top_offdiag_pairs(
    delta_rows: list[dict[str, Any]],
    *,
    domain: str,
    metric: str = "auroc",
    best: bool,
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in delta_rows
        if str(row["domain"]) == str(domain)
        and str(row["metric"]) == str(metric)
        and int(row["source_anchor_pct"]) != int(row["target_anchor_pct"])
        and np.isfinite(_finite_float(row.get("delta_fb_minus_ts")))
    ]
    rows = sorted(
        rows,
        key=lambda row: _finite_float(row["delta_fb_minus_ts"]),
        reverse=bool(best),
    )
    return rows[: int(limit)]


def _load_dense_timing_summary() -> dict[str, dict[str, Any]]:
    path = REPO_ROOT / "results/tables/dense_anchor_main_table.csv"
    if not path.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(path)
    except Exception:
        return {}

    rows: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        rows[str(row["domain"])] = {str(key): row[key] for key in df.columns}
    return rows


def build_note_markdown(
    *,
    bundle_path: Path,
    anchors: tuple[int, ...],
    domains: tuple[str, ...],
    delta_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> str:
    dense_timing = _load_dense_timing_summary()
    lines = [
        "# 17. Dense Cross-Anchor Transfer",
        "",
        "**Experiment script**: `SVDomain/experiments/run_dense_cross_anchor_transfer.py`",
        f"**Source bundle**: `{bundle_path.relative_to(REPO_ROOT)}`",
        f"**Anchors**: `{', '.join(str(v) for v in anchors)}`",
        f"**Domains**: `{', '.join(domains)}`",
        f"**Date**: {_now_utc()[:10]}",
        "",
        "---",
        "",
        "## 1. Claim Under Test",
        "",
        "> Cross-anchor transfer is not only a sparse `10/40/70/100` phenomenon, but can be traced across the full dense `10/20/.../100` reasoning trajectory.",
        "",
        "这轮实验的目的，是把原本四个 anchor 的 frozen-basis transfer 扩展成 **全 10×10 all-to-all dense grid**，直接回答：共享 low-rank basis 在整条 trajectory 上到底能迁移多远、在哪些方向上开始失真。",
        "",
        "## 2. Protocol",
        "",
        "- 条件对比仍保持不变：`frozen_basis` vs `task_specific` vs `no_svd`。",
        "- `frozen_basis`：固定 **source anchor** 的 `scaler + SVD`，只在 **target anchor** 上重训 LR head。",
        "- `task_specific`：在 **target anchor** 上重训 `scaler + SVD + LR head`。",
        "- `no_svd`：在 **target anchor** 上直接训练 `StandardScaler + LR`。",
        "- bundle 切换为 dense `r2`：`es_svd_ms_rr_r2.pkl`，因此 source / target anchors 覆盖 `10/20/.../100` 全部 10 个位置。",
        "- 评估协议与原 `11_CROSS_ANCHOR_TRANSFER.md` 一致：沿用 `GroupKFold` offline protocol，不改 split；主指标关注 `AUROC`，并同步导出 `SelAcc@10%` / `StopAcc`。",
        "",
        "## 3. Headline Summary",
        "",
    ]

    for domain in domains:
        diagonal = _find_summary(summary_rows, summary_type="diagonal_mean", domain=domain)
        offdiag = _find_summary(summary_rows, summary_type="offdiag_all_mean", domain=domain)
        near = _find_summary(summary_rows, summary_type="near_offdiag_mean", domain=domain)
        far = _find_summary(summary_rows, summary_type="far_offdiag_mean", domain=domain)
        source_ranked = _rank_summary(
            summary_rows,
            summary_type="source_anchor_mean",
            domain=domain,
            anchor_field="source_anchor_pct",
        )
        if diagonal is None or offdiag is None:
            continue
        headline = (
            "- `{domain}` diagonal mean: frozen={diag_frozen} vs task-specific={diag_task} "
            "(Δ={diag_delta}); all off-diagonal mean: frozen={off_frozen} vs task-specific={off_task} "
            "(Δ={off_delta}).".format(
                domain=domain,
                diag_frozen=_fmt_pct(diagonal["frozen_basis"]),
                diag_task=_fmt_pct(diagonal["task_specific"]),
                diag_delta=_fmt_signed_pct_points(diagonal["delta_fb_minus_ts"]),
                off_frozen=_fmt_pct(offdiag["frozen_basis"]),
                off_task=_fmt_pct(offdiag["task_specific"]),
                off_delta=_fmt_signed_pct_points(offdiag["delta_fb_minus_ts"]),
            )
        )
        if near is not None and far is not None:
            headline += " Near-gap (`10/20`) mean gap = {near_gap}; far-gap (`50–90`) mean gap = {far_gap}.".format(
                near_gap=_fmt_signed_pct_points(near["delta_fb_minus_ts"]),
                far_gap=_fmt_signed_pct_points(far["delta_fb_minus_ts"]),
            )
        if source_ranked:
            best_anchor, best_delta = source_ranked[0]
            headline += f" Best reusable source anchor is `{best_anchor}%` (mean Δ={_fmt_signed_pct_points(best_delta)})."
        lines.append(headline)

    lines.extend([
        "",
        "## 4. Dense Stability Slices",
        "",
        "| Domain | Slice | Frozen | Task-specific | No-SVD | Δ(Frozen−Task) | Δ(Frozen−NoSVD) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for summary_type, label in (
        ("diagonal_mean", "diagonal"),
        ("offdiag_all_mean", "offdiag all"),
        ("near_offdiag_mean", "offdiag near (10/20)"),
        ("far_offdiag_mean", "offdiag far (50–90)"),
        ("forward_all_mean", "forward all"),
        ("backward_all_mean", "backward all"),
    ):
        for domain in domains:
            row = _find_summary(summary_rows, summary_type=summary_type, domain=domain)
            if row is None:
                continue
            lines.append(
                "| {domain} | {label} | {frozen} | {task} | {nosvd} | {delta_task} | {delta_nosvd} |".format(
                    domain=domain,
                    label=label,
                    frozen=_fmt_pct(row["frozen_basis"]),
                    task=_fmt_pct(row["task_specific"]),
                    nosvd=_fmt_pct(row["no_svd"]),
                    delta_task=_fmt_signed_pct_points(row["delta_fb_minus_ts"]),
                    delta_nosvd=_fmt_signed_pct_points(row["delta_fb_minus_nosvd"]),
                )
            )

    lines.extend([
        "",
        "## 5. Gap Profile by Anchor Distance",
        "",
        "| Domain | |Δanchor| | Combined Δ(Frozen−Task) | Forward Δ | Backward Δ |",
        "|---|---:|---:|---:|---:|",
    ])
    for domain in domains:
        gap_rows = sorted(
            (
                row for row in summary_rows
                if str(row["summary_type"]) == "gap_mean"
                and str(row["domain"]) == str(domain)
                and str(row["metric"]) == "auroc"
                and row["gap_pct"] not in {"", None}
            ),
            key=lambda row: int(row["gap_pct"]),
        )
        for combined in gap_rows:
            forward = _find_summary(
                summary_rows,
                summary_type="forward_gap_mean",
                domain=domain,
                gap_pct=int(combined["gap_pct"]),
            )
            backward = _find_summary(
                summary_rows,
                summary_type="backward_gap_mean",
                domain=domain,
                gap_pct=int(combined["gap_pct"]),
            )
            lines.append(
                "| {domain} | {gap} | {combined_delta} | {forward_delta} | {backward_delta} |".format(
                    domain=domain,
                    gap=int(combined["gap_pct"]),
                    combined_delta=_fmt_signed_pct_points(combined["delta_fb_minus_ts"]),
                    forward_delta=_fmt_signed_pct_points(
                        forward["delta_fb_minus_ts"] if forward is not None else float("nan")
                    ),
                    backward_delta=_fmt_signed_pct_points(
                        backward["delta_fb_minus_ts"] if backward is not None else float("nan")
                    ),
                )
            )

    lines.extend([
        "",
        "## 6. Which Anchors Reuse Best?",
        "",
    ])
    for domain in domains:
        source_ranked = _rank_summary(
            summary_rows,
            summary_type="source_anchor_mean",
            domain=domain,
            anchor_field="source_anchor_pct",
        )
        target_ranked = _rank_summary(
            summary_rows,
            summary_type="target_anchor_mean",
            domain=domain,
            anchor_field="target_anchor_pct",
        )
        if source_ranked:
            source_text = ", ".join(
                f"{anchor}% ({delta * 100.0:+.2f} pts)" for anchor, delta in source_ranked
            )
            lines.append(
                f"- `{domain}` source-anchor ranking by mean off-diagonal Δ(frozen−task_specific): {source_text}."
            )
        if target_ranked:
            hardest_target, hardest_delta = target_ranked[-1]
            lines.append(
                f"- `{domain}` hardest target anchor by incoming mean transfer gap is `{hardest_target}%` (mean Δ={hardest_delta * 100.0:+.2f} pts)."
            )

    lines.extend([
        "",
        "## 7. Highlighted Pairs",
        "",
        "| Domain | Best off-diagonal pair | Δ(Frozen−Task) | Worst off-diagonal pair | Δ(Frozen−Task) |",
        "|---|---|---:|---|---:|",
    ])
    for domain in domains:
        best_rows = _top_offdiag_pairs(delta_rows, domain=domain, best=True, limit=1)
        worst_rows = _top_offdiag_pairs(delta_rows, domain=domain, best=False, limit=1)
        if not best_rows or not worst_rows:
            continue
        best_row = best_rows[0]
        worst_row = worst_rows[0]
        lines.append(
            "| {domain} | {best_pair} | {best_delta} | {worst_pair} | {worst_delta} |".format(
                domain=domain,
                best_pair=f"{int(best_row['source_anchor_pct'])}->{int(best_row['target_anchor_pct'])}",
                best_delta=_fmt_signed_pct_points(best_row["delta_fb_minus_ts"]),
                worst_pair=f"{int(worst_row['source_anchor_pct'])}->{int(worst_row['target_anchor_pct'])}",
                worst_delta=_fmt_signed_pct_points(worst_row["delta_fb_minus_ts"]),
            )
        )

    lines.extend([
        "",
        "## 8. Direct Answers",
        "",
    ])

    math_far = _find_summary(summary_rows, summary_type="far_offdiag_mean", domain="math")
    science_far = _find_summary(summary_rows, summary_type="far_offdiag_mean", domain="science")
    if math_far is not None:
        math_best = _rank_summary(
            summary_rows,
            summary_type="source_anchor_mean",
            domain="math",
            anchor_field="source_anchor_pct",
        )
        math_worst = _top_offdiag_pairs(delta_rows, domain="math", best=False, limit=1)
        text = (
            "- **Dense transfer 是否说明 math 只在少数 anchors 上可复用？** 不是。`math` 的 dense all-to-all 仍然整体接近 task-specific："
            f"diagonal mean gap 为 {_fmt_signed_pct_points(_find_summary(summary_rows, summary_type='diagonal_mean', domain='math')['delta_fb_minus_ts'])}，"
            f"far-gap (`50–90`) mean gap 也只有 {_fmt_signed_pct_points(math_far['delta_fb_minus_ts'])}。"
        )
        if math_best:
            text += f" 最可迁移的 source anchor 是 `{math_best[0][0]}%`。"
        if math_worst:
            row = math_worst[0]
            text += f" 单个最差 pair 是 `{int(row['source_anchor_pct'])}->{int(row['target_anchor_pct'])}`，gap={_fmt_signed_pct_points(row['delta_fb_minus_ts'])}。"
        lines.append(text)

    if science_far is not None:
        science_best = _rank_summary(
            summary_rows,
            summary_type="source_anchor_mean",
            domain="science",
            anchor_field="source_anchor_pct",
        )
        science_worst = _top_offdiag_pairs(delta_rows, domain="science", best=False, limit=1)
        text = (
            "- **Dense transfer 是否说明 science 的 basis 也能均匀复用？** 不能。`science` 的 gap 随 distance 明显放大："
            f"all off-diagonal mean gap 为 {_fmt_signed_pct_points(_find_summary(summary_rows, summary_type='offdiag_all_mean', domain='science')['delta_fb_minus_ts'])}，"
            f"far-gap (`50–90`) mean gap 为 {_fmt_signed_pct_points(science_far['delta_fb_minus_ts'])}。"
        )
        if science_best:
            text += f" 最可迁移的 source anchor 是 `{science_best[0][0]}%`。"
        if science_worst:
            row = science_worst[0]
            text += f" 单个最差 pair 是 `{int(row['source_anchor_pct'])}->{int(row['target_anchor_pct'])}`，gap={_fmt_signed_pct_points(row['delta_fb_minus_ts'])}。"
        lines.append(text)

    math_timing = dense_timing.get("math")
    science_timing = dense_timing.get("science")
    if math_timing is not None and science_timing is not None:
        lines.append(
            "- **与 `16_DENSE_ANCHOR_EARLYSTOP.md` 是否一致？** 是。`math` 在 dense timing 里 `10%` 就达到 95%-of-final，约 `50%` 进入 plateau；这与它在 dense cross-anchor transfer 中的弱距离衰减一致。`science` 在 timing 里 `20%` 达到 95%-of-final、约 `40%` 进入 plateau，但 dense transfer 仍显示 early→late basis reuse 不均匀；这更支持“early coarse signal + late refinement”，而不是“only-at-completion onset”。"
        )

    lines.extend([
        "",
        "## 9. Artifacts",
        "",
        f"- Root matrix CSV: `{artifacts['matrix_csv']}`",
        f"- Root delta CSV: `{artifacts['delta_csv']}`",
        f"- Root summary CSV: `{artifacts['summary_csv']}`",
        f"- Paper-package matrix CSV: `{artifacts['paper_matrix_csv']}`",
        f"- Paper-package delta CSV: `{artifacts['paper_delta_csv']}`",
        f"- Paper-package summary CSV: `{artifacts['paper_summary_csv']}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run dense 10-anchor cross-anchor frozen-basis transfer suite")
    ap.add_argument("--bundle-path", default=DEFAULT_BUNDLE_PATH)
    ap.add_argument("--cache-root", default="MUI_HUB/cache")
    ap.add_argument("--domains", default="math,science")
    ap.add_argument("--source-anchors", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--target-anchors", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--feature-workers", type=int, default=default_feature_workers())
    ap.add_argument("--suite-workers", type=int, default=default_suite_workers())
    ap.add_argument("--feature-chunk-problems", type=int, default=DEFAULT_FEATURE_CHUNK_PROBLEMS)
    ap.add_argument("--feature-cache-dir", default=DEFAULT_FEATURE_CACHE_DIR)
    ap.add_argument("--refresh-feature-cache", action="store_true")
    ap.add_argument("--max-problems-per-cache", type=int, default=0, help="0 means all problems")
    ap.add_argument("--out-matrix", default=DEFAULT_OUT_MATRIX)
    ap.add_argument("--out-deltas", default=DEFAULT_OUT_DELTAS)
    ap.add_argument("--out-summary", default=DEFAULT_OUT_SUMMARY)
    ap.add_argument("--out-note", default=DEFAULT_OUT_NOTE)
    ap.add_argument("--paper-out-matrix", default=DEFAULT_PAPER_OUT_MATRIX)
    ap.add_argument("--paper-out-deltas", default=DEFAULT_PAPER_OUT_DELTAS)
    ap.add_argument("--paper-out-summary", default=DEFAULT_PAPER_OUT_SUMMARY)
    args = ap.parse_args()

    bundle_path = _resolve_path(str(args.bundle_path))
    cache_root = str(_resolve_path(str(args.cache_root)))
    feature_cache_dir = (
        None
        if str(args.feature_cache_dir).strip().lower() in {"", "none", "off"}
        else _resolve_path(str(args.feature_cache_dir))
    )
    source_anchors = _parse_anchor_pcts(str(args.source_anchors))
    target_anchors = _parse_anchor_pcts(str(args.target_anchors))
    domains = tuple(
        domain.strip()
        for domain in str(args.domains).split(",")
        if domain.strip()
    )
    for domain in domains:
        if domain not in DOMAIN_ORDER:
            raise ValueError(f"Unsupported domain {domain!r}; expected subset of {DOMAIN_ORDER}")

    bundle = load_earlystop_svd_bundle(bundle_path)
    required_features: set[str] = set()
    for domain in domains:
        for route in bundle["domains"][domain]["routes"]:
            required_features.update(str(name) for name in route.get("feature_names", []))

    max_problems_per_cache = None if int(args.max_problems_per_cache) <= 0 else int(args.max_problems_per_cache)
    feature_store, feature_cache_path, feature_cache_status = _load_or_build_noncoding_feature_store(
        cache_root=cache_root,
        positions=tuple(float(v) for v in EXTRACTION_POSITIONS),
        required_feature_names=required_features,
        max_problems_per_cache=max_problems_per_cache,
        feature_workers=int(args.feature_workers),
        chunk_problems=int(args.feature_chunk_problems),
        feature_cache_dir=feature_cache_dir,
        refresh_feature_cache=bool(args.refresh_feature_cache),
        included_domains=tuple(domains),
    )

    print(f"Bundle           : {bundle_path}", flush=True)
    print(f"Cache root       : {cache_root}", flush=True)
    print(f"Domains          : {domains}", flush=True)
    print(f"Source anchors   : {source_anchors}", flush=True)
    print(f"Target anchors   : {target_anchors}", flush=True)
    print(f"Feature cache    : {feature_cache_status} | {feature_cache_path}", flush=True)

    route_by_domain_anchor: dict[tuple[str, int], dict[str, Any]] = {}
    for domain in domains:
        routes = list(bundle["domains"][domain]["routes"])
        route_anchor_map = {
            int(round(float(route["training_position"]) * 100.0)): route
            for route in routes
        }
        missing = sorted((set(source_anchors) | set(target_anchors)) - set(route_anchor_map))
        if missing:
            raise ValueError(
                f"Bundle {bundle_path} missing anchors for domain {domain}: {missing}; "
                f"available={sorted(route_anchor_map)}"
            )
        for anchor_pct, route in route_anchor_map.items():
            route_by_domain_anchor[(domain, int(anchor_pct))] = route

    data_cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for domain in domains:
        for target_anchor_pct in target_anchors:
            required_route_keys: dict[str, dict[str, Any]] = {}
            target_route = route_by_domain_anchor[(domain, int(target_anchor_pct))]
            required_route_keys[_route_feature_key(target_route)] = target_route
            for source_anchor_pct in source_anchors:
                source_route = route_by_domain_anchor[(domain, int(source_anchor_pct))]
                required_route_keys[_route_feature_key(source_route)] = source_route

            reference_labels: Optional[np.ndarray] = None
            reference_groups: Optional[np.ndarray] = None
            for feature_key, route in required_route_keys.items():
                x_rep, y, groups = _collect_task_representation(
                    feature_store=feature_store,
                    domain_filter=domain,
                    target_anchor_pct=int(target_anchor_pct),
                    feature_indices=[int(v) for v in route["feature_indices"]],
                    representation=str(route.get("representation", "raw+rank")),
                )
                data_cache[(domain, int(target_anchor_pct), feature_key)] = (x_rep, y, groups)
                if reference_labels is None:
                    reference_labels = y
                    reference_groups = groups
                else:
                    if not np.array_equal(reference_labels, y):
                        raise RuntimeError(
                            f"Label mismatch for {domain}@{target_anchor_pct}% across feature specs"
                        )
                    if not np.array_equal(reference_groups, groups):
                        raise RuntimeError(
                            f"Group mismatch for {domain}@{target_anchor_pct}% across feature specs"
                        )

    matrix_rows: list[dict[str, Any]] = []

    def _run_cell(domain: str, source_anchor_pct: int, target_anchor_pct: int) -> list[dict[str, Any]]:
        source_route = route_by_domain_anchor[(domain, int(source_anchor_pct))]
        target_route = route_by_domain_anchor[(domain, int(target_anchor_pct))]
        source_key = _route_feature_key(source_route)
        target_key = _route_feature_key(target_route)
        x_frozen, y_frozen, groups_frozen = data_cache[(domain, int(target_anchor_pct), source_key)]
        x_target, y_target, groups_target = data_cache[(domain, int(target_anchor_pct), target_key)]
        if not np.array_equal(y_frozen, y_target):
            raise RuntimeError(f"Label mismatch in cell {domain} {source_anchor_pct}->{target_anchor_pct}")
        if not np.array_equal(groups_frozen, groups_target):
            raise RuntimeError(
                f"Group mismatch in cell {domain} {source_anchor_pct}->{target_anchor_pct}"
            )

        print(
            f"[cell] domain={domain:<7s} source={source_anchor_pct:>3d} target={target_anchor_pct:>3d} "
            f"samples={x_target.shape[0]} groups={len(np.unique(groups_target))}",
            flush=True,
        )
        comparison = _cv_compare_metrics(
            x_frozen=x_frozen,
            x_target=x_target,
            y=y_target,
            groups=groups_target,
            frozen_scaler=source_route["model"]["scaler"],
            frozen_svd=source_route["model"]["svd"],
            frozen_whiten=bool(source_route["model"].get("whiten", False)),
            target_rank=int(target_route.get("rank", source_route.get("rank", 16))),
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
        )

        n_samples = int(x_target.shape[0])
        n_groups = int(len(np.unique(groups_target))) if n_samples > 0 else 0
        rows: list[dict[str, Any]] = []
        for condition in CONDITION_ORDER:
            for metric_name in METRIC_ORDER:
                stats = comparison[condition][metric_name]
                rows.append(
                    {
                        "task": "earlystop",
                        "domain": str(domain),
                        "source_anchor_pct": int(source_anchor_pct),
                        "target_anchor_pct": int(target_anchor_pct),
                        "condition": str(condition),
                        "metric": str(metric_name),
                        "mean": _format_float(stats["mean"]),
                        "std": _format_float(stats["std"]),
                        "n_folds": int(stats["n_folds"]),
                        "n_samples": int(n_samples),
                        "n_groups": int(n_groups),
                        "source_bundle": bundle_path.stem,
                        "source_training_position_pct": int(
                            round(float(source_route["training_position"]) * 100.0)
                        ),
                        "target_training_position_pct": int(
                            round(float(target_route["training_position"]) * 100.0)
                        ),
                        "source_route_auroc": _format_float(source_route.get("cv_auroc", float("nan"))),
                        "target_route_auroc": _format_float(target_route.get("cv_auroc", float("nan"))),
                        "source_rank": int(source_route.get("rank", 0)),
                        "target_rank": int(target_route.get("rank", 0)),
                        "source_representation": str(source_route.get("representation", "")),
                        "target_representation": str(target_route.get("representation", "")),
                    }
                )
        return rows

    tasks = [
        (domain, int(source_anchor_pct), int(target_anchor_pct))
        for domain in domains
        for source_anchor_pct in source_anchors
        for target_anchor_pct in target_anchors
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(args.suite_workers))) as executor:
        future_map = {
            executor.submit(_run_cell, domain, source_anchor_pct, target_anchor_pct): (
                domain,
                source_anchor_pct,
                target_anchor_pct,
            )
            for domain, source_anchor_pct, target_anchor_pct in tasks
        }
        for future in as_completed(future_map):
            matrix_rows.extend(future.result())

    condition_order = {name: idx for idx, name in enumerate(CONDITION_ORDER)}
    metric_order = {name: idx for idx, name in enumerate(METRIC_ORDER)}
    domain_order = {name: idx for idx, name in enumerate(DOMAIN_ORDER)}
    matrix_rows = sorted(
        matrix_rows,
        key=lambda row: (
            domain_order.get(str(row["domain"]), 999),
            int(row["source_anchor_pct"]),
            int(row["target_anchor_pct"]),
            condition_order.get(str(row["condition"]), 999),
            metric_order.get(str(row["metric"]), 999),
        ),
    )

    pivot: dict[tuple[str, int, int, str], dict[str, float]] = defaultdict(dict)
    for row in matrix_rows:
        key = (
            str(row["domain"]),
            int(row["source_anchor_pct"]),
            int(row["target_anchor_pct"]),
            str(row["metric"]),
        )
        pivot[key][str(row["condition"])] = _finite_float(row["mean"])

    delta_rows: list[dict[str, Any]] = []
    for (domain, source_anchor_pct, target_anchor_pct, metric_name), cond_values in sorted(
        pivot.items(),
        key=lambda item: (
            domain_order.get(item[0][0], 999),
            item[0][1],
            item[0][2],
            metric_order.get(item[0][3], 999),
        ),
    ):
        frozen_value = cond_values.get("frozen_basis", float("nan"))
        task_value = cond_values.get("task_specific", float("nan"))
        nosvd_value = cond_values.get("no_svd", float("nan"))
        delta_task = (
            frozen_value - task_value
            if np.isfinite(frozen_value) and np.isfinite(task_value)
            else float("nan")
        )
        delta_nosvd = (
            frozen_value - nosvd_value
            if np.isfinite(frozen_value) and np.isfinite(nosvd_value)
            else float("nan")
        )
        delta_rows.append(
            {
                "task": "earlystop",
                "domain": str(domain),
                "source_anchor_pct": int(source_anchor_pct),
                "target_anchor_pct": int(target_anchor_pct),
                "metric": str(metric_name),
                "frozen_basis": _format_float(frozen_value),
                "task_specific": _format_float(task_value),
                "no_svd": _format_float(nosvd_value),
                "delta_fb_minus_ts": _format_float(delta_task),
                "delta_fb_minus_nosvd": _format_float(delta_nosvd),
                "transfer_verdict": _verdict(delta_task),
            }
        )

    summary_rows = build_summary_rows(delta_rows, anchors=source_anchors, domains=domains)
    artifacts = {
        "matrix_csv": str(Path(args.out_matrix)),
        "delta_csv": str(Path(args.out_deltas)),
        "summary_csv": str(Path(args.out_summary)),
        "paper_matrix_csv": str(Path(args.paper_out_matrix)),
        "paper_delta_csv": str(Path(args.paper_out_deltas)),
        "paper_summary_csv": str(Path(args.paper_out_summary)),
        "paper_note": str(Path(args.out_note)),
    }
    note_text = build_note_markdown(
        bundle_path=bundle_path,
        anchors=source_anchors,
        domains=tuple(domains),
        delta_rows=delta_rows,
        summary_rows=summary_rows,
        artifacts=artifacts,
    )

    out_matrix = _resolve_path(str(args.out_matrix))
    out_deltas = _resolve_path(str(args.out_deltas))
    out_summary = _resolve_path(str(args.out_summary))
    out_note = _resolve_path(str(args.out_note))
    paper_out_matrix = _resolve_path(str(args.paper_out_matrix))
    paper_out_deltas = _resolve_path(str(args.paper_out_deltas))
    paper_out_summary = _resolve_path(str(args.paper_out_summary))

    _write_csv(out_matrix, matrix_rows)
    _write_csv(out_deltas, delta_rows)
    _write_csv(out_summary, summary_rows)
    _write_csv(paper_out_matrix, matrix_rows)
    _write_csv(paper_out_deltas, delta_rows)
    _write_csv(paper_out_summary, summary_rows)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text(note_text, encoding="utf-8")

    print(f"Wrote matrix  : {out_matrix}", flush=True)
    print(f"Wrote deltas  : {out_deltas}", flush=True)
    print(f"Wrote summary : {out_summary}", flush=True)
    print(f"Wrote note    : {out_note}", flush=True)
    print(f"Wrote paper matrix  : {paper_out_matrix}", flush=True)
    print(f"Wrote paper deltas  : {paper_out_deltas}", flush=True)
    print(f"Wrote paper summary : {paper_out_summary}", flush=True)


if __name__ == "__main__":
    main()
