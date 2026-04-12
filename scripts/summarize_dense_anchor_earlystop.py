#!/usr/bin/env python3
"""Rewrite dense-anchor EarlyStop note, main table, and main figure from saved summary JSON."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SVDomain import run_dense_anchor_earlystop as dense_mod


B0_LINE_ID = "canonical_legacy"
B1_LINE_ID = "canonical_plus_neuron_meta_min"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pos_map(domain_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(round(float(row["position"]) * 100.0)): row
        for row in domain_payload["aggregate"]["by_position"]
    }


def _takeaway(domain_name: str, b0_payload: dict[str, Any], b1_payload: dict[str, Any]) -> str:
    b0_fixed = b0_payload["earliest_anchor_fixed_threshold"]
    b0_plateau = b0_payload["plateau_anchor_pct"]
    b1_auc = float(b1_payload["aggregate"]["auc_of_auroc"])
    b0_auc = float(b0_payload["aggregate"]["auc_of_auroc"])
    delta_auc = b1_auc - b0_auc
    if str(domain_name) == "math":
        return (
            f"very early strong signal; B0 reaches 95%-of-final at 10% and plateaus by {b0_plateau}%; "
            f"neuron add-on is not helpful overall ({delta_auc:+.003f} AUC)"
        )
    if str(domain_name) == "science":
        return (
            f"usable signal already appears by {b0_fixed}%, but the curve keeps improving into later anchors; "
            f"neuron add-on helps modestly overall ({delta_auc:+.003f} AUC)"
        )
    return (
        "weak curve with no fixed-threshold onset; "
        f"neuron add-on helps mainly late anchors ({delta_auc:+.003f} AUC overall)"
    )


def build_main_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    b0_domains = summary["lines"][B0_LINE_ID]["domains"]
    b1_domains = summary["lines"][B1_LINE_ID]["domains"]
    for domain_name in dense_mod.DOMAIN_ORDER:
        b0 = b0_domains[domain_name]
        b1 = b1_domains[domain_name]
        b0_pos = _pos_map(b0)
        b1_pos = _pos_map(b1)
        b0_auroc_10 = float(b0_pos[10]["auroc"])
        b1_auroc_10 = float(b1_pos[10]["auroc"])
        b0_auroc_100 = float(b0["aggregate"]["auroc@100%"])
        b1_auroc_100 = float(b1["aggregate"]["auroc@100%"])
        rows.append({
            "domain": str(domain_name),
            "b0_auc_of_auroc": float(b0["aggregate"]["auc_of_auroc"]),
            "b1_auc_of_auroc": float(b1["aggregate"]["auc_of_auroc"]),
            "delta_b1_minus_b0_auc_of_auroc": float(b1["aggregate"]["auc_of_auroc"] - b0["aggregate"]["auc_of_auroc"]),
            "b0_auroc_10": float(b0_auroc_10),
            "b0_auroc_100": float(b0_auroc_100),
            "b0_delta_10_to_100_auroc": float(b0_auroc_100 - b0_auroc_10),
            "b1_auroc_10": float(b1_auroc_10),
            "b1_auroc_100": float(b1_auroc_100),
            "b1_delta_10_to_100_auroc": float(b1_auroc_100 - b1_auroc_10),
            "b0_earliest_anchor_fixed_threshold": "" if b0["earliest_anchor_fixed_threshold"] is None else int(b0["earliest_anchor_fixed_threshold"]),
            "b1_earliest_anchor_fixed_threshold": "" if b1["earliest_anchor_fixed_threshold"] is None else int(b1["earliest_anchor_fixed_threshold"]),
            "b0_earliest_anchor_95pct_final_auroc": "" if b0["earliest_anchor_95pct_final_auroc"] is None else int(b0["earliest_anchor_95pct_final_auroc"]),
            "b1_earliest_anchor_95pct_final_auroc": "" if b1["earliest_anchor_95pct_final_auroc"] is None else int(b1["earliest_anchor_95pct_final_auroc"]),
            "b0_plateau_anchor_pct": "" if b0["plateau_anchor_pct"] is None else int(b0["plateau_anchor_pct"]),
            "b1_plateau_anchor_pct": "" if b1["plateau_anchor_pct"] is None else int(b1["plateau_anchor_pct"]),
            "takeaway": _takeaway(domain_name, b0, b1),
        })
    return rows


def _main_table_markdown(main_rows: list[dict[str, Any]]) -> list[str]:
    def _fmt_anchor(value: Any, *, suffix: str = "%") -> str:
        if value in {"", None}:
            return "—"
        return f"{value}{suffix}"

    lines = [
        "| Domain | B0 AUC-AUROC | B1-B0 ΔAUC | B0 onset95 | B0 fixed onset | B0 plateau | Reading |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in main_rows:
        onset95 = _fmt_anchor(row["b0_earliest_anchor_95pct_final_auroc"])
        if str(row["domain"]) == "coding":
            onset95 = f"{onset95}*"
        lines.append(
            f"| `{row['domain']}` | {float(row['b0_auc_of_auroc']):.3f} | {float(row['delta_b1_minus_b0_auc_of_auroc']):+.3f} | "
            f"{onset95} | {_fmt_anchor(row['b0_earliest_anchor_fixed_threshold'])} | "
            f"{_fmt_anchor(row['b0_plateau_anchor_pct'])} | {row['takeaway']} |"
        )
    return lines


def build_note_markdown(summary: dict[str, Any], main_rows: list[dict[str, Any]], compare_rows: list[dict[str, Any]]) -> str:
    compare_lookup = {
        (str(row["domain"]), str(row["anchor_band"])): row
        for row in compare_rows
        if row["row_type"] == "band_summary"
    }
    lines = [
        "# 16. Dense-Anchor EarlyStop",
        "",
        "这份 note 的目标，是把 dense-anchor EarlyStop 结果压缩成一页内可直接转写到论文正文的主结论、主表与主图口径。",
        "",
        "## 1. Main paper assets",
        "",
        "- 主表：`results/tables/dense_anchor_main_table.csv`",
        "- 主图：`results/figures/dense_anchor_earlystop_main.png`",
        "- 完整曲线表：`results/tables/dense_anchor_earlystop.csv`",
        "- onset / plateau 表：`results/tables/onset_of_signal.csv`、`results/tables/plateau_of_signal.csv`",
        "- neuron-vs-legacy 对照：`results/tables/dense_anchor_neuron_vs_legacy.csv`",
        "",
        "## 2. Main table",
        "",
        *_main_table_markdown(main_rows),
        "",
        "* `coding` 这类弱域上，`95% of final AUROC` 不是最稳健的 onset 指标；fixed-threshold onset 与 plateau 更值得优先引用。",
        "",
        "## 3. Main claims",
        "",
        f"- {dense_mod._explanation_math(summary)}",
        f"- {dense_mod._explanation_science(summary)}",
        f"- {dense_mod._explanation_coding(summary)}",
        "",
        "## 4. Neuron add-on readout",
        "",
    ]
    for domain_name in dense_mod.DOMAIN_ORDER:
        early = compare_lookup[(domain_name, "early_10_40")]
        late = compare_lookup[(domain_name, "late_70_100")]
        overall = compare_lookup[(domain_name, "all_10_100")]
        lines.append(
            f"- `{domain_name}`: B1−B0 的 mean ΔAUROC 为 early `{float(early['delta_auroc']):+.3f}` / "
            f"late `{float(late['delta_auroc']):+.3f}` / overall `{float(overall['delta_auroc']):+.3f}`。"
        )
    lines.extend([
        "",
        "## 5. Figure caption candidate",
        "",
        "> Dense-anchor EarlyStop curves show that math is already strongly predictable at very early prefixes, science has usable early signal but continues to improve through later anchors, and coding remains weak under the current feature family; neuron features help mainly on science/coding late anchors rather than shifting the earliest onset.",
        "",
        "## 6. Table sentence candidate",
        "",
        "> The coarse 10/40/70/100 anchors are directionally correct, but denser anchors reveal that math saturates very early, science exhibits early coarse signal followed by late refinement, and coding never develops a comparably stable early-stop signal.",
        "",
    ])
    return "\n".join(lines)


def _plot_domain_panel(ax: Any, *, summary: dict[str, Any], domain_name: str) -> None:
    colors = {
        B0_LINE_ID: "#1f77b4",
        B1_LINE_ID: "#d62728",
    }
    labels = {
        B0_LINE_ID: "B0 legacy canonical",
        B1_LINE_ID: "B1 + neuron_meta_min",
    }
    linestyles = {
        B0_LINE_ID: "-",
        B1_LINE_ID: "--",
    }
    markers = {
        B0_LINE_ID: "o",
        B1_LINE_ID: "s",
    }
    all_values: list[float] = []
    for line_id in (B0_LINE_ID, B1_LINE_ID):
        payload = summary["lines"][line_id]["domains"][domain_name]
        xs = []
        ys = []
        for row in payload["aggregate"]["by_position"]:
            xs.append(int(round(float(row["position"]) * 100.0)))
            ys.append(float(row["auroc"]))
        all_values.extend(ys)
        ax.plot(
            xs,
            ys,
            color=colors[line_id],
            linestyle=linestyles[line_id],
            marker=markers[line_id],
            linewidth=2.0,
            markersize=4.5,
            label=labels[line_id],
        )

    b0_payload = summary["lines"][B0_LINE_ID]["domains"][domain_name]
    threshold = float(b0_payload["threshold_value"])
    onset_95 = b0_payload["earliest_anchor_95pct_final_auroc"]
    plateau = b0_payload["plateau_anchor_pct"]
    ax.axhline(threshold, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    if onset_95 is not None:
        ax.axvline(float(onset_95), color=colors[B0_LINE_ID], linestyle="--", linewidth=1.0, alpha=0.2)
    if plateau is not None:
        ax.axvline(float(plateau), color=colors[B0_LINE_ID], linestyle=":", linewidth=1.0, alpha=0.2)

    value_min = min(all_values)
    value_max = max(all_values)
    pad = max(0.01, 0.15 * (value_max - value_min))
    ax.set_ylim(max(0.0, value_min - pad), min(1.0, value_max + pad))
    ax.set_xticks(dense_mod.ANCHOR_PCTS)
    ax.set_xlabel("Anchor (%)")
    ax.set_title(str(domain_name).capitalize())
    ax.grid(True, alpha=0.2)
    ax.text(
        0.02,
        0.03,
        f"B0 onset95={onset_95 if onset_95 is not None else '—'}%\nB0 plateau={plateau if plateau is not None else '—'}%",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.8},
    )


def write_main_figure(summary: dict[str, Any], out_png: Path, out_pdf: Path | None = None) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), sharex=True, sharey=False)
    for ax, domain_name in zip(axes, dense_mod.DOMAIN_ORDER):
        _plot_domain_panel(ax, summary=summary, domain_name=domain_name)
    axes[0].set_ylabel("AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Dense-anchor EarlyStop: AUROC across 10/20/.../100 anchors", y=1.12, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize dense-anchor EarlyStop outputs from saved summary JSON")
    ap.add_argument("--summary-json", default="results/scans/dense_anchor_earlystop/dense_anchor_earlystop_summary.json")
    ap.add_argument("--out-note", default="docs/16_DENSE_ANCHOR_EARLYSTOP.md")
    ap.add_argument("--out-main-table", default="results/tables/dense_anchor_main_table.csv")
    ap.add_argument("--out-figure-png", default="results/figures/dense_anchor_earlystop_main.png")
    ap.add_argument("--out-figure-pdf", default="results/figures/dense_anchor_earlystop_main.pdf")
    args = ap.parse_args()

    summary_path = REPO_ROOT / str(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    compare_rows = dense_mod._build_compare_rows(summary)
    main_rows = build_main_table_rows(summary)
    note = build_note_markdown(summary=summary, main_rows=main_rows, compare_rows=compare_rows)

    out_note = REPO_ROOT / str(args.out_note)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text(note, encoding="utf-8")

    _write_csv(REPO_ROOT / str(args.out_main_table), main_rows)
    write_main_figure(
        summary=summary,
        out_png=REPO_ROOT / str(args.out_figure_png),
        out_pdf=REPO_ROOT / str(args.out_figure_pdf),
    )
    print(f"[done] rewrote dense-anchor note/table/figure from {summary_path}", flush=True)


if __name__ == "__main__":
    main()
