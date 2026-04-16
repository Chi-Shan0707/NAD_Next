#!/usr/bin/env python3
"""CPU-first linear SSL basis study for EarlyStop / SVDomain.

Focus:
- strict canonical 85/15 grouped holdout
- same-run multiview SSL by default
- shared low-rank basis vs task-specific basis vs no-basis LR
- math/science first; coding only cheap confirmation if requested
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nad.ops.earlystop_svd import _rank_transform_matrix
from nad.ops.earlystop_ssl import (
    AnchorTable,
    BasisBundle,
    build_anchor_tables,
    fit_denoise_svd_basis,
    fit_linear_head,
    fit_regcca_basis,
    fit_rrr_basis,
    linear_head_scores,
    sample_problem_keys,
    shuffle_pairs_within_groups,
    subset_store_by_problem_keys,
    transform_pair_bundle,
    transform_shared_pair_single_input,
    transform_single_bundle,
)
from scripts.run_earlystop_prefix10_svd_round1 import (
    ANCHOR_POSITIONS,
    EARLY_STOP_POSITIONS,
    EXTRACTION_POSITION_INDEX,
    OFFICIAL_SLOT_TO_ANCHOR,
    SEARCH_C_VALUES,
    evaluate_method_from_feature_store,
)
from SVDomain.train_es_svd_ms_rr_r1 import (
    FIXED_FEATURE_INDICES,
    FIXED_FEATURE_NAMES,
    _build_holdout_problem_map,
    _split_feature_store,
)


TOKEN_FEATURE_NAMES = [
    "tok_conf_prefix",
    "tok_conf_recency",
    "tok_gini_prefix",
    "tok_gini_tail",
    "tok_gini_slope",
    "tok_neg_entropy_prefix",
    "tok_neg_entropy_recency",
    "tok_selfcert_prefix",
    "tok_selfcert_recency",
    "tok_logprob_prefix",
    "tok_logprob_recency",
    "has_tok_conf",
    "has_tok_gini",
    "has_tok_neg_entropy",
    "has_tok_selfcert",
    "has_tok_logprob",
]
TOKEN_FIXED_INDICES = [FIXED_FEATURE_NAMES.index(name) for name in TOKEN_FEATURE_NAMES]
ANCHOR_POS_INDICES = [EXTRACTION_POSITION_INDEX[float(p)] for p in ANCHOR_POSITIONS]
POS_TO_ANCHOR_IDX = {
    pos_idx: list(ANCHOR_POSITIONS).index(float(OFFICIAL_SLOT_TO_ANCHOR[float(pos)]))
    for pos_idx, pos in enumerate(EARLY_STOP_POSITIONS)
}
PROTOCOL_TEXT = "85/15 grouped holdout by dataset+problem_id across cache+cache_train; split_seed=42"


@dataclass(frozen=True)
class BasisConfig:
    name: str
    view: str
    method: str
    same_space: bool
    supports_task_specific: bool
    supports_same_problem: bool


BASIS_CONFIGS = [
    BasisConfig(
        name="raw_rank_cca",
        view="raw_rank",
        method="cca",
        same_space=False,
        supports_task_specific=True,
        supports_same_problem=True,
    ),
    BasisConfig(
        name="raw_rank_rrr",
        view="raw_rank",
        method="rrr",
        same_space=False,
        supports_task_specific=True,
        supports_same_problem=False,
    ),
    BasisConfig(
        name="tokenpair_cca",
        view="token_pair",
        method="cca",
        same_space=False,
        supports_task_specific=True,
        supports_same_problem=False,
    ),
    BasisConfig(
        name="tokenpair_rrr",
        view="token_pair",
        method="rrr",
        same_space=False,
        supports_task_specific=True,
        supports_same_problem=False,
    ),
    BasisConfig(
        name="adjacent_cca",
        view="adjacent_full",
        method="cca",
        same_space=True,
        supports_task_specific=False,
        supports_same_problem=True,
    ),
    BasisConfig(
        name="adjacent_rrr",
        view="adjacent_full",
        method="rrr",
        same_space=True,
        supports_task_specific=False,
        supports_same_problem=False,
    ),
    BasisConfig(
        name="denoise_full",
        view="denoise_full",
        method="denoise",
        same_space=True,
        supports_task_specific=True,
        supports_same_problem=False,
    ),
]


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_prebuilt_stores(cache_dir: Path) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    pkl_files = sorted(cache_dir.glob("*.pkl"))
    priority = [p for p in pkl_files if "_all_" in p.name and "noncoding" not in p.name]
    fallback = [p for p in pkl_files if p not in priority]

    for pkl_path in priority + fallback:
        try:
            with pkl_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            print(f"[prebuilt] skip {pkl_path.name}: {exc}")
            continue
        if not isinstance(payload, dict) or "feature_store" not in payload:
            continue
        source_name = str(payload.get("source_name", pkl_path.stem))
        if source_name in seen_sources:
            continue
        seen_sources.add(source_name)
        store = list(payload["feature_store"])
        samples = sum(int(item.get("samples", 0)) for item in store)
        print(f"[prebuilt] loaded source={source_name} payloads={len(store)} samples={samples}")
        combined.extend(store)
    return combined


def _fit_basis(
    config: BasisConfig,
    *,
    tables: dict[int, AnchorTable],
    rank: int,
    seed: int,
    cca_reg: float,
    mask_rate: float,
    basis_scope: str,
    positive_mode: str = "same_run",
) -> dict[int, BasisBundle] | BasisBundle:
    if config.view == "denoise_full":
        if basis_scope == "shared":
            x = np.vstack([tables[a].x_full for a in sorted(tables.keys()) if tables[a].x_full.shape[0] > 0])
            return fit_denoise_svd_basis(x, n_components=rank, mask_rate=mask_rate, seed=seed)
        out: dict[int, BasisBundle] = {}
        for anchor_idx, table in tables.items():
            if table.x_full.shape[0] <= 0:
                continue
            out[anchor_idx] = fit_denoise_svd_basis(
                table.x_full,
                n_components=rank,
                mask_rate=mask_rate,
                seed=seed + anchor_idx,
            )
        return out

    def fit_pair(x_left: np.ndarray, x_right: np.ndarray, *, same_space: bool, local_seed: int, groups: np.ndarray) -> BasisBundle:
        if positive_mode == "same_problem":
            x_left, x_right = shuffle_pairs_within_groups(
                x_left,
                x_right,
                groups,
                seed=local_seed,
            )
        if x_left.shape[0] < 4:
            raise ValueError(f"{config.name}/{basis_scope}/{positive_mode} has too few rows")
        if config.method == "cca":
            return fit_regcca_basis(
                x_left,
                x_right,
                n_components=rank,
                reg=cca_reg,
                same_space=same_space,
            )
        if config.method == "rrr":
            return fit_rrr_basis(
                x_left,
                x_right,
                n_components=rank,
                same_space=same_space,
            )
        raise ValueError(f"Unsupported method: {config.method}")

    if config.view == "raw_rank":
        if basis_scope == "shared":
            x_left = np.vstack([tables[a].x_raw for a in sorted(tables.keys()) if tables[a].x_raw.shape[0] > 0])
            x_right = np.vstack([tables[a].x_rank for a in sorted(tables.keys()) if tables[a].x_rank.shape[0] > 0])
            groups = np.concatenate([tables[a].problem_keys for a in sorted(tables.keys()) if tables[a].x_raw.shape[0] > 0])
            return fit_pair(x_left, x_right, same_space=False, local_seed=seed, groups=groups)
        out: dict[int, BasisBundle] = {}
        for anchor_idx, table in tables.items():
            if table.x_raw.shape[0] <= 0:
                continue
            out[anchor_idx] = fit_pair(
                table.x_raw,
                table.x_rank,
                same_space=False,
                local_seed=seed + anchor_idx,
                groups=table.problem_keys,
            )
        return out

    if config.view == "token_pair":
        if basis_scope == "shared":
            x_left = np.vstack([tables[a].x_token for a in sorted(tables.keys()) if tables[a].x_token.shape[0] > 0])
            x_right = np.vstack([tables[a].x_full for a in sorted(tables.keys()) if tables[a].x_full.shape[0] > 0])
            groups = np.concatenate([tables[a].problem_keys for a in sorted(tables.keys()) if tables[a].x_token.shape[0] > 0])
            return fit_pair(x_left, x_right, same_space=False, local_seed=seed, groups=groups)
        out = {}
        for anchor_idx, table in tables.items():
            if table.x_token.shape[0] <= 0:
                continue
            out[anchor_idx] = fit_pair(
                table.x_token,
                table.x_full,
                same_space=False,
                local_seed=seed + anchor_idx,
                groups=table.problem_keys,
            )
        return out

    if config.view == "adjacent_full":
        x_left_parts: list[np.ndarray] = []
        x_right_parts: list[np.ndarray] = []
        group_parts: list[np.ndarray] = []
        for anchor_idx in range(len(ANCHOR_POSITIONS) - 1):
            left_table = tables[anchor_idx]
            right_table = tables[anchor_idx + 1]
            if left_table.x_full.shape[0] <= 0 or right_table.x_full.shape[0] <= 0:
                continue
            if left_table.x_full.shape[0] != right_table.x_full.shape[0]:
                raise ValueError(f"adjacent anchor row mismatch at {anchor_idx}")
            x_left_parts.append(left_table.x_full)
            x_right_parts.append(right_table.x_full)
            group_parts.append(left_table.problem_keys)
        if not x_left_parts:
            raise ValueError("adjacent_full has no data")
        return fit_pair(
            np.vstack(x_left_parts),
            np.vstack(x_right_parts),
            same_space=True,
            local_seed=seed,
            groups=np.concatenate(group_parts),
        )

    raise ValueError(f"Unsupported view: {config.view}")


def _table_features(
    table: AnchorTable,
    *,
    config: Optional[BasisConfig],
    bundle: BasisBundle,
) -> np.ndarray:
    if config is None:
        raise ValueError("config is required")
    if config.view == "raw_rank":
        return transform_pair_bundle(bundle, table.x_raw, table.x_rank)
    if config.view == "token_pair":
        return transform_pair_bundle(bundle, table.x_token, table.x_full)
    if config.view == "adjacent_full":
        return transform_shared_pair_single_input(bundle, table.x_full)
    if config.view == "denoise_full":
        return transform_single_bundle(bundle, table.x_full)
    raise ValueError(f"Unsupported view: {config.view}")


def _fit_heads_from_tables(
    tables: dict[int, AnchorTable],
    *,
    feature_fn: Callable[[int, AnchorTable], np.ndarray],
    seed: int,
) -> dict[int, Any]:
    heads: dict[int, Any] = {}
    for anchor_idx, table in tables.items():
        if table.y.shape[0] < 4 or np.unique(table.y).shape[0] < 2:
            continue
        x = feature_fn(anchor_idx, table)
        head = fit_linear_head(
            x,
            table.y,
            table.groups,
            c_values=tuple(float(v) for v in SEARCH_C_VALUES),
            random_state=seed,
        )
        if head is not None:
            heads[anchor_idx] = head
    return heads


def _make_score_fn(
    *,
    domain: str,
    config: Optional[BasisConfig],
    bundle_spec: BasisBundle | dict[int, BasisBundle] | None,
    heads: dict[int, Any],
    baseline_mode: str,
) -> Callable[[str, int, np.ndarray], np.ndarray]:
    token_idx_arr = np.asarray(TOKEN_FIXED_INDICES, dtype=np.int32)

    def _score(dom: str, pos_idx: int, x_raw_all: np.ndarray) -> np.ndarray:
        n_rows = int(x_raw_all.shape[0])
        if dom != domain:
            return np.zeros(n_rows, dtype=np.float64)
        anchor_idx = int(POS_TO_ANCHOR_IDX.get(pos_idx, 0))
        head = heads.get(anchor_idx)
        if head is None:
            return np.zeros(n_rows, dtype=np.float64)

        x_fixed = np.asarray(x_raw_all[:, FIXED_FEATURE_INDICES], dtype=np.float64)
        x_rank = _rank_transform_matrix(x_fixed)
        x_full = np.concatenate([x_fixed, x_rank], axis=1)

        if baseline_mode == "raw_only":
            return linear_head_scores(head, x_fixed)
        if baseline_mode == "rank_only":
            return linear_head_scores(head, x_rank)
        if baseline_mode == "token_only":
            x_token = np.concatenate([x_fixed[:, token_idx_arr], x_rank[:, token_idx_arr]], axis=1)
            return linear_head_scores(head, x_token)
        if baseline_mode == "raw+rank":
            return linear_head_scores(head, x_full)

        if config is None or bundle_spec is None:
            return np.zeros(n_rows, dtype=np.float64)

        bundle = bundle_spec if isinstance(bundle_spec, BasisBundle) else bundle_spec.get(anchor_idx)
        if bundle is None:
            return np.zeros(n_rows, dtype=np.float64)
        if config.view == "raw_rank":
            feats = transform_pair_bundle(bundle, x_fixed, x_rank)
        elif config.view == "token_pair":
            x_token = np.concatenate([x_fixed[:, token_idx_arr], x_rank[:, token_idx_arr]], axis=1)
            feats = transform_pair_bundle(bundle, x_token, x_full)
        elif config.view == "adjacent_full":
            feats = transform_shared_pair_single_input(bundle, x_full)
        elif config.view == "denoise_full":
            feats = transform_single_bundle(bundle, x_full)
        else:
            return np.zeros(n_rows, dtype=np.float64)
        return linear_head_scores(head, feats)

    return _score


def _aggregate_row(
    *,
    stage: str,
    domain: str,
    seed: int,
    condition: str,
    basis_scope: str,
    view_name: str,
    method: str,
    positive_mode: str,
    ssl_rank: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    def _safe_float(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except Exception:
            return float("nan")

    return {
        "stage": stage,
        "protocol": PROTOCOL_TEXT,
        "domain": domain,
        "seed": int(seed),
        "condition": condition,
        "basis_scope": basis_scope,
        "view_name": view_name,
        "method": method,
        "positive_mode": positive_mode,
        "ssl_rank": int(ssl_rank),
        "auc_of_auroc": _safe_float(metrics.get("auc_of_auroc", float("nan"))),
        "auc_of_selacc": _safe_float(metrics.get("auc_of_selacc", float("nan"))),
        "earliest_gt_0_6": _safe_float(metrics.get("earliest_gt_0.6", float("nan"))),
        "auroc_at_100": _safe_float(metrics.get("auroc@100%", float("nan"))),
        "stop_acc_at_100": _safe_float(metrics.get("stop_acc@100%", float("nan"))),
    }


def _group_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["stage"],
            row["protocol"],
            row["domain"],
            row["condition"],
            row["basis_scope"],
            row["view_name"],
            row["method"],
            row["positive_mode"],
            row["ssl_rank"],
        )
        buckets.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items()):
        auc_mean, auc_std = _group_mean_std([float(r["auc_of_auroc"]) for r in bucket])
        sel_mean, sel_std = _group_mean_std([float(r["auc_of_selacc"]) for r in bucket])
        au100_mean, au100_std = _group_mean_std([float(r["auroc_at_100"]) for r in bucket])
        stop_mean, stop_std = _group_mean_std([float(r["stop_acc_at_100"]) for r in bucket])
        earliest_mean, earliest_std = _group_mean_std([float(r["earliest_gt_0_6"]) for r in bucket])
        summary_rows.append(
            {
                "stage": key[0],
                "protocol": key[1],
                "domain": key[2],
                "condition": key[3],
                "basis_scope": key[4],
                "view_name": key[5],
                "method": key[6],
                "positive_mode": key[7],
                "ssl_rank": key[8],
                "n_seeds": len(bucket),
                "auc_of_auroc_mean": auc_mean,
                "auc_of_auroc_std": auc_std,
                "auc_of_selacc_mean": sel_mean,
                "auc_of_selacc_std": sel_std,
                "earliest_gt_0_6_mean": earliest_mean,
                "earliest_gt_0_6_std": earliest_std,
                "auroc_at_100_mean": au100_mean,
                "auroc_at_100_std": au100_std,
                "stop_acc_at_100_mean": stop_mean,
                "stop_acc_at_100_std": stop_std,
            }
        )
    return summary_rows


def _eval_method(
    *,
    method_name: str,
    holdout_store: list[dict[str, Any]],
    score_fn: Callable[[str, int, np.ndarray], np.ndarray],
    verbose: bool,
) -> dict[str, Any]:
    if verbose:
        return evaluate_method_from_feature_store(
            method_name=method_name,
            feature_store=holdout_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=score_fn,
        )
    capture = io.StringIO()
    with redirect_stdout(capture):
        return evaluate_method_from_feature_store(
            method_name=method_name,
            feature_store=holdout_store,
            position_values=tuple(float(v) for v in EARLY_STOP_POSITIONS),
            score_fn=score_fn,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate linear EarlyStop SSL bases")
    parser.add_argument("--prebuilt-cache-dir", default="results/cache/es_svd_ms_rr_r1")
    parser.add_argument("--holdout-split", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--domains", nargs="+", default=["math", "science"])
    parser.add_argument("--configs", nargs="+", default=[])
    parser.add_argument("--ssl-ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--cca-reg", type=float, default=0.10)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--verbose-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-summary-csv", default="results/tables/earlystop_ssl_summary.csv")
    parser.add_argument("--out-ablation-csv", default="results/tables/earlystop_ssl_ablation.csv")
    parser.add_argument("--out-detail-json", default="results/scans/earlystop_ssl/ssl_basis_detail.json")
    args = parser.parse_args()

    ssl_ranks = [int(v) for v in args.ssl_ranks]
    seeds = [int(v) for v in args.seeds]
    if args.smoke:
        ssl_ranks = ssl_ranks[:1]
        seeds = seeds[:1]

    config_filter = {str(v) for v in args.configs}
    active_configs = [cfg for cfg in BASIS_CONFIGS if not config_filter or cfg.name in config_filter]
    if not active_configs:
        raise SystemExit("No active configs selected")

    prebuilt_cache_dir = _resolve_path(args.prebuilt_cache_dir)
    feature_store = _load_prebuilt_stores(prebuilt_cache_dir)
    holdout_problem_map, split_summary = _build_holdout_problem_map(
        feature_store,
        holdout_split=float(args.holdout_split),
        split_seed=int(args.split_seed),
    )
    train_payloads, holdout_payloads, _ = _split_feature_store(
        feature_store,
        holdout_problem_map=holdout_problem_map,
    )

    domain_tables_train = {
        domain: build_anchor_tables(
            train_payloads,
            fixed_feature_indices=FIXED_FEATURE_INDICES,
            token_feature_indices=TOKEN_FIXED_INDICES,
            anchor_position_indices=ANCHOR_POS_INDICES,
            domain=domain,
        )
        for domain in args.domains
    }
    holdout_by_domain = {
        domain: [payload for payload in holdout_payloads if str(payload["domain"]) == domain]
        for domain in args.domains
    }

    ablation_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for domain in args.domains:
        tables = domain_tables_train[domain]
        holdout_store = holdout_by_domain.get(domain, [])
        if not holdout_store:
            print(f"[skip] domain={domain}: no holdout payloads")
            continue

        print(f"\n[domain] {domain}")

        baseline_feature_fns = {
            "raw_only": lambda _a, table: table.x_raw,
            "rank_only": lambda _a, table: table.x_rank,
            "token_only": lambda _a, table: table.x_token,
            "raw+rank": lambda _a, table: table.x_full,
        }
        for seed in seeds:
            for baseline_name, feature_fn in baseline_feature_fns.items():
                heads = _fit_heads_from_tables(tables, feature_fn=feature_fn, seed=seed)
                score_fn = _make_score_fn(
                    domain=domain,
                    config=None,
                    bundle_spec=None,
                    heads=heads,
                    baseline_mode=baseline_name,
                )
                eval_result = _eval_method(
                    method_name=f"{baseline_name}/{domain}",
                    holdout_store=holdout_store,
                    score_fn=score_fn,
                    verbose=bool(args.verbose_eval),
                )
                row = _aggregate_row(
                    stage="ssl_basis",
                    domain=domain,
                    seed=seed,
                    condition="baseline_lr",
                    basis_scope="none",
                    view_name=baseline_name,
                    method="lr",
                    positive_mode="n/a",
                    ssl_rank=0,
                    metrics=eval_result["aggregate"],
                )
                ablation_rows.append(row)
                detail_rows.append(
                    {
                        **row,
                        "method_name": eval_result["method_name"],
                        "by_cache": eval_result["by_cache"],
                    }
                )
                print(
                    f"[baseline] domain={domain} seed={seed} view={baseline_name} "
                    f"auc={row['auc_of_auroc']:.4f}"
                )

        for seed in seeds:
            for ssl_rank in ssl_ranks:
                for config in active_configs:
                    positive_modes = ["same_run"]
                    if config.supports_same_problem and ssl_rank == ssl_ranks[0]:
                        positive_modes.append("same_problem")

                    for positive_mode in positive_modes:
                        bundle_shared: BasisBundle | dict[int, BasisBundle] | None = None
                        try:
                            bundle_shared = _fit_basis(
                                config,
                                tables=tables,
                                rank=ssl_rank,
                                seed=seed,
                                cca_reg=float(args.cca_reg),
                                mask_rate=float(args.mask_rate),
                                basis_scope="shared",
                                positive_mode=positive_mode,
                            )
                        except Exception as exc:
                            print(
                                f"[warn] shared fit failed domain={domain} cfg={config.name} "
                                f"r={ssl_rank} pos={positive_mode}: {exc}"
                            )

                        if bundle_shared is not None:
                            heads = _fit_heads_from_tables(
                                tables,
                                feature_fn=lambda anchor_idx, table, bundle_spec=bundle_shared, cfg=config: _table_features(
                                    table,
                                    config=cfg,
                                    bundle=bundle_spec if isinstance(bundle_spec, BasisBundle) else bundle_spec[anchor_idx],
                                ),
                                seed=seed,
                            )
                            score_fn = _make_score_fn(
                                domain=domain,
                                config=config,
                                bundle_spec=bundle_shared,
                                heads=heads,
                                baseline_mode="ssl",
                            )
                            eval_result = _eval_method(
                                method_name=f"{config.name}_shared_r{ssl_rank}/{domain}",
                                holdout_store=holdout_store,
                                score_fn=score_fn,
                                verbose=bool(args.verbose_eval),
                            )
                            row = _aggregate_row(
                                stage="ssl_basis",
                                domain=domain,
                                seed=seed,
                                condition="ssl_basis",
                                basis_scope="shared",
                                view_name=config.view,
                                method=config.method,
                                positive_mode=positive_mode,
                                ssl_rank=ssl_rank,
                                metrics=eval_result["aggregate"],
                            )
                            ablation_rows.append(row)
                            detail_rows.append(
                                {
                                    **row,
                                    "method_name": eval_result["method_name"],
                                    "config_name": config.name,
                                    "by_cache": eval_result["by_cache"],
                                }
                            )
                            print(
                                f"[shared] domain={domain} seed={seed} cfg={config.name} "
                                f"r={ssl_rank} pos={positive_mode} auc={row['auc_of_auroc']:.4f}"
                            )

                        if not config.supports_task_specific or positive_mode != "same_run":
                            continue
                        try:
                            bundle_task = _fit_basis(
                                config,
                                tables=tables,
                                rank=ssl_rank,
                                seed=seed,
                                cca_reg=float(args.cca_reg),
                                mask_rate=float(args.mask_rate),
                                basis_scope="task_specific",
                                positive_mode="same_run",
                            )
                        except Exception as exc:
                            print(
                                f"[warn] task fit failed domain={domain} cfg={config.name} "
                                f"r={ssl_rank}: {exc}"
                            )
                            continue
                        if not isinstance(bundle_task, dict) or not bundle_task:
                            continue
                        heads = _fit_heads_from_tables(
                            tables,
                            feature_fn=lambda anchor_idx, table, bundle_spec=bundle_task, cfg=config: _table_features(
                                table,
                                config=cfg,
                                bundle=bundle_spec[anchor_idx],
                            ),
                            seed=seed,
                        )
                        score_fn = _make_score_fn(
                            domain=domain,
                            config=config,
                            bundle_spec=bundle_task,
                            heads=heads,
                            baseline_mode="ssl",
                        )
                        eval_result = _eval_method(
                            method_name=f"{config.name}_task_r{ssl_rank}/{domain}",
                            holdout_store=holdout_store,
                            score_fn=score_fn,
                            verbose=bool(args.verbose_eval),
                        )
                        row = _aggregate_row(
                            stage="ssl_basis",
                            domain=domain,
                            seed=seed,
                            condition="ssl_basis",
                            basis_scope="task_specific",
                            view_name=config.view,
                            method=config.method,
                            positive_mode="same_run",
                            ssl_rank=ssl_rank,
                            metrics=eval_result["aggregate"],
                        )
                        ablation_rows.append(row)
                        detail_rows.append(
                            {
                                **row,
                                "method_name": eval_result["method_name"],
                                "config_name": config.name,
                                "by_cache": eval_result["by_cache"],
                            }
                        )
                        print(
                            f"[task] domain={domain} seed={seed} cfg={config.name} "
                            f"r={ssl_rank} auc={row['auc_of_auroc']:.4f}"
                        )

    summary_rows = _summarise_rows(ablation_rows)
    out_summary_csv = _resolve_path(args.out_summary_csv)
    out_ablation_csv = _resolve_path(args.out_ablation_csv)
    out_detail_json = _resolve_path(args.out_detail_json)
    _write_csv(summary_rows, out_summary_csv)
    _write_csv(ablation_rows, out_ablation_csv)
    out_detail_json.parent.mkdir(parents=True, exist_ok=True)
    out_detail_json.write_text(
        json.dumps(
            {
                "protocol": {
                    "text": PROTOCOL_TEXT,
                    "prebuilt_cache_dir": str(prebuilt_cache_dir),
                    "holdout_split": float(args.holdout_split),
                    "split_seed": int(args.split_seed),
                    "split_summary": split_summary,
                },
                "rows": detail_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[out] summary -> {out_summary_csv}")
    print(f"[out] ablation -> {out_ablation_csv}")
    print(f"[out] detail -> {out_detail_json}")


if __name__ == "__main__":
    main()
