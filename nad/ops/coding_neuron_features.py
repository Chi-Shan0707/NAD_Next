from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nad.core.views.reader import Agg, CacheReader

DEFAULT_LAYER_STRIDE = 65536
DEFAULT_N_LAYERS = 36


def _sanitize_weight_vector(values: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D weight vector, got shape={arr.shape}")
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        if floor is not None:
            return np.maximum(arr, float(floor))
        return arr
    finite_vals = arr[finite_mask]
    if finite_vals.size <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    pos_fill = float(np.max(finite_vals))
    neg_fill = float(floor) if floor is not None else float(np.min(finite_vals))
    clean = np.nan_to_num(arr, nan=0.0, posinf=pos_fill, neginf=neg_fill)
    if floor is not None:
        clean = np.maximum(clean, float(floor))
    return np.asarray(clean, dtype=np.float32)


def _sanitize_feature_matrix(arr: np.ndarray, *, floor: float | None = 0.0) -> np.ndarray:
    arr_f = np.asarray(arr, dtype=np.float64)
    if arr_f.ndim == 1:
        return _sanitize_weight_vector(arr_f, floor=0.0 if floor is None else float(floor)).astype(np.float64)
    if arr_f.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape={arr_f.shape}")
    out = np.array(arr_f, dtype=np.float64, copy=True)
    bad_rows = np.where(~np.isfinite(out).all(axis=1))[0]
    for row_idx in bad_rows.tolist():
        finite_vals = out[row_idx][np.isfinite(out[row_idx])]
        if finite_vals.size <= 0:
            out[row_idx] = 0.0
            continue
        pos_fill = float(np.max(finite_vals))
        neg_fill = float(floor) if floor is not None else float(np.min(finite_vals))
        out[row_idx] = np.nan_to_num(
            out[row_idx],
            nan=0.0,
            posinf=pos_fill,
            neginf=neg_fill,
        )
    if floor is not None:
        np.maximum(out, float(floor), out=out)
    return out


def _sanitize_layer_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    out = dict(payload)
    changed = False
    for key in ("n_active", "wmax_mean", "wmax_max"):
        if key not in payload:
            continue
        arr = np.asarray(payload[key])
        needs_clean = (not np.isfinite(arr).all()) or np.any(arr < 0)
        if needs_clean:
            changed = True
        out[key] = _sanitize_feature_matrix(arr, floor=0.0).astype(np.float32)
    if "sample_ids" in payload:
        out["sample_ids"] = np.asarray(payload["sample_ids"], dtype=np.int64)
    return out, changed


def extract_layer_summary_arrays(
    reader: CacheReader,
    sample_ids: np.ndarray | None = None,
    *,
    n_layers: int = DEFAULT_N_LAYERS,
    stride: int = DEFAULT_LAYER_STRIDE,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    run_ids = (
        np.arange(reader.num_runs(), dtype=np.int64)
        if sample_ids is None
        else np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    )
    n_runs = int(run_ids.size)
    n_active = np.zeros((n_runs, int(n_layers)), dtype=np.float32)
    wmax_mean = np.zeros((n_runs, int(n_layers)), dtype=np.float32)
    wmax_max = np.zeros((n_runs, int(n_layers)), dtype=np.float32)

    row_ptr = reader.row_ptr
    keys = reader.keys
    weights = reader.weights_for(Agg.MAX)

    for row_idx, sample_id in enumerate(run_ids.tolist()):
        start, end = int(row_ptr[int(sample_id)]), int(row_ptr[int(sample_id) + 1])
        if end <= start:
            continue
        local_keys = np.asarray(keys[start:end], dtype=np.uint32)
        local_weights = np.asarray(weights[start:end], dtype=np.float32)
        layer_ids = np.right_shift(local_keys, 16).astype(np.int64)
        valid_mask = (layer_ids >= 0) & (layer_ids < int(n_layers))
        if not valid_mask.any():
            continue
        layer_ids = layer_ids[valid_mask]
        local_weights = _sanitize_weight_vector(local_weights[valid_mask], floor=0.0)
        counts = np.bincount(layer_ids, minlength=int(n_layers))[: int(n_layers)].astype(np.float32)
        sums = np.bincount(layer_ids, weights=local_weights, minlength=int(n_layers))[: int(n_layers)].astype(np.float32)
        n_active[row_idx] = counts
        valid = counts > 0
        wmax_mean[row_idx, valid] = sums[valid] / counts[valid]
        for layer_id in np.unique(layer_ids).tolist():
            mask = layer_ids == int(layer_id)
            if not mask.any():
                continue
            wmax_max[row_idx, int(layer_id)] = float(np.max(local_weights[mask]))
        if verbose and row_idx > 0 and row_idx % 2000 == 0:
            print(f"  extract_layer_summary_arrays: {row_idx}/{n_runs}", flush=True)

    return {
        "sample_ids": np.asarray(run_ids, dtype=np.int64),
        "n_active": np.asarray(n_active, dtype=np.float32),
        "wmax_mean": np.asarray(wmax_mean, dtype=np.float32),
        "wmax_max": np.asarray(wmax_max, dtype=np.float32),
        "n_layers": int(n_layers),
        "stride": int(stride),
    }


def load_or_build_layer_summary_cache(
    cache_root: str | Path,
    *,
    cache_path: str | Path,
    sample_ids: np.ndarray | None = None,
    n_layers: int = DEFAULT_N_LAYERS,
    stride: int = DEFAULT_LAYER_STRIDE,
    refresh: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    cp = Path(cache_path)
    if cp.exists() and not refresh:
        with cp.open("rb") as fh:
            payload = pickle.load(fh)
        payload, changed = _sanitize_layer_payload(payload)
        if changed:
            cp.parent.mkdir(parents=True, exist_ok=True)
            with cp.open("wb") as fh:
                pickle.dump(payload, fh, protocol=4)
        cached_ids = np.asarray(payload.get("sample_ids", np.zeros(0, dtype=np.int64)), dtype=np.int64)
        desired_ids = (
            np.asarray(sample_ids, dtype=np.int64).reshape(-1)
            if sample_ids is not None
            else cached_ids
        )
        if sample_ids is None or np.array_equal(cached_ids, desired_ids):
            return payload

    reader = CacheReader(str(cache_root))
    payload = extract_layer_summary_arrays(
        reader,
        sample_ids=sample_ids,
        n_layers=n_layers,
        stride=stride,
        verbose=verbose,
    )
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    return payload


def build_global_neuron_feature_matrix(
    layer_payload: dict[str, Any],
    *,
    selected_layers: tuple[int, ...] = (23, 22, 18, 24, 21, 20),
) -> tuple[np.ndarray, list[str]]:
    n_active = _sanitize_feature_matrix(layer_payload["n_active"], floor=0.0)
    wmax_mean = _sanitize_feature_matrix(layer_payload["wmax_mean"], floor=0.0)
    wmax_max = _sanitize_feature_matrix(layer_payload["wmax_max"], floor=0.0)

    cols = [
        n_active.sum(axis=1),
        n_active.mean(axis=1),
        wmax_mean.mean(axis=1),
        wmax_max.max(axis=1),
    ]
    names = [
        "n_active_total",
        "n_active_mean",
        "wmax_mean_global",
        "wmax_max_global",
    ]

    for layer_id in selected_layers:
        cols.append(n_active[:, int(layer_id)])
        names.append(f"n_active_layer_{int(layer_id)}")
    for layer_id in selected_layers:
        cols.append(wmax_mean[:, int(layer_id)])
        names.append(f"wmax_mean_layer_{int(layer_id)}")
    for layer_id in selected_layers[:3]:
        cols.append(wmax_max[:, int(layer_id)])
        names.append(f"wmax_max_layer_{int(layer_id)}")

    X = np.column_stack(cols).astype(np.float64)
    return X, names


def _safe_entropy_rows(arr: np.ndarray) -> np.ndarray:
    arr_f = _sanitize_feature_matrix(arr, floor=0.0)
    if arr_f.ndim != 2 or arr_f.shape[1] <= 1:
        return np.zeros(arr_f.shape[0], dtype=np.float64)
    sums = np.sum(arr_f, axis=1, keepdims=True)
    probs = np.zeros_like(arr_f, dtype=np.float64)
    np.divide(arr_f, np.maximum(sums, 1e-12), out=probs, where=sums > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    return np.nan_to_num(entropy / float(np.log(arr_f.shape[1])), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_center_of_mass(arr: np.ndarray) -> np.ndarray:
    arr_f = _sanitize_feature_matrix(arr, floor=0.0)
    if arr_f.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr_f.shape}")
    layer_idx = np.arange(arr_f.shape[1], dtype=np.float64)
    sums = np.sum(arr_f, axis=1)
    out = np.zeros(arr_f.shape[0], dtype=np.float64)
    np.divide(arr_f @ layer_idx, np.maximum(sums, 1e-12), out=out, where=sums > 0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_tail_fraction(arr: np.ndarray, *, k: int, reverse: bool) -> np.ndarray:
    arr_f = _sanitize_feature_matrix(arr, floor=0.0)
    if arr_f.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr_f.shape}")
    if arr_f.shape[1] <= 0:
        return np.zeros(arr_f.shape[0], dtype=np.float64)
    k_eff = max(1, min(int(k), int(arr_f.shape[1])))
    tail = arr_f[:, -k_eff:] if reverse else arr_f[:, :k_eff]
    denom = np.maximum(np.sum(arr_f, axis=1), 1e-12)
    return np.nan_to_num(np.sum(tail, axis=1) / denom, nan=0.0, posinf=0.0, neginf=0.0)


def build_layer_derived_feature_matrix(
    layer_payload: dict[str, Any],
    *,
    top_k_layers: int = 12,
) -> tuple[np.ndarray, list[str]]:
    n_active = _sanitize_feature_matrix(layer_payload["n_active"], floor=0.0)
    wmax_mean = _sanitize_feature_matrix(layer_payload["wmax_mean"], floor=0.0)
    wmax_max = _sanitize_feature_matrix(layer_payload["wmax_max"], floor=0.0)
    wmax_mean_pos = np.maximum(wmax_mean, 0.0)
    wmax_max_pos = np.maximum(wmax_max, 0.0)

    cols = [
        n_active.sum(axis=1),
        n_active.mean(axis=1),
        n_active.std(axis=1),
        _safe_entropy_rows(n_active),
        _safe_center_of_mass(n_active),
        _safe_tail_fraction(n_active, k=top_k_layers, reverse=True),
        _safe_tail_fraction(n_active, k=top_k_layers, reverse=False),
        _safe_tail_fraction(n_active, k=top_k_layers, reverse=True)
        - _safe_tail_fraction(n_active, k=top_k_layers, reverse=False),
        wmax_mean.mean(axis=1),
        wmax_mean.std(axis=1),
        _safe_entropy_rows(wmax_mean_pos),
        _safe_center_of_mass(wmax_mean_pos),
        _safe_tail_fraction(wmax_mean_pos, k=top_k_layers, reverse=True),
        wmax_max.max(axis=1),
        wmax_max.mean(axis=1),
        wmax_max.std(axis=1),
        _safe_center_of_mass(wmax_max_pos),
        _safe_tail_fraction(wmax_max_pos, k=top_k_layers, reverse=True),
    ]
    names = [
        "n_active_total",
        "n_active_mean",
        "n_active_std",
        "n_active_entropy",
        "n_active_layer_com",
        "n_active_top_frac",
        "n_active_bottom_frac",
        "n_active_top_minus_bottom",
        "wmax_mean_global_mean",
        "wmax_mean_global_std",
        "wmax_mean_entropy",
        "wmax_mean_layer_com",
        "wmax_mean_top_frac",
        "wmax_max_global_max",
        "wmax_max_global_mean",
        "wmax_max_global_std",
        "wmax_max_layer_com",
        "wmax_max_top_frac",
    ]
    X = np.column_stack(cols).astype(np.float64)
    return X, names


def group_rank_matrix(
    X: np.ndarray,
    problem_groups: Mapping[str, Sequence[int]],
    *,
    neutral_singleton: float = 0.5,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={X_arr.shape}")
    out = np.zeros_like(X_arr, dtype=np.float64)
    for problem_id in sorted(problem_groups.keys()):
        idx = np.asarray(problem_groups[problem_id], dtype=np.int64)
        if idx.size <= 0:
            continue
        if idx.size == 1:
            out[idx[0], :] = float(neutral_singleton)
            continue
        local = X_arr[idx]
        order = np.argsort(local, axis=0, kind="stable")
        ranks = np.empty_like(order, dtype=np.float64)
        base = np.arange(idx.size, dtype=np.float64)
        for col_idx in range(local.shape[1]):
            ranks[order[:, col_idx], col_idx] = base
        out[idx] = ranks / float(idx.size - 1)
    return out


def build_activation_hybrid_feature_matrix(
    layer_payload: dict[str, Any],
    problem_groups: Mapping[str, Sequence[int]],
    *,
    selected_layers: tuple[int, ...] = (23, 22, 18, 24, 21, 20),
    include_group_rank: bool = True,
) -> tuple[np.ndarray, list[str]]:
    X_global, global_names = build_global_neuron_feature_matrix(
        layer_payload,
        selected_layers=selected_layers,
    )
    X_derived, derived_names = build_layer_derived_feature_matrix(layer_payload)
    X_base = np.column_stack([X_global, X_derived]).astype(np.float64)
    names = [*global_names, *derived_names]
    if not include_group_rank:
        return X_base, names

    X_rank = group_rank_matrix(X_base, problem_groups)
    rank_names = [f"{name}_group_rank" for name in names]
    X_full = np.column_stack([X_base, X_rank]).astype(np.float64)
    return X_full, [*names, *rank_names]
