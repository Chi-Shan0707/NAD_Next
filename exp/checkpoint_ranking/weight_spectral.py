#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = Path("/home/jovyan/public-ro/NAD_RL/math5000RL_neuron_analysis/model")
DEFAULT_TRUTH_JSON = (
    REPO_ROOT
    / "results/scans/checkpoint_ranking/es_svd_math_rr_r1/es_svd_math_rr_r1__math5000rl_slot100_meanconf_eval.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
OFFICIAL_CHECKPOINTS = (
    "base",
    "step-100",
    "step-200",
    "step-300",
    "step-400",
    "step-500",
    "step-600",
    "step-700",
    "step-800",
    "step-900",
    "step-1000",
)
CHECKPOINT_ORDER = {name: idx for idx, name in enumerate(OFFICIAL_CHECKPOINTS)}
FLOAT_EPS = 1e-12
LAYER_COUNT = 36
DELTA_FAMILIES = ("delta_base", "delta_prev")


@dataclass
class CheckpointInfo:
    checkpoint_name: str
    model_dir: Path
    order_index: int
    step: int
    true_accuracy: float | None = None


@dataclass
class DriftAccumulator:
    numel: int = 0
    delta_sumsq: float = 0.0
    delta_abs_sum: float = 0.0
    ref_sumsq: float = 0.0
    curr_sumsq: float = 0.0
    dot_curr_ref: float = 0.0
    sparse_count: int = 0
    sign_flip_count: int = 0
    active_count: int = 0

    def update(
        self,
        curr: torch.Tensor,
        ref: torch.Tensor,
        *,
        abs_eps: float,
        rel_eps: float,
        sign_eps: float,
        scale: float = 1.0,
    ) -> None:
        curr_f = curr.to(torch.float32)
        ref_f = ref.to(torch.float32)
        delta = curr_f - ref_f
        weight = float(max(scale, 1.0))
        self.numel += int(round(delta.numel() * weight))
        self.delta_sumsq += weight * float(torch.sum(delta * delta, dtype=torch.float64).item())
        self.delta_abs_sum += weight * float(torch.sum(torch.abs(delta), dtype=torch.float64).item())
        self.ref_sumsq += weight * float(torch.sum(ref_f * ref_f, dtype=torch.float64).item())
        self.curr_sumsq += weight * float(torch.sum(curr_f * curr_f, dtype=torch.float64).item())
        self.dot_curr_ref += weight * float(torch.sum(curr_f * ref_f, dtype=torch.float64).item())
        tol = float(abs_eps) + float(rel_eps) * torch.abs(ref_f)
        self.sparse_count += int(round(weight * float(torch.sum(torch.abs(delta) <= tol).item())))
        active = (torch.abs(curr_f) > float(sign_eps)) | (torch.abs(ref_f) > float(sign_eps))
        self.active_count += int(round(weight * float(torch.sum(active).item())))
        flips = ((curr_f >= 0.0) != (ref_f >= 0.0)) & active
        self.sign_flip_count += int(round(weight * float(torch.sum(flips).item())))

    def finalize(self) -> dict[str, float]:
        delta_fro = math.sqrt(max(self.delta_sumsq, 0.0))
        ref_fro = math.sqrt(max(self.ref_sumsq, 0.0))
        curr_fro = math.sqrt(max(self.curr_sumsq, 0.0))
        cosine = self.dot_curr_ref / max(ref_fro * curr_fro, FLOAT_EPS)
        return {
            "fro": float(delta_fro),
            "fro_ratio": float(delta_fro / max(ref_fro, FLOAT_EPS)),
            "mean_abs": float(self.delta_abs_sum / max(self.numel, 1)),
            "sparsity": float(self.sparse_count / max(self.numel, 1)),
            "sign_flip_ratio": float(self.sign_flip_count / max(self.active_count, 1)),
            "cosine_ref": float(np.clip(cosine, -1.0, 1.0)),
        }


@dataclass
class SketchAccumulator:
    weight_sum: float = 0.0
    count: int = 0
    sums: dict[str, float] = field(default_factory=dict)
    max_spectral_proxy: float = 0.0

    def update(self, metrics: dict[str, float], weight: float) -> None:
        weight_value = float(max(weight, 0.0))
        self.weight_sum += weight_value
        self.count += 1
        self.max_spectral_proxy = max(self.max_spectral_proxy, float(metrics.get("spectral_proxy", 0.0)))
        for key, value in metrics.items():
            if key == "spectral_proxy":
                continue
            self.sums[key] = self.sums.get(key, 0.0) + weight_value * float(value)

    def finalize(self) -> dict[str, float]:
        denom = max(self.weight_sum, FLOAT_EPS)
        out = {
            "spec_proxy": float(self.max_spectral_proxy),
            "sketch_count": float(self.count),
        }
        for key, value in self.sums.items():
            out[key] = float(value / denom)
        return out


class ModelTensorStore:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = dict(payload["weight_map"])
        self._stack = contextlib.ExitStack()
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> "ModelTensorStore":
        for filename in sorted(set(self.weight_map.values())):
            path = self.model_dir / filename
            self._handles[filename] = self._stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stack.close()

    def get_slice(self, tensor_name: str) -> Any:
        filename = self.weight_map[tensor_name]
        return self._handles[filename].get_slice(tensor_name)

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        filename = self.weight_map[tensor_name]
        return self._handles[filename].get_tensor(tensor_name)

    def tensor_shape(self, tensor_name: str) -> tuple[int, ...]:
        return tuple(self.get_slice(tensor_name).get_shape())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Checkpoint-weight fallback branch: spectral + drift features for checkpoint ranking."
    )
    ap.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    ap.add_argument("--truth-json", type=Path, default=DEFAULT_TRUTH_JSON)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--scenario-name", default="math5000rl_qwen3_4b")
    ap.add_argument("--sketch-rows", type=int, default=192)
    ap.add_argument("--sketch-cols", type=int, default=192)
    ap.add_argument("--svd-rank", type=int, default=6)
    ap.add_argument("--svd-oversamples", type=int, default=4)
    ap.add_argument("--svd-iters", type=int, default=2)
    ap.add_argument("--probe-count", type=int, default=4)
    ap.add_argument("--exact-drift", action="store_true")
    ap.add_argument("--drift-sample-rows", type=int, default=96)
    ap.add_argument("--drift-sample-cols", type=int, default=96)
    ap.add_argument("--drift-sample-vector", type=int, default=8192)
    ap.add_argument("--full-load-numel", type=int, default=30_000_000)
    ap.add_argument("--chunk-rows", type=int, default=512)
    ap.add_argument("--feature-cap", type=int, default=96)
    ap.add_argument("--ridge-alpha", type=float, default=2.0)
    ap.add_argument("--logreg-c", type=float, default=0.35)
    ap.add_argument("--smooth-lambda", type=float, default=0.35)
    ap.add_argument("--abs-sparsity-eps", type=float, default=1e-6)
    ap.add_argument("--rel-sparsity-eps", type=float, default=1e-3)
    ap.add_argument("--sign-eps", type=float, default=1e-7)
    ap.add_argument("--torch-threads", type=int, default=min(4, os.cpu_count() or 4))
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def _checkpoint_name_from_dir(model_dir_name: str) -> str:
    if model_dir_name == "Qwen3-4B-Base_base":
        return "base"
    match = re.fullmatch(r"Qwen3-4B-Base_math7500-step-(\d+)", model_dir_name)
    if match is None:
        raise ValueError(f"Unrecognized checkpoint directory: {model_dir_name}")
    name = f"step-{int(match.group(1))}"
    if name not in CHECKPOINT_ORDER:
        raise ValueError(f"Unexpected checkpoint step: {model_dir_name}")
    return name


def _discover_checkpoints(model_root: Path, truth_map: dict[str, float] | None) -> list[CheckpointInfo]:
    rows: list[CheckpointInfo] = []
    for child in sorted(Path(model_root).iterdir()):
        if not child.is_dir():
            continue
        checkpoint_name = _checkpoint_name_from_dir(child.name)
        step = 0 if checkpoint_name == "base" else int(checkpoint_name.split("-")[1])
        rows.append(
            CheckpointInfo(
                checkpoint_name=checkpoint_name,
                model_dir=child,
                order_index=int(CHECKPOINT_ORDER[checkpoint_name]),
                step=step,
                true_accuracy=None if truth_map is None else truth_map.get(checkpoint_name),
            )
        )
    rows.sort(key=lambda item: item.order_index)
    found_names = [row.checkpoint_name for row in rows]
    if found_names != list(OFFICIAL_CHECKPOINTS):
        raise ValueError(f"Checkpoint set mismatch: found={found_names}")
    return rows


def _load_truth_map(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "checkpoints" in payload and isinstance(payload["checkpoints"], list):
        return {str(row["checkpoint_name"]): float(row["true_accuracy"]) for row in payload["checkpoints"]}
    if "true_checkpoint_accuracy" in payload and isinstance(payload["true_checkpoint_accuracy"], dict):
        return {str(key): float(value) for key, value in payload["true_checkpoint_accuracy"].items()}
    if isinstance(payload, dict):
        try:
            return {str(key): float(value) for key, value in payload.items()}
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported truth JSON format: {path}") from exc
    raise ValueError(f"Unsupported truth JSON format: {path}")


def _load_config_flag(model_dir: Path, key: str, default: Any) -> Any:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return default
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload.get(key, default)


def _classify_tensor_name(name: str) -> dict[str, Any]:
    if name.startswith("model.embed_tokens."):
        return {
            "layer": None,
            "module": "embedding",
            "principal": True,
            "principal_role": "embedding",
        }
    if name.startswith("lm_head."):
        return {
            "layer": None,
            "module": "lm_head",
            "principal": True,
            "principal_role": "lm_head",
        }

    layer_match = re.match(r"model\.layers\.(\d+)\.(.+)", name)
    layer = None if layer_match is None else int(layer_match.group(1))
    suffix = name if layer_match is None else layer_match.group(2)

    if ".self_attn." in name:
        role = None
        if name.endswith("q_proj.weight"):
            role = "attn_q"
        elif name.endswith("o_proj.weight"):
            role = "attn_o"
        return {
            "layer": layer,
            "module": "attention",
            "principal": role is not None,
            "principal_role": role,
        }
    if ".mlp." in name:
        role = None
        if name.endswith("down_proj.weight"):
            role = "mlp_down"
        elif name.endswith("up_proj.weight"):
            role = "mlp_up"
        return {
            "layer": layer,
            "module": "mlp",
            "principal": role is not None,
            "principal_role": role,
        }
    if "layernorm" in suffix or suffix.endswith("q_norm.weight") or suffix.endswith("k_norm.weight"):
        return {
            "layer": layer,
            "module": "norm",
            "principal": False,
            "principal_role": None,
        }
    return {
        "layer": layer,
        "module": "other",
        "principal": False,
        "principal_role": None,
    }


def _group_keys_for_tensor(info: dict[str, Any], *, tie_word_embeddings: bool) -> list[str]:
    keys = ["global"]
    module = str(info["module"])
    keys.append(f"module_{module}")
    if module == "embedding" and tie_word_embeddings:
        keys.append("module_lm_head")
    if info["layer"] is not None:
        keys.append(f"layer_{int(info['layer']):02d}")
    return keys


def _zero_feature_row(checkpoint: CheckpointInfo) -> dict[str, float | int | str]:
    return {
        "scenario": "",
        "checkpoint_name": checkpoint.checkpoint_name,
        "order_index": checkpoint.order_index,
        "step": checkpoint.step,
        "true_accuracy": checkpoint.true_accuracy if checkpoint.true_accuracy is not None else np.nan,
    }


def _sample_indices(length: int, limit: int) -> np.ndarray:
    if limit <= 0 or length <= limit:
        return np.arange(length, dtype=np.int64)
    raw = np.linspace(0, length - 1, num=limit, dtype=np.int64)
    return np.unique(raw)


def _extract_sketch(matrix_or_slice: Any, shape: tuple[int, int], rows: int, cols: int) -> torch.Tensor:
    row_idx = _sample_indices(int(shape[0]), int(rows))
    col_idx = _sample_indices(int(shape[1]), int(cols))
    if isinstance(matrix_or_slice, torch.Tensor):
        return matrix_or_slice[row_idx][:, col_idx].to(torch.float32)
    return matrix_or_slice[row_idx.tolist()][:, col_idx.tolist()].to(torch.float32)


def _extract_drift_sample(
    tensor_or_slice: Any,
    shape: tuple[int, ...],
    *,
    sample_rows: int,
    sample_cols: int,
    sample_vector: int,
) -> torch.Tensor:
    if len(shape) == 1:
        idx = _sample_indices(int(shape[0]), int(sample_vector))
        if isinstance(tensor_or_slice, torch.Tensor):
            return tensor_or_slice[idx].to(torch.float32)
        return tensor_or_slice[idx.tolist()].to(torch.float32)
    if len(shape) == 2:
        return _extract_sketch(tensor_or_slice, (int(shape[0]), int(shape[1])), sample_rows, sample_cols)
    raise ValueError(f"Unsupported tensor shape for drift sample: {shape}")


def _probe_seed(name: str, probe_idx: int) -> int:
    payload = f"{name}|{probe_idx}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _compute_sketch_metrics(
    delta_sketch: torch.Tensor,
    *,
    tensor_name: str,
    svd_rank: int,
    svd_oversamples: int,
    svd_iters: int,
    probe_count: int,
) -> dict[str, float]:
    arr = np.asarray(delta_sketch.detach().cpu(), dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    fro_sq = float(np.sum(arr * arr, dtype=np.float64))
    if arr.size == 0 or fro_sq <= FLOAT_EPS:
        return {
            "spectral_proxy": 0.0,
            "energy_topk": 0.0,
            "sv_top1": 0.0,
            "sv_top2_ratio": 0.0,
            "sv_mean_topk": 0.0,
            "anisotropy": 0.0,
            "stable_rank": 0.0,
            "effective_rank": 0.0,
            "probe_mean": 0.0,
            "probe_std": 0.0,
            "hutch_trace_mean": 0.0,
            "hutch_trace_std": 0.0,
        }

    min_dim = int(min(arr.shape))
    if min_dim <= 1:
        n_components = 1
    else:
        n_components = max(1, min(int(svd_rank), min_dim - 1))
    try:
        _, singular_values, _ = randomized_svd(
            arr,
            n_components=n_components,
            n_oversamples=min(int(svd_oversamples), max(arr.shape) - n_components + 1),
            n_iter=int(svd_iters),
            random_state=0,
            flip_sign=False,
        )
    except Exception:
        singular_values = np.linalg.svd(arr, compute_uv=False, full_matrices=False)[:n_components]

    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.size == 0:
        singular_values = np.zeros((1,), dtype=np.float64)
    energy_topk = float(np.sum(singular_values**2) / max(fro_sq, FLOAT_EPS))
    sv_top1 = float(singular_values[0])
    sv_top2_ratio = float(singular_values[1] / max(singular_values[0], FLOAT_EPS)) if singular_values.size > 1 else 0.0
    sv_mean_topk = float(np.mean(singular_values))
    anisotropy = float(singular_values[0] / max(sv_mean_topk, FLOAT_EPS))
    stable_rank = float(fro_sq / max(singular_values[0] ** 2, FLOAT_EPS))

    top_energy = np.asarray(singular_values**2, dtype=np.float64)
    tail_energy = max(fro_sq - float(np.sum(top_energy)), 0.0)
    energy_parts = top_energy.tolist()
    if tail_energy > FLOAT_EPS:
        energy_parts.append(float(tail_energy))
    energy_arr = np.asarray(energy_parts, dtype=np.float64)
    probs = energy_arr / max(float(np.sum(energy_arr)), FLOAT_EPS)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, FLOAT_EPS, None))))
    effective_rank = float(np.exp(entropy))

    probe_responses: list[float] = []
    hutch_traces: list[float] = []
    for probe_idx in range(int(probe_count)):
        rng = np.random.default_rng(_probe_seed(tensor_name, probe_idx))
        vec = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=arr.shape[1], replace=True)
        response = arr @ vec
        qform = float(np.sum(response * response, dtype=np.float64))
        probe_responses.append(float(np.linalg.norm(response)))
        hutch_traces.append(qform)

    return {
        "spectral_proxy": sv_top1,
        "energy_topk": energy_topk,
        "sv_top1": sv_top1,
        "sv_top2_ratio": sv_top2_ratio,
        "sv_mean_topk": sv_mean_topk,
        "anisotropy": anisotropy,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "probe_mean": float(np.mean(probe_responses)),
        "probe_std": float(np.std(probe_responses)),
        "hutch_trace_mean": float(np.mean(hutch_traces)),
        "hutch_trace_std": float(np.std(hutch_traces)),
    }


def _ensure_nested_accumulator(
    root: dict[str, dict[str, DriftAccumulator | SketchAccumulator]],
    checkpoint_name: str,
    key: str,
    factory,
):
    checkpoint_dict = root.setdefault(checkpoint_name, {})
    if key not in checkpoint_dict:
        checkpoint_dict[key] = factory()
    return checkpoint_dict[key]


def _iter_row_chunks(length: int, chunk_rows: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < length:
        end = min(length, start + int(chunk_rows))
        yield start, end
        start = end


def _update_drift_sets(
    accumulators: dict[str, dict[str, DriftAccumulator]],
    checkpoint_name: str,
    group_keys: list[str],
    *,
    curr: torch.Tensor,
    ref: torch.Tensor,
    family: str,
    abs_eps: float,
    rel_eps: float,
    sign_eps: float,
    scale: float = 1.0,
) -> None:
    for group_key in group_keys:
        key = f"{group_key}__{family}"
        acc = _ensure_nested_accumulator(accumulators, checkpoint_name, key, DriftAccumulator)
        assert isinstance(acc, DriftAccumulator)
        acc.update(curr, ref, abs_eps=abs_eps, rel_eps=rel_eps, sign_eps=sign_eps, scale=scale)


def _finalize_feature_rows(
    checkpoints: list[CheckpointInfo],
    *,
    scenario_name: str,
    drift_accs: dict[str, dict[str, DriftAccumulator]],
    sketch_accs: dict[str, dict[str, SketchAccumulator]],
    tie_word_embeddings: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        row: dict[str, Any] = {
            "scenario": scenario_name,
            "checkpoint_name": checkpoint.checkpoint_name,
            "order_index": checkpoint.order_index,
            "step": checkpoint.step,
            "true_accuracy": np.nan if checkpoint.true_accuracy is None else float(checkpoint.true_accuracy),
        }
        checkpoint_drift = drift_accs.get(checkpoint.checkpoint_name, {})
        checkpoint_sketch = sketch_accs.get(checkpoint.checkpoint_name, {})

        for key, acc in checkpoint_drift.items():
            finalized = acc.finalize()
            group_key, family = key.split("__", 1)
            prefix = f"{group_key}_{family}"
            for metric_name, value in finalized.items():
                row[f"{prefix}_{metric_name}"] = float(value)

        for key, acc in checkpoint_sketch.items():
            finalized = acc.finalize()
            group_key, family = key.split("__", 1)
            prefix = f"{group_key}_{family}"
            for metric_name, value in finalized.items():
                row[f"{prefix}_{metric_name}"] = float(value)

        global_base_fro = float(row.get("global_delta_base_fro", 0.0))
        global_prev_fro = float(row.get("global_delta_prev_fro", 0.0))
        for module_name in ("attention", "mlp", "embedding", "lm_head", "norm", "other"):
            base_key = f"module_{module_name}_delta_base_fro"
            prev_key = f"module_{module_name}_delta_prev_fro"
            row[f"module_{module_name}_delta_base_share"] = float(row.get(base_key, 0.0) / max(global_base_fro, FLOAT_EPS))
            row[f"module_{module_name}_delta_prev_share"] = float(row.get(prev_key, 0.0) / max(global_prev_fro, FLOAT_EPS))

        if tie_word_embeddings:
            for family in DELTA_FAMILIES:
                for suffix in (
                    "fro",
                    "fro_ratio",
                    "mean_abs",
                    "sparsity",
                    "sign_flip_ratio",
                    "cosine_ref",
                    "share",
                    "spec_proxy",
                    "energy_topk",
                    "sv_top1",
                    "sv_top2_ratio",
                    "sv_mean_topk",
                    "anisotropy",
                    "stable_rank",
                    "effective_rank",
                    "probe_mean",
                    "probe_std",
                    "hutch_trace_mean",
                    "hutch_trace_std",
                    "sketch_count",
                ):
                    embed_key = f"module_embedding_{family}_{suffix}"
                    lm_key = f"module_lm_head_{family}_{suffix}"
                    if embed_key in row and lm_key not in row:
                        row[lm_key] = float(row[embed_key])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["order_index", "checkpoint_name"]).reset_index(drop=True)
    numeric_cols = [c for c in df.columns if c not in {"scenario", "checkpoint_name"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0 if col != "true_accuracy" else np.nan)
    return df


def _extract_feature_frame(
    checkpoints: list[CheckpointInfo],
    args: argparse.Namespace,
    *,
    scenario_name: str,
    tie_word_embeddings: bool,
) -> pd.DataFrame:
    drift_accs: dict[str, dict[str, DriftAccumulator]] = {}
    sketch_accs: dict[str, dict[str, SketchAccumulator]] = {}

    stores = {row.checkpoint_name: ModelTensorStore(row.model_dir) for row in checkpoints}
    with contextlib.ExitStack() as stack:
        open_stores = {name: stack.enter_context(store) for name, store in stores.items()}
        base_store = open_stores["base"]
        tensor_names = list(base_store.weight_map.keys())

        for tensor_name in tensor_names:
            info = _classify_tensor_name(tensor_name)
            group_keys = _group_keys_for_tensor(info, tie_word_embeddings=tie_word_embeddings)
            shape = base_store.tensor_shape(tensor_name)
            numel = int(math.prod(shape))

            if not bool(args.exact_drift):
                base_source = base_store.get_tensor(tensor_name) if numel <= int(args.full_load_numel) else base_store.get_slice(tensor_name)
                base_sample = _extract_drift_sample(
                    base_source,
                    shape,
                    sample_rows=int(args.drift_sample_rows),
                    sample_cols=int(args.drift_sample_cols),
                    sample_vector=int(args.drift_sample_vector),
                )
                prev_source = base_source
                prev_sample = base_sample
                for checkpoint in checkpoints[1:]:
                    curr_source = (
                        open_stores[checkpoint.checkpoint_name].get_tensor(tensor_name)
                        if numel <= int(args.full_load_numel)
                        else open_stores[checkpoint.checkpoint_name].get_slice(tensor_name)
                    )
                    curr_sample = _extract_drift_sample(
                        curr_source,
                        shape,
                        sample_rows=int(args.drift_sample_rows),
                        sample_cols=int(args.drift_sample_cols),
                        sample_vector=int(args.drift_sample_vector),
                    )
                    scale = float(numel / max(curr_sample.numel(), 1))
                    _update_drift_sets(
                        drift_accs,
                        checkpoint.checkpoint_name,
                        group_keys,
                        curr=curr_sample,
                        ref=base_sample,
                        family="delta_base",
                        abs_eps=float(args.abs_sparsity_eps),
                        rel_eps=float(args.rel_sparsity_eps),
                        sign_eps=float(args.sign_eps),
                        scale=scale,
                    )
                    _update_drift_sets(
                        drift_accs,
                        checkpoint.checkpoint_name,
                        group_keys,
                        curr=curr_sample,
                        ref=prev_sample,
                        family="delta_prev",
                        abs_eps=float(args.abs_sparsity_eps),
                        rel_eps=float(args.rel_sparsity_eps),
                        sign_eps=float(args.sign_eps),
                        scale=scale,
                    )

                    if info["principal"] and len(shape) == 2:
                        curr_sketch = _extract_sketch(curr_source, shape, int(args.sketch_rows), int(args.sketch_cols))
                        base_sketch = _extract_sketch(base_source, shape, int(args.sketch_rows), int(args.sketch_cols))
                        prev_sketch = _extract_sketch(prev_source, shape, int(args.sketch_rows), int(args.sketch_cols))
                        delta_base_metrics = _compute_sketch_metrics(
                            curr_sketch - base_sketch,
                            tensor_name=f"{tensor_name}|{checkpoint.checkpoint_name}|delta_base",
                            svd_rank=int(args.svd_rank),
                            svd_oversamples=int(args.svd_oversamples),
                            svd_iters=int(args.svd_iters),
                            probe_count=int(args.probe_count),
                        )
                        delta_prev_metrics = _compute_sketch_metrics(
                            curr_sketch - prev_sketch,
                            tensor_name=f"{tensor_name}|{checkpoint.checkpoint_name}|delta_prev",
                            svd_rank=int(args.svd_rank),
                            svd_oversamples=int(args.svd_oversamples),
                            svd_iters=int(args.svd_iters),
                            probe_count=int(args.probe_count),
                        )
                        weight_base = float(torch.sum((curr_sketch - base_sketch) ** 2, dtype=torch.float64).item())
                        weight_prev = float(torch.sum((curr_sketch - prev_sketch) ** 2, dtype=torch.float64).item())
                        for group_key in group_keys:
                            acc_base = _ensure_nested_accumulator(
                                sketch_accs, checkpoint.checkpoint_name, f"{group_key}__delta_base", SketchAccumulator
                            )
                            acc_prev = _ensure_nested_accumulator(
                                sketch_accs, checkpoint.checkpoint_name, f"{group_key}__delta_prev", SketchAccumulator
                            )
                            assert isinstance(acc_base, SketchAccumulator)
                            assert isinstance(acc_prev, SketchAccumulator)
                            acc_base.update(delta_base_metrics, weight_base)
                            acc_prev.update(delta_prev_metrics, weight_prev)
                    prev_source = curr_source
                    prev_sample = curr_sample
            elif numel <= int(args.full_load_numel):
                base_tensor = base_store.get_tensor(tensor_name)
                prev_tensor = base_tensor
                for checkpoint in checkpoints[1:]:
                    curr_tensor = open_stores[checkpoint.checkpoint_name].get_tensor(tensor_name)
                    _update_drift_sets(
                        drift_accs,
                        checkpoint.checkpoint_name,
                        group_keys,
                        curr=curr_tensor,
                        ref=base_tensor,
                        family="delta_base",
                        abs_eps=float(args.abs_sparsity_eps),
                        rel_eps=float(args.rel_sparsity_eps),
                        sign_eps=float(args.sign_eps),
                    )
                    _update_drift_sets(
                        drift_accs,
                        checkpoint.checkpoint_name,
                        group_keys,
                        curr=curr_tensor,
                        ref=prev_tensor,
                        family="delta_prev",
                        abs_eps=float(args.abs_sparsity_eps),
                        rel_eps=float(args.rel_sparsity_eps),
                        sign_eps=float(args.sign_eps),
                    )

                    if info["principal"] and len(shape) == 2:
                        curr_sketch = _extract_sketch(curr_tensor, shape, int(args.sketch_rows), int(args.sketch_cols))
                        base_sketch = _extract_sketch(base_tensor, shape, int(args.sketch_rows), int(args.sketch_cols))
                        prev_sketch = _extract_sketch(prev_tensor, shape, int(args.sketch_rows), int(args.sketch_cols))
                        delta_base_metrics = _compute_sketch_metrics(
                            curr_sketch - base_sketch,
                            tensor_name=f"{tensor_name}|{checkpoint.checkpoint_name}|delta_base",
                            svd_rank=int(args.svd_rank),
                            svd_oversamples=int(args.svd_oversamples),
                            svd_iters=int(args.svd_iters),
                            probe_count=int(args.probe_count),
                        )
                        delta_prev_metrics = _compute_sketch_metrics(
                            curr_sketch - prev_sketch,
                            tensor_name=f"{tensor_name}|{checkpoint.checkpoint_name}|delta_prev",
                            svd_rank=int(args.svd_rank),
                            svd_oversamples=int(args.svd_oversamples),
                            svd_iters=int(args.svd_iters),
                            probe_count=int(args.probe_count),
                        )
                        weight_base = float(torch.sum((curr_sketch - base_sketch) ** 2, dtype=torch.float64).item())
                        weight_prev = float(torch.sum((curr_sketch - prev_sketch) ** 2, dtype=torch.float64).item())
                        for group_key in group_keys:
                            acc_base = _ensure_nested_accumulator(
                                sketch_accs, checkpoint.checkpoint_name, f"{group_key}__delta_base", SketchAccumulator
                            )
                            acc_prev = _ensure_nested_accumulator(
                                sketch_accs, checkpoint.checkpoint_name, f"{group_key}__delta_prev", SketchAccumulator
                            )
                            assert isinstance(acc_base, SketchAccumulator)
                            assert isinstance(acc_prev, SketchAccumulator)
                            acc_base.update(delta_base_metrics, weight_base)
                            acc_prev.update(delta_prev_metrics, weight_prev)
                    prev_tensor = curr_tensor
            else:
                base_slice = base_store.get_slice(tensor_name)
                prev_slice = base_slice
                for checkpoint in checkpoints[1:]:
                    curr_slice = open_stores[checkpoint.checkpoint_name].get_slice(tensor_name)
                    for start, end in _iter_row_chunks(int(shape[0]), int(args.chunk_rows)):
                        if len(shape) == 1:
                            curr_chunk = curr_slice[start:end]
                            base_chunk = base_slice[start:end]
                            prev_chunk = prev_slice[start:end]
                        else:
                            curr_chunk = curr_slice[start:end, :]
                            base_chunk = base_slice[start:end, :]
                            prev_chunk = prev_slice[start:end, :]
                        _update_drift_sets(
                            drift_accs,
                            checkpoint.checkpoint_name,
                            group_keys,
                            curr=curr_chunk,
                            ref=base_chunk,
                            family="delta_base",
                            abs_eps=float(args.abs_sparsity_eps),
                            rel_eps=float(args.rel_sparsity_eps),
                            sign_eps=float(args.sign_eps),
                        )
                        _update_drift_sets(
                            drift_accs,
                            checkpoint.checkpoint_name,
                            group_keys,
                            curr=curr_chunk,
                            ref=prev_chunk,
                            family="delta_prev",
                            abs_eps=float(args.abs_sparsity_eps),
                            rel_eps=float(args.rel_sparsity_eps),
                            sign_eps=float(args.sign_eps),
                        )

                    if info["principal"] and len(shape) == 2:
                        curr_sketch = _extract_sketch(curr_slice, shape, int(args.sketch_rows), int(args.sketch_cols))
                        base_sketch = _extract_sketch(base_slice, shape, int(args.sketch_rows), int(args.sketch_cols))
                        prev_sketch = _extract_sketch(prev_slice, shape, int(args.sketch_rows), int(args.sketch_cols))
                        delta_base_metrics = _compute_sketch_metrics(
                            curr_sketch - base_sketch,
                            tensor_name=f"{tensor_name}|{checkpoint.checkpoint_name}|delta_base",
                            svd_rank=int(args.svd_rank),
                            svd_oversamples=int(args.svd_oversamples),
                            svd_iters=int(args.svd_iters),
                            probe_count=int(args.probe_count),
                        )
                        delta_prev_metrics = _compute_sketch_metrics(
                            curr_sketch - prev_sketch,
                            tensor_name=f"{tensor_name}|{checkpoint.checkpoint_name}|delta_prev",
                            svd_rank=int(args.svd_rank),
                            svd_oversamples=int(args.svd_oversamples),
                            svd_iters=int(args.svd_iters),
                            probe_count=int(args.probe_count),
                        )
                        weight_base = float(torch.sum((curr_sketch - base_sketch) ** 2, dtype=torch.float64).item())
                        weight_prev = float(torch.sum((curr_sketch - prev_sketch) ** 2, dtype=torch.float64).item())
                        for group_key in group_keys:
                            acc_base = _ensure_nested_accumulator(
                                sketch_accs, checkpoint.checkpoint_name, f"{group_key}__delta_base", SketchAccumulator
                            )
                            acc_prev = _ensure_nested_accumulator(
                                sketch_accs, checkpoint.checkpoint_name, f"{group_key}__delta_prev", SketchAccumulator
                            )
                            assert isinstance(acc_base, SketchAccumulator)
                            assert isinstance(acc_prev, SketchAccumulator)
                            acc_base.update(delta_base_metrics, weight_base)
                            acc_prev.update(delta_prev_metrics, weight_prev)
                    prev_slice = curr_slice

    return _finalize_feature_rows(
        checkpoints,
        scenario_name=scenario_name,
        drift_accs=drift_accs,
        sketch_accs=sketch_accs,
        tie_word_embeddings=tie_word_embeddings,
    )


def _spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= FLOAT_EPS or np.std(y) <= FLOAT_EPS:
        return 0.0
    value = getattr(spearmanr(x, y), "statistic", None)
    if value is None or not np.isfinite(float(value)):
        return 0.0
    return abs(float(value))


def _select_features(df: pd.DataFrame, y: np.ndarray, *, feature_cap: int) -> tuple[list[str], pd.DataFrame]:
    metadata = {"scenario", "checkpoint_name", "order_index", "step", "true_accuracy"}
    numeric_cols = [c for c in df.columns if c not in metadata]
    rows = []
    for col in numeric_cols:
        arr = np.asarray(df[col], dtype=np.float64)
        variance = float(np.var(arr))
        corr = _spearman_abs(arr, y)
        rows.append({"feature": col, "variance": variance, "abs_spearman": corr})
    rank_df = pd.DataFrame(rows).sort_values(["abs_spearman", "variance", "feature"], ascending=[False, False, True])
    keep = rank_df[rank_df["variance"] > FLOAT_EPS]["feature"].tolist()[: int(feature_cap)]

    for must_have in (
        "global_delta_base_fro_ratio",
        "global_delta_prev_fro_ratio",
        "global_delta_prev_cosine_ref",
        "module_attention_delta_prev_share",
        "module_mlp_delta_prev_share",
    ):
        if must_have in numeric_cols and must_have not in keep:
            keep.append(must_have)

    keep = list(dict.fromkeys(keep))
    rank_df["selected"] = rank_df["feature"].isin(keep).astype(int)
    return keep, rank_df


def _fit_pointwise_model(X_train: np.ndarray, y_train: np.ndarray, alpha: float) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=None)
    model.fit(X_scaled, y_train)
    return scaler, model


def _build_pairwise_examples(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diffs: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    n = int(X.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            if float(y[i]) == float(y[j]):
                continue
            margin = abs(float(y[i]) - float(y[j]))
            target = 1 if float(y[i]) > float(y[j]) else 0
            diff = X[i] - X[j]
            diffs.append(diff)
            labels.append(target)
            weights.append(max(margin, 1e-4))
            diffs.append(-diff)
            labels.append(1 - target)
            weights.append(max(margin, 1e-4))
    if not diffs:
        return (
            np.zeros((0, X.shape[1]), dtype=np.float64),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.asarray(diffs, dtype=np.float64),
        np.asarray(labels, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
    )


def _fit_pairwise_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float,
) -> tuple[StandardScaler | None, LogisticRegression | None]:
    X_pairs, y_pairs, w_pairs = _build_pairwise_examples(X_train, y_train)
    if X_pairs.shape[0] <= 0 or np.unique(y_pairs).size < 2:
        return None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pairs)
    model = LogisticRegression(
        C=float(C),
        max_iter=5000,
        solver="liblinear",
        class_weight="balanced",
        random_state=0,
    )
    model.fit(X_scaled, y_pairs, sample_weight=w_pairs)
    return scaler, model


def _pairwise_scores(
    scaler: StandardScaler | None,
    model: LogisticRegression | None,
    X_query: np.ndarray,
    X_ref: np.ndarray,
    *,
    include_self: bool,
) -> np.ndarray:
    if scaler is None or model is None:
        return np.zeros((X_query.shape[0],), dtype=np.float64)
    diff = X_query[:, None, :] - X_ref[None, :, :]
    flat = diff.reshape(-1, diff.shape[-1])
    probs = model.predict_proba(scaler.transform(flat))[:, 1].reshape(X_query.shape[0], X_ref.shape[0])
    if include_self and X_query.shape[0] == X_ref.shape[0]:
        np.fill_diagonal(probs, 0.5)
    return np.mean(probs, axis=1)


def _safe_stat(value: Any) -> float | None:
    raw = getattr(value, "statistic", value)
    if raw is None:
        return None
    raw = float(raw)
    if not np.isfinite(raw):
        return None
    return raw


def _smooth_scores(scores: np.ndarray, lam: float) -> np.ndarray:
    lam_value = float(max(lam, 0.0))
    if lam_value <= 0.0 or scores.size <= 2:
        return scores.astype(np.float64, copy=True)
    n = int(scores.size)
    lap = np.zeros((n, n), dtype=np.float64)
    for idx in range(n):
        if idx > 0:
            lap[idx, idx - 1] = -1.0
            lap[idx, idx] += 1.0
        if idx + 1 < n:
            lap[idx, idx + 1] = -1.0
            lap[idx, idx] += 1.0
    system = np.eye(n, dtype=np.float64) + lam_value * lap
    return np.linalg.solve(system, np.asarray(scores, dtype=np.float64))


def _zscore_like(train_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    mean = float(np.mean(train_scores)) if train_scores.size else 0.0
    std = float(np.std(train_scores)) if train_scores.size else 0.0
    if std <= FLOAT_EPS:
        return np.asarray(test_scores, dtype=np.float64) - mean
    return (np.asarray(test_scores, dtype=np.float64) - mean) / std


def _clip_to_train_range(train_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    train_arr = np.asarray(train_scores, dtype=np.float64)
    test_arr = np.asarray(test_scores, dtype=np.float64)
    if train_arr.size <= 0:
        return test_arr
    lo = float(np.min(train_arr))
    hi = float(np.max(train_arr))
    return np.clip(test_arr, lo, hi)


def _choose_alpha(y_train: np.ndarray, reg_train_scores: np.ndarray, pair_train_scores: np.ndarray) -> float:
    reg_metric = max(_spearman_abs(reg_train_scores, y_train), 1e-6)
    pair_metric = max(_spearman_abs(pair_train_scores, y_train), 1e-6)
    return float(reg_metric / (reg_metric + pair_metric))


def _run_loo(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    ridge_alpha: float,
    logreg_c: float,
    smooth_lambda: float,
) -> pd.DataFrame:
    if "true_accuracy" not in feature_df.columns:
        raise ValueError("feature_df requires true_accuracy for OOF")
    y = np.asarray(feature_df["true_accuracy"], dtype=np.float64)
    X = np.asarray(feature_df[feature_cols], dtype=np.float64)

    pred_reg = np.zeros((X.shape[0],), dtype=np.float64)
    pred_pair = np.zeros((X.shape[0],), dtype=np.float64)
    pred_combo = np.zeros((X.shape[0],), dtype=np.float64)
    alpha_vals = np.zeros((X.shape[0],), dtype=np.float64)

    for holdout in range(X.shape[0]):
        mask = np.ones((X.shape[0],), dtype=bool)
        mask[holdout] = False
        X_train = X[mask]
        y_train = y[mask]
        X_test = X[~mask]

        reg_scaler, reg_model = _fit_pointwise_model(X_train, y_train, alpha=ridge_alpha)
        reg_train_scores = reg_model.predict(reg_scaler.transform(X_train))
        reg_test_score = reg_model.predict(reg_scaler.transform(X_test))
        reg_test_score = _clip_to_train_range(reg_train_scores, reg_test_score)

        pair_scaler, pair_model = _fit_pairwise_model(X_train, y_train, C=logreg_c)
        pair_train_scores = _pairwise_scores(pair_scaler, pair_model, X_train, X_train, include_self=True)
        pair_test_score = _pairwise_scores(pair_scaler, pair_model, X_test, X_train, include_self=False)
        pair_test_score = _clip_to_train_range(pair_train_scores, pair_test_score)

        alpha = _choose_alpha(y_train, reg_train_scores, pair_train_scores)
        pred_reg[holdout] = float(reg_test_score[0])
        pred_pair[holdout] = float(pair_test_score[0])
        pred_combo[holdout] = float(
            alpha * _zscore_like(reg_train_scores, reg_test_score)[0]
            + (1.0 - alpha) * _zscore_like(pair_train_scores, pair_test_score)[0]
        )
        alpha_vals[holdout] = alpha

    smoothed = _smooth_scores(pred_combo, lam=smooth_lambda)
    out = feature_df[["scenario", "checkpoint_name", "order_index", "step", "true_accuracy"]].copy()
    out["oof_pointwise"] = pred_reg
    out["oof_pairwise"] = pred_pair
    out["oof_combined_raw"] = pred_combo
    out["oof_combined_smoothed"] = smoothed
    out["blend_alpha"] = alpha_vals
    return out


def _fit_full_scores(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    ridge_alpha: float,
    logreg_c: float,
    smooth_lambda: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | float | list[str]]]:
    X = np.asarray(feature_df[feature_cols], dtype=np.float64)
    y = np.asarray(feature_df["true_accuracy"], dtype=np.float64)

    reg_scaler, reg_model = _fit_pointwise_model(X, y, alpha=ridge_alpha)
    reg_scores = reg_model.predict(reg_scaler.transform(X))
    pair_scaler, pair_model = _fit_pairwise_model(X, y, C=logreg_c)
    pair_scores = _pairwise_scores(pair_scaler, pair_model, X, X, include_self=True)
    alpha = _choose_alpha(y, reg_scores, pair_scores)
    combined_raw = alpha * _zscore_like(reg_scores, reg_scores) + (1.0 - alpha) * _zscore_like(pair_scores, pair_scores)
    combined_smoothed = _smooth_scores(combined_raw, lam=smooth_lambda)

    out = feature_df[["scenario", "checkpoint_name", "order_index", "step", "true_accuracy"]].copy()
    out["fit_pointwise"] = reg_scores
    out["fit_pairwise"] = pair_scores
    out["fit_combined_raw"] = combined_raw
    out["fit_combined_smoothed"] = combined_smoothed

    reg_coef = np.zeros((len(feature_cols),), dtype=np.float64)
    pair_coef = np.zeros((len(feature_cols),), dtype=np.float64)
    reg_coef[:] = np.asarray(reg_model.coef_, dtype=np.float64).reshape(-1)
    if pair_model is not None and pair_scaler is not None:
        pair_coef[:] = np.asarray(pair_model.coef_, dtype=np.float64).reshape(-1)

    return out, {
        "alpha": float(alpha),
        "reg_coef": reg_coef,
        "pair_coef": pair_coef,
        "feature_cols": list(feature_cols),
    }


def _rank_order(score_map: dict[str, float]) -> list[str]:
    return sorted(score_map.keys(), key=lambda key: (-float(score_map[key]), int(CHECKPOINT_ORDER[key])))


def _ranking_metrics(score_col: str, df: pd.DataFrame) -> dict[str, Any]:
    score_map = {str(row["checkpoint_name"]): float(row[score_col]) for _, row in df.iterrows()}
    true_map = {str(row["checkpoint_name"]): float(row["true_accuracy"]) for _, row in df.iterrows()}
    score_vec = np.asarray([score_map[name] for name in OFFICIAL_CHECKPOINTS], dtype=np.float64)
    true_vec = np.asarray([true_map[name] for name in OFFICIAL_CHECKPOINTS], dtype=np.float64)
    pred_rank = _rank_order(score_map)
    true_rank = _rank_order(true_map)
    return {
        "spearman_rho": _safe_stat(spearmanr(score_vec, true_vec)),
        "pearson_r": _safe_stat(pearsonr(score_vec, true_vec)),
        "kendall_tau": _safe_stat(kendalltau(score_vec, true_vec)),
        "top1_hit": int(pred_rank[0] == true_rank[0]),
        "top3_hit": int(len(set(pred_rank[:3]) & set(true_rank[:3]))),
        "pred_rank_order": pred_rank,
        "true_rank_order": true_rank,
    }


def _parse_feature_info(name: str) -> dict[str, Any]:
    info = {
        "feature": name,
        "scope": "global",
        "group": "global",
        "layer": None,
        "module": None,
        "family": None,
        "metric": None,
    }
    family = "delta_prev" if "_delta_prev_" in name else "delta_base" if "_delta_base_" in name else None
    info["family"] = family
    parts = name.split("_")
    if name.startswith("layer_") and len(parts) >= 4:
        info["scope"] = "layer"
        info["layer"] = int(parts[1])
        info["group"] = f"layer_{parts[1]}"
        metric_start = 3 if family is not None else 2
        info["metric"] = "_".join(parts[metric_start:])
    elif name.startswith("module_"):
        info["scope"] = "module"
        body = name[len("module_") :]
        family_token = f"_{family}_" if family is not None else "_"
        if family is not None and family_token in body:
            module_name, metric = body.split(family_token, 1)
        else:
            chunks = body.split("_", 1)
            module_name = chunks[0]
            metric = "" if len(chunks) == 1 else chunks[1]
        info["module"] = module_name
        info["group"] = f"module_{module_name}"
        info["metric"] = metric
    elif name.startswith("global_"):
        info["scope"] = "global"
        info["group"] = "global"
        metric_start = 2 if family is not None else 1
        info["metric"] = "_".join(parts[metric_start:])
    return info


def _build_feature_importance(
    feature_df: pd.DataFrame,
    feature_rank_df: pd.DataFrame,
    selected_feature_cols: list[str],
    fit_payload: dict[str, np.ndarray | float | list[str]],
) -> pd.DataFrame:
    selected = list(selected_feature_cols)
    y = np.asarray(feature_df["true_accuracy"], dtype=np.float64)
    reg_coef = np.asarray(fit_payload["reg_coef"], dtype=np.float64)
    pair_coef = np.asarray(fit_payload["pair_coef"], dtype=np.float64)
    alpha = float(fit_payload["alpha"])
    rows = []
    rank_lookup = feature_rank_df.set_index("feature").to_dict(orient="index")

    coef_lookup_reg = {feature: float(reg_coef[idx]) for idx, feature in enumerate(selected)}
    coef_lookup_pair = {feature: float(pair_coef[idx]) for idx, feature in enumerate(selected)}

    metadata = {"scenario", "checkpoint_name", "order_index", "step", "true_accuracy"}
    for feature in [c for c in feature_df.columns if c not in metadata]:
        arr = np.asarray(feature_df[feature], dtype=np.float64)
        info = _parse_feature_info(feature)
        rows.append(
            {
                **info,
                "selected": int(feature in selected),
                "variance": float(np.var(arr)),
                "abs_spearman": float(_spearman_abs(arr, y)),
                "reg_coef": coef_lookup_reg.get(feature, 0.0),
                "pair_coef": coef_lookup_pair.get(feature, 0.0),
                "combined_importance": float(
                    alpha * abs(coef_lookup_reg.get(feature, 0.0))
                    + (1.0 - alpha) * abs(coef_lookup_pair.get(feature, 0.0))
                ),
                "selection_rank_abs_spearman": rank_lookup.get(feature, {}).get("abs_spearman", 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["combined_importance", "abs_spearman", "feature"], ascending=[False, False, True]
    )


def _subset_features(cols: list[str], prefix: str) -> list[str]:
    return [col for col in cols if f"_{prefix}_" in col]


def _compute_ablation_metrics(
    feature_df: pd.DataFrame,
    *,
    ridge_alpha: float,
    logreg_c: float,
    smooth_lambda: float,
    feature_cap: int,
) -> pd.DataFrame:
    y = np.asarray(feature_df["true_accuracy"], dtype=np.float64)
    metadata = {"scenario", "checkpoint_name", "order_index", "step", "true_accuracy"}
    numeric_cols = [c for c in feature_df.columns if c not in metadata]
    rows = []
    for name, cols in (
        ("delta_base_only", _subset_features(numeric_cols, "delta_base")),
        ("delta_prev_only", _subset_features(numeric_cols, "delta_prev")),
        ("full", numeric_cols),
    ):
        if not cols:
            continue
        subset_df = feature_df[["scenario", "checkpoint_name", "order_index", "step", "true_accuracy"] + cols].copy()
        keep, _ = _select_features(subset_df, y, feature_cap=min(int(feature_cap), len(cols)))
        oof_df = _run_loo(
            subset_df,
            keep,
            ridge_alpha=ridge_alpha,
            logreg_c=logreg_c,
            smooth_lambda=smooth_lambda,
        )
        metrics = _ranking_metrics("oof_combined_smoothed", oof_df)
        rows.append(
            {
                "subset": name,
                "n_features": len(cols),
                "selected_features": len(keep),
                "spearman_rho": metrics["spearman_rho"],
                "pearson_r": metrics["pearson_r"],
                "kendall_tau": metrics["kendall_tau"],
                "top1_hit": metrics["top1_hit"],
                "top3_hit": metrics["top3_hit"],
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def _build_report(
    *,
    scenario_name: str,
    feature_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    fit_payload: dict[str, np.ndarray | float | list[str]],
    output_paths: dict[str, Path],
    tie_word_embeddings: bool,
    exact_drift: bool,
) -> str:
    oof_metrics = _ranking_metrics("oof_combined_smoothed", oof_df)
    fit_metrics = _ranking_metrics("fit_combined_smoothed", fit_df)
    selected_features = feature_importance_df[feature_importance_df["selected"] > 0].copy()

    module_rows = (
        feature_importance_df[feature_importance_df["scope"] == "module"]
        .groupby("module", as_index=False)["combined_importance"]
        .sum()
        .sort_values("combined_importance", ascending=False)
    )
    layer_rows = (
        feature_importance_df[feature_importance_df["scope"] == "layer"]
        .groupby("layer", as_index=False)["combined_importance"]
        .sum()
        .sort_values("combined_importance", ascending=False)
    )
    probe_layer_rows = (
        feature_importance_df[
            (feature_importance_df["scope"] == "layer")
            & (feature_importance_df["metric"].astype(str).str.contains("probe", na=False))
        ]
        .groupby("layer", as_index=False)["combined_importance"]
        .sum()
        .sort_values("combined_importance", ascending=False)
    )

    later_mask = fit_df["order_index"] >= int(len(fit_df) // 2)
    later_df = fit_df[later_mask].copy()
    later_threshold = float(later_df["true_accuracy"].median()) if not later_df.empty else 0.0
    late_good = later_df[later_df["true_accuracy"] >= later_threshold]["checkpoint_name"].tolist()

    score_map = {str(row["checkpoint_name"]): float(row["fit_combined_smoothed"]) for _, row in fit_df.iterrows()}
    true_map = {str(row["checkpoint_name"]): float(row["true_accuracy"]) for _, row in fit_df.iterrows()}

    ranking_rows = []
    pred_order = _rank_order(score_map)
    for rank, checkpoint_name in enumerate(pred_order, start=1):
        ranking_rows.append(
            [
                rank,
                checkpoint_name,
                f"{score_map[checkpoint_name]:.6f}",
                f"{true_map[checkpoint_name]:.6f}",
            ]
        )

    top_feature_rows = []
    for _, row in selected_features.head(15).iterrows():
        top_feature_rows.append(
            [
                row["feature"],
                f"{float(row['combined_importance']):.4f}",
                f"{float(row['abs_spearman']):.4f}",
            ]
        )

    module_table_rows = [
        [str(row["module"]), f"{float(row['combined_importance']):.4f}"] for _, row in module_rows.head(8).iterrows()
    ]
    layer_table_rows = [
        [int(row["layer"]), f"{float(row['combined_importance']):.4f}"] for _, row in layer_rows.head(10).iterrows()
    ]
    probe_layer_table_rows = [
        [int(row["layer"]), f"{float(row['combined_importance']):.4f}"] for _, row in probe_layer_rows.head(10).iterrows()
    ]
    ablation_rows = []
    for _, row in ablation_df.iterrows():
        ablation_rows.append(
            [
                row["subset"],
                int(row["selected_features"]),
                f"{float(row['spearman_rho'] or 0.0):.4f}",
                f"{float(row['pearson_r'] or 0.0):.4f}",
                f"{float(row['kendall_tau'] or 0.0):.4f}",
            ]
        )

    return "\n".join(
        [
            "# Weight Spectral Fallback Report",
            "",
            "## Branch Status",
            "",
            "- This line is implemented as a **fallback / parallel branch** for checkpoint-centric ranking.",
            "- It does **not** replace the existing response / activation branch; it only adds a weights-only path.",
            f"- Scenario: `{scenario_name}`",
            f"- Checkpoints: `{len(feature_df)}` ({', '.join(feature_df['checkpoint_name'].tolist())})",
            f"- `lm_head` handling: `{'tied-to-embedding' if tie_word_embeddings else 'standalone'}`",
            f"- Drift extraction mode: `{'exact full-tensor scan' if exact_drift else 'sampled drift / exact spectral sketches'}`",
            "",
            "## Modeling Stack",
            "",
            "- Weight drift features cover `delta-to-base`, `delta-to-prev`, cosine-to-reference, per-layer Fro summaries, module drift, sparsity, and sign flips.",
            "- Randomized spectral features come from principal-matrix sketches with `randomized_svd`, random probe responses, and Hutchinson-style quadratic summaries.",
            "- Lightweight modeling uses a pointwise linear ridge head plus a within-scenario pairwise logistic ranking head.",
            f"- Score fusion uses pointwise/pairwise blend alpha `={float(fit_payload['alpha']):.4f}` and a separate 1-D temporal smoothing pass on the fused trajectory.",
            "",
            "## Full-Fit Ranking",
            "",
            f"- Spearman ρ: `{float(fit_metrics['spearman_rho'] or 0.0):.4f}`",
            f"- Pearson r: `{float(fit_metrics['pearson_r'] or 0.0):.4f}`",
            f"- Kendall τ: `{float(fit_metrics['kendall_tau'] or 0.0):.4f}`",
            f"- Top-1 hit: `{int(fit_metrics['top1_hit'])}`",
            f"- Top-3 overlap: `{int(fit_metrics['top3_hit'])}`",
            "",
            _markdown_table(["Pred Rank", "Checkpoint", "Pred Score", "True Accuracy"], ranking_rows),
            "",
            "## OOF Quality",
            "",
            f"- Spearman ρ: `{float(oof_metrics['spearman_rho'] or 0.0):.4f}`",
            f"- Pearson r: `{float(oof_metrics['pearson_r'] or 0.0):.4f}`",
            f"- Kendall τ: `{float(oof_metrics['kendall_tau'] or 0.0):.4f}`",
            f"- Top-1 hit: `{int(oof_metrics['top1_hit'])}`",
            f"- Top-3 overlap: `{int(oof_metrics['top3_hit'])}`",
            "",
            "## Most Predictive Layers",
            "",
            _markdown_table(["Layer", "Importance"], layer_table_rows),
            "",
            "## Random Probe Layers",
            "",
            f"- Later-half checkpoints above median accuracy are treated as `late-good`: `{', '.join(late_good) if late_good else 'n/a'}`.",
            "- The table below sums importance only over probe-response features.",
            "",
            _markdown_table(["Layer", "Probe Importance"], probe_layer_table_rows),
            "",
            "## Module Dependence (Math-Heavy Scenario)",
            "",
            "- Only a math-heavy RL scenario is locally available, so this section is within-scenario rather than cross-scenario.",
            "- Higher module importance means the weight-only fallback leaned more on that module family.",
            "",
            _markdown_table(["Module", "Importance"], module_table_rows),
            "",
            "## Delta-to-Prev vs Delta-to-Base",
            "",
            "- Subset ablations compare the same lightweight stack under restricted feature families.",
            "",
            _markdown_table(["Subset", "Selected", "Spearman ρ", "Pearson r", "Kendall τ"], ablation_rows),
            "",
            "## Top Features",
            "",
            _markdown_table(["Feature", "Importance", "|Spearman|"], top_feature_rows),
            "",
            "## Artifacts",
            "",
            f"- OOF CSV: `{output_paths['oof_csv']}`",
            f"- Feature importance CSV: `{output_paths['importance_csv']}`",
            f"- Feature frame CSV: `{output_paths['feature_frame_csv']}`",
            f"- Report: `{output_paths['report_md']}`",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(args.torch_threads))

    truth_map = _load_truth_map(args.truth_json)
    checkpoints = _discover_checkpoints(args.model_root, truth_map)
    tie_word_embeddings = bool(_load_config_flag(checkpoints[0].model_dir, "tie_word_embeddings", True))

    feature_df = _extract_feature_frame(
        checkpoints,
        args,
        scenario_name=str(args.scenario_name),
        tie_word_embeddings=tie_word_embeddings,
    )
    if feature_df["true_accuracy"].isna().any():
        raise ValueError("Current report generation requires true_accuracy for all checkpoints.")

    y = np.asarray(feature_df["true_accuracy"], dtype=np.float64)
    selected_cols, feature_rank_df = _select_features(feature_df, y, feature_cap=int(args.feature_cap))
    oof_df = _run_loo(
        feature_df,
        selected_cols,
        ridge_alpha=float(args.ridge_alpha),
        logreg_c=float(args.logreg_c),
        smooth_lambda=float(args.smooth_lambda),
    )
    fit_df, fit_payload = _fit_full_scores(
        feature_df,
        selected_cols,
        ridge_alpha=float(args.ridge_alpha),
        logreg_c=float(args.logreg_c),
        smooth_lambda=float(args.smooth_lambda),
    )

    oof_merged = oof_df.merge(
        fit_df.drop(columns=["scenario", "true_accuracy", "order_index", "step"]),
        on="checkpoint_name",
        how="left",
    ).sort_values(["order_index", "checkpoint_name"])
    feature_importance_df = _build_feature_importance(feature_df, feature_rank_df, selected_cols, fit_payload)
    ablation_df = _compute_ablation_metrics(
        feature_df,
        ridge_alpha=float(args.ridge_alpha),
        logreg_c=float(args.logreg_c),
        smooth_lambda=float(args.smooth_lambda),
        feature_cap=int(args.feature_cap),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_csv = output_dir / "weight_spectral_oof.csv"
    importance_csv = output_dir / "weight_spectral_feature_importance.csv"
    feature_frame_csv = output_dir / "weight_spectral_feature_frame.csv"
    report_md = output_dir / "weight_spectral_report.md"
    output_paths = {
        "oof_csv": oof_csv,
        "importance_csv": importance_csv,
        "feature_frame_csv": feature_frame_csv,
        "report_md": report_md,
    }

    oof_merged.to_csv(oof_csv, index=False)
    feature_importance_df.to_csv(importance_csv, index=False)
    feature_df.to_csv(feature_frame_csv, index=False)
    report_text = _build_report(
        scenario_name=str(args.scenario_name),
        feature_df=feature_df,
        oof_df=oof_df,
        fit_df=fit_df,
        feature_importance_df=feature_importance_df,
        ablation_df=ablation_df,
        fit_payload=fit_payload,
        output_paths=output_paths,
        tie_word_embeddings=tie_word_embeddings,
        exact_drift=bool(args.exact_drift),
    )
    report_md.write_text(report_text, encoding="utf-8")

    print(f"[weight_spectral] wrote {oof_csv}")
    print(f"[weight_spectral] wrote {importance_csv}")
    print(f"[weight_spectral] wrote {feature_frame_csv}")
    print(f"[weight_spectral] wrote {report_md}")


if __name__ == "__main__":
    main()
