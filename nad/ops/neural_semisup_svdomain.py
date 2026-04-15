from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from nad.ops.earlystop_svd import (
    AVAILABILITY_FEATURES,
    FULL_FEATURE_NAMES,
    TOKEN_FEATURES,
    TRAJ_FEATURES,
    _fit_svd_transform,
    _rank_transform_matrix,
)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    DataLoader = Any
    Dataset = Any
    TORCH_AVAILABLE = False

DATASET_BASE = Dataset if TORCH_AVAILABLE else object
MODULE_BASE = nn.Module if TORCH_AVAILABLE else object


ANCHOR_POSITIONS = (0.10, 0.40, 0.70, 1.00)
OFFICIAL_SLOT_TO_ANCHOR = {
    0.10: 0.10,
    0.20: 0.10,
    0.30: 0.10,
    0.40: 0.40,
    0.50: 0.40,
    0.60: 0.40,
    0.70: 0.70,
    0.80: 0.70,
    0.90: 0.70,
    1.00: 1.00,
}
DOMAIN_TO_ID = {"math": 0, "science": 1, "coding": 2}
ID_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_ID.items()}
CANONICAL_FEATURE_NAMES = tuple(
    list(TOKEN_FEATURES)
    + list(TRAJ_FEATURES)
    + [
        "has_tok_conf",
        "has_tok_gini",
        "has_tok_neg_entropy",
        "has_tok_selfcert",
        "has_tok_logprob",
        "has_rows_bank",
    ]
)
FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(FULL_FEATURE_NAMES)}
CANONICAL_FEATURE_INDICES = tuple(FEATURE_TO_INDEX[name] for name in CANONICAL_FEATURE_NAMES)
FEATURE_FAMILY_INDEX_BLOCKS = (
    tuple(FEATURE_TO_INDEX[name] for name in TOKEN_FEATURES if name in CANONICAL_FEATURE_NAMES),
    tuple(FEATURE_TO_INDEX[name] for name in TRAJ_FEATURES if name in CANONICAL_FEATURE_NAMES),
    tuple(FEATURE_TO_INDEX[name] for name in AVAILABILITY_FEATURES if name in CANONICAL_FEATURE_NAMES),
)


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for neural semi-supervised SVDomain. "
            "Install `torch` first."
        )


def _stable_hash(text: str) -> int:
    value = 2166136261
    for char in str(text):
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def split_raw_rank_representation(x_rep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_rep, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D representation, got shape={x_arr.shape}")
    if x_arr.shape[1] % 2 != 0:
        raise ValueError(f"Expected even feature width for raw+rank representation, got {x_arr.shape[1]}")
    half = x_arr.shape[1] // 2
    return (
        np.asarray(x_arr[:, :half], dtype=np.float64),
        np.asarray(x_arr[:, half:], dtype=np.float64),
    )


@dataclass(frozen=True)
class NeuralPretrainConfig:
    latent_dim: int
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    min_delta: float = 1e-4
    feature_mask_prob: float = 0.15
    family_mask_prob: float = 0.10
    anchor_mask_prob: float = 0.10
    align_weight: float = 0.25
    future_weight: float = 0.25
    consistency_weight: float = 0.10
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.20
    random_state: int = 42
    device: str = "cpu"


@dataclass
class RunMatrix:
    raw: np.ndarray
    rank: np.ndarray
    labels: np.ndarray
    domain_names: np.ndarray
    domain_ids: np.ndarray
    dataset_names: np.ndarray
    cv_groups: np.ndarray
    rank_groups: np.ndarray
    cache_keys: np.ndarray
    sample_ids: np.ndarray
    problem_ids: np.ndarray
    anchor_positions: tuple[float, ...]
    feature_names: tuple[str, ...]

    def subset(self, mask: np.ndarray) -> "RunMatrix":
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        return RunMatrix(
            raw=np.asarray(self.raw[keep], dtype=np.float64, copy=False),
            rank=np.asarray(self.rank[keep], dtype=np.float64, copy=False),
            labels=np.asarray(self.labels[keep], dtype=np.int32, copy=False),
            domain_names=np.asarray(self.domain_names[keep], dtype=object, copy=False),
            domain_ids=np.asarray(self.domain_ids[keep], dtype=np.int64, copy=False),
            dataset_names=np.asarray(self.dataset_names[keep], dtype=object, copy=False),
            cv_groups=np.asarray(self.cv_groups[keep], dtype=object, copy=False),
            rank_groups=np.asarray(self.rank_groups[keep], dtype=object, copy=False),
            cache_keys=np.asarray(self.cache_keys[keep], dtype=object, copy=False),
            sample_ids=np.asarray(self.sample_ids[keep], dtype=np.int32, copy=False),
            problem_ids=np.asarray(self.problem_ids[keep], dtype=object, copy=False),
            anchor_positions=tuple(float(v) for v in self.anchor_positions),
            feature_names=tuple(str(v) for v in self.feature_names),
        )


@dataclass(frozen=True)
class AnchorExamples:
    x_raw: np.ndarray
    x_rank: np.ndarray
    x_rep: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    datasets: np.ndarray
    domain_names: np.ndarray


def build_run_matrix_from_feature_store(
    feature_store: list[dict[str, Any]],
    *,
    anchor_positions: tuple[float, ...] = ANCHOR_POSITIONS,
    feature_indices: Sequence[int] = CANONICAL_FEATURE_INDICES,
) -> RunMatrix:
    raw_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    domain_parts: list[np.ndarray] = []
    domain_id_parts: list[np.ndarray] = []
    dataset_parts: list[np.ndarray] = []
    cv_group_parts: list[np.ndarray] = []
    rank_group_parts: list[np.ndarray] = []
    cache_key_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    problem_parts: list[np.ndarray] = []

    feat_idx = list(int(v) for v in feature_indices)

    for payload in feature_store:
        positions = [float(v) for v in payload["positions"]]
        position_to_idx = {float(position): idx for idx, position in enumerate(positions)}
        if any(float(position) not in position_to_idx for position in anchor_positions):
            continue

        tensor = np.asarray(payload["tensor"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int32)
        sample_ids = np.asarray(payload["sample_ids"], dtype=np.int32)
        problem_ids = [str(v) for v in payload["problem_ids"]]
        problem_offsets = [int(v) for v in payload["problem_offsets"]]
        anchor_idx = [position_to_idx[float(position)] for position in anchor_positions]
        raw_tensor = np.asarray(tensor[:, anchor_idx, :][:, :, feat_idx], dtype=np.float64)
        rank_tensor = np.zeros_like(raw_tensor, dtype=np.float64)

        cv_groups_rows: list[str] = []
        rank_groups_rows: list[str] = []
        problem_rows: list[str] = []
        for problem_idx, problem_id in enumerate(problem_ids):
            start = int(problem_offsets[problem_idx])
            end = int(problem_offsets[problem_idx + 1])
            if end <= start:
                continue
            dataset_group = f"{payload['dataset_name']}::{problem_id}"
            rank_group = f"{payload['cache_key']}::{problem_id}"
            for local_anchor_idx in range(len(anchor_positions)):
                rank_tensor[start:end, local_anchor_idx, :] = _rank_transform_matrix(
                    raw_tensor[start:end, local_anchor_idx, :]
                )
            width = int(end - start)
            cv_groups_rows.extend([dataset_group] * width)
            rank_groups_rows.extend([rank_group] * width)
            problem_rows.extend([str(problem_id)] * width)

        if raw_tensor.shape[0] != len(cv_groups_rows):
            raise ValueError(
                f"Group row mismatch for cache={payload['cache_key']}: "
                f"{raw_tensor.shape[0]} vs {len(cv_groups_rows)}"
            )

        domain_name = str(payload["domain"])
        domain_id = DOMAIN_TO_ID[domain_name]
        rows = int(raw_tensor.shape[0])
        raw_parts.append(raw_tensor)
        rank_parts.append(rank_tensor)
        label_parts.append(labels)
        domain_parts.append(np.asarray([domain_name] * rows, dtype=object))
        domain_id_parts.append(np.asarray([domain_id] * rows, dtype=np.int64))
        dataset_parts.append(np.asarray([str(payload["dataset_name"])] * rows, dtype=object))
        cv_group_parts.append(np.asarray(cv_groups_rows, dtype=object))
        rank_group_parts.append(np.asarray(rank_groups_rows, dtype=object))
        cache_key_parts.append(np.asarray([str(payload["cache_key"])] * rows, dtype=object))
        sample_parts.append(sample_ids)
        problem_parts.append(np.asarray(problem_rows, dtype=object))

    if not raw_parts:
        feature_count = len(feat_idx)
        return RunMatrix(
            raw=np.zeros((0, len(anchor_positions), feature_count), dtype=np.float64),
            rank=np.zeros((0, len(anchor_positions), feature_count), dtype=np.float64),
            labels=np.zeros((0,), dtype=np.int32),
            domain_names=np.asarray([], dtype=object),
            domain_ids=np.zeros((0,), dtype=np.int64),
            dataset_names=np.asarray([], dtype=object),
            cv_groups=np.asarray([], dtype=object),
            rank_groups=np.asarray([], dtype=object),
            cache_keys=np.asarray([], dtype=object),
            sample_ids=np.zeros((0,), dtype=np.int32),
            problem_ids=np.asarray([], dtype=object),
            anchor_positions=tuple(float(v) for v in anchor_positions),
            feature_names=tuple(FULL_FEATURE_NAMES[idx] for idx in feat_idx),
        )

    return RunMatrix(
        raw=np.concatenate(raw_parts, axis=0).astype(np.float64, copy=False),
        rank=np.concatenate(rank_parts, axis=0).astype(np.float64, copy=False),
        labels=np.concatenate(label_parts, axis=0).astype(np.int32, copy=False),
        domain_names=np.concatenate(domain_parts).astype(object, copy=False),
        domain_ids=np.concatenate(domain_id_parts).astype(np.int64, copy=False),
        dataset_names=np.concatenate(dataset_parts).astype(object, copy=False),
        cv_groups=np.concatenate(cv_group_parts).astype(object, copy=False),
        rank_groups=np.concatenate(rank_group_parts).astype(object, copy=False),
        cache_keys=np.concatenate(cache_key_parts).astype(object, copy=False),
        sample_ids=np.concatenate(sample_parts).astype(np.int32, copy=False),
        problem_ids=np.concatenate(problem_parts).astype(object, copy=False),
        anchor_positions=tuple(float(v) for v in anchor_positions),
        feature_names=tuple(FULL_FEATURE_NAMES[idx] for idx in feat_idx),
    )


def filter_run_matrix(
    matrix: RunMatrix,
    *,
    domains: Optional[Sequence[str]] = None,
) -> RunMatrix:
    if domains is None:
        return matrix.subset(np.ones(matrix.labels.shape[0], dtype=bool))
    keep_domains = {str(domain) for domain in domains}
    mask = np.asarray([str(value) in keep_domains for value in matrix.domain_names.tolist()], dtype=bool)
    return matrix.subset(mask)


def subsample_run_matrix_by_group_fraction(
    matrix: RunMatrix,
    *,
    fraction: float,
    random_state: int,
    domains: Optional[Sequence[str]] = None,
) -> RunMatrix:
    if float(fraction) >= 0.999999:
        return filter_run_matrix(matrix, domains=domains)

    filtered = filter_run_matrix(matrix, domains=domains)
    if filtered.labels.shape[0] <= 0:
        return filtered

    keep_mask = np.zeros(filtered.labels.shape[0], dtype=bool)
    dataset_names = np.asarray(filtered.dataset_names, dtype=object)
    groups = np.asarray(filtered.cv_groups, dtype=object)
    for dataset_name in sorted({str(v) for v in dataset_names.tolist()}):
        dataset_mask = dataset_names == str(dataset_name)
        dataset_groups = np.unique(groups[dataset_mask])
        if dataset_groups.shape[0] <= 1:
            keep_mask[dataset_mask] = True
            continue
        rng = np.random.RandomState(int(random_state) + _stable_hash(dataset_name))
        order = rng.permutation(dataset_groups.shape[0])
        n_keep = int(round(float(fraction) * float(dataset_groups.shape[0])))
        n_keep = max(1, n_keep)
        n_keep = min(int(dataset_groups.shape[0]), n_keep)
        ordered_groups = [str(dataset_groups[idx]) for idx in order.tolist()]
        keep_groups = {group for group in ordered_groups[:n_keep]}
        dataset_groups_arr = np.asarray(groups[dataset_mask], dtype=object)
        dataset_labels_arr = np.asarray(filtered.labels[dataset_mask], dtype=np.int32)
        full_label_set = {int(value) for value in np.unique(dataset_labels_arr).tolist()}
        if len(full_label_set) >= 2:
            selected_labels = {
                int(value)
                for value in dataset_labels_arr[
                    np.asarray([str(group) in keep_groups for group in dataset_groups_arr.tolist()], dtype=bool)
                ].tolist()
            }
            if len(selected_labels) < 2:
                missing_labels = full_label_set - selected_labels
                for group in ordered_groups[n_keep:]:
                    group_mask = dataset_groups_arr == str(group)
                    group_labels = {int(value) for value in dataset_labels_arr[group_mask].tolist()}
                    if not (group_labels & missing_labels):
                        continue
                    keep_groups.add(str(group))
                    selected_labels |= group_labels
                    if len(selected_labels) >= 2:
                        break
                    missing_labels = full_label_set - selected_labels
        keep_mask[dataset_mask] = np.asarray(
            [str(group) in keep_groups for group in dataset_groups_arr.tolist()],
            dtype=bool,
        )
    return filtered.subset(keep_mask)


def extract_anchor_examples(
    matrix: RunMatrix,
    *,
    anchor_position: float,
    domains: Sequence[str],
) -> AnchorExamples:
    position = float(anchor_position)
    if position not in matrix.anchor_positions:
        raise ValueError(f"Unknown anchor position: {position}")
    pos_idx = matrix.anchor_positions.index(position)
    domain_set = {str(domain) for domain in domains}
    mask = np.asarray([str(name) in domain_set for name in matrix.domain_names.tolist()], dtype=bool)
    subset = matrix.subset(mask)
    x_raw = np.asarray(subset.raw[:, pos_idx, :], dtype=np.float64)
    x_rank = np.asarray(subset.rank[:, pos_idx, :], dtype=np.float64)
    return AnchorExamples(
        x_raw=x_raw,
        x_rank=x_rank,
        x_rep=np.concatenate([x_raw, x_rank], axis=1).astype(np.float64, copy=False),
        y=np.asarray(subset.labels, dtype=np.int32),
        groups=np.asarray(subset.cv_groups, dtype=object),
        datasets=np.asarray(subset.dataset_names, dtype=object),
        domain_names=np.asarray(subset.domain_names, dtype=object),
    )


class _RunDataset(DATASET_BASE):
    def __init__(
        self,
        raw: np.ndarray,
        rank: np.ndarray,
        domain_ids: np.ndarray,
        group_codes: np.ndarray,
    ) -> None:
        self.raw = np.asarray(raw, dtype=np.float32)
        self.rank = np.asarray(rank, dtype=np.float32)
        self.domain_ids = np.asarray(domain_ids, dtype=np.int64)
        self.group_codes = np.asarray(group_codes, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.raw.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.raw[int(idx)],
            self.rank[int(idx)],
            np.asarray(self.domain_ids[int(idx)], dtype=np.int64),
            np.asarray(self.group_codes[int(idx)], dtype=np.int64),
        )


class NeuralSemiSupBottleneck(MODULE_BASE):
    def __init__(
        self,
        *,
        feature_dim: int,
        latent_dim: int,
        domain_count: int,
        anchor_count: int,
        hidden_dim: int = 32,
        encoder_dim: int = 16,
        domain_emb_dim: int = 4,
        anchor_emb_dim: int = 4,
    ) -> None:
        require_torch()
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder_dim = int(encoder_dim)
        self.domain_emb_dim = int(domain_emb_dim)
        self.anchor_emb_dim = int(anchor_emb_dim)

        self.raw_encoder = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(encoder_dim)),
            nn.ReLU(),
        )
        self.rank_encoder = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(encoder_dim)),
            nn.ReLU(),
        )
        self.domain_embedding = nn.Embedding(int(domain_count), int(domain_emb_dim))
        self.anchor_embedding = nn.Embedding(int(anchor_count), int(anchor_emb_dim))
        self.trunk = nn.Sequential(
            nn.Linear(2 * int(encoder_dim) + int(domain_emb_dim) + int(anchor_emb_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )
        self.future_predictor = nn.Linear(
            int(latent_dim) + int(domain_emb_dim) + int(anchor_emb_dim),
            int(latent_dim),
        )
        self.raw_decoder = nn.Linear(
            int(latent_dim) + int(domain_emb_dim) + int(anchor_emb_dim),
            int(feature_dim),
        )
        self.rank_decoder = nn.Linear(
            int(latent_dim) + int(domain_emb_dim) + int(anchor_emb_dim),
            int(feature_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        trunk_last = self.trunk[-1]
        if isinstance(trunk_last, nn.Linear):
            nn.init.orthogonal_(trunk_last.weight)
            if trunk_last.bias is not None:
                nn.init.zeros_(trunk_last.bias)

    def _context(
        self,
        *,
        domain_ids: torch.Tensor,
        anchor_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.domain_embedding(domain_ids), self.anchor_embedding(anchor_ids)

    def encode_single(
        self,
        raw: torch.Tensor,
        rank: torch.Tensor,
        *,
        domain_ids: torch.Tensor,
        anchor_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_h = self.raw_encoder(raw)
        rank_h = self.rank_encoder(rank)
        domain_h, anchor_h = self._context(domain_ids=domain_ids, anchor_ids=anchor_ids)
        z = self.trunk(torch.cat([raw_h, rank_h, domain_h, anchor_h], dim=-1))
        return z, raw_h, rank_h

    def decode_single(
        self,
        z: torch.Tensor,
        *,
        domain_ids: torch.Tensor,
        anchor_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        domain_h, anchor_h = self._context(domain_ids=domain_ids, anchor_ids=anchor_ids)
        ctx = torch.cat([z, domain_h, anchor_h], dim=-1)
        return self.raw_decoder(ctx), self.rank_decoder(ctx)

    def predict_future(
        self,
        z: torch.Tensor,
        *,
        domain_ids: torch.Tensor,
        target_anchor_ids: torch.Tensor,
    ) -> torch.Tensor:
        domain_h, anchor_h = self._context(domain_ids=domain_ids, anchor_ids=target_anchor_ids)
        return self.future_predictor(torch.cat([z, domain_h, anchor_h], dim=-1))


def _make_feature_masks(
    *,
    shape: tuple[int, int, int],
    config: NeuralPretrainConfig,
    device: torch.device,
) -> torch.Tensor:
    batch_size, anchor_count, feature_dim = shape
    mask = torch.rand(shape, device=device) < float(config.feature_mask_prob)
    if float(config.family_mask_prob) > 0.0:
        for family_indices in FEATURE_FAMILY_INDEX_BLOCKS:
            if not family_indices:
                continue
            local_indices = [CANONICAL_FEATURE_NAMES.index(FULL_FEATURE_NAMES[idx]) for idx in family_indices]
            family_pick = torch.rand((batch_size, anchor_count, 1), device=device) < float(config.family_mask_prob)
            mask[:, :, local_indices] = torch.logical_or(mask[:, :, local_indices], family_pick.expand(-1, -1, len(local_indices)))
    if float(config.anchor_mask_prob) > 0.0:
        anchor_pick = torch.rand((batch_size, anchor_count, 1), device=device) < float(config.anchor_mask_prob)
        mask = torch.logical_or(mask, anchor_pick.expand(-1, -1, feature_dim))
    return mask


def _contrastive_loss(
    z_flat: torch.Tensor,
    *,
    batch_size: int,
    anchor_count: int,
    group_codes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    device = z_flat.device
    total = int(batch_size * anchor_count)
    if total <= 1:
        return torch.zeros((), device=device, dtype=z_flat.dtype)

    z_norm = F.normalize(z_flat, dim=-1)
    sim = torch.matmul(z_norm, z_norm.T) / float(temperature)
    eye = torch.eye(total, dtype=torch.bool, device=device)
    sample_ids = torch.arange(batch_size, device=device).repeat_interleave(anchor_count)
    groups = group_codes.repeat_interleave(anchor_count)
    positive_mask = sample_ids[:, None].eq(sample_ids[None, :]) & ~eye
    candidate_mask = (~eye) & (positive_mask | ~groups[:, None].eq(groups[None, :]))
    if not bool(torch.any(positive_mask)):
        return torch.zeros((), device=device, dtype=z_flat.dtype)

    sim = sim - torch.max(sim.masked_fill(~candidate_mask, -1e9), dim=1, keepdim=True).values
    exp_sim = torch.exp(sim) * candidate_mask.float()
    denom = exp_sim.sum(dim=1).clamp_min(1e-8)
    pos_mass = (exp_sim * positive_mask.float()).sum(dim=1)
    valid = positive_mask.sum(dim=1) > 0
    if not bool(torch.any(valid)):
        return torch.zeros((), device=device, dtype=z_flat.dtype)
    return -torch.mean(torch.log((pos_mass[valid] / denom[valid]).clamp_min(1e-8)))


@dataclass
class NeuralEncoderSnapshot:
    latent_dim: int
    feature_dim: int
    state_dict: dict[str, np.ndarray]
    domain_names: tuple[str, ...] = ("math", "science", "coding")
    anchor_positions: tuple[float, ...] = ANCHOR_POSITIONS
    hidden_dim: int = 32
    encoder_dim: int = 16
    domain_emb_dim: int = 4
    anchor_emb_dim: int = 4
    _model_cache: Any = field(default=None, init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_model_cache"] = None
        return state

    def _build_model(self) -> NeuralSemiSupBottleneck:
        require_torch()
        model = NeuralSemiSupBottleneck(
            feature_dim=int(self.feature_dim),
            latent_dim=int(self.latent_dim),
            domain_count=len(self.domain_names),
            anchor_count=len(self.anchor_positions),
            hidden_dim=int(self.hidden_dim),
            encoder_dim=int(self.encoder_dim),
            domain_emb_dim=int(self.domain_emb_dim),
            anchor_emb_dim=int(self.anchor_emb_dim),
        )
        state = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in self.state_dict.items()
        }
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _get_model(self) -> NeuralSemiSupBottleneck:
        if self._model_cache is None:
            self._model_cache = self._build_model()
        return self._model_cache

    def encode(
        self,
        raw: np.ndarray,
        rank: np.ndarray,
        *,
        domain_name: str,
        anchor_position: float,
    ) -> np.ndarray:
        require_torch()
        raw_arr = np.asarray(raw, dtype=np.float32)
        rank_arr = np.asarray(rank, dtype=np.float32)
        if raw_arr.shape != rank_arr.shape:
            raise ValueError(f"raw/rank shape mismatch: {raw_arr.shape} vs {rank_arr.shape}")
        model = self._get_model()
        domain_id = self.domain_names.index(str(domain_name))
        anchor_id = self.anchor_positions.index(float(anchor_position))
        with torch.no_grad():
            raw_tensor = torch.as_tensor(raw_arr, dtype=torch.float32)
            rank_tensor = torch.as_tensor(rank_arr, dtype=torch.float32)
            domain_tensor = torch.full((raw_tensor.shape[0],), int(domain_id), dtype=torch.long)
            anchor_tensor = torch.full((raw_tensor.shape[0],), int(anchor_id), dtype=torch.long)
            z, _, _ = model.encode_single(
                raw_tensor,
                rank_tensor,
                domain_ids=domain_tensor,
                anchor_ids=anchor_tensor,
            )
        return np.asarray(z.cpu().numpy(), dtype=np.float64)


@dataclass
class IdentityFeatureTransform:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)


@dataclass
class FrozenSVDFeatureTransform:
    scaler: StandardScaler
    svd: TruncatedSVD
    whiten: bool

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        z = self.svd.transform(self.scaler.transform(x_arr))
        if self.whiten:
            singular_values = np.asarray(self.svd.singular_values_, dtype=np.float64)
            singular_values = np.where(np.abs(singular_values) < 1e-8, 1.0, singular_values)
            z = z / singular_values
        return np.asarray(z, dtype=np.float64)


@dataclass
class FrozenNeuralFeatureTransform:
    snapshot: NeuralEncoderSnapshot
    domain_name: str
    anchor_position: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        raw, rank = split_raw_rank_representation(x)
        return self.snapshot.encode(
            raw,
            rank,
            domain_name=str(self.domain_name),
            anchor_position=float(self.anchor_position),
        )


@dataclass
class ProjectedPointwiseScorer:
    basis: Any
    scaler: StandardScaler
    clf: LogisticRegression

    def score_group(self, x: np.ndarray) -> np.ndarray:
        z = self.basis.transform(x)
        return np.asarray(self.clf.decision_function(self.scaler.transform(z)), dtype=np.float64).reshape(-1)


@dataclass
class ProjectedPairwiseScorer:
    basis: Any
    scaler: StandardScaler
    clf: LogisticRegression

    def score_group(self, x: np.ndarray) -> np.ndarray:
        z = self.basis.transform(x)
        return np.asarray(self.clf.decision_function(self.scaler.transform(z)), dtype=np.float64).reshape(-1)


@dataclass
class ProjectedMLPScorer:
    basis: Any
    scaler: StandardScaler
    clf: MLPClassifier

    def score_group(self, x: np.ndarray) -> np.ndarray:
        z = self.basis.transform(x)
        z_scaled = self.scaler.transform(z)
        if hasattr(self.clf, "predict_proba"):
            return np.asarray(self.clf.predict_proba(z_scaled)[:, 1], dtype=np.float64).reshape(-1)
        return np.asarray(self.clf.decision_function(z_scaled), dtype=np.float64).reshape(-1)


@dataclass
class LowRankAdapterScorer:
    snapshot: NeuralEncoderSnapshot
    domain_name: str
    anchor_position: float
    mean: np.ndarray
    scale: np.ndarray
    adapter_left: np.ndarray
    adapter_right: np.ndarray
    linear_weight: np.ndarray
    bias: float

    def _adapt_latent(self, z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=np.float64)
        centered = (z_arr - np.asarray(self.mean, dtype=np.float64)) / np.asarray(self.scale, dtype=np.float64)
        left = np.asarray(self.adapter_left, dtype=np.float64)
        right = np.asarray(self.adapter_right, dtype=np.float64)
        if left.size <= 0 or right.size <= 0:
            return centered
        return centered + np.matmul(np.matmul(centered, right), left.T)

    def score_group(self, x: np.ndarray) -> np.ndarray:
        raw, rank = split_raw_rank_representation(x)
        z = self.snapshot.encode(
            raw,
            rank,
            domain_name=str(self.domain_name),
            anchor_position=float(self.anchor_position),
        )
        adapted = self._adapt_latent(z)
        weight = np.asarray(self.linear_weight, dtype=np.float64).reshape(-1)
        bias = float(self.bias)
        return np.asarray(np.matmul(adapted, weight) + bias, dtype=np.float64).reshape(-1)


def fit_frozen_svd_basis(
    x_rep: np.ndarray,
    *,
    rank: int,
    whiten: bool,
    random_state: int,
) -> Optional[FrozenSVDFeatureTransform]:
    transform = _fit_svd_transform(
        np.asarray(x_rep, dtype=np.float64),
        rank=int(rank),
        whiten=bool(whiten),
        random_state=int(random_state),
    )
    if transform is None:
        return None
    return FrozenSVDFeatureTransform(
        scaler=transform["scaler"],
        svd=transform["svd"],
        whiten=bool(whiten),
    )


def _pairwise_logistic_score_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_codes: torch.Tensor,
) -> torch.Tensor:
    device = scores.device
    total = torch.zeros((), device=device, dtype=scores.dtype)
    group_count = 0
    for group_code in torch.unique(group_codes):
        mask = group_codes == group_code
        group_scores = scores[mask]
        group_labels = labels[mask]
        pos_scores = group_scores[group_labels > 0.5]
        neg_scores = group_scores[group_labels <= 0.5]
        if pos_scores.numel() <= 0 or neg_scores.numel() <= 0:
            continue
        diffs = pos_scores[:, None] - neg_scores[None, :]
        total = total + F.softplus(-diffs).mean()
        group_count += 1
    if group_count <= 0:
        return torch.zeros((), device=device, dtype=scores.dtype)
    return total / float(group_count)


def fit_lowrank_latent_scorer(
    x_rep: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    snapshot: NeuralEncoderSnapshot,
    domain_name: str,
    anchor_position: float,
    adapter_rank: int,
    pairwise_weight: float = 0.0,
    pointwise_weight: float = 1.0,
    adapter_penalty: float = 0.02,
    learning_rate: float = 0.03,
    weight_decay: float = 1e-4,
    epochs: int = 120,
    patience: int = 18,
    random_state: int = 42,
) -> LowRankAdapterScorer:
    require_torch()
    x_arr = np.asarray(x_rep, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int32).reshape(-1)
    groups_arr = np.asarray(groups, dtype=object).reshape(-1)
    if x_arr.shape[0] <= 0:
        raise ValueError("Cannot fit latent scorer on an empty dataset")
    if np.unique(y_arr).shape[0] < 2:
        raise ValueError("Latent scorer requires at least two classes")

    raw, rank = split_raw_rank_representation(x_arr)
    z = snapshot.encode(
        raw,
        rank,
        domain_name=str(domain_name),
        anchor_position=float(anchor_position),
    )
    mean = np.mean(z, axis=0).astype(np.float64, copy=False)
    scale = np.std(z, axis=0).astype(np.float64, copy=False)
    scale = np.where(scale < 1e-6, 1.0, scale)
    z_scaled = ((z - mean) / scale).astype(np.float32, copy=False)

    seed = int(random_state)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    z_tensor = torch.as_tensor(z_scaled, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_arr.astype(np.float32), dtype=torch.float32, device=device)
    unique_groups = sorted({str(group) for group in groups_arr.tolist()})
    group_to_code = {group: idx for idx, group in enumerate(unique_groups)}
    group_codes = torch.as_tensor(
        np.asarray([group_to_code[str(group)] for group in groups_arr.tolist()], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )

    latent_dim = int(z_tensor.shape[1])
    rank_dim = max(0, int(adapter_rank))
    if rank_dim > 0:
        adapter_left = torch.nn.Parameter(0.05 * torch.randn(latent_dim, rank_dim, device=device))
        adapter_right = torch.nn.Parameter(torch.zeros(latent_dim, rank_dim, device=device))
        params = [adapter_left, adapter_right]
    else:
        adapter_left = None
        adapter_right = None
        params = []
    linear_weight = torch.nn.Parameter(torch.zeros(latent_dim, device=device))
    bias = torch.nn.Parameter(torch.zeros((), device=device))
    params.extend([linear_weight, bias])
    optimizer = torch.optim.AdamW(params, lr=float(learning_rate), weight_decay=float(weight_decay))

    pos_count = max(1, int((y_arr > 0).sum()))
    neg_count = max(1, int((y_arr <= 0).sum()))
    pos_weight = torch.as_tensor(float(neg_count) / float(pos_count), dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state: dict[str, np.ndarray] | None = None
    patience_left = int(patience)
    for _ in range(int(epochs)):
        if adapter_left is not None and adapter_right is not None:
            adapted = z_tensor + torch.matmul(torch.matmul(z_tensor, adapter_right), adapter_left.T)
        else:
            adapted = z_tensor
        scores = torch.matmul(adapted, linear_weight) + bias
        point_loss = F.binary_cross_entropy_with_logits(scores, y_tensor, pos_weight=pos_weight)
        if float(pairwise_weight) > 0.0:
            pair_loss = _pairwise_logistic_score_loss(scores, y_tensor, group_codes)
        else:
            pair_loss = torch.zeros((), device=device, dtype=torch.float32)
        if adapter_left is not None and adapter_right is not None:
            adapter_reg = ((adapted - z_tensor) ** 2).mean()
        else:
            adapter_reg = torch.zeros((), device=device, dtype=torch.float32)
        total_loss = (
            float(pointwise_weight) * point_loss
            + float(pairwise_weight) * pair_loss
            + float(adapter_penalty) * adapter_reg
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        loss_value = float(total_loss.detach().cpu().item())
        if loss_value + 1e-5 < best_loss:
            best_loss = loss_value
            patience_left = int(patience)
            best_state = {
                "adapter_left": (
                    np.asarray(adapter_left.detach().cpu().numpy(), dtype=np.float32)
                    if adapter_left is not None
                    else np.zeros((latent_dim, 0), dtype=np.float32)
                ),
                "adapter_right": (
                    np.asarray(adapter_right.detach().cpu().numpy(), dtype=np.float32)
                    if adapter_right is not None
                    else np.zeros((latent_dim, 0), dtype=np.float32)
                ),
                "linear_weight": np.asarray(linear_weight.detach().cpu().numpy(), dtype=np.float32),
                "bias": np.asarray(bias.detach().cpu().numpy(), dtype=np.float32),
            }
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = {
            "adapter_left": np.zeros((latent_dim, 0), dtype=np.float32),
            "adapter_right": np.zeros((latent_dim, 0), dtype=np.float32),
            "linear_weight": np.asarray(linear_weight.detach().cpu().numpy(), dtype=np.float32),
            "bias": np.asarray(bias.detach().cpu().numpy(), dtype=np.float32),
        }

    return LowRankAdapterScorer(
        snapshot=snapshot,
        domain_name=str(domain_name),
        anchor_position=float(anchor_position),
        mean=np.asarray(mean, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        adapter_left=np.asarray(best_state["adapter_left"], dtype=np.float32),
        adapter_right=np.asarray(best_state["adapter_right"], dtype=np.float32),
        linear_weight=np.asarray(best_state["linear_weight"], dtype=np.float32),
        bias=float(np.asarray(best_state["bias"]).reshape(())),
    )


def pretrain_neural_encoder(
    matrix: RunMatrix,
    *,
    config: NeuralPretrainConfig,
) -> tuple[NeuralEncoderSnapshot, dict[str, Any]]:
    require_torch()
    if matrix.raw.shape[0] <= 0:
        raise ValueError("Cannot pretrain neural encoder on an empty matrix")

    seed = int(config.random_state)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    group_values = [str(group) for group in matrix.cv_groups.tolist()]
    group_to_code = {group: idx for idx, group in enumerate(sorted(set(group_values)))}
    group_codes = np.asarray([group_to_code[group] for group in group_values], dtype=np.int64)
    dataset = _RunDataset(
        raw=np.asarray(matrix.raw, dtype=np.float32),
        rank=np.asarray(matrix.rank, dtype=np.float32),
        domain_ids=np.asarray(matrix.domain_ids, dtype=np.int64),
        group_codes=group_codes,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        drop_last=False,
    )

    device = torch.device(str(config.device))
    model = NeuralSemiSupBottleneck(
        feature_dim=int(matrix.raw.shape[-1]),
        latent_dim=int(config.latent_dim),
        domain_count=len(DOMAIN_TO_ID),
        anchor_count=len(matrix.anchor_positions),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    best_loss = float("inf")
    best_state: dict[str, np.ndarray] | None = None
    best_epoch = 0
    patience_left = int(config.patience)
    history: list[dict[str, float]] = []

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_recon: list[float] = []
        epoch_align: list[float] = []
        epoch_future: list[float] = []
        epoch_consistency: list[float] = []
        epoch_contrastive: list[float] = []
        for raw_batch, rank_batch, domain_batch, group_batch in loader:
            raw = torch.as_tensor(raw_batch, dtype=torch.float32, device=device)
            rank = torch.as_tensor(rank_batch, dtype=torch.float32, device=device)
            domain_ids = torch.as_tensor(domain_batch, dtype=torch.long, device=device).reshape(-1)
            group_codes_t = torch.as_tensor(group_batch, dtype=torch.long, device=device).reshape(-1)
            batch_size, anchor_count, feature_dim = raw.shape
            anchor_ids = torch.arange(anchor_count, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            domain_expand = domain_ids.unsqueeze(1).expand(-1, anchor_count)
            mask = _make_feature_masks(
                shape=(int(batch_size), int(anchor_count), int(feature_dim)),
                config=config,
                device=device,
            )
            raw_masked = raw.masked_fill(mask, 0.0)
            rank_masked = rank.masked_fill(mask, 0.0)
            z_flat, raw_h, rank_h = model.encode_single(
                raw_masked.reshape(-1, feature_dim),
                rank_masked.reshape(-1, feature_dim),
                domain_ids=domain_expand.reshape(-1),
                anchor_ids=anchor_ids.reshape(-1),
            )
            raw_hat, rank_hat = model.decode_single(
                z_flat,
                domain_ids=domain_expand.reshape(-1),
                anchor_ids=anchor_ids.reshape(-1),
            )

            mask_float = mask.reshape(-1, feature_dim).float()
            denom = mask_float.sum().clamp_min(1.0)
            recon_raw = (((raw_hat - raw.reshape(-1, feature_dim)) ** 2) * mask_float).sum() / denom
            recon_rank = (((rank_hat - rank.reshape(-1, feature_dim)) ** 2) * mask_float).sum() / denom
            recon_loss = 0.5 * (recon_raw + recon_rank)

            z_view = z_flat.reshape(batch_size, anchor_count, -1)
            z_mean = z_view.mean(dim=1, keepdim=True)
            align_loss = ((z_view - z_mean) ** 2).mean()

            if anchor_count > 1:
                pred_future = model.predict_future(
                    z_view[:, :-1, :].reshape(-1, z_view.shape[-1]),
                    domain_ids=domain_expand[:, :-1].reshape(-1),
                    target_anchor_ids=anchor_ids[:, 1:].reshape(-1),
                )
                target_future = z_view[:, 1:, :].detach().reshape(-1, z_view.shape[-1])
                future_loss = F.mse_loss(pred_future, target_future)
            else:
                future_loss = torch.zeros((), device=device, dtype=z_flat.dtype)

            consistency_loss = F.mse_loss(raw_h, rank_h)
            if float(config.contrastive_weight) > 0.0:
                contrastive_loss = _contrastive_loss(
                    z_flat,
                    batch_size=int(batch_size),
                    anchor_count=int(anchor_count),
                    group_codes=group_codes_t,
                    temperature=float(config.contrastive_temperature),
                )
            else:
                contrastive_loss = torch.zeros((), device=device, dtype=z_flat.dtype)

            total_loss = (
                recon_loss
                + float(config.align_weight) * align_loss
                + float(config.future_weight) * future_loss
                + float(config.consistency_weight) * consistency_loss
                + float(config.contrastive_weight) * contrastive_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_losses.append(float(total_loss.detach().cpu().item()))
            epoch_recon.append(float(recon_loss.detach().cpu().item()))
            epoch_align.append(float(align_loss.detach().cpu().item()))
            epoch_future.append(float(future_loss.detach().cpu().item()))
            epoch_consistency.append(float(consistency_loss.detach().cpu().item()))
            epoch_contrastive.append(float(contrastive_loss.detach().cpu().item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        history.append(
            {
                "epoch": float(epoch),
                "loss": mean_loss,
                "recon": float(np.mean(epoch_recon)) if epoch_recon else float("nan"),
                "align": float(np.mean(epoch_align)) if epoch_align else float("nan"),
                "future": float(np.mean(epoch_future)) if epoch_future else float("nan"),
                "consistency": float(np.mean(epoch_consistency)) if epoch_consistency else float("nan"),
                "contrastive": float(np.mean(epoch_contrastive)) if epoch_contrastive else float("nan"),
            }
        )

        if mean_loss + float(config.min_delta) < best_loss:
            best_loss = mean_loss
            best_epoch = int(epoch)
            patience_left = int(config.patience)
            best_state = {
                key: np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
                for key, value in model.state_dict().items()
            }
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = {
            key: np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
            for key, value in model.state_dict().items()
        }

    snapshot = NeuralEncoderSnapshot(
        latent_dim=int(config.latent_dim),
        feature_dim=int(matrix.raw.shape[-1]),
        state_dict=best_state,
        domain_names=tuple(ID_TO_DOMAIN[idx] for idx in sorted(ID_TO_DOMAIN.keys())),
        anchor_positions=tuple(float(v) for v in matrix.anchor_positions),
    )
    return snapshot, {
        "latent_dim": int(config.latent_dim),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "num_samples": int(matrix.raw.shape[0]),
        "num_groups": int(np.unique(matrix.cv_groups).shape[0]),
        "history": history,
    }


def build_anchor_route_bundle(
    *,
    method_id: str,
    routes_by_domain: dict[str, dict[float, dict[str, Any]]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    positions = tuple(sorted(OFFICIAL_SLOT_TO_ANCHOR.keys()))
    domains: dict[str, dict[str, Any]] = {}
    for domain_name, anchor_routes in routes_by_domain.items():
        expanded = [
            anchor_routes[float(OFFICIAL_SLOT_TO_ANCHOR[float(position)])]
            for position in positions
        ]
        domains[str(domain_name)] = {"routes": expanded}
    return {
        "bundle_version": str(method_id),
        "created_at_utc": str(np.datetime64("now")),
        "feature_names": list(FULL_FEATURE_NAMES),
        "positions": [float(v) for v in positions],
        "anchor_positions": [float(v) for v in ANCHOR_POSITIONS],
        "protocol": dict(protocol),
        "domains": domains,
    }


def compute_latent_reuse_matrix(
    snapshot: NeuralEncoderSnapshot,
    matrix: RunMatrix,
) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    vectors: list[np.ndarray] = []
    for domain_name in ("math", "science", "coding"):
        domain_mask = np.asarray([str(v) == domain_name for v in matrix.domain_names.tolist()], dtype=bool)
        if not bool(np.any(domain_mask)):
            continue
        domain_matrix = matrix.subset(domain_mask)
        for anchor_position in matrix.anchor_positions:
            anchor_idx = matrix.anchor_positions.index(float(anchor_position))
            raw = np.asarray(domain_matrix.raw[:, anchor_idx, :], dtype=np.float64)
            rank = np.asarray(domain_matrix.rank[:, anchor_idx, :], dtype=np.float64)
            if raw.shape[0] <= 0:
                continue
            z = snapshot.encode(
                raw,
                rank,
                domain_name=domain_name,
                anchor_position=float(anchor_position),
            )
            names.append(f"{domain_name}@{int(round(float(anchor_position) * 100.0))}")
            vectors.append(np.mean(z, axis=0))

    if not vectors:
        return [], np.zeros((0, 0), dtype=np.float64)
    mat = np.vstack(vectors).astype(np.float64, copy=False)
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.where(denom < 1e-8, 1.0, denom)
    norm = mat / denom
    sim = np.matmul(norm, norm.T)
    return names, np.asarray(sim, dtype=np.float64)


def decoder_feature_strengths(
    snapshot: NeuralEncoderSnapshot,
    *,
    feature_names: Sequence[str] = CANONICAL_FEATURE_NAMES,
) -> dict[str, float]:
    model = snapshot._get_model()
    raw_w = np.asarray(model.raw_decoder.weight.detach().cpu().numpy(), dtype=np.float64)
    rank_w = np.asarray(model.rank_decoder.weight.detach().cpu().numpy(), dtype=np.float64)
    latent_slice = slice(0, int(snapshot.latent_dim))
    raw_latent = raw_w[:, latent_slice]
    rank_latent = rank_w[:, latent_slice]
    strengths = np.sqrt(np.sum(raw_latent ** 2, axis=1) + np.sum(rank_latent ** 2, axis=1))
    if strengths.shape[0] != len(feature_names):
        raise ValueError(
            f"Decoder/feature mismatch: decoder={strengths.shape[0]} vs feature_names={len(feature_names)}"
        )
    return {
        str(name): float(strength)
        for name, strength in zip(feature_names, strengths.tolist())
    }
