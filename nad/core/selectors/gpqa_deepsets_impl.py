from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import SelectorContext
from .gpqa_pairwise_impl import (
    GPQA_PAIRWISE_FEATURE_NAMES,
    build_gpqa_pairwise_features,
    extract_gpqa_pairwise_raw,
)

GPQA_DEEPSETS_FEATURE_NAMES = list(GPQA_PAIRWISE_FEATURE_NAMES)


def default_gpqa_deepsets_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    v1_path = models_dir / "gpqa_deepsets_v1.pkl"
    if v1_path.exists():
        return v1_path
    return models_dir / "gpqa_deepsets_round1.pkl"


def extract_gpqa_deepsets_raw(context: SelectorContext) -> dict[str, np.ndarray]:
    return extract_gpqa_pairwise_raw(context)


def build_gpqa_deepsets_features(raw: dict[str, np.ndarray]) -> np.ndarray:
    return build_gpqa_pairwise_features(raw)


@dataclass(frozen=True)
class GPQADeepSetsConfig:
    pooling: str = "mean"
    hidden_dim: int = 16
    embed_dim: int = 8
    head_hidden_dim: int = 8
    epochs: int = 120
    lr: float = 2e-3
    weight_decay: float = 1e-4
    pairwise_aux_weight: float = 0.0
    seed: int = 42

    def validate(self) -> "GPQADeepSetsConfig":
        if self.pooling not in {"mean", "max"}:
            raise ValueError(f"Unknown GPQADeepSets pooling: {self.pooling}")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(self.embed_dim) <= 0:
            raise ValueError("embed_dim must be positive")
        if int(self.head_hidden_dim) <= 0:
            raise ValueError("head_hidden_dim must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.pairwise_aux_weight) < 0.0:
            raise ValueError("pairwise_aux_weight must be non-negative")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_network(config: GPQADeepSetsConfig, input_dim: int):
    import torch
    from torch import nn

    class _TorchDeepSets(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(int(input_dim), int(config.hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(config.hidden_dim), int(config.embed_dim)),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(2 * int(config.embed_dim), int(config.head_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(config.head_hidden_dim), 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            embed = self.encoder(x)
            if config.pooling == "mean":
                pooled = embed.mean(dim=1)
            elif config.pooling == "max":
                pooled = embed.max(dim=1).values
            else:
                raise ValueError(f"Unsupported pooling: {config.pooling}")
            pooled_expand = pooled.unsqueeze(1).expand(-1, embed.shape[1], -1)
            logits = self.head(torch.cat([embed, pooled_expand], dim=-1)).squeeze(-1)
            return logits

    return _TorchDeepSets()


class GPQADeepSetsScorer:
    def __init__(self, *, config: GPQADeepSetsConfig | None = None) -> None:
        self.config = (config or GPQADeepSetsConfig()).validate()
        self.model = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.training_summary: dict[str, float] = {}

    def fit_problem_batches(
        self,
        X_groups: np.ndarray,
        y_groups: np.ndarray,
        *,
        torch_threads: int | None = None,
    ) -> None:
        import torch
        import torch.nn.functional as F
        from torch import nn

        X_arr = np.asarray(X_groups, dtype=np.float32)
        y_arr = np.asarray(y_groups, dtype=np.float32)
        if X_arr.ndim != 3:
            raise ValueError(f"X_groups must have shape (B, N, F), got {X_arr.shape}")
        if y_arr.shape != X_arr.shape[:2]:
            raise ValueError(f"y_groups must have shape (B, N), got {y_arr.shape} for X_groups={X_arr.shape}")

        if torch_threads is not None and int(torch_threads) > 0:
            torch.set_num_threads(int(torch_threads))
        torch.manual_seed(int(self.config.seed))

        feature_mean = X_arr.reshape(-1, X_arr.shape[-1]).mean(axis=0, dtype=np.float32)
        feature_std = X_arr.reshape(-1, X_arr.shape[-1]).std(axis=0, dtype=np.float32)
        feature_std = np.where(feature_std > 1e-6, feature_std, 1.0).astype(np.float32)
        X_norm = ((X_arr - feature_mean[None, None, :]) / feature_std[None, None, :]).astype(np.float32)

        X_tensor = torch.from_numpy(X_norm)
        y_tensor = torch.from_numpy(y_arr)

        pos_count = float(y_arr.sum())
        neg_count = float(y_arr.size - pos_count)
        pos_weight_value = float(neg_count / max(pos_count, 1.0))
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32)

        model = _build_network(self.config, int(X_arr.shape[-1]))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_loss = float("inf")
        best_state = None

        def _pairwise_aux_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            losses = []
            for group_logits, group_labels in zip(logits, labels):
                pos_logits = group_logits[group_labels > 0.5]
                neg_logits = group_logits[group_labels <= 0.5]
                if pos_logits.numel() <= 0 or neg_logits.numel() <= 0:
                    continue
                diff = pos_logits[:, None] - neg_logits[None, :]
                losses.append(F.softplus(-diff).mean())
            if not losses:
                return logits.new_tensor(0.0)
            return torch.stack(losses).mean()

        model.train()
        for _ in range(int(self.config.epochs)):
            optimizer.zero_grad()
            logits = model(X_tensor)
            loss = criterion(logits, y_tensor)
            pairwise_aux = _pairwise_aux_loss(logits, y_tensor)
            total_loss = loss + float(self.config.pairwise_aux_weight) * pairwise_aux
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_value = float(total_loss.detach().cpu().item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("GPQADeepSetsScorer failed to train a valid model state")

        model.load_state_dict(best_state)
        model.eval()

        self.model = model
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.training_summary = {
            "train_loss": float(best_loss),
            "pos_weight": float(pos_weight_value),
            "pairwise_aux_weight": float(self.config.pairwise_aux_weight),
            "n_groups": float(X_arr.shape[0]),
            "n_runs_per_group": float(X_arr.shape[1]),
        }

    def fit(
        self,
        X_groups: np.ndarray,
        y_groups: np.ndarray,
        *,
        torch_threads: int | None = None,
    ) -> None:
        self.fit_problem_batches(X_groups, y_groups, torch_threads=torch_threads)

    def score_group(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("GPQADeepSetsScorer.fit() must be called before score_group()")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must have shape (N, F), got {X_arr.shape}")
        X_norm = ((X_arr - self.feature_mean[None, :]) / self.feature_std[None, :]).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_norm[None, :, :]))
        return np.asarray(logits.detach().cpu().numpy()[0], dtype=np.float64)

    def score_context(self, context: SelectorContext) -> np.ndarray:
        raw = extract_gpqa_deepsets_raw(context)
        X = build_gpqa_deepsets_features(raw)
        return self.score_group(X)

    def save(self, path: str | Path) -> None:
        import torch

        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("GPQADeepSetsScorer.fit() must be called before save()")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.as_dict(),
            "feature_names": list(GPQA_DEEPSETS_FEATURE_NAMES),
            "feature_mean": np.asarray(self.feature_mean, dtype=np.float32),
            "feature_std": np.asarray(self.feature_std, dtype=np.float32),
            "state_dict": self.model.state_dict(),
            "training_summary": dict(self.training_summary),
        }
        torch.save(payload, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "GPQADeepSetsScorer":
        import torch

        payload = torch.load(Path(path), map_location="cpu")
        config = GPQADeepSetsConfig(**payload["config"]).validate()
        obj = cls(config=config)
        obj.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        obj.feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
        model = _build_network(config, int(obj.feature_mean.shape[0]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        obj.model = model
        obj.training_summary = {
            str(key): float(value)
            for key, value in dict(payload.get("training_summary", {})).items()
        }
        return obj
