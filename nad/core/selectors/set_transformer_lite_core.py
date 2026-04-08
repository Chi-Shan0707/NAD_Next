from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SetTransformerLiteConfig:
    pooling: str = "mean"
    model_dim: int = 16
    num_heads: int = 2
    ff_hidden_dim: int = 32
    head_hidden_dim: int = 8
    epochs: int = 120
    lr: float = 2e-3
    weight_decay: float = 1e-4
    pairwise_aux_weight: float = 0.0
    seed: int = 42

    def validate(self) -> "SetTransformerLiteConfig":
        if self.pooling not in {"mean", "max"}:
            raise ValueError(f"Unknown SetTransformerLite pooling: {self.pooling}")
        if int(self.model_dim) <= 0:
            raise ValueError("model_dim must be positive")
        if int(self.num_heads) <= 0:
            raise ValueError("num_heads must be positive")
        if int(self.model_dim) % int(self.num_heads) != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if int(self.ff_hidden_dim) <= 0:
            raise ValueError("ff_hidden_dim must be positive")
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


def _build_network(config: SetTransformerLiteConfig, input_dim: int):
    import torch
    from torch import nn

    class _TorchSetTransformerLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(int(input_dim), int(config.model_dim))
            self.attn = nn.MultiheadAttention(
                embed_dim=int(config.model_dim),
                num_heads=int(config.num_heads),
                batch_first=True,
                dropout=0.0,
            )
            self.norm1 = nn.LayerNorm(int(config.model_dim))
            self.ff = nn.Sequential(
                nn.Linear(int(config.model_dim), int(config.ff_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(config.ff_hidden_dim), int(config.model_dim)),
            )
            self.norm2 = nn.LayerNorm(int(config.model_dim))
            self.head = nn.Sequential(
                nn.Linear(2 * int(config.model_dim), int(config.head_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(config.head_hidden_dim), 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.input_proj(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            h = self.norm1(h + attn_out)
            ff_out = self.ff(h)
            h = self.norm2(h + ff_out)
            if config.pooling == "mean":
                pooled = h.mean(dim=1)
            elif config.pooling == "max":
                pooled = h.max(dim=1).values
            else:
                raise ValueError(f"Unsupported pooling: {config.pooling}")
            pooled_expand = pooled.unsqueeze(1).expand(-1, h.shape[1], -1)
            logits = self.head(torch.cat([h, pooled_expand], dim=-1)).squeeze(-1)
            return logits

    return _TorchSetTransformerLite()


class SetTransformerLiteScorer:
    def __init__(
        self,
        *,
        config: SetTransformerLiteConfig | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.config = (config or SetTransformerLiteConfig()).validate()
        self.model = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.training_summary: dict[str, float] = {}
        self.feature_names = list(feature_names or [])

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
            raise RuntimeError("SetTransformerLiteScorer failed to train a valid model state")

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
            raise RuntimeError("SetTransformerLiteScorer.fit() must be called before score_group()")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must have shape (N, F), got {X_arr.shape}")
        X_norm = ((X_arr - self.feature_mean[None, :]) / self.feature_std[None, :]).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_norm[None, :, :]))
        return np.asarray(logits.detach().cpu().numpy()[0], dtype=np.float64)

    def save(
        self,
        path: str | Path,
        *,
        feature_names: list[str] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        import torch

        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("SetTransformerLiteScorer.fit() must be called before save()")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.as_dict(),
            "feature_names": list(feature_names if feature_names is not None else self.feature_names),
            "feature_mean": np.asarray(self.feature_mean, dtype=np.float32),
            "feature_std": np.asarray(self.feature_std, dtype=np.float32),
            "state_dict": self.model.state_dict(),
            "training_summary": dict(self.training_summary),
        }
        if extra_payload:
            payload.update(dict(extra_payload))
        torch.save(payload, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "SetTransformerLiteScorer":
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = SetTransformerLiteConfig(**payload["config"]).validate()
        obj = cls(config=config, feature_names=list(payload.get("feature_names") or []))
        obj.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        obj.feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
        model = _build_network(config, int(obj.feature_mean.shape[0]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        obj.model = model
        obj.training_summary = dict(payload.get("training_summary") or {})
        return obj
