from __future__ import annotations

from pathlib import Path

from .base import SelectorContext
from .gpqa_deepsets_impl import (
    GPQA_DEEPSETS_FEATURE_NAMES,
    build_gpqa_deepsets_features,
    extract_gpqa_deepsets_raw,
)
from .set_transformer_lite_core import SetTransformerLiteConfig, SetTransformerLiteScorer


def default_gpqa_set_transformer_lite_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"
    return models_dir / "gpqa_set_transformer_lite_round1.pkl"


class GPQASetTransformerLiteScorer(SetTransformerLiteScorer):
    def __init__(self, *, config: SetTransformerLiteConfig | None = None) -> None:
        super().__init__(config=config, feature_names=GPQA_DEEPSETS_FEATURE_NAMES)

    def score_context(self, context: SelectorContext):
        raw = extract_gpqa_deepsets_raw(context)
        X = build_gpqa_deepsets_features(raw)
        return self.score_group(X)

    @classmethod
    def load(cls, path: str | Path) -> "GPQASetTransformerLiteScorer":
        loaded = SetTransformerLiteScorer.load(path)
        obj = cls(config=loaded.config)
        obj.model = loaded.model
        obj.feature_mean = loaded.feature_mean
        obj.feature_std = loaded.feature_std
        obj.training_summary = dict(loaded.training_summary)
        return obj

