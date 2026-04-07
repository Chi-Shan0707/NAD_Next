"""
GPQA Group Pairwise Selector — Plugin

Loads a pre-trained GPQAPairwiseScorer and uses it to select the best run
from a group via exhaustive O(N²) pairwise scoring.

Usage (CLI):
    python3 -m nad.cli analyze \
        --cache-root <path> \
        --selectors "file:plugins/gpqa_pairwise_selector.py:GPQAPairwiseSelector" \
        --out result.json

The selector loads `models/ml_selectors/gpqa_pairwise_v1.pkl` when present,
otherwise falls back to `models/ml_selectors/gpqa_pairwise_round1.pkl`.
Pass `model_path=` to override.

Requires the model to have been trained first:
    python scripts/run_gpqa_pairwise_round1.py --cache-root <gpqa_cache> --out ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.core.selectors.gpqa_pairwise_impl import (
    GPQAPairwiseScorer,
    build_gpqa_pairwise_features_configurable,
    extract_gpqa_pairwise_raw,
)


def _default_model_path() -> Path:
    models_dir = Path(__file__).resolve().parent.parent / "models" / "ml_selectors"
    v1_path = models_dir / "gpqa_pairwise_v1.pkl"
    if v1_path.exists():
        return v1_path
    return models_dir / "gpqa_pairwise_round1.pkl"


class GPQAPairwiseSelector(Selector):
    """Full-group pairwise selector using exhaustive O(N²) Bradley-Terry scoring.

    Architecturally distinct from Extreme8/9/10 (tuple-sampling):
    this selector simultaneously observes all N runs and deterministically
    scores every ordered pair.
    """

    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        path = Path(model_path) if model_path is not None else _default_model_path()
        self._scorer: GPQAPairwiseScorer = GPQAPairwiseScorer.load(path)
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        if self._context is None:
            raise RuntimeError(
                "GPQAPairwiseSelector requires bind() to be called before select()"
            )

        n = len(self._context.run_ids)
        if n <= 1:
            return 0

        raw = extract_gpqa_pairwise_raw(self._context)
        X = build_gpqa_pairwise_features_configurable(
            raw,
            include_margin=bool(getattr(self._scorer, "include_margin", False)),
            include_dominance=bool(getattr(self._scorer, "include_dominance", False)),
        )
        scores = self._scorer.score_group(X)

        return int(np.argmax(scores))
