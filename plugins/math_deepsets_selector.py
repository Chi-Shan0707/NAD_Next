from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.core.selectors.math_deepsets_impl import (
    MathDeepSetsScorer,
    build_math_deepsets_features,
    default_math_deepsets_model_path,
)


class MathDeepSetsSelector(Selector):
    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        path = Path(model_path) if model_path is not None else default_math_deepsets_model_path()
        self._scorer: MathDeepSetsScorer = MathDeepSetsScorer.load(path)
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        if self._context is None:
            raise RuntimeError("MathDeepSetsSelector requires bind() before select()")
        if len(self._context.run_ids) <= 1:
            return 0
        X = build_math_deepsets_features(D, run_stats, context=self._context)
        scores = self._scorer.score_group(X)
        return int(np.argmax(scores))
