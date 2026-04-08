from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.core.selectors.gpqa_deepsets_impl import (
    GPQADeepSetsScorer,
    build_gpqa_deepsets_features,
    default_gpqa_deepsets_model_path,
    extract_gpqa_deepsets_raw,
)


class GPQADeepSetsSelector(Selector):
    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        path = Path(model_path) if model_path is not None else default_gpqa_deepsets_model_path()
        self._scorer: GPQADeepSetsScorer = GPQADeepSetsScorer.load(path)
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        if self._context is None:
            raise RuntimeError("GPQADeepSetsSelector requires bind() before select()")

        n = len(self._context.run_ids)
        if n <= 1:
            return 0

        raw = extract_gpqa_deepsets_raw(self._context)
        X = build_gpqa_deepsets_features(raw)
        scores = self._scorer.score_group(X)
        return int(np.argmax(scores))
