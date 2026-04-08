from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.core.selectors.code_deepsets_impl import (
    CodeDeepSetsScorer,
    default_code_deepsets_model_path,
)


class CodeDeepSetsSelector(Selector):
    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        path = Path(model_path) if model_path is not None else default_code_deepsets_model_path()
        self._scorer: CodeDeepSetsScorer = CodeDeepSetsScorer.load(path)
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        if self._context is None:
            raise RuntimeError("CodeDeepSetsSelector requires bind() before select()")
        if len(self._context.run_ids) <= 1:
            return 0
        return int(self._scorer.select_best_index(self._context, D))
