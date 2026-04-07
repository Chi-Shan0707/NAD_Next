from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.core.selectors.gpqa_pairwise_impl import GPQAPairwiseScorer
from nad.core.selectors.science_hybrid_impl import (
    ScienceHybridConfig,
    compute_science_hybrid_decision_for_context,
    default_gpqa_pairwise_model_path,
)


class ScienceHybridSelector(Selector):
    """Narrow GPQA/science hybrid: baseline-first with optional pairwise rerank/override."""

    def __init__(
        self,
        family: str = "hard_override",
        backend: str = "mean",
        tau: float = 0.031746031746031744,
        k: int = 3,
        alpha: float = 0.50,
        m: float = 0.02,
        temperature: float = 0.75,
        model_path: Optional[str | Path] = None,
    ) -> None:
        path = Path(model_path) if model_path is not None else default_gpqa_pairwise_model_path()
        self._scorer: GPQAPairwiseScorer = GPQAPairwiseScorer.load(path)
        self._config = ScienceHybridConfig(
            family=str(family),
            backend=str(backend),
            tau=float(tau),
            k=int(k),
            alpha=float(alpha),
            m=float(m),
            temperature=float(temperature),
        ).validate()
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        if self._context is None:
            raise RuntimeError("ScienceHybridSelector requires bind() before select()")

        n = len(self._context.run_ids)
        if n <= 1:
            return 0

        decision = compute_science_hybrid_decision_for_context(
            self._context,
            np.asarray(D, dtype=np.float64),
            self._scorer,
            config=self._config,
        )
        return int(decision.hybrid_order[0]) if decision.hybrid_order.size else 0
