from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.core.selectors.code_dynamic_impl import order_code_dynamic_group_indices
from nad.core.selectors.science_dynamic_impl import (
    DEFAULT_SCIENCE_DYNAMIC_WEIGHTS,
    DEFAULT_SCIENCE_PREFIX_FRACTION,
    DEFAULT_SCIENCE_RECENCY_EXP,
    DEFAULT_SCIENCE_TAIL_FRACTION,
    DEFAULT_SCIENCE_WINDOW_TOKENS,
    compute_science_dynamic_primary_scores,
)


class ScienceCommitmentSelector(Selector):
    """Science-oriented selector with a recency-dominant baseline and late-window controls."""

    def __init__(
        self,
        w_prefix: float = DEFAULT_SCIENCE_DYNAMIC_WEIGHTS["prefix_conf_mean"],
        w_recency: float = DEFAULT_SCIENCE_DYNAMIC_WEIGHTS["recency_conf_mean"],
        w_late_worst: float = DEFAULT_SCIENCE_DYNAMIC_WEIGHTS["late_worst_window"],
        w_late_recovery: float = DEFAULT_SCIENCE_DYNAMIC_WEIGHTS["late_recovery"],
        prefix_fraction: float = DEFAULT_SCIENCE_PREFIX_FRACTION,
        tail_fraction: float = DEFAULT_SCIENCE_TAIL_FRACTION,
        recency_exp: float = DEFAULT_SCIENCE_RECENCY_EXP,
        window_tokens: int = DEFAULT_SCIENCE_WINDOW_TOKENS,
    ):
        self.w_prefix = float(w_prefix)
        self.w_recency = float(w_recency)
        self.w_late_worst = float(w_late_worst)
        self.w_late_recovery = float(w_late_recovery)
        self.prefix_fraction = float(prefix_fraction)
        self.tail_fraction = float(tail_fraction)
        self.recency_exp = float(recency_exp)
        self.window_tokens = int(window_tokens)
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        if self._context is None:
            raise RuntimeError("ScienceCommitmentSelector requires bind() before select()")

        n = len(self._context.run_ids)
        if n <= 1:
            return 0

        scores, _, _ = compute_science_dynamic_primary_scores(
            self._context,
            weights={
                "prefix_conf_mean": self.w_prefix,
                "recency_conf_mean": self.w_recency,
                "late_worst_window": self.w_late_worst,
                "late_recovery": self.w_late_recovery,
            },
            prefix_fraction=self.prefix_fraction,
            tail_fraction=self.tail_fraction,
            recency_exp=self.recency_exp,
            window_tokens=self.window_tokens,
        )
        order = order_code_dynamic_group_indices(
            scores,
            D,
            run_ids=self._context.run_ids,
        )
        return int(order[0]) if order.size else 0
