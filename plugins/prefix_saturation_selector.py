"""
prefix_saturation_selector.py - Code-oriented dynamic structure selector.

This plugin now reuses the shared code-dynamic feature core instead of a
standalone hand-crafted implementation.

Features (group-internal rank-normalized):
1. prefix_best_window_quality_r   lower prefix-window tok_conf = better
2. head_tail_gap_r               tail becomes more confident = better
3. reflection_density_r          fewer reflections per log-length = better
4. tail_variance_r               lower tail variance = better
5. post_reflection_recovery_r    stronger confidence recovery after last reflection = better
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.core.selectors.code_dynamic_impl import (
    DEFAULT_CODE_DYNAMIC_WEIGHTS,
    DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    compute_code_dynamic_primary_scores,
    select_code_dynamic_best_index,
)


class PrefixSaturationSelector(Selector):
    """Code-oriented selector using shared dynamic prefix/tail/reflection features."""

    def __init__(
        self,
        w_prefix: float = DEFAULT_CODE_DYNAMIC_WEIGHTS["prefix_best_window_quality"],
        w_settle: float = DEFAULT_CODE_DYNAMIC_WEIGHTS["head_tail_gap"],
        w_refl: float = DEFAULT_CODE_DYNAMIC_WEIGHTS["reflection_density"],
        w_tail: float = DEFAULT_CODE_DYNAMIC_WEIGHTS["tail_variance"],
        w_recovery: float = DEFAULT_CODE_DYNAMIC_WEIGHTS["post_reflection_recovery"],
        reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
        reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
        prefix_fraction: float = 0.20,
        prefix_window_tokens: int = 128,
    ):
        self.w_prefix = float(w_prefix)
        self.w_settle = float(w_settle)
        self.w_refl = float(w_refl)
        self.w_tail = float(w_tail)
        self.w_recovery = float(w_recovery)
        self.reflection_threshold = float(reflection_threshold)
        self.reflection_lookback_slices = int(reflection_lookback_slices)
        self.prefix_fraction = float(prefix_fraction)
        self.prefix_window_tokens = int(prefix_window_tokens)
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        if self._context is None:
            raise RuntimeError("PrefixSaturationSelector requires bind() before select()")

        n = len(self._context.run_ids)
        if n <= 1:
            return 0

        scores, _, _ = compute_code_dynamic_primary_scores(
            self._context,
            weights={
                "prefix_best_window_quality": self.w_prefix,
                "head_tail_gap": self.w_settle,
                "reflection_density": self.w_refl,
                "tail_variance": self.w_tail,
                "post_reflection_recovery": self.w_recovery,
            },
            reflection_threshold=self.reflection_threshold,
            reflection_lookback_slices=self.reflection_lookback_slices,
            prefix_fraction=self.prefix_fraction,
            prefix_window_tokens=self.prefix_window_tokens,
        )
        return select_code_dynamic_best_index(
            scores,
            D,
            run_ids=self._context.run_ids,
        )
