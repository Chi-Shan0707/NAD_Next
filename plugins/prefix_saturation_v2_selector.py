from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from nad.core.selectors.base import Selector, SelectorContext
from nad.core.selectors.code_dynamic_impl import (
    DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
    DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
    order_code_dynamic_group_indices,
)
from nad.core.selectors.code_v2_impl import (
    DEFAULT_CODE_V2_WEIGHTS,
    compute_code_v2_primary_scores,
)


class PrefixSaturationV2Selector(Selector):
    """Experimental coding selector with last-block instability replacing reflection density."""

    def __init__(
        self,
        w_prefix: float = DEFAULT_CODE_V2_WEIGHTS["prefix_best_window_quality"],
        w_settle: float = DEFAULT_CODE_V2_WEIGHTS["head_tail_gap"],
        w_tail: float = DEFAULT_CODE_V2_WEIGHTS["tail_variance"],
        w_recovery: float = DEFAULT_CODE_V2_WEIGHTS["post_reflection_recovery"],
        w_instability: float = DEFAULT_CODE_V2_WEIGHTS["last_block_instability"],
        reflection_threshold: float = DEFAULT_CODE_DYNAMIC_REFLECTION_THRESHOLD,
        reflection_lookback_slices: int = DEFAULT_CODE_DYNAMIC_REFLECTION_LOOKBACK,
        prefix_fraction: float = 0.30,
        prefix_window_tokens: int = 128,
    ):
        self.w_prefix = float(w_prefix)
        self.w_settle = float(w_settle)
        self.w_tail = float(w_tail)
        self.w_recovery = float(w_recovery)
        self.w_instability = float(w_instability)
        self.reflection_threshold = float(reflection_threshold)
        self.reflection_lookback_slices = int(reflection_lookback_slices)
        self.prefix_fraction = float(prefix_fraction)
        self.prefix_window_tokens = int(prefix_window_tokens)
        self._context: Optional[SelectorContext] = None

    def bind(self, context: SelectorContext) -> None:
        self._context = context

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        if self._context is None:
            raise RuntimeError("PrefixSaturationV2Selector requires bind() before select()")

        n = len(self._context.run_ids)
        if n <= 1:
            return 0

        scores, _, _ = compute_code_v2_primary_scores(
            self._context,
            weights={
                "prefix_best_window_quality": self.w_prefix,
                "head_tail_gap": self.w_settle,
                "tail_variance": self.w_tail,
                "post_reflection_recovery": self.w_recovery,
                "last_block_instability": self.w_instability,
            },
            reflection_threshold=self.reflection_threshold,
            reflection_lookback_slices=self.reflection_lookback_slices,
            prefix_fraction=self.prefix_fraction,
            prefix_window_tokens=self.prefix_window_tokens,
        )
        order = order_code_dynamic_group_indices(
            scores,
            D,
            run_ids=self._context.run_ids,
        )
        return int(order[0]) if order.size else 0
