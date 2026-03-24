"""
Smart slice boundary alignment via dynamic programming.

Chooses how to merge consecutive 32-token slices into super-slices of
1–4 original slices (32, 64, 96, or 128 tokens) such that group boundaries
fall as close as possible to natural language boundary tokens (\\n, ., ?, : …).
"""
from __future__ import annotations

import numpy as np


def smart_slice_grouping(
    token_ids: np.ndarray,            # int32 (T,) — all token IDs for the run
    slice_token_row_ptr: np.ndarray,  # int64 (S+1,) — local 0-based token CSR ptr per slice
    boundary_token_ids: np.ndarray,   # int32 (N,) — token IDs that count as NL boundaries
    allowed_group_sizes: tuple = (1, 2, 3, 4),
) -> np.ndarray:                      # int32 (S,) — group ID per slice, 0-indexed
    """
    Partition S slices into contiguous groups whose sizes are drawn from
    ``allowed_group_sizes``, minimising the total distance from each group
    start boundary to the nearest boundary token.

    Parameters
    ----------
    token_ids:
        All token IDs for this run, shape (T,), dtype int32.
    slice_token_row_ptr:
        CSR-style row pointer giving the start token index of each slice
        (plus a sentinel at the end).  Shape (S+1,), dtype int64, 0-based
        within the run.
    boundary_token_ids:
        Token IDs that count as natural-language boundaries (e.g. the IDs
        for ``\\n``, ``.``, ``?``, ``:``).
    allowed_group_sizes:
        Tuple of allowed sizes (in slices) for a single group.  Defaults to
        ``(1, 2, 3, 4)`` which corresponds to 32–128 tokens per group.

    Returns
    -------
    np.ndarray of shape (S,), dtype int32.
        group_ids[i] is the 0-indexed group number assigned to slice i.
        Slices within the same group are contiguous and all get the same ID.

    Raises
    ------
    ValueError
        If S cannot be perfectly tiled by the given ``allowed_group_sizes``.
    """
    S = len(slice_token_row_ptr) - 1

    # ------------------------------------------------------------------
    # Step 1 — edge cases
    # ------------------------------------------------------------------
    if S == 0:
        return np.empty((0,), dtype=np.int32)

    min_k = min(allowed_group_sizes)
    if S < min_k:
        return np.zeros(S, dtype=np.int32)

    # ------------------------------------------------------------------
    # Step 2 — locate boundary token positions (sorted indices into the
    #          run's token sequence)
    # ------------------------------------------------------------------
    boundary_token_ids = np.asarray(boundary_token_ids, dtype=np.int32)
    boundary_positions = np.nonzero(np.isin(token_ids, boundary_token_ids))[0].astype(np.int64)

    # ------------------------------------------------------------------
    # Step 3 — compute per-slice boundary cost
    #
    # cost[j] = penalty for placing a group boundary *just before* slice j
    #          (= at token index slice_token_row_ptr[j] within the run).
    # cost[0] is always 0 (the first group boundary is free).
    # ------------------------------------------------------------------
    slice_starts = np.asarray(slice_token_row_ptr[:-1], dtype=np.int64)  # (S,)

    if len(boundary_positions) == 0:
        # No boundary tokens at all → all placements equally penalised
        cost = np.ones(S, dtype=np.float64)
    else:
        # For each slice start find the nearest boundary token distance
        idx = np.searchsorted(boundary_positions, slice_starts)  # (S,)

        right_dist = np.full(S, np.inf, dtype=np.float64)
        mask_r = idx < len(boundary_positions)
        right_dist[mask_r] = (
            boundary_positions[idx[mask_r]] - slice_starts[mask_r]
        ).astype(np.float64)

        left_dist = np.full(S, np.inf, dtype=np.float64)
        mask_l = idx > 0
        left_dist[mask_l] = (
            slice_starts[mask_l] - boundary_positions[idx[mask_l] - 1]
        ).astype(np.float64)

        cost = np.minimum(right_dist, left_dist)

    cost[0] = 0.0  # first boundary is always free

    # ------------------------------------------------------------------
    # Step 4 — DP recurrence
    #
    # dp[i]   = minimum total penalty to partition slices 0 .. i-1
    # back[i] = size k of the last group in the optimal solution for dp[i]
    # ------------------------------------------------------------------
    INF = np.inf
    dp = np.full(S + 1, INF, dtype=np.float64)
    back = np.zeros(S + 1, dtype=np.int32)
    dp[0] = 0.0

    for i in range(1, S + 1):
        for k in allowed_group_sizes:
            if k > i:
                continue
            val = dp[i - k] + cost[i - k]
            if val < dp[i]:
                dp[i] = val
                back[i] = k

    # ------------------------------------------------------------------
    # Step 5 — check feasibility
    # ------------------------------------------------------------------
    if np.isinf(dp[S]):
        raise ValueError(
            f"Cannot tile {S} slices into groups using allowed_group_sizes="
            f"{allowed_group_sizes}. "
            "Adjust allowed_group_sizes so that S can be expressed as a sum "
            "of values from that set."
        )

    # ------------------------------------------------------------------
    # Step 6 — backtrack to recover group sizes, then assign group IDs
    # ------------------------------------------------------------------
    group_sizes: list[int] = []
    pos = S
    while pos > 0:
        k = int(back[pos])
        group_sizes.append(k)
        pos -= k
    group_sizes.reverse()  # now ordered from first group to last

    result = np.empty(S, dtype=np.int32)
    idx_out = 0
    for g, size in enumerate(group_sizes):
        result[idx_out : idx_out + size] = g
        idx_out += size

    return result
