#!/usr/bin/env python3
"""Problem catalog helpers shared by visualization front-ends."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np

RunEntry = Dict[str, object]
# Support both int and str problem_ids (e.g., "gpqa-0" or "123")
ProblemCatalog = Dict[Union[int, str], Dict[str, list]]


def build_problem_catalog(index: Mapping[str, np.ndarray],
                          correctness_map: Optional[Dict[Tuple[Union[int, str], int], bool]] = None
                          ) -> ProblemCatalog:
    """Group runs by problem using the lightweight index structure."""
    sample_ids = index["sample_ids"]
    samples_row_ptr = index["samples_row_ptr"]

    problem_ids = index.get("problem_ids")
    if problem_ids is None or len(problem_ids) != len(sample_ids):
        problem_ids = np.zeros_like(sample_ids, dtype=np.int32)

    run_indices = index.get("run_indices")
    if run_indices is None or len(run_indices) != len(sample_ids):
        run_indices = np.arange(len(sample_ids), dtype=np.int32)

    catalog: ProblemCatalog = defaultdict(lambda: {"correct_runs": [], "incorrect_runs": []})

    for i in range(len(sample_ids)):
        sample_id = int(sample_ids[i])
        # Keep problem_id as string to support formats like "gpqa-0" or "123"
        problem_id = str(problem_ids[i])
        run_index = int(run_indices[i]) if i < len(run_indices) else i

        start_row = int(samples_row_ptr[i])
        end_row = int(samples_row_ptr[i + 1])

        entry: RunEntry = {
            "sample_id": sample_id,
            "viz_row_start": start_row,
            "viz_row_end": end_row,
            "max_entropy_slice_index": None,
            "token_activations": None,
            "problem_id": problem_id,
            "run_index": run_index,
            "has_tokens": True,
        }

        if correctness_map and (problem_id, run_index) in correctness_map:
            is_correct = bool(correctness_map[(problem_id, run_index)])
            bucket = "correct_runs" if is_correct else "incorrect_runs"
            catalog[problem_id][bucket].append(entry)
        else:
            catalog[problem_id]["correct_runs"].append(entry)

    return dict(catalog)
