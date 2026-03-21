
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ..views.reader import RunView

# Optional Roaring Bitmaps backend for JA distance (large sets)
try:
    from .roaring import has_roaring, to_bitmap, jaccard_distance_bitmap
except Exception:
    def has_roaring() -> bool:
        return False
    def to_bitmap(_):
        raise RuntimeError("roaring backend not available")
    def jaccard_distance_bitmap(*args, **kwargs):  # type: ignore
        raise RuntimeError("roaring backend not available")

@dataclass(frozen=True)
class DistanceSpec:
    name: str              # "ja" or "wj"
    normalize: bool = False  # for wj
    # agg is already applied in RunView via ViewSpec
    num_threads: int = 16      # Number of threads for parallel computation
    assume_unique: bool = True  # Assume keys are unique within each run (faster intersect1d)
    # JA backend routing (NEW: Roaring Bitmaps for large sets)
    ja_backend: str = "auto"    # "auto" | "numpy" | "roaring"
    roaring_min_size: int = 4096  # Prefer roaring when max(|A|,|B|) >= this threshold

def _jaccard_distance(keys1: np.ndarray, keys2: np.ndarray, assume_unique: bool = False) -> float:
    """
    Vectorized Jaccard distance using NumPy's C implementation.

    Distance = 1 - |A ∩ B| / |A ∪ B|

    Args:
        keys1, keys2: Sorted numpy arrays of keys
        assume_unique: If True, assume keys are unique (faster)

    Returns:
        float: Jaccard distance [0, 1]
    """
    if keys1.size == 0 and keys2.size == 0:
        return 0.0

    # Use NumPy's optimized C implementation for set intersection
    inter_sz = np.intersect1d(keys1, keys2, assume_unique=assume_unique).size
    denom = keys1.size + keys2.size - inter_sz

    if denom == 0:
        return 0.0

    return float(1.0 - (inter_sz / denom))

def _weighted_jaccard_distance(
    keys1: np.ndarray, w1: np.ndarray,
    keys2: np.ndarray, w2: np.ndarray,
    sum_w1: float, sum_w2: float,
    assume_unique: bool = False
) -> float:
    """
    Vectorized weighted Jaccard distance using NumPy's C implementation.

    Distance = 1 - sum(min_weights) / (sum(w1) + sum(w2) - sum(min_weights))

    Uses the identity: sum(max_weights) = sum(w1) + sum(w2) - sum(min_weights)

    Args:
        keys1, keys2: Sorted numpy arrays of keys
        w1, w2: Weight arrays
        sum_w1, sum_w2: Pre-computed weight sums (for efficiency)
        assume_unique: If True, assume keys are unique (faster)

    Returns:
        float: Weighted Jaccard distance [0, 1]
    """
    if keys1.size == 0 and keys2.size == 0:
        return 0.0
    if keys1.size == 0 or keys2.size == 0:
        return 1.0

    # Use NumPy's intersect1d with return_indices to align weights
    _, i1, i2 = np.intersect1d(keys1, keys2, assume_unique=assume_unique, return_indices=True)

    if i1.size == 0:
        return 1.0

    # Use float64 for accumulation to avoid precision loss on large arrays
    min_sum = float(np.minimum(w1[i1].astype(np.float64, copy=False),
                               w2[i2].astype(np.float64, copy=False)).sum())

    denom = (float(sum_w1) + float(sum_w2) - min_sum)

    if denom <= 0.0:
        return 0.0

    return float(1.0 - (min_sum / denom))

# Legacy Python implementations (kept for reference/fallback)
def _jaccard_pair(keys1: np.ndarray, keys2: np.ndarray) -> float:
    """Original Python implementation using two-pointer merge."""
    i = j = 0
    inter = 0
    n1, n2 = keys1.size, keys2.size
    while i < n1 and j < n2:
        a, b = keys1[i], keys2[j]
        if a == b:
            inter += 1
            i += 1; j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    uni = n1 + n2 - inter
    if uni == 0: return 1.0
    return 1.0 - (inter / uni)

def _weighted_jaccard_pair(keys1: np.ndarray, w1: np.ndarray,
                           keys2: np.ndarray, w2: np.ndarray) -> float:
    """Original Python implementation using two-pointer merge."""
    i = j = 0
    s_min = 0.0
    s_max = 0.0
    n1, n2 = keys1.size, keys2.size
    while i < n1 and j < n2:
        a, b = keys1[i], keys2[j]
        if a == b:
            wa, wb = float(w1[i]), float(w2[j])
            s_min += min(wa, wb)
            s_max += max(wa, wb)
            i += 1; j += 1
        elif a < b:
            s_max += float(w1[i])
            i += 1
        else:
            s_max += float(w2[j])
            j += 1
    # tails
    if i < n1: s_max += float(w1[i:].sum())
    if j < n2: s_max += float(w2[j:].sum())
    if s_max <= 0.0: return 1.0
    return 1.0 - (s_min / s_max)

class DistanceEngine:
    def __init__(self, spec: DistanceSpec):
        self.spec = spec
        # Validate thread count
        self._num_threads = max(1, int(spec.num_threads))

    def dense_matrix(self, views: List[RunView]) -> np.ndarray:
        """
        Compute pairwise distance matrix with NumPy vectorization and parallelization.

        Args:
            views: List of RunView objects

        Returns:
            Symmetric distance matrix D[i,j] = distance(views[i], views[j])
        """
        n = len(views)
        D = np.zeros((n, n), dtype=np.float32)

        if n <= 1:
            return D

        # Pre-fetch and ensure contiguous memory layout
        keys = [np.ascontiguousarray(v.keys) for v in views]
        weights = [np.ascontiguousarray(v.weights) for v in views]
        sum_w = [float(w.sum(dtype=np.float64)) for w in weights]

        # --- JA backend selection (auto / numpy / roaring) ---
        use_roaring = False
        if self.spec.name == "ja":
            # Priority: environment variable > spec config
            backend = (os.getenv("NAD_JA_BACKEND", "") or self.spec.ja_backend).strip().lower() or "auto"

            if backend == "roaring":
                # Force roaring if available
                if has_roaring():
                    use_roaring = True
                else:
                    warnings.warn(
                        "Roaring backend requested but pyroaring is not available. "
                        "Falling back to NumPy backend. "
                        "Install with: pip install pyroaring",
                        RuntimeWarning,
                        stacklevel=2
                    )
                    use_roaring = False
            elif backend == "auto":
                # Auto: use roaring if available AND largest set >= threshold
                max_key_size = max(int(k.size) for k in keys)
                if max_key_size >= int(self.spec.roaring_min_size):
                    if has_roaring():
                        use_roaring = True
                    else:
                        warnings.warn(
                            f"Large key sets detected (max size: {max_key_size} >= {self.spec.roaring_min_size}). "
                            f"Roaring backend would improve performance but pyroaring is not available. "
                            f"Using NumPy backend instead. "
                            f"Install with: pip install pyroaring",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        use_roaring = False
                else:
                    use_roaring = False
            else:
                # "numpy" or other: use NumPy backend
                use_roaring = False

        # If using roaring, build bitmaps once and cache cardinalities
        bitmaps = None
        bcard = None
        if use_roaring:
            bitmaps = [to_bitmap(k) for k in keys]
            bcard = [len(b) for b in bitmaps]

        # Generate upper triangle pairs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        def worker(pair: Tuple[int, int]) -> Tuple[int, int, float]:
            """Compute distance for a single pair (i, j)."""
            i, j = pair

            if self.spec.name == "ja":
                # Route to Roaring bitmap or NumPy backend
                if use_roaring:
                    d = jaccard_distance_bitmap(bitmaps[i], bitmaps[j], bcard[i], bcard[j])  # type: ignore[index]
                else:
                    d = _jaccard_distance(keys[i], keys[j], self.spec.assume_unique)
            elif self.spec.name == "wj":
                # WJ always uses NumPy backend (weights not supported by roaring)
                d = _weighted_jaccard_distance(
                    keys[i], weights[i], keys[j], weights[j],
                    sum_w[i], sum_w[j], self.spec.assume_unique
                )
            else:
                raise ValueError(f"Unknown distance: {self.spec.name}")

            return (i, j, d)

        # For small matrices or single-threaded, skip parallelization overhead
        if len(pairs) < 256 or self._num_threads == 1:
            for i, j in pairs:
                _, _, d = worker((i, j))
                D[i, j] = D[j, i] = d
        else:
            # Both NumPy set ops and Roaring C-level ops release GIL, enabling true parallelization
            with ThreadPoolExecutor(max_workers=self._num_threads) as ex:
                for i, j, d in ex.map(worker, pairs, chunksize=8):
                    D[i, j] = D[j, i] = d

        return D
