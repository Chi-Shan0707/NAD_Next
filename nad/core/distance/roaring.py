from __future__ import annotations
from typing import Optional
import numpy as np

try:
    # pip install pyroaring
    from pyroaring import BitMap  # type: ignore
    HAS_ROARING = True
except Exception:  # ImportError or runtime disable
    BitMap = None  # type: ignore
    HAS_ROARING = False

def has_roaring() -> bool:
    """Whether pyroaring is available."""
    return HAS_ROARING

def to_bitmap(keys: np.ndarray) -> "BitMap":
    """
    Convert a sorted np.uint32/int array of keys to a Roaring bitmap.
    Assumes keys are non-negative and fit into 32-bit.

    Args:
        keys: Sorted numpy array of non-negative integer keys

    Returns:
        BitMap: Roaring bitmap containing the keys

    Raises:
        AssertionError: If pyroaring is not available
    """
    assert HAS_ROARING, "pyroaring is not available"
    # BitMap constructor does C-level batch insertion
    # Converting to Python list has acceptable overhead for large arrays
    # Future optimization: chunked iteration if needed
    return BitMap(keys.astype(np.uint32, copy=False).tolist())

def jaccard_distance_bitmap(
    b1: "BitMap",
    b2: "BitMap",
    c1: Optional[int] = None,
    c2: Optional[int] = None
) -> float:
    """
    Compute Jaccard distance using Roaring bitmaps.

    Distance = 1 - |A ∩ B| / |A ∪ B|

    Uses PyRoaring's native jaccard_index() method for optimal performance.
    Falls back to manual calculation if jaccard_index() is unavailable.

    Args:
        b1: Roaring bitmap for first set
        b2: Roaring bitmap for second set
        c1: Pre-computed cardinality of b1 (optional, used for edge case checks)
        c2: Pre-computed cardinality of b2 (optional, used for edge case checks)

    Returns:
        float: Jaccard distance in [0, 1]
    """
    import warnings

    # Use pre-computed cardinalities if provided (for edge case checks)
    s1 = int(c1 if c1 is not None else len(b1))
    s2 = int(c2 if c2 is not None else len(b2))

    # Edge cases
    if s1 == 0 and s2 == 0:
        return 0.0
    if s1 == 0 or s2 == 0:
        return 1.0

    # Use PyRoaring's native jaccard_index() for optimal performance
    # jaccard_index() returns similarity (0-1), so distance = 1 - similarity
    try:
        jaccard_sim = b1.jaccard_index(b2)
        return float(1.0 - jaccard_sim)
    except (AttributeError, Exception) as e:
        # Fallback: manual calculation using set identity
        warnings.warn(
            f"PyRoaring jaccard_index() failed ({type(e).__name__}), "
            f"falling back to manual calculation. "
            f"This may indicate an outdated pyroaring version.",
            RuntimeWarning,
            stacklevel=2
        )
        inter = len(b1 & b2)
        union = s1 + s2 - inter
        if union == 0:
            return 0.0
        return float(1.0 - (inter / union))

def union_many(bitmaps: "list[BitMap]") -> "BitMap":
    """
    Compute N-ary union of Roaring bitmaps using PyRoaring's optimized static method.

    This avoids creating intermediate union objects by using BitMap.union(*bitmaps),
    which internally uses CRoaring's "or-many" fast path.

    Args:
        bitmaps: Sequence of Roaring bitmaps to union

    Returns:
        BitMap: Union of all input bitmaps

    Raises:
        RuntimeWarning: If PyRoaring's union() method is unavailable (falls back)
    """
    import warnings

    if not bitmaps:
        if HAS_ROARING:
            return BitMap()
        else:
            return set()

    # Validate all inputs are BitMaps
    if not all(isinstance(bm, BitMap) for bm in bitmaps):
        warnings.warn(
            "union_many() received non-BitMap inputs, falling back to set operations. "
            "This is slower than native Roaring operations.",
            RuntimeWarning,
            stacklevel=2
        )
        # Fallback: convert to sets and union
        result = set()
        for bm in bitmaps:
            if isinstance(bm, BitMap):
                result |= set(bm)
            else:
                result |= set(bm)
        return result

    # Use PyRoaring's static N-ary union (CRoaring or-many fast path)
    try:
        return BitMap.union(*bitmaps)
    except (AttributeError, Exception) as e:
        warnings.warn(
            f"PyRoaring BitMap.union() failed ({type(e).__name__}), "
            f"falling back to iterative union. "
            f"This may indicate an outdated pyroaring version.",
            RuntimeWarning,
            stacklevel=2
        )
        # Fallback: iterative union
        result = bitmaps[0].copy()
        for bm in bitmaps[1:]:
            result |= bm
        return result

def intersection_many(bitmaps: "list[BitMap]") -> "BitMap":
    """
    Compute N-ary intersection of Roaring bitmaps using PyRoaring's optimized static method.

    This function sorts bitmaps by cardinality (smallest first) to minimize intermediate
    result sizes, then uses BitMap.intersection(*bitmaps) which internally uses CRoaring's
    "and-many" fast path.

    Args:
        bitmaps: Sequence of Roaring bitmaps to intersect

    Returns:
        BitMap: Intersection of all input bitmaps

    Raises:
        RuntimeWarning: If PyRoaring's intersection() method is unavailable (falls back)
    """
    import warnings

    if not bitmaps:
        if HAS_ROARING:
            return BitMap()
        else:
            return set()

    # Validate all inputs are BitMaps
    if not all(isinstance(bm, BitMap) for bm in bitmaps):
        warnings.warn(
            "intersection_many() received non-BitMap inputs, falling back to set operations. "
            "This is slower than native Roaring operations.",
            RuntimeWarning,
            stacklevel=2
        )
        # Fallback: convert to sets and intersect
        ordered = sorted(
            [set(bm) if isinstance(bm, BitMap) else set(bm) for bm in bitmaps],
            key=len
        )
        result = ordered[0].copy()
        for s in ordered[1:]:
            result &= s
        return result

    # Sort by cardinality (smallest first) to reduce intermediate result sizes
    ordered = sorted(bitmaps, key=len)

    # Use PyRoaring's static N-ary intersection (CRoaring and-many fast path)
    try:
        return BitMap.intersection(*ordered)
    except (AttributeError, Exception) as e:
        warnings.warn(
            f"PyRoaring BitMap.intersection() failed ({type(e).__name__}), "
            f"falling back to iterative intersection. "
            f"This may indicate an outdated pyroaring version.",
            RuntimeWarning,
            stacklevel=2
        )
        # Fallback: iterative intersection on sorted bitmaps
        result = ordered[0].copy()
        for bm in ordered[1:]:
            result &= bm
        return result
