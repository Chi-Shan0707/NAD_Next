"""
Graph topology features derived from the 64-run Jaccard distance matrix.

Inspired by CodeCircuit: correct-answer clusters tend to be denser and more
connected in the activation similarity graph.  Three features are extracted
per run from the full n×n distance matrix D:

  local_cc         : local clustering coefficient — fraction of neighbors
                     that are mutual neighbors (higher = better)
  norm_degree      : normalised degree — fraction of all other runs within eps
                     (higher = better)
  cluster_size_frac: DBSCAN cluster size / n; noise points get 1/n
                     (higher = better)

Adaptive eps = 30th percentile of off-diagonal distances (same heuristic as
DBSCANMedoidSelector in impl.py).

`GraphDegreeSelector` is a zero-training baseline that selects argmax(norm_degree).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Selector, SelectorContext

GRAPH_TOPO_FEATURE_NAMES = [
    "local_cc",
    "norm_degree",
    "cluster_size_frac",
]


def _dbscan_cluster_labels(
    D: np.ndarray,
    eps: float,
    min_samples: int = 3,
) -> np.ndarray:
    """
    BFS DBSCAN returning integer label array (-1 = noise).

    Mirrors the BFS kernel from DBSCANMedoidSelector (impl.py L103-154).
    """
    n = D.shape[0]
    nbr = D <= eps
    core = np.asarray(nbr.sum(axis=1) >= int(min_samples))
    labels = np.full(n, -1, dtype=np.int64)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i] or not core[i]:
            continue
        queue = [i]
        visited[i] = True
        while queue:
            u = queue.pop()
            labels[u] = cluster_id
            if core[u]:
                for v in np.where(nbr[u])[0]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(int(v))
        cluster_id += 1
    return labels


def extract_graph_topo_raw(
    D: np.ndarray,
    eps: float | None = None,
    min_samples: int = 3,
) -> dict[str, np.ndarray]:
    """
    Compute per-run graph topology features from distance matrix D.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Pairwise distance matrix (Jaccard distances, 0 = identical).
    eps : float or None
        Neighbourhood radius.  If None, uses the 30th percentile of
        all upper-triangular pairwise distances (adaptive).
    min_samples : int
        Minimum neighbourhood size for DBSCAN core points.

    Returns
    -------
    dict with keys "local_cc", "norm_degree", "cluster_size_frac",
    each an np.ndarray of shape (n,).
    """
    n = D.shape[0]
    zeros = {name: np.zeros(n, dtype=np.float64) for name in GRAPH_TOPO_FEATURE_NAMES}
    if n <= 1:
        zeros["cluster_size_frac"][:] = 1.0 / max(n, 1)
        return zeros

    # Adaptive eps
    triu_idx = np.triu_indices(n, k=1)
    triu = D[triu_idx]
    if eps is None:
        eps = float(np.quantile(triu, 0.30)) if triu.size else 0.5

    adj = (D <= float(eps)).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    degree = adj.sum(axis=1)  # (n,)

    # local clustering coefficient via matrix multiply
    # adj @ adj gives number of closed walks of length 2 through each node
    AA = adj @ adj                       # (n, n)
    triangles = np.diag(AA)              # (n,) — length-2 paths i→v→i
    denom = degree * (degree - 1.0)
    safe_denom = np.where(denom > 0, denom, 1.0)
    local_cc = np.where(denom > 0, triangles / safe_denom, 0.0)

    norm_degree = degree / max(n - 1, 1)

    labels = _dbscan_cluster_labels(D, eps=float(eps), min_samples=int(min_samples))
    cluster_size_frac = np.ones(n, dtype=np.float64) / n  # noise default
    max_label = int(labels.max())
    for cid in range(max_label + 1):
        mask = labels == cid
        if mask.any():
            cluster_size_frac[mask] = float(mask.sum()) / n

    return {
        "local_cc": local_cc,
        "norm_degree": norm_degree,
        "cluster_size_frac": cluster_size_frac,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Zero-training baseline selector
# ─────────────────────────────────────────────────────────────────────────────

class GraphDegreeSelector(Selector):
    """
    Zero-training graph topology baseline.

    Selects the run with the highest normalised in-graph degree (norm_degree).
    High degree = many activation-similar neighbours = likely in the consensus
    cluster of correct answers.

    Purpose: validate that the graph topology features carry additive signal
    beyond dc_r / dbscan-medoid before training Extreme10.
    """

    name = "graph-degree"

    def __init__(
        self,
        eps: float | None = None,
        min_samples: int = 3,
    ) -> None:
        self.eps = eps
        self.min_samples = int(min_samples)

    def bind(self, context: SelectorContext) -> None:
        pass  # no per-run data needed; D is available in select()

    def select(self, D: np.ndarray, run_stats: dict) -> int:
        graph_raw = extract_graph_topo_raw(
            D,
            eps=self.eps,
            min_samples=self.min_samples,
        )
        scores = graph_raw["norm_degree"]
        if not np.isfinite(scores).any():
            return 0
        return int(np.argmax(scores))
