"""
detectors/gcad.py  —  GCAD components and their combinations.

Three detectors are exposed, designed as an ablation of Prof. Ting's framework:

  IDK        Isolation Distributional Kernel on raw features.
             No graph used. Baseline: does IDK alone work?

  GCAD       Subgraph centralization + plain IsolationForest scorer.
             Graph used (centralization), but basic scorer.
             Tests: does the graph structure component help?

  GCAD+IDK   Subgraph centralization + IDK scorer. The full method.
             Tests: do graph structure AND data-dependent kernel together
             outperform either alone?

Reference: Zhuang et al., SDM 2023  (GCAD)
           Ting et al.,  KDD 2020   (IDK)
"""

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data
from torch_geometric.utils import degree

from detectors.base import Detector


# ── Shared building blocks ────────────────────────────────────────────────────

def _centralized_embeddings(data: Data, h: int) -> np.ndarray:
    """
    For each node v:
      1. Compute the mean of neighbour features across h hops (v excluded).
      2. Subtract v's own feature vector.

    The result encodes relative deviation from the local neighbourhood,
    which is harder to camouflage than absolute feature values.

    Implemented as h rounds of mean-aggregation message passing (no self-loops),
    equivalent to the h-hop neighbourhood mean in the original BFS formulation
    but fully vectorized.

    Note: repeated mean-pooling re-normalises at each hop, so for h > 1 the
    result is a weighted average that down-weights distant nodes geometrically.
    This matches the spirit of GCAD (local context) while being tractable on
    large graphs.
    """
    x          = torch.tensor(data.x.numpy(), dtype=torch.float)
    edge_index = data.edge_index          # no self-loops — v excluded from its own mean
    N          = x.size(0)

    src, dst = edge_index                 # PyG convention: message flows src → dst
    deg = degree(dst, N).clamp(min=1).view(-1,1)  # in-degree; isolated nodes get deg=1 so agg/deg=0, giving embedding=-x (intentional)
    current_x = x.clone()
    for _ in range(h):
        agg = torch.zeros_like(current_x)
        agg.index_add_(0, dst, current_x[src])
        current_x = agg / deg

    # centralize: how much do my h-hop neighbours differ from me?
    return (current_x - x).numpy()


def _idk_scores(X: np.ndarray, n_trees: int = 200, psi: int = 64, seed: int = 42) -> np.ndarray:
    """
    IDK anomaly score: 1 - mean_similarity(v, all others).

    Similarity = fraction of isolation trees where two points share the same
    leaf (Voronoi cell). Data-dependent: adapts to the actual distribution.

    Denominator per tree is psi (the subsample size each tree was built on),
    matching Ting et al. 2020.  Using (n - 1) instead would be a scaling error
    that shifts all similarities by a constant — rankings and AUC are identical,
    but the values would not match the paper's definition.
    """
    n      = X.shape[0]
    forest = IsolationForest(n_estimators=n_trees, max_samples=min(psi, n), random_state=seed)
    forest.fit(X)
    psi_actual = forest.max_samples_     # actual subsample size resolved by sklearn

    leaf_ids = np.stack([tree.apply(X) for tree in forest.estimators_], axis=1)  # (N, T)
    mean_sim = np.zeros(n, dtype=np.float64)
    for t in range(n_trees):
        _, inv, counts = np.unique(leaf_ids[:, t], return_inverse=True, return_counts=True)
        mean_sim += counts[inv] / psi_actual   # Ting et al. 2020: divide by psi, not n-1

    mean_sim /= n_trees
    return (1.0 - mean_sim).astype(np.float32)


def _ik_scores(X: np.ndarray, n_trees: int = 200, psi: int = 256, seed: int = 42) -> np.ndarray:
    """Plain IsolationForest score (path-length based, no kernel)."""
    forest = IsolationForest(n_estimators=n_trees, max_samples=min(psi, X.shape[0]), random_state=seed)
    forest.fit(X)
    return (-forest.decision_function(X)).astype(np.float32)


# ── The three detectors ───────────────────────────────────────────────────────

class IDK(Detector):
    """
    IDK on raw node features. No graph structure used at all.
    Group: none  →  expected to collapse at high gamma.
    """
    def __init__(self):
        super().__init__(name="IDK", group="none")

    def fit_predict(self, data: Data) -> np.ndarray:
        X = RobustScaler().fit_transform(data.x.numpy())
        return _idk_scores(X)


class GCAD(Detector):
    """
    Subgraph centralization + plain IsolationForest scorer.
    Graph is used (centralization encodes neighbourhood deviation),
    but the scorer is data-independent (path-length based).
    Group: indirect  →  partially resistant to feature camouflage.
    """
    def __init__(self, h: int = 2):
        super().__init__(name="GCAD", group="indirect")
        self.h = h

    def fit_predict(self, data: Data) -> np.ndarray:
        embeddings = _centralized_embeddings(data, self.h)
        embeddings = RobustScaler().fit_transform(embeddings)
        return _ik_scores(embeddings)


class GCAD_IDK(Detector):
    """
    Subgraph centralization + IDK scorer. The full method.
    Combines graph-aware embeddings with a data-dependent kernel scorer.
    Group: indirect  →  most robust of the three to feature camouflage.
    """
    def __init__(self, h: int = 2):
        super().__init__(name="GCAD+IDK", group="indirect")
        self.h = h

    def fit_predict(self, data: Data) -> np.ndarray:
        embeddings = _centralized_embeddings(data, self.h)
        embeddings = RobustScaler().fit_transform(embeddings)
        return _idk_scores(embeddings)