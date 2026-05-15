"""
detectors/base.py  —  Base class + PyGOD compatibility patch.

The NeighborLoader patch makes PyGOD work without pyg-lib or torch-sparse
by replacing mini-batch loading with a simple full-batch loader.
This module must be imported before any pygod.detector import.
"""

import numpy as np
import torch
from torch_geometric.data import Data

# ── PyGOD compatibility patch ────────────────────────────────────────────────
import torch_geometric.loader.neighbor_loader as _nl

class _FullBatch:
    """Yields the whole graph as a single batch (no pyg-lib needed)."""
    def __init__(self, data, num_neighbors, batch_size=None, **kw):
        data.n_id       = torch.arange(data.x.shape[0])
        data.batch_size = data.x.shape[0]
        self._data      = data
    def __iter__(self): yield self._data
    def __len__(self):  return 1

_nl.NeighborLoader = _FullBatch

import pygod.detector.base as _pgbase
_pgbase.NeighborLoader = _FullBatch
# ─────────────────────────────────────────────────────────────────────────────


class Detector:
    """
    Common interface for all detectors.

    Attributes
    ----------
    name  : str   display name, e.g. "DOMINANT"
    group : str   "none" | "indirect" | "direct"
                  describes how (if at all) the detector uses graph structure
    """

    def __init__(self, name: str, group: str):
        self.name  = name
        self.group = group

    def fit_predict(self, data: Data) -> np.ndarray:
        """
        Fit on `data` and return anomaly scores (np.ndarray, float32).
        Higher score = more anomalous.
        """
        raise NotImplementedError