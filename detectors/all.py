"""
detectors/all.py  —  All detectors for the feature camouflage stress test.

Three groups based on how each detector uses graph structure:

  GROUP A  "none"     — attributes only, no graph used at all
  GROUP B  "indirect" — uses both attributes and graph, score is feature-driven
  GROUP C  "direct"   — score derived from graph topology only

GROUP A — Attribute-only detectors
  LOF       density estimation (k-nearest neighbours)
  IDK       isolation distributional kernel  (Ting et al. 2020)
  iForest   path-length isolation forest

GROUP B — Attribute + structure detectors
  ONE       outlier-aware network embedding  (aligns structure + attribute)
  Radar     residual analysis using graph Laplacian
  DOMINANT  GCN reconstruction of attributes + adjacency
  AnomalyDAE dual autoencoder (attributes + structure)
  DONE      deep dual-objective: attribute + structure residuals
  CoLA      contrastive subgraph sampling
  GCAD      subgraph centralization + IsolationForest scorer
  GCAD+IDK  subgraph centralization + IDK scorer  (Zhuang et al. 2023)

GROUP C — Structure-only detectors
  GAE       graph autoencoder reconstructing adjacency only (recon_s=True)

Note on ONE and Radar classification:
  Both are classified as Group B (not Group A) because they explicitly use
  graph structure. ONE aligns network embeddings with attribute embeddings.
  Radar solves an optimization problem involving the graph Laplacian.
  Classifying them as attribute-only would corrupt the Group A baseline:
  if either performs well at high gamma, the conclusion that "attribute-only
  methods are fragile" would be incorrect.

Expected behaviour under increasing gamma (Condition A):
  Group A  collapses first  — pure attribute signal destroyed by camouflage
  Group B  collapses later  — graph provides partial resistance
             global camouflage: local inconsistency still detectable by GNNs
             local camouflage:  local inconsistency eliminated, Group B collapses
  Group C  stays ~0.5       — no structural signal in Condition A by design

Epoch control:
  Set SBM_EPOCHS env variable to control training epochs for PyGOD detectors.
  Default: 20 (good balance of speed and accuracy for sweeps).
  Production: SBM_EPOCHS=50 python run_sbm.py
  Development: SBM_EPOCHS=10 python run_sbm.py --conditions A --seeds 2
"""

import os
import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data

from detectors.base import Detector

# Controllable via environment variable for fast development runs.
_EPOCHS = int(os.environ.get("SBM_EPOCHS", os.environ.get("REAL_EPOCHS", "20")))

# ── GROUP A: attribute-only ───────────────────────────────────────────────────

class LOF(Detector):
    """Density-based. Scores anomalies by local density contrast."""
    def __init__(self, k: int = 200):
        super().__init__(name="LOF", group="none")
        self.k = k

    def fit_predict(self, data: Data) -> np.ndarray:
        X = RobustScaler().fit_transform(data.x.numpy())
        if X.std() < 1e-6:
            return np.zeros(data.num_nodes, dtype=np.float32)
        lof = LocalOutlierFactor(n_neighbors=min(self.k, data.num_nodes - 1))
        lof.fit(X)
        return (-lof.negative_outlier_factor_).astype(np.float32)


class IForest(Detector):
    """Path-length isolation. Scores anomalies by how quickly they are isolated."""
    def __init__(self, n_estimators: int = 200):
        super().__init__(name="iForest", group="none")
        self.n_estimators = n_estimators

    def fit_predict(self, data: Data) -> np.ndarray:
        X = RobustScaler().fit_transform(data.x.numpy())
        clf = IsolationForest(n_estimators=self.n_estimators, random_state=42)
        clf.fit(X)
        return (-clf.decision_function(X)).astype(np.float32)


# ── GROUP B & C: PyGOD wrapper ────────────────────────────────────────────────

class _PyGOD(Detector):
    """Thin wrapper around any PyGOD detector."""
    def __init__(self, name: str, group: str, cls, **kwargs):
        super().__init__(name, group)
        self._cls    = cls
        self._kwargs = kwargs

    def fit_predict(self, data: Data) -> np.ndarray:
        contamination = float(data.y.sum().item()) / data.num_nodes
        model = self._cls(
            epoch=_EPOCHS, verbose=0, contamination=contamination,
            **self._kwargs
        )
        model.fit(data)
        return model.decision_score_.numpy().astype(np.float32)


class GAEDetector(Detector):
    """
    Graph Autoencoder with featureless encoder (identity matrix as input).
    Reconstructs adjacency only (recon_s=True); the GCN encoder sees only
    node IDs, not actual attributes, matching Kipf & Welling's structure-only
    variant.
    """
    def __init__(self):
        super().__init__(name="GAE", group="direct")

    def fit_predict(self, data: Data) -> np.ndarray:
        from pygod.detector import GAE
        # Featureless encoder: identity matrix replaces node features
        featureless = Data(
            x=torch.eye(data.num_nodes, dtype=torch.float),
            edge_index=data.edge_index,
            y=data.y,
        )
        contamination = float(data.y.sum().item()) / data.num_nodes
        model = GAE(
            epoch=_EPOCHS, verbose=0,
            contamination=contamination, recon_s=True
        )
        model.fit(featureless)
        return model.decision_score_.numpy().astype(np.float32)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_detectors() -> list:
    from pygod.detector import DOMINANT, AnomalyDAE, CoLA, GAE, DONE, ONE, Radar
    from detectors.gcad import IDK, GCAD, GCAD_IDK

    return [
        # ── A: attribute only ─────────────────────────────────────────────────
        LOF(),
        IDK(),
        IForest(),

        # ── B: attribute + structure ──────────────────────────────────────────
        _PyGOD("ONE",        "indirect", ONE),    # network + attribute embedding
        _PyGOD("Radar",      "indirect", Radar),  # graph Laplacian residuals
        _PyGOD("DOMINANT",   "indirect", DOMINANT),
        _PyGOD("AnomalyDAE", "indirect", AnomalyDAE),
        _PyGOD("DONE",       "indirect", DONE),
        _PyGOD("CoLA",       "indirect", CoLA),
        GCAD(),
        GCAD_IDK(),

        # ── C: structure only ─────────────────────────────────────────────────
        GAEDetector(),
    ]