"""
utils/diagnose_radar_one_lof.py  —  Check Radar and ONE score direction.

Run:
    python utils/diagnose_radar_one_lof.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

import detectors.base  # noqa: F401
from data.sbm import make_sbm

# Simple graph with clear feature signal
data = make_sbm(gamma=0.0, n_nodes=1000, seed=42)
y    = data.y.numpy()
contamination = float(data.y.sum().item()) / data.num_nodes

print(f"Graph: {len(y)} nodes, {y.sum()} anomalies\n")


def check(name, scores, y):
    auc     = roc_auc_score(y, scores)
    auc_inv = roc_auc_score(y, -scores)
    std     = scores.std()

    print(f"── {name}")
    print(f"   score range : [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   score std   : {std:.4f}")
    print(f"   mean anomaly: {scores[y==1].mean():.4f}")
    print(f"   mean normal : {scores[y==0].mean():.4f}")
    print(f"   AUC         : {auc:.4f}")
    print(f"   AUC inverted: {auc_inv:.4f}")

    if std < 1e-4:
        print("   → PROBLEM: scores are uniform — detector is not learning")
    elif auc_inv > auc + 0.05:
        print(f"   → INVERTED: use -scores (real AUC = {auc_inv:.4f})")
    else:
        print("   → OK")
    print()


# ── Radar ─────────────────────────────────────────────────────────────────────
print("Checking Radar...")
try:
    from pygod.detector import Radar
    m = Radar(epoch=50, verbose=1, contamination=contamination)
    m.fit(data.clone())
    check("Radar", m.decision_score_.numpy(), y)
except Exception as e:
    print(f"Radar failed: {e}\n")


# ── ONE ───────────────────────────────────────────────────────────────────────
print("Checking ONE...")
try:
    from pygod.detector import ONE
    m = ONE(epoch=50, verbose=1, contamination=contamination)
    m.fit(data.clone())
    check("ONE", m.decision_score_.numpy(), y)
except Exception as e:
    print(f"ONE failed: {e}\n")

# ── LOF ───────────────────────────────────────────────────────────────────────
print("Checking LOF...")
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

X   = RobustScaler().fit_transform(data.x.numpy())
lof = LocalOutlierFactor(n_neighbors=200)
lof.fit(X)
scores_lof = (-lof.negative_outlier_factor_).astype(float)
check("LOF", scores_lof, y)