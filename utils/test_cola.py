"""
utils/test_cola.py  —  Diagnostic tests for CoLA's behaviour under feature camouflage.

Three tests:
  Test 1 — Feature distance check
    Verifies camouflage is working: anomaly features should move
    closer to the normal mean as gamma increases.
    Runs both global and local camouflage modes.

  Test 2 — CoLA with NO edges (self-loops only)
    Removes structural signal to isolate attribute contribution.
    - positive delta → graph helps CoLA
    - negative delta → graph hurts CoLA

  Test 3 — CoLA data.x mutation check
    Checks whether CoLA modifies data.x in-place during fit().

Output (in results/):
  cola_feature_distance.csv     feature distance per (gamma, camouflage)
  cola_graph_vs_noedge.csv      AUC full vs bare per (gamma, camouflage)

USAGE:
    python utils/test_cola.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

import detectors.base  # noqa: F401
from pygod.detector import CoLA
from data.sbm import make_sbm

GAMMAS  = [round(g * 0.1, 1) for g in range(11)]  # 0.0 → 1.0
N_NODES = 1000
N_SEEDS = 5   # Test 2 seeds; Test 1 and 3 use a single deterministic seed
SEED    = 42
EPOCHS  = 50
RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def auc_from_scores(y_true, scores):
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return float("nan")


def run_cola(data: Data) -> float:
    contamination = float(data.y.sum().item()) / data.num_nodes
    model = CoLA(epoch=EPOCHS, verbose=0, contamination=contamination)
    model.fit(data)
    return auc_from_scores(data.y.numpy(),
                           model.decision_score_.numpy().astype(np.float32))


def self_loops_only(data: Data) -> Data:
    """Self-loops only — isolates attribute signal."""
    n   = data.x.shape[0]
    idx = torch.arange(n, dtype=torch.long)
    return Data(
        x          = data.x.clone(),
        edge_index = torch.stack([idx, idx], dim=0),
        y          = data.y.clone(),
    )


def feature_distance(data: Data) -> float:
    mean_n = data.x[data.y == 0].mean(0)
    mean_a = data.x[data.y == 1].mean(0)
    return (mean_a - mean_n).norm().item()


# ── Test 1: Feature distance ──────────────────────────────────────────────────

print("\n" + "="*62)
print("TEST 1 — Feature distance: global vs local camouflage")
print("="*62)
print(f"  {'gamma':>6}  {'global_dist':>12}  {'local_dist':>11}")
print("  " + "-"*36)

dist_records = []
for gamma in GAMMAS:
    dg = feature_distance(make_sbm(gamma=gamma, n_nodes=N_NODES, seed=SEED, camouflage="global"))
    dl = feature_distance(make_sbm(gamma=gamma, n_nodes=N_NODES, seed=SEED, camouflage="local"))
    print(f"  gamma={gamma:.1f}  global={dg:6.3f}       local={dl:6.3f}")
    dist_records.append(dict(gamma=gamma, global_dist=dg, local_dist=dl))

dist_df   = pd.DataFrame(dist_records)
dist_path = RESULTS / "cola_feature_distance.csv"
dist_df.to_csv(dist_path, index=False)
print(f"\nSaved: {dist_path.name}")


# ── Test 2: CoLA full graph vs no edges ──────────────────────────────────────

print("\n" + "="*62)
print(f"TEST 2 — CoLA AUC: full graph vs self-loops only  (N_SEEDS={N_SEEDS})")
print("="*62)

records = []
for cam in ["global", "local"]:
    print(f"\n  Camouflage: {cam}")
    print(f"  {'gamma':>6}  {'full':>8}  {'±':>6}  {'no_edges':>10}  {'±':>6}  {'delta':>8}")
    print("  " + "-"*54)
    for gamma in GAMMAS:
        auc_full_list, auc_bare_list = [], []
        for s in range(N_SEEDS):
            seed      = s * 100
            data      = make_sbm(gamma=gamma, n_nodes=N_NODES, seed=seed, camouflage=cam)
            data_bare = self_loops_only(data)
            auc_full_list.append(run_cola(data))
            auc_bare_list.append(run_cola(data_bare))

        auc_full = float(np.mean(auc_full_list))
        auc_bare = float(np.mean(auc_bare_list))
        std_full = float(np.std(auc_full_list))
        std_bare = float(np.std(auc_bare_list))
        delta    = auc_full - auc_bare
        print(f"  gamma={gamma:.1f}  full={auc_full:.3f} ±{std_full:.3f}  "
              f"bare={auc_bare:.3f} ±{std_bare:.3f}  d={delta:+.3f}")
        records.append(dict(
            camouflage=cam, gamma=gamma,
            auc_full=auc_full, auc_full_std=std_full,
            auc_bare=auc_bare, auc_bare_std=std_bare,
            delta=delta,
        ))

df2       = pd.DataFrame(records)
cola_path = RESULTS / "cola_graph_vs_noedge.csv"
df2.to_csv(cola_path, index=False)
print(f"\nSaved: {cola_path.name}")

# ── Test 3: In-place mutation ─────────────────────────────────────────────────

print("\n" + "="*62)
print("TEST 3 — Does CoLA mutate data.x in-place?")
print("="*62)

data     = make_sbm(gamma=0.3, n_nodes=N_NODES, seed=SEED)
x_before = data.x.clone()
_        = run_cola(data)
max_diff = (data.x - x_before).abs().max().item()
mutated  = max_diff > 1e-6

print(f"  Max element-wise change in data.x: {max_diff:.2e}")
print(f"  {'WARNING: CoLA MUTATES data.x — use data.clone()' if mutated else 'OK: no in-place mutation.'}")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*62)
print("SUMMARY")
print("="*62)

for cam in ["global", "local"]:
    dists    = [feature_distance(make_sbm(gamma=g, n_nodes=N_NODES, seed=SEED, camouflage=cam)) for g in GAMMAS]
    monotone = all(dists[i] >= dists[i+1] for i in range(len(dists)-1))
    print(f"  Test 1 [{cam}] — Monotone: {'YES' if monotone else 'NO'}")

for cam in ["global", "local"]:
    mean_delta = df2[df2["camouflage"] == cam]["delta"].mean()
    if abs(mean_delta) < 0.05:
        verdict = "CoLA largely ignores graph"
    elif mean_delta > 0.05:
        verdict = "CoLA benefits from graph"
    else:
        verdict = "CoLA hurt by graph"
    print(f"  Test 2 [{cam}] — Mean delta: {mean_delta:+.3f}  →  {verdict}")

print(f"  Test 3         — Mutation: {'YES — FIX NEEDED' if mutated else 'NO — OK'}")
print("\nDone.\n")