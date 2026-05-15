"""
utils/test_structure_reliance.py  —  Quantifies how much each Group B detector
relies on graph structure vs node attributes under feature camouflage.

For each detector in Group B (DOMINANT, AnomalyDAE, DONE, CoLA) we run
three conditions at every gamma level, for both camouflage modes:

  full        — normal data (attributes + graph)
  no_edges    — same features, self-loops only  (attributes only)
  rand_feats  — same graph, Gaussian noise features  (structure only)

Output (in results/):
  structure_reliance_raw.csv      one row per (detector, camouflage, gamma, seed, condition)
  structure_reliance.csv          aggregated — one row per (detector, camouflage, gamma)

USAGE:
  python utils/test_structure_reliance.py
  python utils/test_structure_reliance.py --detector CoLA
  python utils/test_structure_reliance.py --camouflage local
  python utils/test_structure_reliance.py --detector DOMINANT --seeds 3
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

import detectors.base  # noqa: F401
from pygod.detector import DOMINANT, AnomalyDAE, DONE, CoLA
from data.sbm import make_sbm

# ── Config ────────────────────────────────────────────────────────────────────
GAMMAS   = [round(g * 0.1, 1) for g in range(11)]  # 0.0 → 1.0
N_NODES  = 1000
N_SEEDS  = 5
EPOCHS   = 50
RESULTS  = PROJECT_ROOT / "results"
RAW_CSV  = RESULTS / "structure_reliance_raw.csv"
AGG_CSV  = RESULTS / "structure_reliance.csv"

CLASS_MAP = {
    "DOMINANT":   DOMINANT,
    "AnomalyDAE": AnomalyDAE,
    "DONE":       DONE,
    "CoLA":       CoLA,
}
ALL_CAMOUFLAGES = ["global", "local"]
# ─────────────────────────────────────────────────────────────────────────────

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--detector",   default=None, choices=list(CLASS_MAP.keys()))
parser.add_argument("--seeds",      type=int, default=N_SEEDS)
parser.add_argument("--camouflage", nargs="+", default=ALL_CAMOUFLAGES,
                    choices=ALL_CAMOUFLAGES)
args = parser.parse_args()

SINGLE      = args.detector
N_SEEDS     = args.seeds
CAMOUFLAGES = args.camouflage
run_map     = {SINGLE: CLASS_MAP[SINGLE]} if SINGLE else CLASS_MAP

print(f"\nDetectors:  {list(run_map.keys())}  |  seeds={N_SEEDS}")
print(f"Camouflage: {CAMOUFLAGES}")
RESULTS.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return float("nan")


def run_detector(cls, data: Data) -> float:
    contamination = float(data.y.sum().item()) / data.num_nodes
    model = cls(epoch=EPOCHS, verbose=0, contamination=contamination)
    model.fit(data)
    return auc(data.y.numpy(), model.decision_score_.numpy().astype(np.float32))


def no_edges(data: Data) -> Data:
    """Self-loops only — isolates attribute signal."""
    n   = data.x.shape[0]
    idx = torch.arange(n, dtype=torch.long)
    return Data(
        x          = data.x.clone(),
        edge_index = torch.stack([idx, idx], dim=0),
        y          = data.y.clone(),
    )


def rand_feats(data: Data, seed: int = 0) -> Data:
    """Random Gaussian features, real graph — isolates structural signal."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return Data(
        x          = torch.randn(data.x.shape, generator=rng),
        edge_index = data.edge_index.clone(),
        y          = data.y.clone(),
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

new_records = []

for cam in CAMOUFLAGES:
    for det_name, det_cls in run_map.items():
        print(f"\n{'='*62}")
        print(f"Detector: {det_name}  |  camouflage={cam}  |  seeds={N_SEEDS}")
        print(f"{'='*62}")
        print(f"  {'gamma':>6}  {'full':>8}  {'no_edges':>10}  {'rand_f':>8}  {'reliance':>10}")
        print("  " + "-"*52)

        for gamma in GAMMAS:
            auc_full_list, auc_bare_list, auc_rand_list = [], [], []

            for seed in range(N_SEEDS):
                data   = make_sbm(gamma=gamma, n_nodes=N_NODES,
                                  seed=seed * 100, camouflage=cam)
                a_full = run_detector(det_cls, data.clone())
                a_bare = run_detector(det_cls, no_edges(data))
                a_rand = run_detector(det_cls, rand_feats(data, seed=seed))

                auc_full_list.append(a_full)
                auc_bare_list.append(a_bare)
                auc_rand_list.append(a_rand)

                for condition, val in [
                    ("full",       a_full),
                    ("no_edges",   a_bare),
                    ("rand_feats", a_rand),
                ]:
                    new_records.append(dict(
                        detector   = det_name,
                        camouflage = cam,
                        gamma      = gamma,
                        seed       = seed,
                        n_seeds    = N_SEEDS,
                        condition  = condition,
                        auc        = val,
                    ))

            mf  = np.mean(auc_full_list)
            mb  = np.mean(auc_bare_list)
            mr  = np.mean(auc_rand_list)
            print(f"  gamma={gamma:.1f}  full={mf:.3f}   no_edges={mb:.3f}   rand={mr:.3f}   reliance={mf-mb:+.3f}")


# ── Save raw CSV ──────────────────────────────────────────────────────────────

new_df = pd.DataFrame(new_records)

if SINGLE is not None and RAW_CSV.exists():
    existing = pd.read_csv(RAW_CSV)
    mask     = (existing["detector"] == SINGLE) & (existing["camouflage"].isin(CAMOUFLAGES))
    existing = existing[~mask]
    raw      = pd.concat([existing, new_df], ignore_index=True)
    print(f"\nPatched {RAW_CSV.name}: updated {SINGLE}")
else:
    raw = new_df

raw = raw.sort_values(["detector", "camouflage", "gamma", "seed", "condition"]).reset_index(drop=True)
raw.to_csv(RAW_CSV, index=False)
print(f"Saved {RAW_CSV.name}  ({len(raw)} rows)")


# ── Aggregate ─────────────────────────────────────────────────────────────────

mean_auc = (
    raw.groupby(["detector", "camouflage", "gamma", "condition"])["auc"]
    .mean().reset_index()
    .pivot(index=["detector", "camouflage", "gamma"], columns="condition", values="auc")
    .reset_index()
)
mean_auc.columns.name = None

std_auc = (
    raw.groupby(["detector", "camouflage", "gamma", "condition"])["auc"]
    .std().reset_index()
    .pivot(index=["detector", "camouflage", "gamma"], columns="condition", values="auc")
    .reset_index()
    .rename(columns={
        "full":       "full_std",
        "no_edges":   "no_edges_std",
        "rand_feats": "rand_feats_std",
    })
)
std_auc.columns.name = None

n_seeds_col = (
    raw[raw["condition"] == "full"]
    .groupby(["detector", "camouflage", "gamma"])["seed"]
    .count().reset_index()
    .rename(columns={"seed": "n_seeds"})
)

agg = (
    mean_auc
    .merge(std_auc[["detector", "camouflage", "gamma",
                     "full_std", "no_edges_std", "rand_feats_std"]],
           on=["detector", "camouflage", "gamma"])
    .merge(n_seeds_col, on=["detector", "camouflage", "gamma"])
)

agg["reliance"]     = agg["full"] - agg["no_edges"]
agg["reliance_std"] = np.sqrt(agg["full_std"]**2 + agg["no_edges_std"]**2)
# attr_share: fraction of above-chance performance explained by attributes.
# Clipped to [-1, 1] — denominator is near-zero when full AUC ≈ 0.5 (detector
# at chance level), which makes the ratio undefined and numerically explosive.
denom               = (agg["full"] - 0.5).abs().clip(lower=1e-3)
agg["attr_share"]   = ((agg["full"] - agg["rand_feats"]) / denom).clip(-1, 1)

agg = agg[[
    "detector", "camouflage", "gamma", "n_seeds",
    "full", "full_std",
    "no_edges", "no_edges_std",
    "rand_feats", "rand_feats_std",
    "reliance", "reliance_std",
    "attr_share",
]].sort_values(["detector", "camouflage", "gamma"]).reset_index(drop=True)

agg.to_csv(AGG_CSV, index=False)
print(f"Saved {AGG_CSV.name}  ({len(agg)} rows)")


# ── Summary tables ────────────────────────────────────────────────────────────

for cam in CAMOUFLAGES:
    sub = agg[agg["camouflage"] == cam]
    print(f"\n── Reliance [{cam}] (full - no_edges) ─────────────────────────")
    print(sub.pivot(index="detector", columns="gamma", values="reliance").round(3).to_string())

print("\nDone.\n")