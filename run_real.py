import os, sys

os.environ['REAL_EPOCHS'] = '50'
sys.path.insert(0, '.')

import torch, torch_geometric.data.storage

# Required for PyTorch 2.x safe deserialization of PyG Data objects
torch.serialization.add_safe_globals([torch_geometric.data.storage.GlobalStorage])
import numpy as np, random
from data.real import load
from detectors.all import get_detectors
from utils.metrics import evaluate
import pandas as pd
from pathlib import Path


N_SEEDS = 5

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

Path("results").mkdir(exist_ok=True)
detectors = get_detectors()

for ds_name in ["inj_cora", "reddit", "weibo", "inj_amazon"]:
    print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")
    data = load(ds_name)
    if data is None:
        print(f"Skipping {ds_name}"); continue

    y_true  = data.y.numpy()
    records = []

    for det in detectors:
        seed_aucs, seed_aps = [], []

        for seed in range(N_SEEDS):
            set_seed(seed)
            try:
                scores = det.fit_predict(data.clone())
                if hasattr(scores, 'numpy'):
                    scores = scores.numpy()
                metrics = evaluate(y_true, scores)
                seed_aucs.append(metrics['auc'])
                seed_aps.append(metrics['ap'])
            except Exception as e:
                import traceback; traceback.print_exc()
                seed_aucs.append(float("nan"))
                seed_aps.append(float("nan"))

        auc_mean = np.nanmean(seed_aucs)
        auc_std  = np.nanstd(seed_aucs)
        ap_mean  = np.nanmean(seed_aps)
        ap_std   = np.nanstd(seed_aps)

        print(f"  {det.name:<14}  AUC={auc_mean:.3f}±{auc_std:.3f}  AP={ap_mean:.3f}±{ap_std:.3f}")

        records.append(dict(
            detector = det.name,
            group    = det.group,
            dataset  = ds_name,
            auc      = auc_mean,
            auc_std  = auc_std,
            ap       = ap_mean,
            ap_std   = ap_std,
            n_seeds  = N_SEEDS,
        ))

    df = pd.DataFrame(records)
    df.to_csv(f"results/real_{ds_name}.csv", index=False)
    print(f"\n{df[['detector','auc','auc_std','ap','ap_std']].round(3).to_string(index=False)}")

# Combinar — excluye real_results.csv del glob
all_files = [f for f in Path("results").glob("real_*.csv") if f.name != "real_results.csv"]
combined = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
combined = combined.drop_duplicates(subset=["detector", "dataset"], keep="last")
combined.to_csv("results/real_results.csv", index=False)

print("\n── AUC summary (mean) ───────────────────────────────────────────")
print(combined.pivot(index="detector", columns="dataset", values="auc").round(3).to_string())
print("\n── AUC std ──────────────────────────────────────────────────────")
print(combined.pivot(index="detector", columns="dataset", values="auc_std").round(3).to_string())