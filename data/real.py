"""
data/real.py  —  Real-world dataset loaders (PyGOD).

Each function returns a torch_geometric.data.Data object with binary labels y.
Returns None and prints a message if the dataset cannot be loaded.
"""

import torch
from pygod.utils import load_data


def load(name: str):
    """Load a named PyGOD dataset. Labels binarized: 0=normal, 1=anomaly."""
    try:
        data = load_data(name)
        # PyGOD returns multiclass labels {0,1,2,3,...} where 0=normal.
        # Binarize: any anomaly type → 1.
        data.y = (data.y > 0).long()
        n_anom = int(data.y.sum())
        print(
            f"[{name}]  nodes={data.num_nodes}  "
            f"edges={data.num_edges}  "
            f"features={data.x.shape[1]}  "
            f"anomalies={n_anom}  "
            f"ratio={n_anom/data.num_nodes:.1%}"
        )
        return data
    except Exception as e:
        print(f"[{name}]  failed to load: {e}")
        return None


def load_all() -> dict:
    """Return dict of {name: Data} for all real-world datasets."""
    names  = ["inj_cora", "inj_amazon", "weibo", "reddit"]
    loaded = {n: load(n) for n in names}
    return {n: d for n, d in loaded.items() if d is not None}