"""
data/sbm.py  —  SBM generator with controllable feature camouflage.

Three anomaly conditions:

  Condition A — Feature-only (structural_anomaly=False)
    Anomalies are structurally indistinguishable from normal nodes.
    Only signal is in features — destroyed as gamma increases.
    → All detectors degrade with gamma.
    → GAE stays ~0.5 (no structural signal).

  Condition B — Bridge nodes (structural_anomaly=True, pattern="bridge")
    Anomalies connect to ALL communities with high probability (p_bridge).
    Feature signal destroyed by gamma; structural signal always present.
    → Attribute detectors degrade with gamma.
    → GAE stays high regardless of gamma.
    → Proves: structural anomalies cannot be camouflaged by features.

  Condition C — Clique anomalies (structural_anomaly=True, pattern="clique")
    Anomalies form a dense clique. Features are NORMAL (same as community).
    Tests pure structural detection — no feature signal at any gamma.
    → Only structure-aware detectors (GAE) should detect these.
    → Proves: some anomalies are purely structural, features are irrelevant.

Camouflage modes (Conditions A and B only):
  global  — blend toward global mean of all normal nodes (naive attacker)
  local   — blend toward mean of assigned community (informed attacker)

Performance:
  Edge generation is fully vectorized with numpy (no Python loops over pairs).
  Two separate RNGs for graph and features ensure determinism within each domain.

References:
  Holland et al. (1983). Stochastic blockmodels. Social Networks.
  Karrer & Newman (2011). Stochastic blockmodels. Physical Review E.
  Ting et al. (2025). What are anomalies in a network? ACM TKDD.
  Liu et al. (2022). BOND benchmark. IJCAI 2022.
"""

import numpy as np
import torch
from torch_geometric.data import Data


def _sbm_edges(rng, n, comm_of, p_intra, p_inter):
    """
    Sample edges for n nodes following SBM probabilities.
    Fully vectorized — no Python loop over pairs.

    Parameters
    ----------
    rng     : np.random.RandomState
    n       : number of nodes (indices 0..n-1)
    comm_of : dict mapping node index → community index
    p_intra : edge probability within same community
    p_inter : edge probability between communities

    Returns
    -------
    List of (u, v) edge tuples.
    """
    u, v  = np.triu_indices(n, k=1)
    cu    = np.array([comm_of[i] for i in u])
    cv    = np.array([comm_of[j] for j in v])
    probs = np.where(cu == cv, p_intra, p_inter)
    mask  = rng.rand(len(probs)) < probs
    return list(zip(u[mask].tolist(), v[mask].tolist()))


def make_sbm(
    n_nodes:            int   = 1000,
    n_communities:      int   = 3,
    anomaly_ratio:      float = 0.05,
    p_intra:            float = 0.10,
    p_inter:            float = 0.01,
    p_bridge:           float = 0.15,
    p_clique:           float = 0.80,
    n_features:         int   = 32,
    separation:         float = 5.0,
    gamma:              float = 0.0,
    structural_anomaly: bool  = False,
    pattern:            str   = "bridge",
    camouflage:         str   = "global",
    seed:               int   = 42,
) -> Data:
    """
    Generate a Stochastic Block Model graph with controllable anomalies
    and feature camouflage.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes.
    n_communities : int
        Number of communities for normal nodes.
    anomaly_ratio : float
        Fraction of anomalous nodes (default 5%).
    p_intra : float
        Edge probability within a community (default 0.10, per Ting et al.).
    p_inter : float
        Edge probability between communities.
    p_bridge : float
        For bridge pattern: cross-community edge probability for anomalies.
        Should be clearly > p_inter but not extreme (default 0.15).
    p_clique : float
        For clique pattern: within-clique edge probability (default 0.80).
        Also used for anomaly-to-anomaly backbone in bridge pattern.
    n_features : int
        Feature dimensionality.
    separation : float
        Distance between anomaly and normal feature centers (default 5.0).
        Lower than 10.0 avoids AUC ceiling at gamma=0.
    gamma : float
        Camouflage intensity: 0.0 = no camouflage, 1.0 = full camouflage.
        Ignored for Condition C (features already normal).
    structural_anomaly : bool
        If True, anomalies have a structural pattern (bridge or clique).
    pattern : str
        Structural pattern: "bridge" (Condition B) or "clique" (Condition C).
    camouflage : str
        Camouflage target: "global" (naive attacker, blends toward global mean)
        or "local" (informed attacker, blends toward community mean).
        Ignored for Condition C.
    seed : int
        Random seed. Two derived RNGs are used internally:
          rng_graph    = RandomState(seed)     — controls all edge sampling
          rng_features = RandomState(seed + 1) — controls all feature sampling
        This ensures graph and feature generation are independently reproducible.

    Returns
    -------
    Data
        PyG Data with fields: x, edge_index, y, gamma, camouflage.
    """
    # Two separate RNGs: graph topology and node features are independent.
    # Changing edge generation logic does not affect features and vice versa.
    rng_graph    = np.random.RandomState(seed)
    rng_features = np.random.RandomState(seed + 1)

    # Guard: Condition C (clique anomalies) has no feature signal by design —
    # gamma and camouflage are both irrelevant and silently ignored.  Callers
    # that accidentally pass gamma > 0 here will get wrong results without any
    # error; raise early instead.
    if structural_anomaly and pattern == "clique" and gamma != 0.0:
        raise ValueError(
            f"make_sbm: Condition C (pattern='clique') has no feature signal — "
            f"gamma must be 0.0, got {gamma}.  "
            f"Set gamma=0.0 or use pattern='bridge'."
        )

    n_anomalies = int(n_nodes * anomaly_ratio)
    n_normal    = n_nodes - n_anomalies

    # ── Communities ──────────────────────────────────────────────────────────
    sizes = [n_normal // n_communities] * n_communities
    sizes[-1] += n_normal - sum(sizes)
    communities, start = [], 0
    for s in sizes:
        communities.append(list(range(start, start + s)))
        start += s

    anomaly_nodes = list(range(n_normal, n_nodes))
    anomaly_set   = set(anomaly_nodes)
    normal_nodes  = list(range(n_normal))

    # Community assignment.
    # Normal nodes: their actual community.
    # Condition C anomalies: round-robin assignment to avoid feature leakage.
    #   Random assignment with small n_anomalies can shift empirical community
    #   means, creating spurious feature signal in a purely structural condition.
    #   Round-robin guarantees exactly n_anomalies // n_communities per community.
    # Conditions A/B anomalies: random assignment (structurally invisible).
    comm_of = {n: i for i, comm in enumerate(communities) for n in comm}

    if structural_anomaly and pattern == "clique":
        for i, a in enumerate(anomaly_nodes):
            comm_of[a] = i % n_communities
    else:
        for a in anomaly_nodes:
            comm_of[a] = rng_graph.randint(0, n_communities)

    # ── Edges ─────────────────────────────────────────────────────────────────
    edge_list = []

    if structural_anomaly and pattern == "bridge":
        # ── Condition B: Bridge nodes ─────────────────────────────────────────
        # Anomalies have high cross-community connectivity (p_bridge >> p_inter).
        # This creates a structural signal that gamma cannot destroy.

        # Normal-to-normal: vectorized SBM
        edge_list += _sbm_edges(rng_graph, n_normal, comm_of, p_intra, p_inter)

        # Anomaly-to-normal: vectorized per anomaly node.
        # Same community → p_intra (structurally consistent within community).
        # Different community → p_bridge (the structural anomaly signal).
        normal_arr  = np.array(normal_nodes)
        comm_normal = np.array([comm_of[v] for v in normal_nodes])

        for a in anomaly_nodes:
            probs = np.where(comm_normal == comm_of[a], p_intra, p_bridge)
            mask  = rng_graph.rand(n_normal) < probs
            edge_list += [(a, int(v)) for v in normal_arr[mask]]

        # Anomaly-to-anomaly: dense backbone among bridge nodes.
        # Makes the structural signal unambiguous for GAE.
        n_anom = len(anomaly_nodes)
        if n_anom > 1:
            ai, aj = np.triu_indices(n_anom, k=1)
            mask   = rng_graph.rand(len(ai)) < p_clique
            edge_list += [
                (anomaly_nodes[i], anomaly_nodes[j])
                for i, j, m in zip(ai, aj, mask) if m
            ]

    elif structural_anomaly and pattern == "clique":
        # ── Condition C: Clique anomalies ─────────────────────────────────────
        # Anomalies form a dense subgraph (p_clique).
        # Normal nodes follow standard SBM.
        # Features are NORMAL — no feature signal at any gamma.
        all_nodes  = list(range(n_nodes))
        comm_arr   = np.array([comm_of[i] for i in all_nodes])
        is_anomaly = np.zeros(n_nodes, dtype=bool)
        is_anomaly[list(anomaly_set)] = True

        u, v         = np.triu_indices(n_nodes, k=1)
        both_anomaly = is_anomaly[u] & is_anomaly[v]
        probs        = np.where(
            both_anomaly,
            p_clique,
            np.where(comm_arr[u] == comm_arr[v], p_intra, p_inter)
        )
        mask      = rng_graph.rand(len(probs)) < probs
        edge_list = list(zip(u[mask].tolist(), v[mask].tolist()))

    else:
        # ── Condition A: Feature-only anomalies ───────────────────────────────
        # All nodes follow standard SBM. Anomalies are structurally invisible.
        edge_list = _sbm_edges(rng_graph, n_nodes, comm_of, p_intra, p_inter)

    # ── Features ──────────────────────────────────────────────────────────────
    # Each community has its own center; nodes = center + Gaussian noise.
    centers  = rng_features.randn(n_communities, n_features)
    features = np.zeros((n_nodes, n_features))
    for i, comm in enumerate(communities):
        noise = rng_features.randn(len(comm), n_features)
        features[np.array(comm)] = centers[i] + noise

    # Global mean computed once from normal nodes only.
    global_mean = features[:n_normal].mean(axis=0)

    if structural_anomaly and pattern == "clique":
        # Condition C: anomalies have NORMAL community features.
        # No feature signal at any gamma — purely structural anomalies.
        # Round-robin community assignment above ensures no empirical mean shift.
        for a in anomaly_nodes:
            features[a] = centers[comm_of[a]] + rng_features.randn(n_features)
    else:
        # Conditions A and B: anomalies are offset from global mean.
        # separation controls how far — lower values give more informative curves.
        direction      = rng_features.randn(n_features)
        direction     /= np.linalg.norm(direction)
        anomaly_center = global_mean + direction * separation
        noise          = rng_features.randn(n_anomalies, n_features)
        features[np.array(anomaly_nodes)] = anomaly_center + noise

    # ── Feature Camouflage ────────────────────────────────────────────────────
    # Only applied to Conditions A and B.
    # Condition C has no feature signal to camouflage.
    #
    # global (naive attacker):
    #   Blends toward the mean of ALL normal nodes.
    #   Destroys global feature signal but leaves local inconsistency:
    #   node still differs from its immediate neighbors.
    #   GNN-based detectors can exploit this residual contrast.
    #
    # local (informed attacker):
    #   Blends toward the mean of the anomaly's own assigned community.
    #   Eliminates local inconsistency — node looks like its neighbors.
    #   Harder test for GNN-based detectors (message passing cannot help).
    #   Supported by Ting et al. (2025): normality is community-relative.
    #
    if gamma > 0.0 and not (structural_anomaly and pattern == "clique"):
        for a in anomaly_nodes:
            target      = centers[comm_of[a]] if camouflage == "local" else global_mean
            features[a] = gamma * target + (1 - gamma) * features[a]

    # ── Labels + PyG ──────────────────────────────────────────────────────────
    labels = np.zeros(n_nodes, dtype=int)
    labels[anomaly_nodes] = 1

    if len(edge_list) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edges      = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edges, edges.flip(0)], dim=1)

    return Data(
        x          = torch.tensor(features, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(labels,   dtype=torch.long),
        gamma      = gamma,
        camouflage = camouflage,
    )