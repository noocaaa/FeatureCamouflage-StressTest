# Stress Testing Detection Under Feature Camouflage in Graph Networks

## What this project does

Runs 12 graph anomaly detectors under increasing levels of **feature camouflage** (γ) and measures how their AUC degrades. The score question: which detectors survive when anomaly features are made indistinguishable from normal features?

Camouflage works by blending anomaly node features toward the normal mean.  At γ = 0 anomalies look different; at γ = 1 they look identical to normal nodes. Detectors that rely purely on features collapse. Detectors that also use graph structure may survive — depending on *how* they use it.

---
## Structure

```
├── data/
│   ├── sbm.py                    Synthetic SBM graph generator (gamma, conditions A/B/C)
│   └── real.py                   Loaders for inj_cora, inj_amazon, weibo, reddit
│
├── detectors/
│   ├── base.py                   Base Detector class + PyGOD NeighborLoader patch
│   ├── gcad.py                   IDK / GCAD / GCAD+IDK  (ablation of Prof. Ting's framework)
│   └── all.py                    All 11 detectors + get_detectors() factory
│
├── utils/
│   ├── metrics.py                evaluate() → AUC, AP
│   ├── test_cola.py              CoLA diagnostic: graph vs no-edges AUC
│   ├── test_structure_reliance.py  Group B ablation: full / no-edges / rand-feats
│   └── diagnose_radar_one_lof.py   Score-direction check for Radar and ONE
│
├── run_sbm.py                    Experiment 1 — gamma sweep on synthetic SBM
├── run_real.py                   Experiment 2 — real-world datasets
├── dashboard.py                  Interactive results dashboard (Dash/Plotly)
├── Stress_Testing_Feature        Google Colab Run Notebook
    Anomalies_RealWorld.ipynb                  
└── results/                      Auto-generated CSVs and PNGs
```

---
## The 11 detectors

Three groups based on how each detector uses graph structure.

| Detector   | Group    | How it works                                                    |
|------------|----------|-----------------------------------------------------------------|
| LOF        | none     | Local density contrast on raw attributes                        |
| IDK        | none     | Isolation distributional kernel on raw attributes               |
| iForest    | none     | Path-length isolation forest on raw attributes                  |
| ONE        | indirect | Aligns network embeddings with attribute embeddings             |
| Radar      | indirect | Residual analysis using the graph Laplacian                     |
| DOMINANT   | indirect | GCN message passing, score = feature reconstruction error       |
| AnomalyDAE | indirect | Dual autoencoder on attributes + structure                      |
| DONE       | indirect | Deep dual-objective: attribute + structure residuals            |
| CoLA       | indirect | Contrastive subgraph pairs — partial structural signal          |
| GCAD       | indirect | Subgraph centralization + plain IsolationForest scorer          |
| GCAD+IDK   | indirect | Subgraph centralization + IDK scorer  ← full method             |
| GAE        | direct   | Reconstructs adjacency only — never sees features               |

**Expected behaviour under increasing γ (Condition A — feature-only anomalies):**

- Group `none` collapses first — pure attribute signal is destroyed by camouflage.
- Group `indirect` collapses later — graph provides partial resistance.
  - Global camouflage: local inconsistency is still detectable by GNN-based detectors.
  - Local camouflage: local inconsistency is eliminated, Group `indirect` also collapses.
- Group `direct` stays near 0.5 — no structural signal exists in Condition A by design.

> **Note on ONE and Radar:** both appear in raw CSVs and plots (dashed lines) but are excluded from group-mean curves. 
> 
> Their `decision_score_` direction is inconsistent
> across runs — higher scores are sometimes anomalous, sometimes normal — so aggregating
> across seeds is unreliable without a per-seed inversion check. They are still classified
> as Group `indirect` because they explicitly use graph structure.
> 
> Run `python utils/diagnose_radar_one_lof.py` to check score direction on any graph.

---
## The three experimental conditions

| Condition | Anomaly type        | Feature signal | Structural signal | Gamma sweep? |
|-----------|---------------------|----------------|-------------------|--------------|
| A         | Feature-only        | Yes → destroyed by γ | None         | Yes          |
| B         | Bridge nodes        | Yes → destroyed by γ | Always present | Yes        |
| C         | Clique              | None (features are normal) | Always present | No (γ=0 only) |

- **Condition A** — Anomalies are structurally invisible. Only feature signal exists, and it is progressively destroyed. All detectors should degrade; GAE stays near 0.5.

- **Condition B** — Anomalies connect to all communities with high probability. Feature signal is destroyed by γ but structural signal always remains. GAE stays high regardless of γ. Proves structural anomalies cannot be camouflaged.

- **Condition C** — Anomalies form a dense clique. Features are indistinguishable from normal nodes at γ = 0. Tests pure structural detection; feature-based detectors fail.

**Two camouflage modes (Conditions A and B only):**

- `global` — Naive attacker: blends toward the global mean of all normal nodes. Destroys global feature signal but leaves local inconsistency detectable by GNNs.

- `local` — Informed attacker: blends toward the anomaly's own community mean. Eliminates local inconsistency — the hardest test for GNN-based detectors.

---
## Setup

```bash
pip install -r requirements.txt
```

---
## Usage

### Experiment 1 — Synthetic SBM (gamma sweep)

```bash
# Full run — all conditions, both camouflage modes, 5 seeds
python run_sbm.py

# Single detector
python run_sbm.py --detector LOF
python run_sbm.py --detector GCAD --seeds 10

# Specific conditions or camouflage modes
python run_sbm.py --conditions A B
python run_sbm.py --conditions A --camouflage local

# Control parallelism
python run_sbm.py --workers 4
python run_sbm.py --workers 1    # sequential — easier to debug

# Faster development example run (fewer training epochs)
SBM_EPOCHS=10 python run_sbm.py --conditions A --seeds 2
```

### Experiment 2 — Real-world datasets

```bash
python run_real.py
```

Runs on the datasets: `inj_cora`, `inj_amazon`, `weibo`, `reddit`.

### Dashboard

```bash
python dashboard.py
# Open http://127.0.0.1:8050
```

Tabs: Overview · AP · Heatmap · Robustness · Per-seed · CoLA · Structure · Reliance

### Diagnostics

```bash
# Check whether CoLA benefits from graph structure
python utils/test_cola.py

# Quantify how much each Group B detector relies on structure vs attributes
python utils/test_structure_reliance.py
python utils/test_structure_reliance.py --detector CoLA --camouflage local

# Check Radar and ONE score direction (higher = anomalous or normal?)
python utils/diagnose_radar_one_lof.py
```

---
## Outputs

All files are saved to `results/`.

**Experiment 1 CSVs:**

| File | Contents |
|------|----------|
| `sbm_raw.csv` | One row per (detector, condition, camouflage, gamma, seed) |
| `sbm_results.csv` | Mean AUC + std per (detector, condition, camouflage, gamma) |

**Experiment 2 CSVs:**

| File | Contents |
|------|----------|
| `real_<dataset>.csv` | Per-detector AUC/AP for each dataset |
| `real_results.csv` | Combined results across all datasets |

---
## References

- Holland et al. (1983). Stochastic blockmodels. *Social Networks.*
- Karrer & Newman (2011). Stochastic blockmodels and community structure. *Physical Review E.*
- Ting et al. (2020). Isolation distributional kernel. *KDD 2020.*
- Zhuang et al. (2023). GCAD. *SDM 2023.*
- Ting et al. (2025). What are anomalies in a network? *ACM TKDD.*
- Liu et al. (2022). BOND benchmark. *IJCAI 2022.*