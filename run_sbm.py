"""
run_sbm.py  —  Stress test: how does feature camouflage affect anomaly detectors?

Three conditions:
  A — Feature-only anomalies (structural_anomaly=False)
      Full gamma sweep × both camouflage modes.
  B — Bridge nodes           (structural_anomaly=True, pattern="bridge")
      Full gamma sweep × both camouflage modes.
  C — Clique anomalies       (structural_anomaly=True, pattern="clique")
      Single gamma=0.0 only — features already normal, camouflage irrelevant.

Two camouflage modes (Conditions A and B only):
  global — naive attacker, blends toward global mean
  local  — informed attacker, blends toward community mean

USAGE:
  Full run:
      python run_sbm.py

  Single detector:
      python run_sbm.py --detector LOF
      python run_sbm.py --detector GCAD --seeds 10

  Specific conditions or camouflage:
      python run_sbm.py --conditions A B
      python run_sbm.py --conditions A --camouflage local
      python run_sbm.py --conditions C

  Control parallelism:
      python run_sbm.py --workers 4
      python run_sbm.py --workers 1   # sequential, easier to debug

  Fast development run (fewer epochs):
      SBM_EPOCHS=20 python run_sbm.py --conditions A --seeds 2

Output (in results/):
  sbm_raw.csv        one row per (detector, condition, camouflage, gamma, seed)
  sbm_results.csv    mean AUC + std per (detector, condition, camouflage, gamma)
"""

import os
import sys
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
GAMMAS            = [round(g * 0.1, 1) for g in range(11)]  # 0.0 → 1.0
CONDITION_C_GAMMA = [0.0]   # C has no gamma sweep — features already normal
N_NODES           = 1000
N_SEEDS           = 5
RAW_CSV           = Path("results/sbm_raw.csv")
AGG_CSV           = Path("results/sbm_results.csv")
ALL_CONDITIONS    = ["A", "B", "C"]
ALL_CAMOUFLAGES   = ["global", "local"]
# ─────────────────────────────────────────────────────────────────────────────


# ── Worker function ───────────────────────────────────────────────────────────

def _run_seed(task):
    """
    Execute all detectors for one (condition, camouflage, gamma, seed).
    One graph generated, all detectors run on it — no redundant regeneration.
    Runs in a child process: all imports happen inside.
    CUDA disabled to avoid GPU memory conflicts between workers.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    cond, cam, gamma, seed, sbm_kwargs, det_names = task

    from data.sbm      import make_sbm
    from detectors.all import get_detectors
    from utils.metrics import evaluate

    # One import, one graph, all detectors
    det_map = {d.name: d for d in get_detectors()}
    data    = make_sbm(n_nodes=N_NODES, seed=seed * 100, **sbm_kwargs)
    y_true  = data.y.numpy()

    results = []
    for det_name in det_names:
        if det_name not in det_map:
            continue
        det = det_map[det_name]
        try:
            scores  = det.fit_predict(data.clone())
            metrics = evaluate(y_true, scores)
        except Exception as e:
            print(f"\n  FAILED [{det_name}] cond={cond} cam={cam} "
                  f"γ={gamma:.1f} seed={seed} — {e}")
            metrics = {"auc": float("nan"), "ap": float("nan")}

        results.append(dict(
            detector   = det_name,
            group      = det.group,
            condition  = cond,
            camouflage = cam,
            gamma      = gamma,
            seed       = seed,
            n_nodes    = N_NODES,
            **metrics,
        ))
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stress test anomaly detectors under feature camouflage."
    )
    parser.add_argument("--detector",   default=None,
                        help="Single detector to rerun. Omit for full run.")
    parser.add_argument("--seeds",      type=int, default=N_SEEDS,
                        help=f"Number of seeds (default: {N_SEEDS})")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS,
                        help="Conditions to run: A, B, C")
    parser.add_argument("--camouflage", nargs="+", default=ALL_CAMOUFLAGES,
                        choices=ALL_CAMOUFLAGES,
                        help="Camouflage modes: global, local (ignored for C)")
    parser.add_argument("--workers",    type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help="Parallel workers (default: cpu_count-1, use 1 to debug)")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    SINGLE      = args.detector
    N_SEEDS_RUN = args.seeds
    CONDITIONS  = args.conditions
    CAMOUFLAGES = args.camouflage
    N_WORKERS   = args.workers

    from detectors.all import get_detectors
    all_detectors = get_detectors()

    if SINGLE is not None:
        det_names = [d.name for d in all_detectors]
        if SINGLE not in det_names:
            print(f"ERROR: '{SINGLE}' not found. Available: {det_names}")
            sys.exit(1)
        detectors = [d for d in all_detectors if d.name == SINGLE]
        print(f"\nSingle-detector mode: {SINGLE}")
    else:
        detectors = all_detectors
        print(f"\nFull run: {len(detectors)} detectors")

    print(f"Conditions:  {CONDITIONS}")
    print(f"Camouflage:  {CAMOUFLAGES}  (ignored for condition C)")
    print(f"Seeds:       {N_SEEDS_RUN}")
    print(f"Workers:     {N_WORKERS}")

    # ── Build run list ────────────────────────────────────────────────────────
    # Each entry: (condition, camouflage, gamma, sbm_kwargs)
    # Condition C: single gamma=0.0, no camouflage sweep.
    # Conditions A/B: full gamma sweep × both camouflage modes.
    run_list = []

    for cond in CONDITIONS:
        if cond == "C":
            for gamma in CONDITION_C_GAMMA:
                run_list.append((cond, "none", gamma, dict(
                    structural_anomaly = True,
                    pattern            = "clique",
                    gamma              = gamma,
                )))
        else:
            structural = (cond == "B")
            for cam in CAMOUFLAGES:
                for gamma in GAMMAS:
                    run_list.append((cond, cam, gamma, dict(
                        structural_anomaly = structural,
                        pattern            = "bridge",
                        gamma              = gamma,
                        camouflage         = cam,
                    )))

    # ── Build task list ───────────────────────────────────────────────────────
    # One task per (condition, camouflage, gamma, seed).
    # Each task runs ALL detectors on ONE graph — avoids redundant regeneration.
    det_names = [d.name for d in detectors]
    tasks = [
        (cond, cam, gamma, seed, sbm_kwargs, det_names)
        for cond, cam, gamma, sbm_kwargs in run_list
        for seed in range(N_SEEDS_RUN)
    ]

    total_runs = len(tasks) * len(det_names)
    print(f"Tasks:       {len(tasks)}  ({total_runs} detector runs total)\n")

    # ── Execute ───────────────────────────────────────────────────────────────
    records = []

    if N_WORKERS == 1:
        # Sequential — easier to debug, same results as parallel
        with tqdm(total=len(tasks), desc="Running") as pbar:
            for task in tasks:
                cond, cam, gamma, seed, _, _ = task
                pbar.set_description(
                    f"cond={cond} cam={cam} γ={gamma:.1f} s={seed+1}/{N_SEEDS_RUN}"
                )
                results = _run_seed(task)
                records.extend(results)
                pbar.update(1)
    else:
        # Parallel by (condition, gamma, seed) — each worker imports once
        # and runs all detectors. Much less import overhead than per-detector tasks.
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_run_seed, t): t for t in tasks}
            with tqdm(total=len(tasks), desc="Running") as pbar:
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        records.extend(results)
                    except Exception as e:
                        print(f"\nTask failed: {e}")
                    pbar.update(1)

    # ── Patch or create raw CSV ───────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    new_df = pd.DataFrame(records)

    if SINGLE is not None and RAW_CSV.exists():
        existing = pd.read_csv(RAW_CSV)
        mask = (
            (existing["detector"]   == SINGLE) &
            (existing["condition"].isin(CONDITIONS)) &
            (existing["camouflage"].isin(CAMOUFLAGES + ["none"]))
        )
        removed  = mask.sum()
        existing = existing[~mask]
        raw      = pd.concat([existing, new_df], ignore_index=True)
        print(f"\nPatched {RAW_CSV.name}: replaced {removed} rows for {SINGLE}")
    else:
        raw = new_df
        if SINGLE is not None:
            print(f"\nNo existing {RAW_CSV.name} — creating fresh.")

    raw = raw.sort_values(
        ["condition", "camouflage", "detector", "gamma", "seed"]
    ).reset_index(drop=True)

    raw.to_csv(RAW_CSV, index=False)
    print(f"Saved {RAW_CSV}  ({len(raw)} rows)")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = (
        raw.groupby(["detector", "group", "condition", "camouflage", "gamma"])
        .agg(
            auc_mean = ("auc", "mean"),
            auc_std  = ("auc", "std"),
            auc_min  = ("auc", "min"),
            auc_max  = ("auc", "max"),
            ap_mean  = ("ap",  "mean"),
            ap_std   = ("ap",  "std"),
            n_seeds  = ("seed", "count"),
        )
        .reset_index()
    )
    agg["auc"] = agg["auc_mean"]
    agg["ap"]  = agg["ap_mean"]

    agg.to_csv(AGG_CSV, index=False)
    print(f"Saved {AGG_CSV}  ({len(agg)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── AUC summary ──────────────────────────────────────────────────")
    for cond in CONDITIONS:
        cams = ["none"] if cond == "C" else CAMOUFLAGES
        for cam in cams:
            subset = agg[
                (agg["condition"]  == cond) &
                (agg["camouflage"] == cam)
            ]
            if subset.empty:
                continue
            print(f"\nCondition {cond}  |  camouflage={cam}")
            pivot = subset.pivot(
                index="detector", columns="gamma", values="auc_mean"
            ).round(3)
            print(pivot.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()