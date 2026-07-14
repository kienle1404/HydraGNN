#!/usr/bin/env python3
"""
revalidate_hpo_best.py
======================

Re-validate the Optuna winner at the FULL training budget, then compare it -- on
identical preprocessing -- against the comparison run's default TemporalSAGE and
the standalone T-GCN baseline.

The HPO search used a cheap proxy (coarse stride, few epochs). This confirms the
winning config holds at the same budget as the definitive comparison
(``runs/full5``: stride 10 = 100% coverage, 5 epochs, 3 seeds), so the tuned
number is directly comparable to that table.

Steps
-----
1. Read the best params from the Optuna study.
2. Train that config at the full budget, one run per seed (resumable: existing
   runs are skipped), into ``runs/full5_tuned/hydragnn/TemporalSAGE_s<seed>``.
3. Recompute per-channel test/pred metrics with the shared ``masked_metrics`` and
   print tuned-SAGE vs default-SAGE (from ``runs/full5``) vs standalone.

Usage
-----
    python revalidate_hpo_best.py                 # uses study best + runs/full5
    python revalidate_hpo_best.py --num_epoch 5 --seeds 42 1 2
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

import optuna

from comparison_common import (
    RECOMPUTED_SPLITS,
    STANDALONE_TARGETS,
    load_hydra_split,
    load_standalone_split,
    per_channel_metrics,
)

HERE = Path(__file__).resolve().parent
HYDRA_SCRIPT = HERE / "fnet_temporal_anomaly_detection.py"
CHANNELS = ("all",) + tuple(STANDALONE_TARGETS)


def _env():
    env = os.environ.copy()
    env.setdefault("USE_LIBUV", "0")
    return env


def train_full_budget(params, seed, args, out_dir: Path) -> bool:
    if (out_dir / "metrics.json").exists():
        print(f"[skip] {out_dir.name} already complete")
        return True
    cmd = [
        sys.executable, "-u", str(HYDRA_SCRIPT),
        "--date", args.date, "--format", "pickle",
        "--cache_dir", str(args.cache_dir),
        "--dataset_mode", "ondemand", "--window_stride", str(args.window_stride),
        "--skip_downstream_arrays",
        "--num_epoch", str(args.num_epoch),
        "--seed", str(seed),
        "--out_dir", str(out_dir), "--log", f"tuned_SAGE_s{seed}",
        "--mpnn_type", params["mpnn_type"],
        "--hidden_dim", str(params["hidden_dim"]),
        "--num_conv_layers", str(params["num_conv_layers"]),
        "--learning_rate", str(params["learning_rate"]),
        "--batch_size", str(params["batch_size"]),
    ]
    log = out_dir.parent / f"{out_dir.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] tuned {params['mpnn_type']} seed={seed} -> {out_dir}")
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=str(HERE), env=_env(), stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[FAIL] seed {seed} rc={proc.returncode}; see {log}")
        return False
    return True


def agg_channel(loader, run_dirs):
    """{(split, channel, metric): (mean, std)} over the given run dirs."""
    rows = {}
    for d in run_dirs:
        for split in RECOMPUTED_SPLITS:
            p, y, m = loader(Path(d), split)
            for chan, met in per_channel_metrics(p, y, m).items():
                for k in ("mae", "rmse", "r2"):
                    rows.setdefault((split, chan, k), []).append(met[k])
    return {k: (st.fmean(v), st.pstdev(v) if len(v) > 1 else 0.0) for k, v in rows.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study_name", default="fnet_sage_gin")
    p.add_argument("--storage", default=f"sqlite:///{HERE / 'fnet_hpo.db'}")
    p.add_argument("--cache_dir", type=Path, default=HERE / "dataset_parity")
    p.add_argument("--date", default="2024-06-01")
    p.add_argument("--window_stride", type=int, default=10)   # full coverage, matches runs/full5
    p.add_argument("--num_epoch", type=int, default=5)         # matches runs/full5
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    p.add_argument("--ref_run", type=Path, default=HERE / "runs" / "full5")
    p.add_argument("--out_run", type=Path, default=HERE / "runs" / "full5_tuned")
    args = p.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    best = study.best_params
    print(f"[hpo] best config (val MAE {study.best_value:.5f}): {best}\n")

    tuned_dirs = []
    for s in args.seeds:
        od = args.out_run / "hydragnn" / f"{best['mpnn_type']}_s{s}"
        if train_full_budget(best, s, args, od):
            tuned_dirs.append(od)
    if not tuned_dirs:
        raise SystemExit("No tuned runs completed.")

    # Reference runs from the definitive comparison.
    def_dirs = [args.ref_run / "hydragnn" / f"TemporalSAGE_s{s}" for s in args.seeds
                if (args.ref_run / "hydragnn" / f"TemporalSAGE_s{s}" / "metrics.json").exists()]
    std_dirs = [args.ref_run / "standalone" / f"seed{s}" for s in args.seeds
                if (args.ref_run / "standalone" / f"seed{s}" / "config.json").exists()]

    tuned = agg_channel(load_hydra_split, tuned_dirs)
    default = agg_channel(load_hydra_split, def_dirs) if def_dirs else {}
    standalone = agg_channel(load_standalone_split, std_dirs) if std_dirs else {}

    def cell(a, split, chan, k):
        v = a.get((split, chan, k))
        return "-" if v is None else (f"{v[0]:.4g}" + (f"±{v[1]:.2g}" if v[1] else ""))

    print("Full-budget re-validation (stride 10, 100% coverage, "
          f"{args.num_epoch} epochs, seeds {args.seeds}); original units, per channel.\n")
    for split in RECOMPUTED_SPLITS:
        print(f"=== {split} ===")
        print(f"  {'channel':11s} {'metric':5s} {'standalone':>16s} {'SAGE default':>16s} {'SAGE tuned':>16s}")
        print("  " + "-" * 70)
        for chan in CHANNELS:
            for k in ("mae", "rmse", "r2"):
                print(f"  {chan:11s} {k:5s} {cell(standalone,split,chan,k):>16s} "
                      f"{cell(default,split,chan,k):>16s} {cell(tuned,split,chan,k):>16s}")
            print()
    print("SAGE default = hidden 64 / 2 layers / lr 1e-3 (comparison PARITY).")
    print(f"SAGE tuned   = {best['mpnn_type']} hidden {best['hidden_dim']} / "
          f"{best['num_conv_layers']} layers / lr {best['learning_rate']:.2e} / batch {best['batch_size']}.")


if __name__ == "__main__":
    main()
