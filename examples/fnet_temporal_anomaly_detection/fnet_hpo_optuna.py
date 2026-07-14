#!/usr/bin/env python3
"""
fnet_hpo_optuna.py
==================

Local, sequential hyperparameter optimization for the FNET temporal forecaster,
using Optuna. Tunes the model/optimizer knobs that the comparison established as
worth exploring, on the same parity cache, and mirrors the search-space layout of
the DeepHyper driver (``examples/multidataset_hpo_sc26``) so the winning space
ports to a cluster run later.

How it works
------------
Each trial launches the existing single-trial script
``fnet_temporal_anomaly_detection.py`` as a subprocess with sampled
``--{flag}={value}`` arguments, then reads that run's ``metrics.json`` and returns
the objective. The HPO driver never imports the model code -- anything the
training script accepts as a CLI flag can become a search dimension.

Design choices
--------------
- **Reuse the parity cache** (``dataset_parity/``, on-demand windowing). Only
  train-time knobs are tuned, so no re-preprocessing per trial.
- **Cheap proxy for search.** Trials run at a coarse ``--hpo_stride`` (fewer
  windows) and few ``--hpo_epochs`` so a broad search is affordable; re-validate
  the best config at the full budget afterwards (stride 10, more epochs, 3 seeds)
  via ``run_comparison.py`` / the single-trial script.
- **Objective = validation MAE in original units** (``metrics.json``
  ``val.original.mae``). MAE is absolute and, being pooled, is dominated by
  ``volt_dev`` -- the channel with the most headroom (see README_COMPARISON.md).
  Lower is better; Optuna minimizes it.
- **OOM / failure handling.** A trial whose subprocess fails (e.g. a
  ``batch_size`` too large for 6 GB) is pruned rather than killing the study.
- **Resumable.** State is a SQLite study (``--storage``); re-running continues it.

Search space (edit ``suggest_params`` to change)
------------------------------------------------
  mpnn_type        {TemporalSAGE, TemporalGIN}   -- the comparison's top backbones
  hidden_dim       {32, 64, 96, 128}
  num_conv_layers  1..4
  learning_rate    1e-4 .. 1e-2   (log)
  batch_size       {16, 32, 48}

Usage
-----
    python fnet_hpo_optuna.py --n_trials 40
    python fnet_hpo_optuna.py --n_trials 40 --hpo_stride 50 --hpo_epochs 5
    # resume / inspect:
    python fnet_hpo_optuna.py --n_trials 20        # continues the same study
    optuna-dashboard sqlite:///fnet_hpo.db          # optional GUI
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import optuna

HERE = Path(__file__).resolve().parent
HYDRA_SCRIPT = HERE / "fnet_temporal_anomaly_detection.py"


def _env() -> dict:
    env = os.environ.copy()
    env.setdefault("USE_LIBUV", "0")  # Windows PyTorch DDP without libuv
    return env


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "mpnn_type": trial.suggest_categorical("mpnn_type", ["TemporalSAGE", "TemporalGIN"]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 96, 128]),
        "num_conv_layers": trial.suggest_int("num_conv_layers", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 48]),
    }


def make_objective(args):
    trials_root = Path(args.out_root)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        out_dir = trials_root / f"trial_{trial.number:04d}"
        cmd = [
            sys.executable, "-u", str(HYDRA_SCRIPT),
            "--date", args.date,
            "--format", "pickle",
            "--cache_dir", str(args.cache_dir),
            "--dataset_mode", "ondemand",
            "--window_stride", str(args.hpo_stride),
            "--skip_downstream_arrays",
            "--num_epoch", str(args.hpo_epochs),
            "--seed", str(args.seed),
            "--out_dir", str(out_dir),
            "--log", f"hpo_t{trial.number:04d}",
            "--mpnn_type", params["mpnn_type"],
            "--hidden_dim", str(params["hidden_dim"]),
            "--num_conv_layers", str(params["num_conv_layers"]),
            "--learning_rate", str(params["learning_rate"]),
            "--batch_size", str(params["batch_size"]),
        ]
        if args.cpu:
            cmd.append("--cpu")

        log_path = trials_root / f"trial_{trial.number:04d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[trial {trial.number}] {params}", flush=True)
        with open(log_path, "w") as fh:
            proc = subprocess.run(cmd, cwd=str(HERE), env=_env(),
                                  stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            # Most common cause is CUDA OOM for a large batch/hidden combo. Prune
            # so the study continues; the log has the traceback.
            tail = log_path.read_text(errors="replace")[-400:]
            print(f"[trial {trial.number}] FAILED rc={proc.returncode}; pruning. "
                  f"tail:\n{tail}", flush=True)
            raise optuna.TrialPruned()

        mpath = out_dir / "metrics.json"
        if not mpath.exists():
            raise optuna.TrialPruned()
        metrics = json.loads(mpath.read_text())
        val_mae = metrics["val"]["original"]["mae"]
        if val_mae is None or math.isnan(val_mae):
            raise optuna.TrialPruned()

        # Record the useful extras so the study DB is self-describing.
        for k, v in params.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("val_original_rmse", metrics["val"]["original"]["rmse"])
        trial.set_user_attr("test_original_mae", metrics["test"]["original"]["mae"])
        print(f"[trial {trial.number}] val MAE = {val_mae:.6f}", flush=True)
        return float(val_mae)

    return objective


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--timeout", type=int, default=None, help="Wall-clock seconds (optional).")
    p.add_argument("--cache_dir", type=Path, default=HERE / "dataset_parity")
    p.add_argument("--date", type=str, default="2024-06-01")
    p.add_argument("--hpo_stride", type=int, default=100,
                   help="Coarse window stride for cheap trials (higher = fewer windows).")
    p.add_argument("--hpo_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--study_name", type=str, default="fnet_sage_gin")
    p.add_argument("--storage", type=str, default=f"sqlite:///{HERE / 'fnet_hpo.db'}")
    p.add_argument("--out_root", type=str, default=str(HERE / "hpo_trials"))
    args = p.parse_args()

    if not (args.cache_dir / f"fnet_{args.date}_meta.pkl").exists():
        raise SystemExit(
            f"Parity cache not found at {args.cache_dir}. Build it first, e.g.:\n"
            f"  python run_comparison.py --stage cache"
        )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    print(f"[hpo] study={args.study_name} storage={args.storage}")
    print(f"[hpo] proxy budget: stride={args.hpo_stride}, epochs={args.hpo_epochs}, "
          f"seed={args.seed}  (re-validate the winner at full budget)")
    print(f"[hpo] existing trials in study: {len(study.trials)}")

    study.optimize(make_objective(args), n_trials=args.n_trials, timeout=args.timeout,
                   gc_after_trial=True, show_progress_bar=False)

    print("\n" + "=" * 70)
    print(f"[hpo] completed. best val MAE = {study.best_value:.6f}")
    print("[hpo] best params:")
    for k, v in study.best_params.items():
        print(f"      {k} = {v}")

    # Importances (needs >1 completed trial).
    try:
        imp = optuna.importance.get_param_importances(study)
        print("[hpo] parameter importances:")
        for k, v in imp.items():
            print(f"      {v:6.3f}  {k}")
    except Exception as e:
        print(f"[hpo] importances unavailable: {e}")

    # Dump a tidy leaderboard.
    df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    csv = Path(args.out_root) / "leaderboard.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("value").to_csv(csv, index=False)
    print(f"[hpo] leaderboard -> {csv}")
    print(f"\n[next] re-validate the best config at full budget (stride 10, 3 seeds), e.g.:")
    bp = study.best_params
    print(f"  python run_comparison.py --stage sweep --run_dir runs/hpo_best "
          f"--models {bp['mpnn_type']} --seeds 42 1 2   # after setting hidden/layers/lr in PARITY")


if __name__ == "__main__":
    main()
