#!/usr/bin/env python3
"""
run_comparison.py
=================

Runner for the HydraGNN-vs-standalone-T-GCN predictive-performance comparison.

Executes three stages, all governed by the single ``PARITY`` dict in
``comparison_common.py`` so the two pipelines cannot silently diverge:

  1. ``cache``    Build the HydraGNN preprocessed cache with parity settings
                  (robust scaling + clip, 2-channel targets, no coverage drop).
  2. ``baseline`` Run the standalone T-GCN once per seed, with strides derived
                  from the cache's actual aligned length so both sides see a
                  comparable number of windows.
  3. ``sweep``    Run each HydraGNN temporal backbone once per seed on that cache.

Everything lands under ``--run_root/<timestamp>/`` together with a
``manifest.json`` recording the settings, both repos' git SHAs, and where each
artifact went. ``make_report.py`` consumes that manifest -- it never guesses.

This script does NOT compute or print metrics. Producing numbers is
``make_report.py``'s job, deliberately: one script runs experiments, one script
reads them.

Usage
-----
    # everything, end to end
    python run_comparison.py --stage all

    # or stage by stage (each is resumable / re-runnable)
    python run_comparison.py --stage cache
    python run_comparison.py --stage baseline --run_dir runs/20260704_120000
    python run_comparison.py --stage sweep    --run_dir runs/20260704_120000

    python make_report.py --run_dir runs/20260704_120000

Windows note: HydraGNN's setup_ddp() crashes on PyTorch builds without libuv;
USE_LIBUV=0 is injected into every child process automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from comparison_common import (
    BACKBONES,
    BACKBONES_ALL,
    PARITY,
    SEEDS,
    git_sha,
    load_hydra_meta,
    standalone_strides,
)

HERE = Path(__file__).resolve().parent
HYDRA_SCRIPT = HERE / "fnet_temporal_anomaly_detection.py"
HYDRA_REPO = HERE.parents[1]  # .../HydraGNN
DEFAULT_STANDALONE = (HERE.parents[2] / "t-gcn").resolve()
DEFAULT_DATA_ROOT = (HERE.parents[2] / "data").resolve()


def _env() -> dict:
    """USE_LIBUV=0 avoids a Windows-only PyTorch DDP crash; no-op elsewhere."""
    env = os.environ.copy()
    env.setdefault("USE_LIBUV", "0")
    return env


def _run(cmd: list, cwd: Path, log: Path) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[run] {' '.join(str(c) for c in cmd)}\n      cwd={cwd}  log={log}", flush=True)
    with open(log, "w") as fh:
        try:
            subprocess.run(cmd, cwd=str(cwd), env=_env(), check=True, stdout=fh, stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[FAIL] exit {exc.returncode}; see {log}", flush=True)
            return False


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_cache(run_dir: Path, args) -> dict:
    """Build the HydraGNN parity cache. Returns cache facts (T, N) for the baseline."""
    cache_dir = (HERE / args.cache_name).resolve()
    meta_file = cache_dir / f"fnet_{PARITY['date']}_meta.pkl"

    if meta_file.exists() and not args.force_cache:
        print(f"[cache] reusing existing {cache_dir} (pass --force_cache to rebuild)")
    else:
        cmd = [
            sys.executable, "-u", str(HYDRA_SCRIPT), "--preonly",
            "--data_root", str(args.data_root),
            "--date", PARITY["date"],
            "--scaling", PARITY["scaling"],
            "--scaled_clip_value", str(PARITY["scaled_clip_value"]),
            "--min_device_coverage", str(PARITY["min_device_coverage"]),
            "--min_step_coverage", str(PARITY["min_step_coverage"]),
            "--short_gap_steps", str(PARITY["short_gap_steps"]),
            "--medium_gap_steps", str(PARITY["medium_gap_steps"]),
            "--Tin", str(PARITY["Tin"]),
            "--horizon", str(PARITY["horizon"]),
            "--out_features", *[str(i) for i in PARITY["out_features"]],
            "--train_frac", str(PARITY["train_frac"]),
            "--val_frac", str(PARITY["val_frac"]),
            "--test_frac", str(PARITY["test_frac"]),
            "--pred_frac", str(PARITY["pred_frac"]),
            "--dataset_mode", PARITY["dataset_mode"],
            "--window_stride", str(PARITY["window_stride"]),
            "--cache_dir", str(cache_dir),
        ]
        if not _run(cmd, HERE, run_dir / "logs" / "cache.log"):
            raise SystemExit("cache stage failed")

    meta = load_hydra_meta(cache_dir, PARITY["date"])
    facts = {
        "cache_dir": str(cache_dir),
        "aligned_timesteps": int(meta["X"].shape[0]),
        "n_nodes": int(len(meta["fdr_ids"])),
        "F_out": int(meta["F_out"]),
        "out_idx": [int(i) for i in meta["out_idx"]],
    }
    print(f"[cache] T={facts['aligned_timesteps']}  N={facts['n_nodes']}  F_out={facts['F_out']}")
    return facts


def stage_baseline(run_dir: Path, cache_facts: dict, args) -> dict:
    """Run the standalone T-GCN once per seed, using the same window stride."""
    strides = standalone_strides()
    print(f"[baseline] strides (matched to PARITY['window_stride']): {strides}")

    out = {"strides": strides, "runs": {}}
    for seed in args.seeds:
        odir = run_dir / "standalone" / f"seed{seed}"
        if (odir / "config.json").exists():
            print(f"[skip] standalone seed{seed} already complete")
            out["runs"][str(seed)] = {"dir": str(odir), "ok": True}
            continue
        cmd = [
            sys.executable, "-u", str(args.standalone_dir / "train.py"),
            "--data_dir", str(args.data_root / PARITY["date"]),
            "--metadata_file", str(args.data_root / "FDRLocation.xlsx"),
            "--Tin", str(PARITY["Tin"]),
            "--H", str(PARITY["horizon"]),
            "--hidden_dim", str(PARITY["hidden_dim"]),
            "--num_layers", str(PARITY["num_conv_layers"]),
            "--epochs", str(PARITY["num_epoch"]),
            "--lr", str(PARITY["learning_rate"]),
            "--batch_size", str(PARITY["batch_size"]),
            "--feature_scaling", PARITY["scaling"],
            "--scaled_clip_value", str(PARITY["scaled_clip_value"]),
            "--time_axis", PARITY["time_axis"],
            "--alignment", PARITY["alignment"],
            "--min_step_coverage", str(PARITY["min_step_coverage"]),
            "--short_gap_steps", str(PARITY["short_gap_steps"]),
            "--medium_gap_steps", str(PARITY["medium_gap_steps"]),
            "--train_frac", str(PARITY["train_frac"]),
            "--val_frac", str(PARITY["val_frac"]),
            "--test_frac", str(PARITY["test_frac"]),
            "--pred_frac", str(PARITY["pred_frac"]),
            *[f"--{k}={v}" for k, v in strides.items()],
            "--seed", str(seed),
            "--output_dir", str(odir),
        ]
        ok = _run(cmd, args.standalone_dir, run_dir / "logs" / f"standalone_s{seed}.log")
        out["runs"][str(seed)] = {"dir": str(odir), "ok": ok}
    return out


def stage_sweep(run_dir: Path, cache_facts: dict, args) -> dict:
    """Run each HydraGNN backbone once per seed on the parity cache."""
    out = {}
    for mpnn in args.models:
        out[mpnn] = {}
        for seed in args.seeds:
            odir = run_dir / "hydragnn" / f"{mpnn}_s{seed}"
            if (odir / "metrics.json").exists():
                print(f"[skip] {mpnn} seed{seed} already complete")
                out[mpnn][str(seed)] = {"dir": str(odir), "ok": True}
                continue
            cmd = [
                sys.executable, "-u", str(HYDRA_SCRIPT),
                "--date", PARITY["date"],
                "--format", "pickle",
                "--cache_dir", cache_facts["cache_dir"],
                "--dataset_mode", PARITY["dataset_mode"],
                "--window_stride", str(PARITY["window_stride"]),
                "--skip_downstream_arrays",  # X.npy (~1.4 GB) not needed for the report
                "--mpnn_type", mpnn,
                "--hidden_dim", str(PARITY["hidden_dim"]),
                "--num_conv_layers", str(PARITY["num_conv_layers"]),
                "--num_epoch", str(PARITY["num_epoch"]),
                "--learning_rate", str(PARITY["learning_rate"]),
                "--batch_size", str(PARITY["batch_size"]),
                "--train_frac", str(PARITY["train_frac"]),
                "--val_frac", str(PARITY["val_frac"]),
                "--test_frac", str(PARITY["test_frac"]),
                "--seed", str(seed),
                "--out_dir", str(odir),
                "--log", f"cmp_{mpnn}_s{seed}",
            ]
            if args.cpu:
                cmd.append("--cpu")
            ok = _run(cmd, HERE, run_dir / "logs" / f"{mpnn}_s{seed}.log")
            out[mpnn][str(seed)] = {"dir": str(odir), "ok": ok}
    return out


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["cache", "baseline", "sweep", "all"], default="all")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Existing run dir to continue (required for staged use).")
    p.add_argument("--run_root", type=str, default="runs")
    p.add_argument("--cache_name", type=str, default="dataset_parity")
    p.add_argument("--force_cache", action="store_true")
    p.add_argument("--models", nargs="+", default=BACKBONES,
                   help=f"Subset of {BACKBONES_ALL}. TemporalGAT needs ~25GB VRAM at batch 32.")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--num_epoch", type=int, default=None,
                   help="Override PARITY['num_epoch'] (both sides). For smoke tests.")
    p.add_argument("--window_stride", type=int, default=None,
                   help="Override PARITY['window_stride'] (both sides). Larger = fewer "
                        "windows = faster/cheaper. For smoke tests.")
    p.add_argument("--standalone_dir", type=Path, default=DEFAULT_STANDALONE)
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    bad = [m for m in args.models if m not in BACKBONES_ALL]
    if bad:
        raise SystemExit(f"Unknown backbone(s) {bad}; valid: {BACKBONES_ALL}")

    # Overrides mutate PARITY so the manifest records the values actually used.
    if args.num_epoch is not None:
        PARITY["num_epoch"] = args.num_epoch
    if args.window_stride is not None:
        PARITY["window_stride"] = args.window_stride

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = (Path(args.run_root) / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}")

    mpath = run_dir / "manifest.json"
    manifest = {}
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text())
        except (json.JSONDecodeError, ValueError):
            # A prior run may have died mid-write (e.g. disk full), leaving an
            # empty/truncated manifest. Rebuild it; all state needed to resume is
            # recoverable from the run dirs + cache, and the stages skip whatever
            # already completed.
            print(f"[warn] {mpath} is corrupt/empty; rebuilding from scratch.")
    manifest.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    manifest["parity"] = PARITY
    manifest["seeds"] = args.seeds
    manifest["models"] = args.models
    manifest["provenance"] = {
        "hydragnn_sha": git_sha(HYDRA_REPO),
        "standalone_sha": git_sha(args.standalone_dir),
        "python": sys.version.split()[0],
        "argv": sys.argv,
    }
    try:
        import torch
        manifest["provenance"]["torch"] = torch.__version__
        manifest["provenance"]["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["provenance"]["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    def save():
        mpath.write_text(json.dumps(manifest, indent=2, default=str))

    save()

    if args.stage in ("cache", "all"):
        manifest["cache"] = stage_cache(run_dir, args); save()
    if "cache" not in manifest:
        raise SystemExit("No cache facts in manifest; run --stage cache first.")

    if args.stage in ("baseline", "all"):
        manifest["baseline"] = stage_baseline(run_dir, manifest["cache"], args); save()
    if args.stage in ("sweep", "all"):
        manifest["hydragnn"] = stage_sweep(run_dir, manifest["cache"], args); save()

    print(f"\n[done] manifest: {mpath}")
    print(f"[next] python make_report.py --run_dir {run_dir}")


if __name__ == "__main__":
    main()
