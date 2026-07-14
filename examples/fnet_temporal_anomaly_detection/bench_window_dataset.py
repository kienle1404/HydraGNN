#!/usr/bin/env python3
"""
bench_window_dataset.py
=======================

Validates and benchmarks ``FNETWindowDataset`` (on-the-fly windowing) against the
materialized ``SimplePickleDataset`` cache.

1. **Correctness first.** For a set of window indices, assert every tensor of the
   on-the-fly ``Data`` is bit-identical to the pickled one. A speedup is
   meaningless if the windows differ.
2. **Throughput.** items/sec for each, plus DataLoader-collated batch throughput.
3. **Feasibility.** Extrapolate RAM / disk / epoch cost at the canonical
   stride-10 budget (~65.5k train windows) that the materialized path cannot reach.

Usage
-----
    python bench_window_dataset.py --cache_dir dataset_robust2k --date 2024-06-01
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from fnet_window_dataset import FNETWindowDataset, window_anchors
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

MB = 1024**2
GB = 1024**3


def compare(a, b, idx) -> list:
    """Return list of mismatching tensor keys between two Data objects."""
    bad = []
    keys = set(a.keys()) | set(b.keys()) if callable(getattr(a, "keys", None)) is False else set(a.keys()) | set(b.keys())
    for k in sorted(keys):
        ta, tb = getattr(a, k, None), getattr(b, k, None)
        if ta is None or tb is None:
            bad.append(f"{k}: present={ta is not None}/{tb is not None}")
            continue
        if isinstance(ta, torch.Tensor):
            if ta.shape != tb.shape:
                bad.append(f"{k}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
            elif not torch.equal(ta, tb):
                d = (ta.float() - tb.float()).abs().max().item()
                bad.append(f"{k}: max|diff|={d:.3e}")
        elif ta != tb:
            bad.append(f"{k}: {ta} vs {tb}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, default=Path("dataset_robust2k"))
    ap.add_argument("--date", type=str, default="2024-06-01")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--max_windows", type=int, default=2000)
    ap.add_argument("--n_check", type=int, default=8)
    ap.add_argument("--n_time", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    with open(args.cache_dir / f"fnet_{args.date}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    X, om = meta["X"], meta["observed_mask"]
    T, N, F = X.shape
    Tin, H = meta["Tin"], meta["horizon"]
    print(f"[meta] X={X.shape} Tin={Tin} H={H} F_out={meta['F_out']} N={N}")
    print(f"[meta] X={X.nbytes/GB:.2f} GB  observed_mask={om.nbytes/GB:.2f} GB "
          f"-> arrays held once: {(X.nbytes+om.nbytes)/GB:.2f} GB\n")

    # The materialized trainset was built from the *train segment* of X.
    n_train = int(T * args.train_frac)
    ds_live = FNETWindowDataset(
        X[0:n_train], om[0:n_train], meta["grid_embed"], meta["edge_index"],
        Tin, H, meta["out_idx"], edge_weight=meta.get("edge_weight"),
        predict_delta=bool(meta.get("predict_delta", False)),
        max_windows=args.max_windows,
    )
    ds_mat = SimplePickleDataset(
        basedir=str(args.cache_dir / f"fnet_{args.date}.pickle"), label="trainset"
    )
    print(f"[len] on-the-fly={len(ds_live)}  materialized={len(ds_mat)}")
    assert len(ds_live) == len(ds_mat), "anchor sets differ -- cannot compare"

    # ---------------- 1. correctness ----------------
    idxs = np.linspace(0, len(ds_live) - 1, args.n_check, dtype=int)
    n_bad = 0
    for i in idxs:
        bad = compare(ds_live[int(i)], ds_mat[int(i)], i)
        if bad:
            n_bad += 1
            print(f"  [MISMATCH] window {i}: {bad}")
    if n_bad == 0:
        print(f"[correctness] OK — {len(idxs)} windows bit-identical to the cache\n")
    else:
        raise SystemExit(f"[correctness] FAILED on {n_bad}/{len(idxs)} windows")

    # ---------------- 2. throughput ----------------
    def timeit(ds, n):
        ii = np.random.default_rng(0).integers(0, len(ds), n)
        _ = ds[int(ii[0])]  # warm
        t0 = time.perf_counter()
        for i in ii:
            _ = ds[int(i)]
        return n / (time.perf_counter() - t0)

    live_ips = timeit(ds_live, args.n_time)
    mat_ips = timeit(ds_mat, args.n_time)
    print(f"[item throughput] on-the-fly {live_ips:8.1f} items/s")
    print(f"[item throughput] materialized {mat_ips:8.1f} items/s  "
          f"(on-the-fly is {live_ips/mat_ips:.2f}x)\n")

    dl = DataLoader(ds_live, batch_size=args.batch_size, shuffle=False)
    t0 = time.perf_counter(); nb = 0
    for b in dl:
        nb += 1
        if nb >= 20:
            break
    bps = nb / (time.perf_counter() - t0)
    print(f"[batch throughput] on-the-fly {bps:.1f} batches/s "
          f"({bps*args.batch_size:.0f} windows/s) at batch={args.batch_size}\n")

    # ---------------- 3. feasibility at stride-10 ----------------
    per_window_mb = 0.39  # measured size of one pickled Data object
    n10 = len(window_anchors(n_train, Tin, H, stride=H))
    print("=== canonical stride-10 budget (target horizons tile the timeline) ===")
    print(f"  train windows        : {n10:,}  (vs {len(ds_live):,} at max_windows={args.max_windows})")
    print(f"  materialized disk    : {n10*per_window_mb/1024:.1f} GB  (train split alone)")
    print(f"  on-the-fly disk      : {(X.nbytes+om.nbytes)/GB:.2f} GB  (arrays held once, all splits)")
    print(f"  on-the-fly RAM       : {(X.nbytes+om.nbytes)/GB:.2f} GB")
    sec = n10 / max(bps * args.batch_size, 1e-9)
    print(f"  data-loading time/epoch (on-the-fly, 1 worker): {sec:.0f} s "
          f"({sec/60:.1f} min) — overlappable with GPU compute via num_workers>0")


if __name__ == "__main__":
    main()
