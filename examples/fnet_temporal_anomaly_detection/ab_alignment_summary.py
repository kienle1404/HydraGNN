#!/usr/bin/env python3
"""
ab_alignment_summary.py
=======================

Summarize the standalone T-GCN A/B: historical alignment vs corrected alignment.

  baseline : --time_axis received --alignment offset   (current default)
  fixed    : --time_axis datetime --alignment timestamp

Both sides are the SAME model and hyperparameters; only preprocessing differs. So
any delta is attributable to the alignment, not to modelling.

Metrics are recomputed here from each run's saved prediction arrays with the same
`masked_metrics` used everywhere else, and reported **per target channel**
(`freq_dev`, `volt_dev` differ ~40x in magnitude, so a pooled MAE is dominated by
`volt_dev` -- see comparison_common.per_channel_metrics).

Caveat baked into this comparison: the two runs do not evaluate identical windows.
Correcting the alignment retains 863,983 timesteps instead of 819,582, so the
splits cover different spans. That is intrinsic to the fix, not a flaw in the A/B:
each model is scored on its own held-out data. Absolute MAE/RMSE remain comparable;
R^2 is normalized per-run and should be read with care.

Usage
-----
    python ab_alignment_summary.py --root ../../../t-gcn/results_ab --seeds 42 1 2
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from comparison_common import (
    RECOMPUTED_SPLITS,
    STANDALONE_TARGETS,
    load_standalone_split,
    per_channel_metrics,
)

METRICS = ("mae", "rmse", "r2")
CHANNELS = ("all",) + tuple(STANDALONE_TARGETS)


def collect(root: Path, arm: str, seeds):
    rows = []
    for s in seeds:
        d = root / f"{arm}_s{s}"
        if not (d / "test_predictions_original.npy").exists():
            print(f"  [missing] {d}")
            continue
        for split in RECOMPUTED_SPLITS:
            p, y, m = load_standalone_split(d, split)
            for chan, met in per_channel_metrics(p, y, m).items():
                rows.append({"arm": arm, "seed": s, "split": split, "channel": chan, **met})
    return rows


def agg(rows):
    b = {}
    for r in rows:
        for k in METRICS:
            b.setdefault((r["arm"], r["split"], r["channel"], k), []).append(r[k])
    return {k: (st.fmean(v), st.pstdev(v) if len(v) > 1 else 0.0, len(v)) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("../../../t-gcn/results_ab"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    args = ap.parse_args()

    rows = collect(args.root, "baseline", args.seeds) + collect(args.root, "fixed", args.seeds)
    if not rows:
        raise SystemExit(f"no runs found under {args.root}")
    A = agg(rows)

    print("Standalone T-GCN: historical alignment vs corrected alignment")
    print("  baseline = received/offset      fixed = datetime/timestamp")
    print(f"  same model + hyperparameters; seeds {args.seeds}; original physical units\n")

    for split in RECOMPUTED_SPLITS:
        print(f"=== {split} ===")
        print(f"  {'channel':12s} {'metric':6s} {'baseline':>18s} {'fixed':>18s} {'change':>12s}")
        print("  " + "-" * 72)
        for chan in CHANNELS:
            for k in METRICS:
                b = A.get(("baseline", split, chan, k))
                f = A.get(("fixed", split, chan, k))
                if not b or not f:
                    continue
                bm, bs, bn = b
                fm, fs, fn = f
                if k == "r2":
                    delta = f"{fm - bm:+.4f}"
                else:  # lower is better
                    delta = f"{100.0 * (fm - bm) / bm:+.1f}%" if bm else "-"
                bcell = f"{bm:.4g}" + (f"±{bs:.2g}" if bn > 1 else "")
                fcell = f"{fm:.4g}" + (f"±{fs:.2g}" if fn > 1 else "")
                print(f"  {chan:12s} {k:6s} {bcell:>18s} {fcell:>18s} {delta:>12s}")
            print()
    print("For MAE/RMSE 'change' is percent (negative = better).")
    print("For R2 it is an absolute difference (positive = better).")


if __name__ == "__main__":
    main()
