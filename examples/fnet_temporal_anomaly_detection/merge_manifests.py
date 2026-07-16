#!/usr/bin/env python3
"""
merge_manifests.py
===================

Combine manifest.json files from independent parallel run_comparison.py
invocations (each in its own --run_dir, on its own GPU) into one manifest
that make_report.py can read.

Why this exists: run_comparison.py's manifest.json has no locking, so two
processes writing to the SAME run_dir race and silently drop each other's
entries. The safe pattern is one run_dir per parallel invocation, each
writing its own manifest; this script merges them afterward by reading
only (never touching a manifest another process might still be writing).

Usage
-----
    python merge_manifests.py --out_dir runs/tier1_merged \\
        runs/20260714_210809 runs/full50_gin runs/full50_sage runs/full50_gat

The first run_dir must contain the "baseline" stage (standalone T-GCN);
later run_dirs are expected to contribute "hydragnn" entries only. Model-name
collisions are resolved last-wins, in the order given on the command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path,
                   help="run_comparison.py --run_dir paths to merge, in priority order.")
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    manifests = []
    for d in args.run_dirs:
        mpath = d / "manifest.json"
        if not mpath.exists():
            raise SystemExit(f"No manifest.json in {d}")
        manifests.append((d, json.loads(mpath.read_text())))

    base_dir, base = manifests[0]
    if "baseline" not in base:
        raise SystemExit(
            f"{base_dir} has no 'baseline' stage in its manifest yet -- "
            f"put the run_dir with the standalone baseline first."
        )
    if "cache" not in base:
        raise SystemExit(f"{base_dir} has no 'cache' stage in its manifest.")

    merged = dict(base)
    merged["hydragnn"] = {}
    merged["_merged_from"] = [str(d) for d, _ in manifests]

    missing = []
    for d, m in manifests:
        hy = m.get("hydragnn")
        if not hy:
            missing.append(str(d))
            continue
        merged["hydragnn"].update(hy)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "manifest.json"
    out_path.write_text(json.dumps(merged, indent=2, default=str))

    n_seed_runs = sum(len(v) for v in merged["hydragnn"].values())
    print(f"[merged] {n_seed_runs} seed-runs across {len(merged['hydragnn'])} model(s) "
          f"-> {out_path}")
    if missing:
        print(f"[warn] no 'hydragnn' stage found yet in: {', '.join(missing)} "
              f"(that run_dir's sweep may still be in progress -- rerun this merge once it finishes)")


if __name__ == "__main__":
    main()
