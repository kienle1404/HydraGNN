#!/usr/bin/env python3
"""
diagnose_time_axis.py
=====================

Self-contained diagnostic: which datetime field in the FNET parquet files is the
GPS measurement clock, and what does aligning on the wrong one cost?

Motivation
----------
The parquet schema is
    ['Frequency','VoltageAngle','VoltageMagnitude','ReceivedTime','__index_level_0__']
The index is *anonymous* (`name: None`). There is no field literally called
``DateTime``. The only self-describing timestamp is ``ReceivedTime``. So which
field is the measurement clock cannot be read off the file, and any pipeline has
to guess. This script decides it from physics instead of naming.

Test
----
Grid frequency is a system-wide quantity: sensors within one interconnection must
track each other almost perfectly. So the correct time axis is the one under
which cross-sensor Frequency correlation is ~1.

It then attributes the damage of the wrong choice across a 2x2:

                     offset from ReceivedTime | offset from index
    row order = RT       (standalone)         |  (ordering only)
    row order = index    (offset only)        |  (positional, done right)

so you can see whether the harm comes from *ordering* rows by arrival time, from
deriving each sensor's *start offset* from arrival time, or both. A reference row
shows timestamp-reindexing (nearest within dt/2), which sidesteps positional
placement entirely.

Depends only on pandas/numpy/pyarrow -- runnable inside the t-gcn repo.

Usage
-----
    python diagnose_time_axis.py \
        --data_dir ../../../data/2024-06-01 \
        --metadata_file ../../../data/FDRLocation.xlsx
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

IDX = "__index_level_0__"
HZ = 10.0
DT = pd.Timedelta("100ms")
TOL = pd.Timedelta("50ms")


def sid_of(path: str) -> int:
    return int(Path(path).stem.split("-")[0])


def report_schema(f: str):
    sch = pq.read_schema(f)
    print("=" * 78)
    print("1. SCHEMA -- the file does not label its clock")
    print("=" * 78)
    print(f"  arrow fields    : {sch.names}")
    meta = (sch.metadata or {}).get(b"pandas")
    if meta:
        import json
        p = json.loads(meta.decode())
        print(f"  index_columns   : {p.get('index_columns')}")
        for c in p.get("columns", []):
            if c.get("field_name") == IDX:
                print(f"  index descriptor: name={c.get('name')!r} type={c.get('pandas_type')!r}")
    print("  -> the only NAMED timestamp is 'ReceivedTime'; the index is anonymous.\n")


def per_device_stats(files, sample_every=50):
    """Start instants + latency for every device."""
    rows = []
    for f in files:
        try:
            t = pq.read_table(f, columns=["ReceivedTime", IDX])
            rt = t.column("ReceivedTime").to_numpy(zero_copy_only=False)
            ix = t.column(IDX).to_numpy(zero_copy_only=False)
            if len(rt) < 1000:
                continue
            lag = (rt[::sample_every] - ix[::sample_every]) / np.timedelta64(1, "s")
            rows.append(dict(
                sid=sid_of(f), n=len(rt),
                ix_min=ix.min(), rt_min=rt.min(),
                lag_mean=float(np.nanmean(lag)),
                ix_monotonic=bool(np.all(np.diff(ix) >= np.timedelta64(0))),
                rt_monotonic=bool(np.all(np.diff(rt) >= np.timedelta64(0))),
                rt_unique_frac=float(len(np.unique(rt)) / len(rt)),
            ))
        except Exception as e:
            print(f"  [skip] {Path(f).name}: {e}")
    return pd.DataFrame(rows)


def _ordered(d, order_key, channel):
    """Rows in the chosen order, and the channel computed IN THAT ORDER.

    Both pipelines compute the dynamic features per device *before* alignment, so
    the row order determines the derivative itself for differenced channels --
    not merely where each sample lands. `rocof` under arrival-order is a diff
    between samples that are not adjacent in measurement time.
    """
    dd = d.sort_index() if order_key == "index" else d.sort_values("ReceivedTime")
    if channel == "angle_delta":
        # Angle is radians wrapped to [0, 2pi) (see data/README.txt), so it must be
        # unwrapped before differencing. np.unwrap assumes CONSECUTIVE samples and is
        # cumulative: one spurious +/-2pi correction persists for the rest of the
        # series. That makes it far more order-sensitive than a plain diff.
        a = np.unwrap(dd["VoltageAngle"].to_numpy(dtype=float))
        v = np.zeros_like(a)
        v[1:] = np.diff(a)
        if len(v) > 1:
            v[0] = v[1]
        return dd.index, v
    v = dd["Frequency"].to_numpy(dtype=float) - 60.0          # freq_dev
    if channel == "rocof":
        r = np.zeros_like(v)
        r[1:] = np.diff(v)
        if len(r) > 1:
            r[0] = r[1]
        v = r
    return dd.index, v


def coherence_matrix(frames, order_key, offset_key, channel):
    """Standalone-style positional placement: sort rows, offset by start, write contiguously."""
    def start(d):
        return d.index.min() if offset_key == "index" else d["ReceivedTime"].min()
    g0 = min(start(d) for d in frames.values())
    L = max(int(round((start(d) - g0).total_seconds() * HZ)) + len(d) for d in frames.values())
    M = np.full((L, len(frames)), np.nan)
    for j, (sid, d) in enumerate(frames.items()):
        off = max(int(round((start(d) - g0).total_seconds() * HZ)), 0)
        _, v = _ordered(d, order_key, channel)
        M[off:off + len(v), j] = v
    return pd.DataFrame(M, columns=list(frames))


def coherence_reindex(frames, channel):
    """HydraGNN-style: features in measurement order, then reindex by own timestamps."""
    cols = {}
    for s, d in frames.items():
        idx, v = _ordered(d, "index", channel)
        ser = pd.Series(v, index=idx)
        cols[s] = ser[~ser.index.duplicated(keep="first")]
    grid = pd.date_range(max(c.index.min() for c in cols.values()),
                         min(c.index.max() for c in cols.values()), freq=DT)
    return pd.DataFrame({s: c.reindex(grid, method="nearest", tolerance=TOL)
                         for s, c in cols.items()})


def mean_corr(M):
    C = M.corr().to_numpy()
    v = C[np.triu_indices_from(C, 1)]
    v = v[np.isfinite(v)]
    return v.mean(), np.median(v), v.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--metadata_file", type=Path, required=True)
    ap.add_argument("--n_coherence", type=int, default=10, help="devices used for the coherence test")
    ap.add_argument("--nrows", type=int, default=180000, help="rows per device for coherence (~5h)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.data_dir / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet under {args.data_dir}")
    report_schema(files[0])

    print("=" * 78)
    print(f"2. PER-DEVICE CLOCKS  ({len(files)} files)")
    print("=" * 78)
    st = per_device_stats(files)
    g_ix, g_rt = st.ix_min.min(), st.rt_min.min()
    st["off_ix"] = ((st.ix_min - g_ix).dt.total_seconds() * HZ).round().astype(int)
    st["off_rt"] = ((st.rt_min - g_rt).dt.total_seconds() * HZ).round().astype(int)
    st["err"] = (st.off_rt - st.off_ix).abs()
    print(f"  devices analysed          : {len(st)}")
    print(f"  latency (RT - index), s   : min={st.lag_mean.min():.2f}  max={st.lag_mean.max():.2f}  "
          f"spread={st.lag_mean.max()-st.lag_mean.min():.2f}")
    print(f"  devices starting at the global index instant : "
          f"{(st.off_ix == 0).sum()}/{len(st)}   (a synchronized clock's signature)")
    print(f"  devices starting at the global RT instant    : {(st.off_rt == 0).sum()}/{len(st)}")
    print(f"  |offset error| samples    : median={st.err.median():.0f}  p90={st.err.quantile(.9):.0f}  max={st.err.max():.0f}")
    print(f"  |offset error| seconds    : median={st.err.median()/HZ:.2f}  p90={st.err.quantile(.9)/HZ:.2f}  max={st.err.max()/HZ:.2f}")
    print(f"  index monotonic as stored : {int(st.ix_monotonic.sum())}/{len(st)}")
    print(f"  RT    monotonic as stored : {int(st.rt_monotonic.sum())}/{len(st)}")
    print(f"  RT distinct-value frac    : median={st.rt_unique_frac.median():.4f}  "
          f"(low => many samples share one arrival stamp => tie-order is arbitrary)\n")

    # --- coherence: pick devices from the largest interconnection ---
    md = pd.read_excel(args.metadata_file)[["FDRID", "GridName"]].dropna()
    md["FDRID"] = md.FDRID.astype(int)
    have = {sid_of(f): f for f in files}
    md = md[md.FDRID.isin(have)]
    top = md.GridName.value_counts().idxmax()
    sel = md[md.GridName == top].FDRID.tolist()[: args.n_coherence]

    frames = {}
    for sid in sel:
        d = pd.read_parquet(have[sid], columns=["Frequency", "VoltageAngle", "ReceivedTime"]).iloc[: args.nrows]
        d.index = pd.to_datetime(d.index)
        d["ReceivedTime"] = pd.to_datetime(d["ReceivedTime"])
        frames[sid] = d

    print("=" * 78)
    print(f"3. CROSS-SENSOR COHERENCE  (grid={top}, {len(frames)} sensors, {args.nrows} rows)")
    print("=" * 78)
    print("   Interconnection quantities are system-wide: correct alignment => high corr.")
    print("   'freq_dev' is a smooth level; 'rocof' is its per-sample derivative, computed")
    print("   per device BEFORE alignment -- so row order changes the derivative itself.\n")

    variants = [
        ("reindex on index timestamps (HydraGNN)", ("reindex", None, None)),
        ("order=index, offset=index  (positional, correct)", ("pos", "index", "index")),
        ("order=index, offset=RT     (offset error only)", ("pos", "index", "rt")),
        ("order=RT,    offset=index  (ordering error only)", ("pos", "rt", "index")),
        ("order=RT,    offset=RT     (standalone T-GCN)", ("pos", "rt", "rt")),
    ]
    for channel in ("freq_dev", "rocof", "angle_delta"):
        print(f"   --- channel: {channel} ---")
        print(f"   {'variant':52s} {'mean':>8s} {'median':>8s} {'min':>8s}")
        print("   " + "-" * 78)
        for name, (kind, ok, fk) in variants:
            M = coherence_reindex(frames, channel) if kind == "reindex" \
                else coherence_matrix(frames, ok, fk, channel)
            m, med, mn = mean_corr(M)
            print(f"   {name:52s} {m:8.4f} {med:8.4f} {mn:8.4f}")
        print()


if __name__ == "__main__":
    main()
