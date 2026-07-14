#!/usr/bin/env python3
"""
make_report.py
==============

Extraction + reporting for the HydraGNN-vs-standalone-T-GCN comparison.

Reads the ``manifest.json`` written by ``run_comparison.py``, then:

  1. Runs ``check_parity`` -- verifies both pipelines solved the SAME task
     (same nodes, same target channels, same Tin/H, comparable window counts).
     If any check fails, the report is still written but is stamped INVALID and
     the process exits non-zero. Numbers from a failed-parity run must not be
     circulated.
  2. Recomputes every metric here, from each side's raw saved arrays, using the
     single ``masked_metrics`` implementation -- never from the pipelines' own
     metrics.json. This is what makes the comparison auditable.
  3. Aggregates across seeds (mean +/- std) for BOTH sides.
  4. Emits, into ``<run_dir>/report/``:
       - ``metrics_raw.csv``   one row per (side, model, seed, split)
       - ``summary.csv``       mean/std per (model, split, metric)
       - ``parity_check.txt``  the parity table
       - ``results.md``        human-readable tables
       - ``results.tex``       booktabs table for the paper

Only ``test`` and ``pred`` are reported: the standalone does not save validation
arrays, so ``val`` cannot be independently recomputed for both sides. ``val`` is
a model-selection split and is not part of the claim.

Usage
-----
    python make_report.py --run_dir runs/20260704_120000
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

from comparison_common import (
    RECOMPUTED_SPLITS,
    STANDALONE_TARGETS,
    check_parity,
    format_parity,
    load_hydra_split,
    load_standalone_split,
    parity_ok,
    per_channel_metrics,
)

METRICS = ("mae", "rmse", "r2")
STANDALONE = "Standalone T-GCN"
# "all" (pooled) is reported for continuity, but the per-channel rows are the
# primary view: freq_dev and volt_dev differ ~40x in magnitude, so a pooled MAE
# is dominated by volt_dev. See per_channel_metrics() for the rationale.
CHANNELS = ("all",) + tuple(STANDALONE_TARGETS)


def collect(manifest: dict) -> list:
    """One row per (side, model, seed, split, channel); metrics recomputed here."""
    rows = []

    def add(side, model, seed, split, p, y, m):
        for chan, met in per_channel_metrics(p, y, m).items():
            rows.append({"side": side, "model": model, "seed": int(seed),
                         "split": split, "channel": chan, **met})

    for seed, info in manifest.get("baseline", {}).get("runs", {}).items():
        if not info.get("ok"):
            continue
        for split in RECOMPUTED_SPLITS:
            add("standalone", STANDALONE, seed, split, *load_standalone_split(Path(info["dir"]), split))
    for model, seeds in manifest.get("hydragnn", {}).items():
        for seed, info in seeds.items():
            if not info.get("ok"):
                continue
            for split in RECOMPUTED_SPLITS:
                add("hydragnn", model, seed, split, *load_hydra_split(Path(info["dir"]), split))
    return rows


def aggregate(rows: list) -> dict:
    """(model, split, channel, metric) -> (mean, std, n_seeds)."""
    buckets = {}
    for r in rows:
        for metric in METRICS:
            buckets.setdefault((r["model"], r["split"], r["channel"], metric), []).append(r[metric])
    return {
        k: (st.fmean(v), st.pstdev(v) if len(v) > 1 else 0.0, len(v))
        for k, v in buckets.items()
    }


def _cell(agg, model, split, channel, metric, prec=4):
    got = agg.get((model, split, channel, metric))
    if not got:
        return "-"
    mean, std, n = got
    return f"{mean:.{prec}g}" + (f" ± {std:.2g}" if n > 1 else "")


def order_models(rows: list) -> list:
    models = {r["model"] for r in rows}
    rest = sorted(m for m in models if m != STANDALONE)
    return ([STANDALONE] if STANDALONE in models else []) + rest


def write_markdown(path: Path, agg, models, manifest, checks):
    P = manifest["parity"]
    prov = manifest.get("provenance", {})
    valid = parity_ok(checks)
    L = []
    L.append("# HydraGNN vs. Standalone T-GCN — Results\n")
    if not valid:
        L.append("> **INVALID — PARITY CHECK FAILED.** The two pipelines did not solve the\n"
                 "> same task. Do not circulate these numbers. See `parity_check.txt`.\n")
    L.append(f"- Generated: {manifest.get('created')}")
    L.append(f"- HydraGNN @ `{prov.get('hydragnn_sha')}`, standalone @ `{prov.get('standalone_sha')}`")
    L.append(f"- torch {prov.get('torch')} (cuda={prov.get('cuda')}, {prov.get('gpu', 'cpu')})")
    L.append(f"- Day `{P['date']}`, Tin={P['Tin']}, H={P['horizon']}, "
             f"targets={P['out_features']}, scaling={P['scaling']}+clip{P['scaled_clip_value']}")
    cov = 100.0 * P["horizon"] / P["window_stride"]
    L.append(f"- hidden={P['hidden_dim']}, layers={P['num_conv_layers']}, epochs={P['num_epoch']}, "
             f"lr={P['learning_rate']}, batch={P['batch_size']}")
    L.append(f"- window stride={P['window_stride']} (both sides) → **{cov:.0f}% target coverage**; "
             f"dataset_mode={P['dataset_mode']}")
    L.append(f"- Seeds: {manifest.get('seeds')} (mean ± std, both sides)")
    L.append(f"- Nodes: {manifest['cache']['n_nodes']}, aligned timesteps: {manifest['cache']['aligned_timesteps']}\n")
    L.append("All metrics are recomputed by `make_report.py` from raw prediction arrays in "
             "original (physical) units, masked to observed targets, using one shared "
             "`masked_metrics()` for both sides.\n")
    L.append("> **Read the per-channel rows, not the pooled one.** `freq_dev` and `volt_dev`\n"
             "> differ ~40x in magnitude, so the pooled (`all`) MAE is dominated by `volt_dev`\n"
             "> and says almost nothing about `freq_dev`. Pooled values are shown for continuity\n"
             "> with the reference report only.\n>\n"
             "> **MAE and RMSE are absolute and directly comparable.** R² is normalized by each\n"
             "> pipeline's own target variance, and the two sample different windows, so R² is\n"
             "> only loosely comparable across sides.\n")
    for split in RECOMPUTED_SPLITS:
        L.append(f"\n## {split}\n")
        for chan in CHANNELS:
            label = "pooled (all channels)" if chan == "all" else f"channel: `{chan}`"
            L.append(f"\n### {split} — {label}\n")
            L.append("| Model | MAE | RMSE | R² |")
            L.append("|---|---|---|---|")
            for m in models:
                L.append(f"| {m} | {_cell(agg, m, split, chan, 'mae')} | "
                         f"{_cell(agg, m, split, chan, 'rmse')} | {_cell(agg, m, split, chan, 'r2')} |")
    L.append("\n## Parity check\n\n```\n" + format_parity(checks) + "\n```\n")
    L.append("\n## Window budget\n")
    L.append(f"Both sides take every {P['window_stride']}-th window anchor "
             f"(`arange(Tin-1, T-H, {P['window_stride']})`), giving **{cov:.0f}% target coverage** "
             f"— each timestep appears in a forecast horizon. Because the sampling *scheme* now "
             f"matches (not just the count), R² denominators are far closer than under the old "
             f"linspace-vs-stride setup.\n")
    if P["num_epoch"] != 50:
        L.append(f"> Trained for {P['num_epoch']} epochs, not the reference run's 50, to hold the "
                 f"gradient-step budget (and wall clock) reasonable at this stride. Both sides use "
                 f"the same value, so the comparison is internally valid; absolute numbers will sit "
                 f"below `t-gcn/report_2026-06-03.md`.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def write_latex(path: Path, agg, models, manifest, checks):
    """One table per target channel (the per-channel view is the primary one)."""
    P = manifest["parity"]

    def cell(m, s, c, k):
        got = agg.get((m, s, c, k))
        if not got:
            return "--"
        mean, std, n = got
        return f"${mean:.4g}$" + (f"{{\\scriptsize$\\pm{std:.2g}$}}" if n > 1 else "")

    L = [
        "% Auto-generated by make_report.py -- do not edit by hand.",
        f"% run: {manifest.get('created')}  hydragnn@{manifest['provenance'].get('hydragnn_sha')}",
        f"% parity: {'OK' if parity_ok(checks) else 'FAILED -- DO NOT USE'}",
        f"% window stride={P['window_stride']} both sides "
        f"({100.0*P['horizon']/P['window_stride']:.0f}% target coverage), "
        f"{P['num_epoch']} epochs",
        "% Requires \\usepackage{booktabs}",
    ]
    for chan in CHANNELS:
        label = "pooled over both channels" if chan == "all" else f"channel \\texttt{{{chan}}}"
        L += [
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\caption{{Forecasting accuracy on FNET PMU data ({P['date']}), {label}. "
            f"Matched preprocessing ({P['scaling']} scaling, clip $\\pm{int(P['scaled_clip_value'])}$; "
            f"$T_\\text{{in}}={P['Tin']}$, $H={P['horizon']}$; hidden {P['hidden_dim']}, "
            f"{P['num_conv_layers']} conv layers, {P['num_epoch']} epochs, "
            f"window stride {P['window_stride']}). Original physical units, "
            f"mean~$\\pm$~std over {len(manifest['seeds'])} seeds. Lower MAE/RMSE is better.}}",
            f"  \\label{{tab:cmp_{chan}}}",
            "  \\begin{tabular}{l" + "ccc" * len(RECOMPUTED_SPLITS) + "}",
            "    \\toprule",
            "    & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{s}}}}}" for s in RECOMPUTED_SPLITS) + " \\\\",
            "    " + "".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(RECOMPUTED_SPLITS))),
            "    Model & " + " & ".join(["MAE & RMSE & $R^2$"] * len(RECOMPUTED_SPLITS)) + " \\\\",
            "    \\midrule",
        ]
        for m in models:
            cells = " & ".join(cell(m, s, chan, k) for s in RECOMPUTED_SPLITS for k in METRICS)
            L.append(f"    {m.replace('_', chr(92) + '_')} & {cells} \\\\")
            if m == STANDALONE:
                L.append("    \\midrule")
        L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    out = args.run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)

    # --- parity: pick any completed hydragnn trial + any completed baseline ---
    h_trial = next((i["dir"] for s in manifest.get("hydragnn", {}).values()
                    for i in s.values() if i.get("ok")), None)
    s_run = next((i["dir"] for i in manifest.get("baseline", {}).get("runs", {}).values()
                  if i.get("ok")), None)
    if not h_trial or not s_run:
        raise SystemExit("Need at least one successful run on each side.")
    checks = check_parity(
        Path(h_trial), Path(s_run),
        hydra_aligned_timesteps=manifest.get("cache", {}).get("aligned_timesteps"),
        parity=manifest.get("parity"),
    )
    (out / "parity_check.txt").write_text(format_parity(checks), encoding="utf-8")
    print(format_parity(checks))

    rows = collect(manifest)
    if not rows:
        raise SystemExit("No metrics collected.")
    agg = aggregate(rows)
    models = order_models(rows)

    with open(out / "metrics_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["side", "model", "seed", "split", "channel", *METRICS, "mse", "n"])
        w.writeheader(); w.writerows(rows)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "split", "channel", "metric", "mean", "std", "n_seeds"])
        for (m, s, c, k), (mean, sd, n) in sorted(agg.items()):
            w.writerow([m, s, c, k, f"{mean:.6g}", f"{sd:.6g}", n])

    write_markdown(out / "results.md", agg, models, manifest, checks)
    write_latex(out / "results.tex", agg, models, manifest, checks)

    print(f"\n[out] {out}/  (metrics_raw.csv, summary.csv, parity_check.txt, results.md, results.tex)")
    for split in RECOMPUTED_SPLITS:
        for chan in CHANNELS:
            tag = "pooled" if chan == "all" else chan
            print(f"\n--- {split} / {tag} (original units, mean ± std) ---")
            for m in models:
                print(f"  {m:20s} MAE={_cell(agg, m, split, chan, 'mae'):>18s}  "
                      f"RMSE={_cell(agg, m, split, chan, 'rmse'):>18s}  "
                      f"R2={_cell(agg, m, split, chan, 'r2'):>18s}")

    if not parity_ok(checks):
        print("\n[INVALID] parity check FAILED — do not circulate these numbers.", file=sys.stderr)
        sys.exit(1)
    print("\n[OK] parity check passed.")


if __name__ == "__main__":
    main()
