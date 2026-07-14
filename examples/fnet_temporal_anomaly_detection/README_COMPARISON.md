# HydraGNN temporal backbones vs. the standalone T-GCN — comparison harness

Reproduce-from-scratch guide and methodology for the predictive-performance
comparison. Any number in the report can be regenerated from two commands and
audited down to the raw arrays.

- **Question:** does the T-GCN → HydraGNN integration regress predictive
  performance, and can HydraGNN's alternative MPNN backbones exceed plain T-GCN?
- **Baseline:** Haoran's standalone T-GCN (`t-gcn/`).
- **Candidate:** HydraGNN's `Temporal*` backbones (`TemporalGCN/GIN/SAGE`).
- **Data:** FNET PMU, one day, geographic k-NN sensor graph.
- **Task:** multi-step forecasting, `Tin=100` history → `H=10` horizon.

---

## 1. Files

Everything is in `HydraGNN/examples/fnet_temporal_anomaly_detection/` unless noted.

**Entry points (what you run):**

| file | role |
|---|---|
| `run_comparison.py` | Runs the experiment: cache → standalone baseline → HydraGNN sweep. Writes `manifest.json`. **Computes no metrics.** |
| `make_report.py` | Reads the manifest, recomputes metrics from raw arrays, emits CSV/MD/TeX + parity check. **Trains nothing.** |

**Supporting modules:**

| file | role |
|---|---|
| `comparison_common.py` | Single source of truth: the `PARITY` dict, one `masked_metrics()`, array loaders, `check_parity()` |
| `fnet_window_dataset.py` | On-the-fly sliding-window dataset (see §5) |

**Diagnostics & docs:**

| file | role |
|---|---|
| `diagnose_time_axis.py` | The time-axis / alignment finding (see §6): schema, per-device clocks, coherence 2×2 |
| `ab_alignment_summary.py` | Summarizes the baseline A/B (old vs corrected alignment) |
| `bench_window_dataset.py` | Proves the on-the-fly dataset is bit-identical to the materialized cache, and benchmarks it |
| `NOTE_TO_HAORAN_time_axis.md` | Write-up of the alignment finding for Haoran |
| `README_COMPARISON.md` | This file |

**Modified existing code:**

- `fnet_temporal_anomaly_detection.py` (HydraGNN) — added `--dataset_mode`/`--window_stride`
  (on-the-fly windowing), `--skip_downstream_arrays`, and a seed-variation fix (§8).
- `t-gcn/train.py` + `t-gcn/data_processing.py` (**branch `kien/timestamp-alignment`**) —
  added `--time_axis`, `--alignment`, `--align_tolerance_ms`. Defaults reproduce Haoran's
  published numbers exactly.

Run outputs and caches (`runs/`, `dataset_*/`) are git-ignored — large and regenerable.
Reproduce by running the commands below, not by copying artifacts.

---

## 2. Quick start

Environment: `tgcn-gpu` (torch 2.5.0+cu124; `t-gcn` on branch `kien/timestamp-alignment`).
On Windows, `USE_LIBUV=0` is injected into child processes automatically (HydraGNN's
`setup_ddp()` crashes otherwise).

```bash
cd HydraGNN/examples/fnet_temporal_anomaly_detection

# End to end (cache → baseline → sweep). Hours on one GPU.
python run_comparison.py --stage all

# Produce every table from the saved arrays (no training).
python make_report.py --run_dir runs/<stamp>
```

Stage-by-stage (each is re-runnable and **resumable** — completed runs are skipped):

```bash
python run_comparison.py --stage cache
python run_comparison.py --stage baseline --run_dir runs/<stamp>
python run_comparison.py --stage sweep    --run_dir runs/<stamp>
```

Useful flags: `--models TemporalGCN TemporalGIN TemporalSAGE`, `--seeds 42 1 2`,
`--num_epoch N`, `--window_stride N` (larger = fewer windows = faster), `--force_cache`,
`--cpu`. Overrides are recorded in the manifest, so provenance stays honest.

The current definitive run is `runs/full5/` (stride 10, 5 epochs, 3 seeds).

---

## 3. Why this needs a harness

The two codebases have *independent* preprocessing and *independent* metric code.
Naively running both and diffing their reported numbers is wrong. Four real traps bit
us before this harness existed — every one produced *plausible-looking* numbers:

| Trap | What happened | Effect |
|---|---|---|
| **Scaling mismatch** | HydraGNN defaulted to `global` (unclipped); standalone uses `robust` + clip ±10. Raw PMU spikes to ~1800σ survived unclipped. | `pred` RMSE exploded to 4.3, R²→0. Pure artifact. |
| **Target-channel mismatch** | HydraGNN predicted 3 channels incl. `angle_delta`; standalone predicts 2. `angle_delta` is small-magnitude. | Deflated HydraGNN's pooled MAE, inverting the `TemporalGCN` result. |
| **Node-set mismatch** | HydraGNN's `--min_device_coverage 0.5` dropped 6 sensors → 101 nodes; standalone keeps 107. | Different graph, different scored population. |
| **Pooled-metric domination** | `volt_dev` is ~40× `freq_dev`; pooled MAE ≈ mean of the two. | `freq_dev` — the disturbance-relevant channel — contributes ~2% and is invisible. |

Two structural rules follow:

1. **One settings dict.** All shared settings live in `PARITY` (`comparison_common.py`).
   Both pipelines launch from it — there is no second place to change a hyperparameter.
2. **One metric implementation.** `make_report.py` recomputes every metric itself, from
   each side's raw saved prediction arrays, via a single `masked_metrics()`. Neither
   pipeline's own `metrics.json`/`config.json` numbers are trusted.

And a hard gate: `check_parity()` verifies `Tin`, `H`, node count, target-channel count
and names, scaling, clip, **time_axis, alignment**, hidden dim, conv layers, epochs,
batch size, learning rate, split fractions, retained-timestep count, and per-split window
counts. Any failure ⇒ the report is stamped `INVALID` and `make_report.py` exits non-zero.

### What the standalone's cov80 pipeline does (read from source, not inferred)

| | standalone | HydraGNN knob to match it |
|---|---|---|
| dynamic inputs | `F_dyn=5` (Δf, RoCoF, Δθ, ΔV, **observed-mask**) | `F_DYN=4` + `F_MASK=1` (automatic) |
| static inputs | `F_static=4` (grid embedding) | `--grid_embed_dim 4` (default) |
| **targets** | `F_out=2` — Δf, ΔV (**Δθ dropped**) | `--out_features 0 3` |
| device coverage | **no device-level drop** | `--min_device_coverage 0.0` |
| timestep coverage | `min_step_coverage=0.8` ← *this* is "cov80" | `--min_step_coverage 0.8` |
| time axis | GPS index (`--time_axis datetime`, branch) | reindex by timestamp (built in) |

---

## 4. Alignment (both sides matched to the corrected version)

Historically the two pipelines aligned sensors differently, which confounded model
quality with preprocessing quality. `PARITY` now pins **both** sides to the corrected
alignment (`time_axis=datetime`, `alignment=timestamp`); see §6 and
`NOTE_TO_HAORAN_time_axis.md` for why. Concretely:

- **Time axis** — the GPS measurement clock (the parquet's unnamed index), not
  `ReceivedTime`. `data/README.txt`: *"The received time … can be ignored, as it does not
  reflect the true measurement timestamp."*
- **Placement** — each sensor is reindexed onto a uniform grid by its own timestamps
  (nearest within ±50 ms; gaps recorded in `observed_mask`), not written contiguously.

With this, both sides retain ~863,983 timesteps (the `kept timesteps` parity check
passes), and both sample windows with the same rule (`arange(Tin-1, T-H, stride)`), so
window sampling is identical rather than merely similar.

---

## 5. Windowing (on-the-fly, `--dataset_mode ondemand`)

The original example materialized one PyG `Data` object per window (~0.39 MB each), so
the cache scaled with window count — stride-10 (full coverage) would need ~26 GB.
`fnet_window_dataset.py` instead caches only the `[T,N,F]` arrays (~1.7 GB) and builds
each window in `__getitem__`, making disk/RAM independent of the window count. It is
bit-identical to the materialized path (`bench_window_dataset.py` asserts it).

**Window budget = coverage.** A window supervises only its `H`-step horizon, so the
fraction of the timeline ever used as a target is `H / stride`. `PARITY["window_stride"]
= H = 10` ⇒ **100% coverage** (every timestep is predicted). A coarse stride (e.g. the
old `max_windows=2000` ≈ stride 328) covers only ~3%, leaving disturbances in the gaps.
Cost scales with `windows/stride × epochs` (i.e. total gradient steps); at equal steps a
small stride with few epochs dominates a large stride with many.

---

## 6. The time-axis finding (context for the alignment choice)

`diagnose_time_axis.py` establishes, from physics rather than naming, that the parquet's
anonymous index is the GPS clock and `ReceivedTime` is packet arrival:

- **115/125 devices** start at the same instant under the index; **1/125** under
  `ReceivedTime` (which lags 3.9–57.7 s, per device).
- Cross-sensor `freq_dev` correlation (same interconnection): **0.98** reindexing on the
  index vs **0.77** with contiguous placement; `angle_delta` **0.89 vs 0.13** (arrival-
  order breaks `np.unwrap`, which is cumulative).
- A/B on the standalone (`ab_alignment_summary.py`): correcting the alignment **halves
  `freq_dev` MAE** (−53% test) while `volt_dev` shifts ~+5%. Per-channel matters.

`t-gcn` branch `kien/timestamp-alignment` adds the flags; defaults reproduce Haoran's
report exactly. `NOTE_TO_HAORAN_time_axis.md` is the write-up to send upstream.

---

## 7. How metrics are defined

For each split, over **observed elements only** (`mask > 0.5`), pooled across every
(window, node, horizon, channel) element, in **original physical units**:

- `MAE = mean(|pred − true|)`, `RMSE = sqrt(mean((pred−true)²))`, `R² = 1 − SS_res/SS_tot`

HydraGNN saves predictions standardized; `load_hydra_split()` denormalizes with the
train-only scaler first. The standalone saves `*_original.npy` already denormalized.
Pooling is order-independent, so node *ordering* doesn't matter — only the node *set*,
which the parity check verifies. Metrics are emitted **per target channel** (`freq_dev`,
`volt_dev`) and pooled (`all`).

### Reading the numbers

**Read the per-channel rows. The pooled row is nearly meaningless** — the two targets
differ ~40× in magnitude, so pooled MAE is ~50% `volt_dev`, ~2% `freq_dev`. Conclusions
flip per channel (e.g. `TemporalGIN` beats the baseline on `volt_dev` but loses on
`freq_dev`). Pooled values are retained only for continuity with `report_2026-06-03.md`.

**MAE and RMSE are absolute and directly comparable — prefer them.** **R²** is normalized
by each pipeline's own target variance; now that window sampling matches, R² is far more
comparable across sides than before, but MAE/RMSE remain the primary quantities.

### Reported splits
Only **`test`** and **`pred`** are reported: the standalone writes prediction arrays only
for those, so `val` cannot be recomputed for both sides. `val` is model-selection; `pred`
(final held-out 10%) is the most realistic figure.

### Harness validation
`per_channel_metrics` on a standalone run reproduces the reference report's feature-level
`freq_dev` R² to within 0.001 (0.8925 vs 0.8933), confirming the loaders and metric code.

---

## 8. Output layout

```
runs/<stamp>/
  manifest.json              every setting, both repos' git SHAs, torch/GPU, artifact paths
  logs/                      one log per training run
  standalone/seed{42,1,2}/   baseline arrays + config.json
  hydragnn/<Backbone>_s<seed>/  preds/ys/masks .npy, feat_mean/std, out_idx
  report/
    parity_check.txt         the parity table (must be all OK)
    metrics_raw.csv          one row per (side, model, seed, split, channel)
    summary.csv              mean/std per (model, split, channel, metric)
    results.md               human-readable per-channel tables
    results.tex              booktabs tables (one per channel) for the paper
```

`manifest.json` records both repos' git SHAs (with a `-dirty` marker), torch version, and
GPU — so any result traces to the code that made it.

---

## 9. Known limitations

- **Training budget.** The definitive run is stride 10 (100% coverage) but only 5 epochs
  (the on-the-fly windowing at stride 10 is ~21 min/epoch on a laptop 4050). Both sides
  use the same value, so the comparison is internally valid; the absolute magnitudes
  would firm up with more epochs.
- **Single day.** Short-horizon PMU forecasting is highly autocorrelated, so absolute R²
  on `test` runs high. The named day folder (`2024-06-01`) actually contains
  **2024-01-01** (New Year's Day — atypical load). Validate across days before any
  generalization claim; only one day is currently on disk.
- **`TemporalGAT` excluded.** 6 attention heads × the `Tin=100` unroll need ~25 GB VRAM
  at batch 32 — OOM on a 6 GB card. Including it needs a lower batch, which breaks parity.
- **Seed variance covers weight init only.** `DistributedSampler` shuffles with a fixed
  seed, so batch order is constant across seeds.

### Upstream bug fixed along the way
`hydragnn/models/create.py` calls `torch.manual_seed(0)` unconditionally inside
`create_model()`, immediately before weight init — silently overriding `--seed` for
*every* HydraGNN example, so any reported mean±std was a single run repeated. The FNET
example now re-seeds and re-initializes the model after `create_model_config()` returns.
A proper upstream fix (making the seed configurable in `create.py`) is worth its own PR.

---

## 10. Provenance rule

Any table circulated must come from `runs/<stamp>/report/`, generated by `make_report.py`,
with `parity_check.txt` all-`OK`. If parity fails, the report self-labels `INVALID` and
the script exits non-zero — those numbers are not to be circulated.
