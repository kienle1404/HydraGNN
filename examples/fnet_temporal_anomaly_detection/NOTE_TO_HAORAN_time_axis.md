# FNET sensor alignment — two findings

While matching preprocessing between your T-GCN and the HydraGNN port, we found two things:

---

## 1. `ReceivedTime` is not the measurement clock

The true timestamp is the parquet index. In the
files it's written as an *unnamed* index, which makes it easy to miss — `ReceivedTime` is
the only named datetime column.

`data_processing.load_sensor_data` sorts by `ReceivedTime` and drops the index, and
`train.py` derives each sensor's start offset from `ReceivedTime`. Measured over all 125
devices: `ReceivedTime` lags the true timestamp by **3.9–57.7 s, varying per device**. 115
of 125 devices start at the same instant under the true clock; only 1 does under
`ReceivedTime`. The resulting sensor-to-sensor displacement is a **median of 5 s**.

Two consequences:

- **Propagation timing.** Disturbances cross the interconnection in well under a second, so
  a 5 s inter-sensor displacement makes the propagation ordering reflect network latency
  rather than grid physics.
- **`angle_delta` is corrupted.** `np.unwrap` assumes consecutive samples and is cumulative
  — one out-of-order sample injects a ±2π that persists for the rest of the series. About
  10 samples share each `ReceivedTime` stamp, so tie-order is arbitrary. Cross-sensor
  coherence of `angle_delta` falls from **0.74 to 0.13** purely from the sort order.
  (`freq_dev`, being a level, is unaffected by ordering; `rocof` at 10 Hz is below the PMU
  noise floor and shows no coherence either way.)

## 2. Contiguous placement drops cross-sensor coherence

Writing each device's samples as an unbroken block assumes a gap-free 10 Hz stream. Real
streams have gaps (81 in a day for one device), and each gap permanently shifts the rest of
that device's series.

Mean pairwise correlation across 10 sensors in `01-EI` (5 h):

| | `freq_dev` | `angle_delta` |
|---|---|---|
| reindex each device by its own timestamps | **0.98** | **0.89** |
| contiguous placement, correct clock | 0.78 | 0.74 |
| contiguous placement, `ReceivedTime` (current) | 0.77 | 0.13 |

Note the clock fix alone barely moves `freq_dev` (0.77 → 0.78) — the placement is what
costs the spatial signal. Reindexing each device onto the grid by its own timestamps fixes
it, and the artifact this could introduce (one sample claimed by two adjacent slots) we
measured at 0.03%.

Since T-GCN's value comes from cross-sensor structure, we'd expect this to matter.

We implemented both as flags and ran the 2×2 through `train.py`. Timesteps surviving the
cov80 filter:

| | `alignment=offset` | `alignment=timestamp` |
|---|---|---|
| `time_axis=received` | **94.85%** (819,582 / 864,064 — reproduces your report exactly) | 9.22% |
| `time_axis=datetime` | 94.86% | **100.00%** (863,983 / 864,001) |

Three things this shows. Fixing the
clock alone changes nothing (94.85% → 94.86%) — the coverage loss is caused by contiguous
placement truncating each sensor's tail, not by the clock. And `received` + `timestamp`
collapsing to 9.22% is itself a proof that `ReceivedTime` is not a usable time axis: it is
bursty (~10 samples share one stamp), so it cannot populate a 10 Hz grid at all.

The corrected combination retains essentially the whole day, which also brings the
end-of-day coverage decay (we measured `pred` quintiles falling 0.897 → 0.813) back to flat.

---

## What the change actually does to the model

We implemented both as flags (defaults = current behaviour) and ran T-GCN unchanged, 3 seeds, same hyperparameters, matched ~2000-window
budgets. Only preprocessing differs. Original physical units:

| split | channel | baseline (received/offset) | fixed (datetime/timestamp) | change |
|---|---|---|---|---|
| test | **`freq_dev` MAE** | 0.003811 | **0.001806** | **−52.6%** |
| test | `freq_dev` R² | 0.8923 | **0.9024** | +0.010 |
| test | `volt_dev` MAE | 0.1893 | 0.1994 | +5.3% |
| test | `volt_dev` R² | 0.9056 | 0.8961 | −0.010 |
| pred | **`freq_dev` MAE** | 0.004545 | **0.002888** | **−36.4%** |
| pred | `freq_dev` R² | 0.5266 | **0.5529** | +0.026 |
| pred | `volt_dev` MAE | 0.2521 | 0.2511 | −0.4% |

**`freq_dev` error halves.** That is the grid-coherent channel, and the one
`generate_map_freq_hot_clusters.py` keys the disturbance detector on. `volt_dev` gets ~5%
worse. Because `volt_dev` is ~40× larger in magnitude, the *pooled* MAE therefore rises
4.2% — so a single pooled number would report this change as a regression. We think the
per-channel view is the honest one.

We checked the obvious confounds: `volt_dev` target variance is the same in both arms
(0.899 vs 0.904) and the fixed arm's observed fraction is slightly *higher*, so the
`volt_dev` regression is real, not a harder-test-set artifact.

Our hypothesis for why: contiguous placement keeps each sensor's own consecutive samples
perfectly intact — internally self-consistent, merely misaligned with its neighbours.
Timestamp reindexing adds ±50 ms nearest-match jitter to each sensor's own series (and a
0.03% duplicate/skip rate) while fixing the cross-sensor structure. So a **system-wide
coherent** quantity that the GNN reconstructs from neighbours (`freq_dev`) gains a lot,
while a **local, per-sensor** quantity predicted mostly from its own history (`volt_dev`)
loses a little. That is a hypothesis, not something we've isolated.
