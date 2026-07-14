"""
comparison_common.py
====================

Single source of truth for the HydraGNN-vs-standalone-T-GCN comparison.

Everything that must agree between the two pipelines lives here:
  * ``PARITY``          -- the canonical experiment settings
  * ``masked_metrics``  -- ONE metric implementation, applied to BOTH sides
  * ``load_*_split``    -- loaders that canonicalize each side's arrays to a
                           common layout and to original (physical) units
  * ``check_parity``    -- asserts the two runs actually solved the same task

Why the metrics are recomputed here instead of read from each pipeline's own
output: the two codebases each ship their own ``compute_metrics``. Trusting them
means a silent difference in metric definition (or in *what is being averaged*)
can masquerade as a model difference. An earlier iteration of this comparison
reported HydraGNN beating the baseline on MAE purely because HydraGNN was
predicting a third, small-magnitude channel (``angle_delta``) that the baseline
never predicted -- the extra channel deflated the mean absolute error. Computing
both sides here, from raw saved arrays, with the same function, makes that class
of bug impossible.
"""

from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Canonical experiment settings
# ---------------------------------------------------------------------------
# The 4 dynamic feature channels produced by both pipelines, in index order.
FEATURE_NAMES = ("freq_dev", "rocof", "angle_delta", "volt_dev")

STANDALONE_TARGETS = ("freq_dev", "volt_dev")

PARITY = {
    # --- data / preprocessing (must match; several are baked into the cache) ---
    "date": "2024-06-01",
    "Tin": 100,
    "horizon": 10,
    "out_features": [0, 3],  # freq_dev, volt_dev -> matches STANDALONE_TARGETS
    "scaling": "robust",  # median/IQR, per-channel
    "scaled_clip_value": 10.0,
    # The standalone drops no sensors for coverage; it uses every metadata-matched
    # device. HydraGNN's default (0.5) silently drops the low-coverage sensors,
    # producing a 101-node graph vs the standalone's 107. Force them to agree.
    "min_device_coverage": 0.0,
    "min_step_coverage": 0.8,
    "short_gap_steps": 10,
    "medium_gap_steps": 300,
    "train_frac": 0.8,
    "val_frac": 0.05,
    "test_frac": 0.05,
    "pred_frac": 0.10,
    # --- windowing (must match) ---
    # Both sides now take every `window_stride`-th anchor. With stride == horizon
    # the 10-step forecast horizons tile the timeline exactly: 100% of timesteps
    # appear as a prediction target. This is the standalone's canonical scheme.
    #
    # The old approach (HydraGNN --max_windows 2000, standalone derived strides)
    # gave only H/328 ~= 3% target coverage: 97% of the day was never predicted,
    # so short disturbances fell entirely into the 32.8 s gaps between horizons.
    # It also meant the two sides sampled windows differently (evenly-spaced
    # linspace vs fixed stride), which is why R^2 was not comparable across sides.
    # A stride on both sides removes both problems.
    #
    # Requires --dataset_mode ondemand: the materialized cache would need ~26 GB
    # for the train split alone at this stride (see fnet_window_dataset.py).
    "dataset_mode": "ondemand",
    "window_stride": 10,  # == horizon -> 100% target coverage
    # --- alignment (must match) ---
    # The FNET parquet index is the GPS measurement clock (data/README.txt: the
    # ReceivedTime column "can be ignored, as it does not reflect the true
    # measurement timestamp"). HydraGNN already reindexes each sensor onto the
    # grid by that clock; the standalone T-GCN historically sorted by ReceivedTime
    # and placed samples contiguously. These flags (branch kien/timestamp-alignment
    # in the t-gcn fork) put the baseline on the same footing. Without them the
    # comparison would confound model quality with alignment quality.
    "time_axis": "datetime",   # standalone --time_axis
    "alignment": "timestamp",  # standalone --alignment
    # --- model / training (must match) ---
    "hidden_dim": 64,
    "num_conv_layers": 2,  # standalone calls this --num_layers
    # Cost scales with (windows / stride) * epochs, i.e. with total gradient
    # steps. Going stride 328 -> 10 multiplies windows by ~34x, so epochs are cut
    # to keep the step budget (and wall clock) comparable. At equal step count,
    # a small stride with few epochs strictly dominates a large stride with many
    # epochs: same cost, but every timestep is supervised instead of 3% of them.
    # The reference cov80 run uses stride 10 with 50 epochs (~6.7 h/run here).
    "num_epoch": 10,
    "learning_rate": 1e-3,
    "batch_size": 32,
}

SEEDS = [42, 1, 2]

# HydraGNN temporal backbones. TemporalGAT is excluded by default: with 6
# attention heads back-propagating through the Tin=100 unroll it needs ~25 GB of
# VRAM at batch 32. Add it explicitly (and lower --batch_size) if you have the
# memory -- but note a differing batch size breaks parity.
BACKBONES = ["TemporalGCN", "TemporalGIN", "TemporalSAGE"]
BACKBONES_ALL = BACKBONES + ["TemporalGAT"]

# The standalone only writes prediction arrays for these splits, so these are the
# only splits whose metrics we can recompute independently for BOTH sides.
# `val` is a model-selection split; it is not part of the reported comparison.
RECOMPUTED_SPLITS = ("test", "pred")


# ---------------------------------------------------------------------------
# The one metric implementation
# ---------------------------------------------------------------------------
def masked_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict:
    """Pooled MAE / RMSE / MSE / R^2 over observed elements only (mask > 0.5).

    `pred`/`true`/`mask` must be identically shaped and already in original
    (physical) units. Pooling is over every (window, node, horizon, channel)
    element, so the result does not depend on node ordering -- only on the node
    *set*, which `check_parity` verifies.

    NOTE on R^2: the denominator is the variance of *this pipeline's own*
    targets. Because the two pipelines sample different windows, their R^2
    denominators differ, so R^2 is only loosely comparable across sides. MAE and
    RMSE are absolute and are the primary comparison quantities.
    """
    valid = mask.reshape(-1) > 0.5
    p = pred.reshape(-1)[valid]
    y = true.reshape(-1)[valid]
    if p.size == 0:
        return {k: float("nan") for k in ("mae", "rmse", "mse", "r2", "n")}
    err = p - y
    mse = float(np.mean(err**2))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(p.size),
    }


def per_channel_metrics(pred, true, mask, channels=STANDALONE_TARGETS) -> dict:
    """`masked_metrics` computed separately for each output channel.

    THIS IS THE PRIMARY VIEW, not a nicety. The two target channels differ in
    magnitude by ~40x (train scaler: freq_dev ~0.023, volt_dev ~0.747), so a
    pooled MAE is dominated almost entirely by volt_dev and says nearly nothing
    about freq_dev -- the channel that matters most for disturbance detection.
    The reference T-GCN report (t-gcn/report_2026-06-03.md) reports feature-level
    metrics for exactly this reason; e.g. its test MAE is 0.0034 for freq_dev vs
    0.1429 for volt_dev, against a pooled 0.0732.

    Returns {channel_name: metrics_dict}, plus "all" for the pooled value.
    """
    out = {"all": masked_metrics(pred, true, mask)}
    for i, name in enumerate(channels):
        out[name] = masked_metrics(pred[..., i], true[..., i], mask[..., i])
    return out


# ---------------------------------------------------------------------------
# Loaders -> canonical layout [W, N, H, F], original (physical) units
# ---------------------------------------------------------------------------
def load_hydra_split(trial_dir: Path, split: str):
    """HydraGNN saves preds/ys in *standardized* space as [W, N, H, F_out].

    Denormalize with the train-only scaler (feat_mean/feat_std restricted to the
    output channels) so the result is in original units, matching the standalone.
    """
    trial_dir = Path(trial_dir)
    p = np.load(trial_dir / f"preds_{split}.npy")
    y = np.load(trial_dir / f"ys_{split}.npy")
    m = np.load(trial_dir / f"masks_{split}.npy")
    fm = np.load(trial_dir / "feat_mean.npy")  # [N, 4]
    fs = np.load(trial_dir / "feat_std.npy")  # [N, 4]
    oidx = np.load(trial_dir / "out_idx.npy")  # [F_out]

    center = fm[:, oidx][None, :, None, :]
    scale = fs[:, oidx][None, :, None, :]
    return p * scale + center, y * scale + center, m


def load_standalone_split(run_dir: Path, split: str):
    """Standalone saves already-denormalized arrays as [S, H, N, F_out].

    Transpose to the canonical [W, N, H, F_out].
    """
    run_dir = Path(run_dir)
    p = np.load(run_dir / f"{split}_predictions_original.npy")
    y = np.load(run_dir / f"{split}_targets_original.npy")
    m = np.load(run_dir / f"{split}_target_mask.npy")
    tr = (0, 2, 1, 3)
    return p.transpose(tr), y.transpose(tr), m.transpose(tr)


def hydra_channel_names(trial_dir: Path) -> tuple:
    oidx = np.load(Path(trial_dir) / "out_idx.npy")
    return tuple(FEATURE_NAMES[i] for i in oidx)


def load_hydra_meta(cache_dir: Path, date: str) -> dict:
    with open(Path(cache_dir) / f"fnet_{date}_meta.pkl", "rb") as f:
        return pickle.load(f)


def load_standalone_config(run_dir: Path) -> dict:
    with open(Path(run_dir) / "config.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parity checking
# ---------------------------------------------------------------------------
def _row(name, hydra, standalone, ok, note=""):
    return {
        "check": name,
        "hydragnn": hydra,
        "standalone": standalone,
        "ok": bool(ok),
        "note": note,
    }


def check_parity(hydra_trial: Path, standalone_run: Path, window_tol=0.10,
                 hydra_aligned_timesteps: int | None = None,
                 timestep_tol=0.02, parity: dict | None = None) -> list:
    """Verify the two pipelines solved the same task. Returns a list of check rows.

    Any row with ok=False means the reported comparison is NOT apples-to-apples.

    `hydra_aligned_timesteps` is the HydraGNN cache's retained series length
    (manifest["cache"]["aligned_timesteps"]). The two pipelines align the common
    time grid differently, and HydraGNN's step-coverage filter operates on
    whatever node set survives `min_device_coverage` -- so a coverage mismatch
    shows up here as a differing retained length even when node counts agree.
    The reference cov80 run retains 819,582 of 864,064 timesteps (94.85%).
    """
    # Use the effective parity actually used for the run (recorded in the manifest
    # and passed here), not the module default -- otherwise a --num_epoch /
    # --window_stride override, or make_report running as a separate process,
    # would compare against stale defaults and report spurious mismatches.
    P = parity if parity is not None else PARITY

    cfg = load_standalone_config(standalone_run)
    hp, hy, hm = load_hydra_split(hydra_trial, "test")
    sp, sy, sm = load_standalone_split(standalone_run, "test")

    h_ch = hydra_channel_names(hydra_trial)
    checks = [
        _row("Tin", P["Tin"], cfg.get("Tin"), P["Tin"] == cfg.get("Tin")),
        _row("horizon (H)", hp.shape[2], cfg.get("H"), hp.shape[2] == cfg.get("H")),
        _row("n_nodes", hp.shape[1], sp.shape[1], hp.shape[1] == sp.shape[1]),
        _row("n_out_channels", hp.shape[3], sp.shape[3], hp.shape[3] == sp.shape[3]),
        _row(
            "target channels",
            ",".join(h_ch),
            ",".join(STANDALONE_TARGETS),
            tuple(h_ch) == STANDALONE_TARGETS,
            "standalone targets are hardcoded in t-gcn/train.py",
        ),
        _row(
            "scaling",
            P["scaling"],
            cfg.get("feature_scaling"),
            P["scaling"] == cfg.get("feature_scaling"),
        ),
        _row(
            "time_axis",
            P["time_axis"],
            cfg.get("time_axis", "received"),
            P["time_axis"] == cfg.get("time_axis", "received"),
            "standalone must use the GPS index, not ReceivedTime",
        ),
        _row(
            "alignment",
            P["alignment"],
            cfg.get("alignment", "offset"),
            P["alignment"] == cfg.get("alignment", "offset"),
            "standalone must reindex by timestamp, not place contiguously",
        ),
        _row(
            "clip value",
            P["scaled_clip_value"],
            cfg.get("scaled_clip_value"),
            float(P["scaled_clip_value"]) == float(cfg.get("scaled_clip_value", -1)),
        ),
        _row("hidden_dim", P["hidden_dim"], cfg.get("hidden_dim"),
             P["hidden_dim"] == cfg.get("hidden_dim")),
        _row("conv layers", P["num_conv_layers"], cfg.get("num_layers"),
             P["num_conv_layers"] == cfg.get("num_layers")),
        _row("epochs", P["num_epoch"], cfg.get("epochs"),
             P["num_epoch"] == cfg.get("epochs")),
        _row("batch_size", P["batch_size"], cfg.get("batch_size"),
             P["batch_size"] == cfg.get("batch_size")),
        _row("learning rate", P["learning_rate"], cfg.get("lr"),
             abs(P["learning_rate"] - float(cfg.get("lr", 0))) < 1e-12),
    ]
    for frac in ("train_frac", "val_frac", "test_frac", "pred_frac"):
        checks.append(
            _row(frac, P[frac], cfg.get(frac), P[frac] == cfg.get(frac))
        )

    # Retained series length. The two align the common time grid differently, and
    # HydraGNN's step-coverage filter runs over whatever node set survives
    # min_device_coverage. If HydraGNN drops sensors the standalone keeps, its
    # per-step coverage is inflated, more timesteps pass the 0.8 filter, and the
    # two end up modelling different time ranges (splits are *fractions* of this).
    if hydra_aligned_timesteps is not None:
        s_kept = cfg.get("kept_timesteps")
        if s_kept:
            rel = abs(hydra_aligned_timesteps - s_kept) / max(hydra_aligned_timesteps, s_kept)
            checks.append(
                _row("kept timesteps", hydra_aligned_timesteps, s_kept, rel <= timestep_tol,
                     f"relative diff {rel:.2%} (tol {timestep_tol:.0%}); "
                     f"differing retained length => different time ranges per split")
            )

    # Window counts need not be identical (HydraGNN samples evenly via linspace,
    # the standalone by fixed stride) but must be within tolerance, else the two
    # are effectively evaluating different amounts of data.
    for split in RECOMPUTED_SPLITS:
        hw = load_hydra_split(hydra_trial, split)[0].shape[0]
        sw = load_standalone_split(standalone_run, split)[0].shape[0]
        rel = abs(hw - sw) / max(hw, sw)
        checks.append(
            _row(f"{split} windows", hw, sw, rel <= window_tol,
                 f"relative diff {rel:.1%} (tol {window_tol:.0%})")
        )
    return checks


def parity_ok(checks: list) -> bool:
    return all(c["ok"] for c in checks)


def format_parity(checks: list) -> str:
    w = max(len(c["check"]) for c in checks) + 2
    lines = [f"{'check'.ljust(w)}{'hydragnn':>18}{'standalone':>18}   status"]
    lines.append("-" * (w + 44))
    for c in checks:
        status = "OK " if c["ok"] else "MISMATCH"
        lines.append(
            f"{c['check'].ljust(w)}{str(c['hydragnn']):>18}{str(c['standalone']):>18}   {status}"
            + (f"  ({c['note']})" if c["note"] and not c["ok"] else "")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo), capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def standalone_strides() -> dict:
    """Per-split stride for the standalone -- simply PARITY['window_stride'].

    Both pipelines now use the same anchor rule (`arange(Tin-1, T-H, stride)`),
    so the strides are identical and no derivation from the series length is
    needed. Window counts then agree up to the two sides' small difference in
    aligned series length, which `check_parity` bounds separately.
    """
    d = int(PARITY["window_stride"])
    return {f"stride_{s}": d for s in ("train", "val", "test", "pred")}
