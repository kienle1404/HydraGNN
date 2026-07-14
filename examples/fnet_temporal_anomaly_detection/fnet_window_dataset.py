"""
fnet_window_dataset.py
======================

On-the-fly sliding-window dataset for the FNET temporal example.

Motivation
----------
``make_window_dataset()`` materializes one ``torch_geometric.data.Data`` object
per window and pickles it to disk. Each window costs ~0.39 MB (it stores
``x_seq`` of shape ``[N, Tin, F_dyn+1+F_static]``), so the cache size scales with
the *number of windows*:

    2,000 windows  ->  ~0.8 GB   (what we can afford today)
   65,556 windows  ->  ~26 GB    (the stride-10 budget the standalone T-GCN uses)
  655,556 windows  ->  ~256 GB   (every window)

That forces ``--max_windows`` down to ~2000, at which point consecutive windows'
10-step target horizons sit 32.8 s apart: only ~3% of the timeline is ever a
prediction target. The standalone T-GCN has no such limit because it keeps the
``[T, N, F]`` arrays once (~1.4 GB) and slices windows on demand.

This module does the same for HydraGNN. Nothing is precomputed per window: the
arrays are held once, and ``__getitem__`` builds the ``Data`` object for one
anchor. Disk and RAM become independent of the window count, so stride-10 parity
with the standalone becomes reachable.

The per-window construction is a line-for-line port of the body of
``make_window_dataset``'s loop, so a window produced here is identical to the
materialized one for the same anchor. ``bench_window_dataset.py`` asserts this.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

F_DYN = 4  # freq_dev, rocof, angle_delta, volt_dev


def window_anchors(T: int, Tin: int, H: int, max_windows: int | None = None,
                   stride: int | None = None) -> np.ndarray:
    """Anchor indices `s` for a segment of length T.

    A window at anchor `s` reads inputs `s-Tin+1 .. s` and targets `s+1 .. s+H`.

    - `stride`: take every `stride`-th anchor (the standalone's scheme). With
      `stride == H` the target horizons tile the timeline exactly (100% coverage).
    - `max_windows`: evenly-spaced anchors (HydraGNN's existing scheme), applied
      only if the full set is larger. Matches `make_window_dataset` exactly.
    - neither: every anchor.
    """
    if stride is not None:
        return np.arange(Tin - 1, T - H, stride, dtype=np.int64)
    full = np.arange(Tin - 1, T - H, dtype=np.int64)
    if max_windows is not None and len(full) > max_windows:
        return np.linspace(Tin - 1, T - H - 1, max_windows, dtype=np.int64)
    return full


class FNETWindowDataset(Dataset):
    """Builds one PyG ``Data`` per window on demand.

    Holds references to the (already scaled + imputed) arrays. `X` and
    `observed_mask` may be contiguous slices (views) of the full-day arrays --
    no copy is made, so the four splits share one underlying buffer.
    """

    def __init__(self, X, observed_mask, grid_embed, edge_index, Tin, H, out_idx,
                 edge_weight=None, predict_delta=False,
                 max_windows=None, stride=None):
        T, N, F = X.shape
        assert F == F_DYN, f"expected {F_DYN} dynamic channels, got {F}"

        self.X = X
        self.observed_mask = observed_mask
        self.Tin, self.H, self.N = int(Tin), int(H), int(N)
        self.out_idx = np.asarray(out_idx, dtype=np.int64)
        self.F_out = len(self.out_idx)
        self.predict_delta = bool(predict_delta)

        self.edge_index = edge_index
        self.edge_weight = edge_weight

        # Shared, window-independent tensors. `expand` is a stride-0 view, so the
        # static block costs nothing until torch.cat materializes it per item.
        F_static = grid_embed.shape[1]
        self.static_t = (
            torch.from_numpy(grid_embed).float().unsqueeze(1).expand(N, Tin, F_static)
        )
        self.pos = torch.zeros(N, 3)
        self.batch = torch.zeros(N, dtype=torch.long)
        self.y_loc = torch.tensor([[0, N * H * self.F_out]], dtype=torch.int64)

        self.anchors = window_anchors(T, Tin, H, max_windows=max_windows, stride=stride)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, i: int) -> Data:
        s = int(self.anchors[i])
        Tin, H, N, out_idx = self.Tin, self.H, self.N, self.out_idx
        X, om = self.X, self.observed_mask

        x_dyn = torch.from_numpy(X[s - Tin + 1 : s + 1]).permute(1, 0, 2).contiguous()
        channels = [x_dyn]
        if om is not None:
            mask_win = (
                torch.from_numpy(om[s - Tin + 1 : s + 1])
                .permute(1, 0).unsqueeze(-1).contiguous()
            )
            channels.append(mask_win)
        channels.append(self.static_t)
        x_seq = torch.cat(channels, dim=-1).contiguous()

        y_block = X[s + 1 : s + 1 + H][:, :, out_idx]
        if self.predict_delta:
            y_block = y_block - X[s : s + 1, :, out_idx]
        y = torch.from_numpy(y_block).permute(1, 0, 2).reshape(N, H * self.F_out).contiguous()

        data = Data(
            x=torch.from_numpy(X[s]).contiguous(),
            x_seq=x_seq,
            edge_index=self.edge_index.clone(),
            y=y,
            y_loc=self.y_loc.clone(),
            pos=self.pos,
            batch=self.batch,
            num_nodes=N,
        )
        if self.edge_weight is not None:
            data.edge_weight = self.edge_weight.clone()
        if om is not None:
            m_block = np.repeat(om[s + 1 : s + 1 + H][:, :, None], self.F_out, axis=2)
            y_mask = torch.from_numpy(m_block).permute(1, 0, 2).reshape(N, H * self.F_out).contiguous()
            data.observed_mask = y_mask
        return data


def splits_from_meta(meta: dict, train_frac=0.8, val_frac=0.05, test_frac=0.05,
                     pred_frac=0.10, max_windows=None, stride=None) -> dict:
    """Build the four on-the-fly split datasets from a cached ``*_meta.pkl``.

    The meta pickle already stores everything needed (X, observed_mask,
    grid_embed, edge_index, edge_weight) -- the per-window pickles are redundant.
    Segments are sliced first, then windowed, so no window crosses a boundary
    (same invariant as ``preprocess_stage``).
    """
    X, om = meta["X"], meta["observed_mask"]
    T = X.shape[0]
    n_train, n_val = int(T * train_frac), int(T * val_frac)
    n_test, n_pred = int(T * test_frac), int(T * pred_frac)
    pred_start = n_train + n_val + n_test
    bounds = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, pred_start),
        "pred": (pred_start, min(pred_start + n_pred, T)),
    }
    common = dict(
        grid_embed=meta["grid_embed"],
        edge_index=meta["edge_index"],
        edge_weight=meta.get("edge_weight"),
        Tin=meta["Tin"], H=meta["horizon"], out_idx=meta["out_idx"],
        predict_delta=bool(meta.get("predict_delta", False)),
        max_windows=max_windows, stride=stride,
    )
    return {
        name: FNETWindowDataset(X[lo:hi], om[lo:hi] if om is not None else None, **common)
        for name, (lo, hi) in bounds.items()
    }
