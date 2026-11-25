"""Quality-control attributes: trace RMS, fold per CMP, values gridded by two headers or averaged by one."""

from __future__ import annotations

import numpy as np

from .dataset import SeismicData


def trace_rms(data: SeismicData, tmin: float = 0.0, tmax: float = 0.0) -> np.ndarray:
    """RMS amplitude of every trace between tmin and tmax (tmax = 0: to the end of the trace)."""
    t = data.times
    sel = (t >= tmin) & ((t <= tmax) if tmax > 0 else np.ones(t.size, bool))
    if not sel.any():
        sel[:] = True
    seg = data.traces[:, sel].astype(np.float64)
    return np.sqrt(np.mean(seg * seg, axis=1))


def fold_per_cmp(data: SeismicData) -> tuple[np.ndarray, np.ndarray]:
    cdp, fold = np.unique(data.header("cdp"), return_counts=True)
    return cdp, fold


def mean_by(values: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` for every distinct key."""
    u, inv = np.unique(np.asarray(keys, dtype=np.float64), return_inverse=True)
    sums = np.bincount(inv, weights=np.asarray(values, dtype=np.float64), minlength=u.size)
    counts = np.bincount(inv, minlength=u.size)
    return u, sums / np.maximum(counts, 1)


def grid_by(values: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    """Values arranged on a (row key x column key) grid, e.g. RMS by FFID and channel; NaN where there is no trace."""
    ru, ri = np.unique(np.asarray(rows, dtype=np.float64), return_inverse=True)
    cu, ci = np.unique(np.asarray(cols, dtype=np.float64), return_inverse=True)
    total = np.zeros((ru.size, cu.size))
    count = np.zeros((ru.size, cu.size))
    np.add.at(total, (ri, ci), np.asarray(values, dtype=np.float64))
    np.add.at(count, (ri, ci), 1.0)
    grid = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
    return ru, cu, grid
