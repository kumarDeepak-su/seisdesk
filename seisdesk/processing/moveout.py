"""Moveout equations and trace resampling shared by velocity analysis, NMO and residual moveout."""

from __future__ import annotations

import numpy as np


def moveout_time(t0: np.ndarray, x: np.ndarray, v: np.ndarray, eta: np.ndarray | float = 0.0) -> np.ndarray:
    """Reflection time at offset x: hyperbolic, or Alkhalifah-Tsvankin non-hyperbolic (VTI) when eta != 0.

    t^2 = t0^2 + x^2/v^2 - 2 eta x^4 / (v^2 [t0^2 v^2 + (1 + 2 eta) x^2])
    """
    t0 = np.asarray(t0, dtype=np.float64)
    x2 = np.asarray(x, dtype=np.float64) ** 2
    v2 = np.asarray(v, dtype=np.float64) ** 2
    t2 = t0 ** 2 + x2 / v2
    eta = np.asarray(eta, dtype=np.float64)
    if np.any(eta != 0):
        t2 = t2 - 2.0 * eta * x2 * x2 / (v2 * (t0 ** 2 * v2 + (1.0 + 2.0 * eta) * x2) + 1e-30)
    return np.sqrt(np.maximum(t2, 0.0))


ENERGY_FLOOR = 1e-8


def gate_by_energy(panel: np.ndarray, energy: np.ndarray, floor: float = ENERGY_FLOOR) -> np.ndarray:
    """Zero a semblance-type panel where its moveout-corrected energy is negligible.

    Semblance is blind to amplitude: dead or muted zones, and the identical round-off of noise-free records, look
    perfectly coherent (mostly at the flattest trial curves) and would drag automatic picks there."""
    out = np.array(panel, copy=True)
    out[energy < floor * max(float(energy.max()), 1e-300)] = 0.0
    return out


def sample_at(traces: np.ndarray, times: np.ndarray, t0: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of traces (ntr, nt) at times (ntr, m); returns values and a validity mask."""
    nt = traces.shape[1]
    pos = (times - t0) / dt
    valid = (pos >= 0) & (pos <= nt - 1)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, nt - 2)
    frac = np.clip(pos - i0, 0.0, 1.0)
    a = np.take_along_axis(traces, i0, axis=1)
    b = np.take_along_axis(traces, i0 + 1, axis=1)
    vals = a + frac * (b - a)
    return np.where(valid, vals, 0.0), valid
