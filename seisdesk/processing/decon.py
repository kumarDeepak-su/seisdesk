"""Wiener-Levinson deconvolution: spiking (gap = one sample) and predictive (gap > one sample)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz

from .base import Param, Process, register, taper_window


def prediction_error_filters(traces: np.ndarray, dt: float, op_length: float, gap: float, prewhite: float,
                             tmin: float, tmax: float, t0: float = 0.0) -> np.ndarray:
    """One prediction-error filter per trace, designed on [tmin, tmax]; returns (ntr, nfilter)."""
    ntr, nt = traces.shape
    L = max(2, int(round(op_length / dt)))
    a = max(1, int(round(gap / dt)))
    i0 = max(0, int(round((tmin - t0) / dt)))
    i1 = nt if tmax <= tmin else min(nt, int(round((tmax - t0) / dt)) + 1)
    if i1 - i0 < L + a + 2:
        raise ValueError("design window too short for the operator length and gap")
    seg = traces[:, i0:i1].astype(np.float64) * taper_window(i1 - i0, 0.05)[None, :]
    nfft = int(2 ** np.ceil(np.log2(2 * seg.shape[1])))
    spec = np.fft.rfft(seg, nfft, axis=1)
    acf = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)[:, : L + a]
    filters = np.zeros((ntr, a + L))
    filters[:, 0] = 1.0
    for i in range(ntr):
        r = acf[i]
        if r[0] <= 0:
            continue
        col = r[:L].copy()
        col[0] *= 1.0 + prewhite / 100.0
        try:
            f = solve_toeplitz(col, r[a:a + L])
        except np.linalg.LinAlgError:
            continue
        filters[i, a:a + L] = -f
    return filters


def apply_filters(traces: np.ndarray, filters: np.ndarray) -> np.ndarray:
    ntr, nt = traces.shape
    nfft = int(2 ** np.ceil(np.log2(nt + filters.shape[1])))
    out = np.fft.irfft(np.fft.rfft(traces, nfft, axis=1) * np.fft.rfft(filters, nfft, axis=1), nfft, axis=1)
    return out[:, :nt].astype(np.float32)


@register
class PredictiveDecon(Process):
    name = "decon"
    label = "Deconvolution (spiking / predictive)"
    category = "Deconvolution"
    description = ("Wiener-Levinson prediction-error filter per trace. Gap = one sample gives spiking deconvolution; "
                   "a longer gap (e.g. the water-bottom period) gives predictive deconvolution of short-period multiples.")
    params = (Param("operator_length", "float", 0.16, "Operator length", unit="s", minimum=0.004),
              Param("gap", "float", 0.004, "Prediction gap (dt = spiking)", unit="s", minimum=0.0),
              Param("prewhitening", "float", 0.1, "Pre-whitening", unit="%", minimum=0.0),
              Param("tmin", "float", 0.1, "Design window start", unit="s"),
              Param("tmax", "float", 3.0, "Design window end", unit="s"))

    def run(self, data, context):
        gap = max(self["gap"], data.dt)
        f = prediction_error_filters(data.traces, data.dt, self["operator_length"], gap, self["prewhitening"],
                                     self["tmin"], self["tmax"], data.t0)
        return data.copy(apply_filters(data.traces, f))
