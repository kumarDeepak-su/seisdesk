"""Residual moveout: analysis and correction of remaining (parabolic or quartic) curvature in flattened gathers."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .base import Param, Process, cdp_values, register
from .moveout import gate_by_energy, sample_at


def rmo_scan(gather: np.ndarray, offsets: np.ndarray, times: np.ndarray, dt: float, curvatures: np.ndarray,
             window: float = 0.03, order: int = 2) -> np.ndarray:
    """Semblance after removing dt(x) = r (x / xmax)^order, for each curvature r (s at the far offset)."""
    xn = np.abs(np.asarray(offsets, dtype=np.float64)) / max(np.max(np.abs(offsets)), 1e-9)
    w = xn ** order
    nwin = max(1, int(round(window / dt)))
    out = np.zeros((curvatures.size, times.size))
    energy = np.zeros((curvatures.size, times.size))
    g = gather.astype(np.float64)
    for i, r in enumerate(curvatures):
        tt = times[None, :] + r * w[:, None]
        vals, valid = sample_at(g, tt, times[0], dt)
        num = uniform_filter1d(vals.sum(axis=0) ** 2, nwin)
        den = uniform_filter1d(np.maximum(valid.sum(axis=0), 1) * (vals ** 2).sum(axis=0), nwin)
        out[i] = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
        energy[i] = uniform_filter1d((vals ** 2).sum(axis=0), nwin)
    return gate_by_energy(out, energy)


def rmo_correct(gather: np.ndarray, offsets: np.ndarray, times: np.ndarray, dt: float, r_of_t: np.ndarray,
                order: int = 2) -> np.ndarray:
    xn = np.abs(np.asarray(offsets, dtype=np.float64)) / max(np.max(np.abs(offsets)), 1e-9)
    tt = times[None, :] + r_of_t[None, :] * (xn ** order)[:, None]
    vals, _ = sample_at(gather.astype(np.float64), tt, times[0], dt)
    return vals.astype(np.float32)


@register
class ResidualMoveout(Process):
    name = "residual_moveout"
    label = "Residual moveout correction"
    category = "Moveout"
    description = ("For each CMP (or image) gather: scan the residual curvature r(t0) of flattened events, keep the "
                   "maximum-semblance r smoothed in time, and remove it. Works after NMO or on migrated gathers.")
    params = (Param("rmin", "float", -0.03, "Minimum residual at far offset", unit="s"),
              Param("rmax", "float", 0.03, "Maximum residual at far offset", unit="s"),
              Param("nr", "int", 41, "Number of trials", minimum=3),
              Param("order", "choice", "2", "Curvature order", choices=("2", "4")),
              Param("window", "float", 0.03, "Semblance window", unit="s", minimum=0.001),
              Param("smooth", "float", 0.1, "Smoothing of r(t)", unit="s", minimum=0.0),
              Param("gather_key", "key", "cdp", "Gather key"))

    def run(self, data, context):
        if self["gather_key"] == "cdp":
            cdp_values(data)
        r = np.linspace(self["rmin"], self["rmax"], self["nr"])
        order = int(self["order"])
        out = data.traces.copy()
        groups = list(data.iter_gathers(self["gather_key"]))
        table = {}
        for n, (value, idx) in enumerate(groups):
            if idx.size < 3:
                continue
            off = data.header("offset")[idx]
            spec = rmo_scan(data.traces[idx], off, data.times, data.dt, r, self["window"], order)
            r_t = r[np.argmax(spec, axis=0)]
            power = spec.max(axis=0)
            weight = np.where(power > 0.2 * power.max(), power, 0.0)
            if self["smooth"] > 0:                 # smooth r(t) weighted by semblance: quiet zones do not dilute it
                sigma = self["smooth"] / data.dt
                wsum = gaussian_filter1d(weight, sigma)
                r_t = np.where(wsum > 1e-6, gaussian_filter1d(r_t * weight, sigma) / np.maximum(wsum, 1e-12), 0.0)
            else:
                r_t = np.where(weight > 0, r_t, 0.0)
            out[idx] = rmo_correct(data.traces[idx], off, data.times, data.dt, r_t, order)
            table[value] = r_t[:: max(1, int(0.1 / data.dt))].tolist()
            context.report((n + 1) / len(groups))
        context.results["residual_moveout"] = table
        return data.copy(out)
