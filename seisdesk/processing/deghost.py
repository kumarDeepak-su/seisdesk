"""Removal of the sea-surface ghost of marine sources and receivers (deghosting).

A source or receiver at depth z below the sea surface records the wave and, 2 z cos(theta) / v later, its reflection
from the surface, whose reflection coefficient is close to -1. The ghost operator in the frequency domain is
G = 1 - r exp(-i w 2 z cos(theta) / v), r being the magnitude of the surface reflection coefficient. Deghosting divides
by G with a water-level stabilisation (Jovanovich, Sumner & Akins-Easterlin 1983), either at vertical incidence
(cos(theta) = 1, trace by trace) or for every plane-wave component of a shot gather in the linear tau-p domain
(cos(theta) = sqrt(1 - p^2 v^2); source and receiver take the same slowness, as for flat layers).
"""

from __future__ import annotations

import numpy as np

from .base import Param, Process, register
from .noise import _groups, linear_radon


def ghost_operator(freqs, depth, velocity: float = 1500.0, r: float = 0.95, cos_theta=1.0) -> np.ndarray:
    """G(f) = 1 - r exp(-2 pi i f 2 z cos(theta) / v); arguments broadcast against each other. Where the depth is
    zero (no instrument depth known, or that side not deghosted) the operator is 1."""
    z = np.asarray(depth, dtype=np.float64)
    delay = 2.0 * z * np.asarray(cos_theta, dtype=np.float64) / velocity
    g = 1.0 - r * np.exp(-2j * np.pi * np.asarray(freqs, dtype=np.float64) * delay)
    return np.where(z > 0, g, 1.0 + 0j)


def inverse_operator(g: np.ndarray, water_level: float) -> np.ndarray:
    """Stabilised inverse conj(G) / (|G|^2 + eps), eps = water_level x the largest |G|^2 of the whole operator (a
    common floor, so components with a weak ghost response, e.g. near-horizontal ones, are not boosted)."""
    power = np.abs(g) ** 2
    eps = water_level * float(power.max()) if power.size else 0.0
    return np.conj(g) / (power + max(eps, 1e-12))


def _fft_length(nt: int, delay_max: float, dt: float) -> int:
    return int(2 ** np.ceil(np.log2(nt + int(np.ceil(delay_max / dt)) + 2)))


def add_ghost(traces: np.ndarray, dt: float, depth, velocity: float = 1500.0, r: float = 0.95) -> np.ndarray:
    """Traces (ntr, nt) with the vertical-incidence ghost of a source or receiver at ``depth`` (m, one value or one per
    trace) added; for tests and demonstrations."""
    x = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    ntr, nt = x.shape
    z = np.broadcast_to(np.asarray(depth, dtype=np.float64), (ntr,))
    n = _fft_length(nt, 2.0 * float(np.max(z, initial=0.0)) / velocity, dt)
    f = np.fft.rfftfreq(n, dt)
    g = ghost_operator(f[None, :], z[:, None], velocity, r)
    return np.fft.irfft(np.fft.rfft(x, n, axis=1) * g, n, axis=1)[:, :nt]


@register
class Deghost(Process):
    name = "deghost"
    label = "Deghosting (source/receiver ghost)"
    category = "Filtering"
    description = ("Removes the sea-surface ghost of marine data: the delayed, opposite-polarity copy that makes "
                   "notches in the spectrum at multiples of v / (2 z cos(theta)) and weakens the low frequencies. "
                   "Vertical incidence works trace by trace; the angle-dependent mode works on shot gathers in the "
                   "linear tau-p domain, where every slowness p has its own ghost delay. Depths come from the "
                   "headers (sdepth for the source, minus gelev for the receivers) unless given. The water level "
                   "limits the gain in the ghost notches.")
    params = (Param("side", "choice", "receiver", "Ghost to remove", choices=("receiver", "source", "both")),
              Param("mode", "choice", "vertical incidence", "Mode",
                    choices=("vertical incidence", "angle-dependent (tau-p)")),
              Param("source_depth", "float", 0.0, "Source depth (0 = header sdepth)", unit="m", minimum=0.0),
              Param("receiver_depth", "float", 0.0, "Receiver depth (0 = minus header gelev)", unit="m",
                    minimum=0.0),
              Param("velocity", "float", 1500.0, "Water velocity", unit="m/s", minimum=100.0),
              Param("reflection", "float", 0.95, "Sea-surface reflection coefficient (magnitude)", minimum=0.0,
                    maximum=1.0),
              Param("water_level", "float", 2.0, "Water level", unit="%", minimum=0.0),
              Param("gather_key", "key", "fldr", "Shot gather key (tau-p mode)"),
              Param("np", "int", 61, "Number of p values (tau-p mode)", minimum=11),
              Param("fmin", "float", 2.0, "Lowest frequency (tau-p mode)", unit="Hz", minimum=0.0),
              Param("fmax", "float", 0.0, "Highest frequency (tau-p mode, 0 = Nyquist)", unit="Hz", minimum=0.0))

    def _depths(self, data):
        """Source and receiver depth of every trace (m); zero for a side that is not deghosted."""
        side = self["side"]
        src = np.zeros(data.ntr)
        rec = np.zeros(data.ntr)
        if side in ("source", "both"):
            src = (np.full(data.ntr, self["source_depth"]) if self["source_depth"] > 0
                   else np.maximum(data.header("sdepth"), 0.0))
            if not np.any(src > 0):
                raise ValueError("source depth unknown: set header sdepth or give the source depth")
        if side in ("receiver", "both"):
            ge = data.header("gelev")
            rec = (np.full(data.ntr, self["receiver_depth"]) if self["receiver_depth"] > 0
                   else np.where(ge < 0, -ge, 0.0))
            if not np.any(rec > 0):
                raise ValueError("receiver depth unknown: set header gelev (negative below the sea surface) or give "
                                 "the receiver depth")
        return src, rec

    def run(self, data, context):
        src, rec = self._depths(data)
        v, r, wl = self["velocity"], self["reflection"], self["water_level"] / 100.0
        traces = data.traces.astype(np.float64)
        if self["mode"] == "vertical incidence":
            n = _fft_length(data.nt, 2.0 * float(max(src.max(), rec.max())) / v, data.dt)
            f = np.fft.rfftfreq(n, data.dt)
            g = ghost_operator(f[None, :], src[:, None], v, r) * ghost_operator(f[None, :], rec[:, None], v, r)
            out = np.fft.irfft(np.fft.rfft(traces, n, axis=1) * inverse_operator(g, wl), n, axis=1)[:, :data.nt]
            return data.copy(out.astype(np.float32))
        out = traces.copy()
        groups = _groups(data, self["gather_key"])
        fmax = self["fmax"] if self["fmax"] > 0 else 0.5 / data.dt
        p = np.linspace(-1.0 / v, 1.0 / v, self["np"])
        cos_t = np.sqrt(np.maximum(1.0 - (p * v) ** 2, 0.0))
        for n_g, idx in enumerate(groups):
            zs, zr = float(np.median(src[idx])), float(np.median(rec[idx]))
            if idx.size < 4:                                   # too few traces for a slant stack: vertical incidence
                n = _fft_length(data.nt, 2.0 * max(zs, zr) / v, data.dt)
                f = np.fft.rfftfreq(n, data.dt)
                g = ghost_operator(f, zs, v, r) * ghost_operator(f, zr, v, r)
                out[idx] = np.fft.irfft(np.fft.rfft(traces[idx], n, axis=1) * inverse_operator(g, wl)[None, :], n,
                                        axis=1)[:, :data.nt]
                continue
            model, op, nfft, band = linear_radon(traces[idx], data.header("offset")[idx], data.dt, p, 0.02,
                                                 self["fmin"], fmax)
            if band.size == 0:
                continue
            freqs = np.fft.rfftfreq(nfft, data.dt)[band]
            g = (ghost_operator(freqs[None, :], zs, v, r, cos_t[:, None]) *
                 ghost_operator(freqs[None, :], zr, v, r, cos_t[:, None]))                       # (np, nb)
            gain = inverse_operator(g, wl) - 1.0
            corr = np.zeros((idx.size, model.shape[1]), dtype=np.complex128)
            for k, j in enumerate(band):
                corr[:, j] = op(j) @ (model[:, j] * gain[:, k])
            # only the change is transformed back, so whatever the slant stack does not model stays untouched
            out[idx] = traces[idx] + np.fft.irfft(corr, nfft, axis=1)[:, :data.nt]
            context.report((n_g + 1) / len(groups))
        return data.copy(out.astype(np.float32))
