"""Seismic attributes: the Hilbert transform, instantaneous (complex-trace) attributes (Taner, Koehler & Sheriff 1979;
Barnes 1993) and single-frequency spectral decomposition by the S-transform, the Gabor transform or the Morlet
continuous wavelet transform (see seisdesk.core.timefreq)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert as _scipy_hilbert

from ..core.timefreq import METHODS, band_component, sigma_f
from .base import Param, Process, register

ATTRIBUTES = ("envelope", "instantaneous phase", "cosine of phase", "instantaneous frequency",
              "instantaneous bandwidth", "envelope derivative", "envelope second derivative", "sweetness",
              "apparent polarity")
_CHUNK = 256


def analytic_signal(traces: np.ndarray, pad: bool = True) -> np.ndarray:
    """Analytic signal x + i H{x} of every trace (ntr, nt), complex128; the Hilbert transform H turns cos into sin.
    With ``pad`` the traces are zero-padded to at least twice their length, so the end of a trace does not wrap
    around onto its start."""
    x = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    nt = x.shape[1]
    if nt == 0:
        return x.astype(np.complex128)
    n = int(2 ** np.ceil(np.log2(max(2 * nt, 2)))) if pad else nt
    return _scipy_hilbert(x, N=n, axis=1)[:, :nt]


def _smooth(values: np.ndarray, nwin: int) -> np.ndarray:
    return uniform_filter1d(values, nwin, axis=1, mode="nearest") if nwin > 1 else values


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, zero where den is (relatively) zero."""
    tiny = 1e-12 * float(np.max(np.abs(den))) if den.size else 0.0
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=np.abs(den) > max(tiny, 1e-300))


def instantaneous_frequency(z: np.ndarray, dt: float, window: float = 0.0) -> np.ndarray:
    """Instantaneous frequency (Hz) from the phase increments of the analytic signal z between neighbouring samples,
    averaged over a sliding window with the squared envelope as weight (Barnes 1993). Exact for a pure tone up to the
    Nyquist frequency and stable where the envelope is small."""
    if z.shape[1] < 2:
        return np.zeros(z.shape)
    inc = z[:, 1:] * np.conj(z[:, :-1])                       # |z|^2 exp(i dphi) between samples n and n + 1
    q = np.zeros(z.shape, dtype=np.complex128)
    q[:, :-1] += inc
    q[:, 1:] += inc
    nwin = int(round(window / dt))
    q = _smooth(q.real, nwin) + 1j * _smooth(q.imag, nwin)
    return np.angle(q) / (2.0 * np.pi * dt)


def apparent_polarity(x: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Apparent polarity (Taner et al. 1979): every envelope lobe (between two envelope minima) takes the sign of the
    trace at the envelope maximum times that maximum."""
    ntr, nt = env.shape
    if nt < 3:
        return np.sign(x) * env
    out = np.zeros((ntr, nt))
    for i in range(ntr):
        e = env[i]
        is_min = np.zeros(nt, dtype=bool)
        is_min[1:-1] = (e[1:-1] <= e[:-2]) & (e[1:-1] < e[2:])
        labels = np.cumsum(is_min)                             # lobe number of every sample, 0, 1, 2, ...
        order = np.lexsort((e, labels))                        # by lobe, then by envelope
        ends = np.append(np.flatnonzero(np.diff(labels[order])), nt - 1)
        peaks = order[ends]                                    # envelope maximum of each lobe
        out[i] = (np.sign(x[i, peaks]) * e[peaks])[labels]
    return out


def instantaneous_attribute(traces: np.ndarray, dt: float, name: str, window: float = 0.02) -> np.ndarray:
    """One instantaneous attribute of every trace (ntr, nt), float64."""
    x = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    z = analytic_signal(x)
    env = np.abs(z)
    nwin = int(round(window / dt))
    if name == "envelope":
        return env
    if name == "instantaneous phase":
        return np.degrees(np.angle(z))
    if name == "cosine of phase":
        return _ratio(x, env)
    if name == "instantaneous frequency":
        return instantaneous_frequency(z, dt, window)
    if name in ("instantaneous bandwidth", "envelope derivative", "envelope second derivative"):
        d_env = np.gradient(env, dt, axis=1) if env.shape[1] > 2 else np.zeros_like(env)
        if name == "envelope derivative":
            return d_env
        if name == "envelope second derivative":
            return np.gradient(d_env, dt, axis=1) if env.shape[1] > 2 else np.zeros_like(env)
        # Barnes (1993): |d ln A / dt| / (2 pi), weighted by A^2 over the window
        return _ratio(_smooth(np.abs(d_env) * env, nwin), 2.0 * np.pi * _smooth(env * env, nwin))
    if name == "sweetness":
        floor = max(1.0, 0.01 * 0.5 / dt)                      # Hz: keeps quiet, low-frequency zones finite
        return env / np.sqrt(np.maximum(instantaneous_frequency(z, dt, window), floor))
    if name == "apparent polarity":
        return apparent_polarity(x, env)
    raise ValueError(f"unknown attribute {name!r}")


@register
class HilbertTransform(Process):
    name = "hilbert"
    label = "Hilbert transform (quadrature trace)"
    category = "Attributes"
    description = ("The quadrature trace: every frequency component shifted by -90 degrees, so a cosine becomes a "
                   "sine. With the input trace it forms the analytic (complex) trace x + iH{x}, whose modulus is "
                   "the envelope and whose angle is the instantaneous phase. '-H{x}' gives the opposite shift "
                   "(+90 degrees).")
    params = (Param("sign", "choice", "H{x} (-90 degrees)", "Output",
                    choices=("H{x} (-90 degrees)", "-H{x} (+90 degrees)")),)

    def run(self, data, context):
        out = np.zeros(data.traces.shape)
        for start in range(0, data.ntr, _CHUNK):
            sl = slice(start, min(start + _CHUNK, data.ntr))
            out[sl] = analytic_signal(data.traces[sl]).imag
            context.report(sl.stop / max(data.ntr, 1))
        if self["sign"].startswith("-"):
            out = -out
        return data.copy(out.astype(np.float32))


@register
class InstantaneousAttributes(Process):
    name = "attributes"
    label = "Instantaneous attributes"
    category = "Attributes"
    description = ("Complex-trace attributes from the analytic trace x + iH{x}: envelope (reflection strength), "
                   "instantaneous phase (degrees; shows the continuity of weak events), cosine of phase, "
                   "instantaneous frequency (Hz, envelope-weighted average over the smoothing window), "
                   "instantaneous bandwidth, first and second derivative of the envelope (sharpness of "
                   "reflections), sweetness (envelope / sqrt(frequency), high for bright low-frequency sands) and "
                   "apparent polarity (sign of the trace at each envelope peak times the peak).")
    params = (Param("attribute", "choice", "envelope", "Attribute", choices=ATTRIBUTES),
              Param("window", "float", 0.02, "Smoothing window (frequency, bandwidth, sweetness)", unit="s",
                    minimum=0.0))

    def run(self, data, context):
        out = np.zeros(data.traces.shape)
        for start in range(0, data.ntr, _CHUNK):
            sl = slice(start, min(start + _CHUNK, data.ntr))
            out[sl] = instantaneous_attribute(data.traces[sl], data.dt, self["attribute"], self["window"])
            context.report(sl.stop / max(data.ntr, 1))
        res = data.copy(out.astype(np.float32))
        res.meta["attribute"] = self["attribute"]
        return res


@register
class SpectralDecomposition(Process):
    name = "spectral_decomp"
    label = "Spectral decomposition (single frequency)"
    category = "Attributes"
    description = ("The amplitude (or phase) of every trace at one frequency, as a function of time: a "
                   "single-frequency section for thin-bed tuning, channels and low-frequency shadows. "
                   "S-transform: time window one period long (sharp in time at high frequencies). STFT (Gabor): one "
                   "fixed Gaussian window for all frequencies. CWT (Morlet): like the S-transform, with w0 setting "
                   "the trade-off (larger w0 = sharper in frequency, smoother in time). A unit sinusoid at the "
                   "chosen frequency has amplitude 1.")
    params = (Param("frequency", "float", 30.0, "Frequency", unit="Hz", minimum=0.1),
              Param("method", "choice", "S-transform", "Method", choices=METHODS),
              Param("window", "float", 0.1, "STFT window length (+-2 standard deviations)", unit="s", minimum=0.004),
              Param("morlet_w0", "float", 6.0, "Morlet centre frequency w0 (radians)", minimum=3.0, maximum=20.0),
              Param("output", "choice", "amplitude", "Output", choices=("amplitude", "phase (degrees)")))

    def run(self, data, context):
        f0 = self["frequency"]
        nyq = 0.5 / data.dt
        if f0 >= nyq:
            raise ValueError(f"the frequency {f0:g} Hz is at or above the Nyquist frequency {nyq:g} Hz")
        sf = sigma_f(self["method"], f0, self["window"], self["morlet_w0"])
        out = np.zeros(data.traces.shape)
        for start in range(0, data.ntr, _CHUNK):
            sl = slice(start, min(start + _CHUNK, data.ntr))
            z = band_component(data.traces[sl], data.dt, f0, sf)
            out[sl] = np.abs(z) if self["output"] == "amplitude" else np.degrees(np.angle(z))
            context.report(sl.stop / max(data.ntr, 1))
        res = data.copy(out.astype(np.float32))
        res.meta["spectral_decomposition"] = {"frequency": f0, "method": self["method"], "output": self["output"]}
        return res
