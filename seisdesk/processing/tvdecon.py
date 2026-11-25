"""Time-variant deconvolution: Wiener prediction-error operators designed in several overlapping time windows and
blended between the window centres (multi-window deconvolution), and Gabor deconvolution of non-stationary,
attenuated data (Margrave, Lamoureux & Henley 2011).

Gabor deconvolution works in the Gabor (windowed Fourier) domain. Gaussian windows that sum to one at every sample
(a partition of unity) cut the trace into overlapping pieces; the smoothed amplitude of their spectra estimates the
propagating wavelet, source spectrum times attenuation, at every time and frequency. Every piece is divided by the
minimum-phase (or zero-phase) wavelet of its own time, and the pieces are summed back."""

from __future__ import annotations

import re

import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d

from .base import Param, Process, register
from .decon import apply_filters, prediction_error_filters
from .filters import ormsby_response
from .surface import minimum_phase


# ---------------------------------------------------------------------------------------------- multi-window decon
def parse_windows(text: str) -> list[tuple[float, float]]:
    """Design windows written 'start-end, start-end, ...' in seconds, sorted by their centres."""
    out = []
    for part in re.split(r"[,;]", str(text)):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*(?:-|:|to)\s*([0-9]*\.?[0-9]+)$", part)
        if not m:
            raise ValueError(f"design window {part!r}: write start-end in seconds, e.g. 0.2-1.2")
        a, b = float(m.group(1)), float(m.group(2))
        if b <= a:
            raise ValueError(f"design window {part!r}: the end must come after the start")
        out.append((a, b))
    if not out:
        raise ValueError("give at least one design window, e.g. 0.2-1.2, 1.0-2.5")
    return sorted(out, key=lambda w: w[0] + w[1])


def blend_weights(times: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """(nwin, nt) weights interpolating linearly between the window centres (constant before the first and after the
    last centre); they sum to one at every time."""
    centres = np.asarray(centres, dtype=np.float64)
    if centres.size > 1 and np.any(np.diff(centres) <= 0):
        raise ValueError("the design windows must have different centres")
    eye = np.eye(centres.size)
    return np.stack([np.interp(times, centres, eye[k]) for k in range(centres.size)])


@register
class TimeVariantDecon(Process):
    name = "tv_decon"
    label = "Time-variant deconvolution (multi-window)"
    category = "Deconvolution"
    description = ("Spiking or predictive deconvolution whose operator changes with time: one Wiener "
                   "prediction-error filter per trace is designed in every design window, the whole trace is "
                   "filtered with each, and the outputs are blended linearly between the window centres (before the "
                   "first centre only the first operator is used, after the last only the last). Use it when the "
                   "wavelet changes down the trace, e.g. through absorption: shallow and deep windows each get their "
                   "own operator. Windows are written start-end in seconds, separated by commas.")
    params = (Param("windows", "str", "0.1-1.2, 1.0-2.2, 2.0-3.5", "Design windows (start-end, s)"),
              Param("operator_length", "float", 0.16, "Operator length", unit="s", minimum=0.004),
              Param("gap", "float", 0.004, "Prediction gap (dt = spiking)", unit="s", minimum=0.0),
              Param("prewhitening", "float", 0.1, "Pre-whitening", unit="%", minimum=0.0))

    def run(self, data, context):
        wins = parse_windows(self["windows"])
        gap = max(self["gap"], data.dt)
        centres = np.array([0.5 * (a + b) for a, b in wins])
        weights = blend_weights(data.times, centres)
        out = np.zeros(data.traces.shape, dtype=np.float64)
        for k, (a, b) in enumerate(wins):
            try:
                f = prediction_error_filters(data.traces, data.dt, self["operator_length"], gap,
                                             self["prewhitening"], a, b, data.t0)
            except ValueError as exc:
                raise ValueError(f"design window {a:g}-{b:g} s: {exc}") from None
            out += weights[k][None, :] * apply_filters(data.traces, f)
            context.report((k + 1) / len(wins))
        context.results["tv_decon"] = {"windows": [list(w) for w in wins], "centres": centres.tolist()}
        context.log(f"    {len(wins)} design window(s), operators blended between "
                    + ", ".join(f"{c:.2f}" for c in centres) + " s")
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- Gabor decon
def gabor_windows(nt: int, dt: float, t0: float, half_width: float, spacing: float):
    """Gaussian windows exp(-((t - c) / half_width)^2) with centres every ``spacing`` seconds from the first to past
    the last sample, normalised to a partition of unity: (centres, windows (nwin, nt))."""
    times = t0 + np.arange(nt) * dt
    span = float(times[-1] - times[0]) if nt > 1 else 0.0
    nwin = max(1, int(np.ceil(span / spacing)) + 1)
    centres = times[0] + np.arange(nwin) * spacing
    g = np.exp(-(((times[None, :] - centres[:, None]) / half_width) ** 2))
    return centres, g / np.maximum(g.sum(axis=0), 1e-12)[None, :]


def hyperbolic_smooth(amp: np.ndarray, centres: np.ndarray, freqs: np.ndarray, nf_smooth: int,
                      nbins: int = 64) -> np.ndarray:
    """Wavelet amplitude |W(t, f)| = S(f) A(t f) from a Gabor amplitude spectrum (nwin, nf): the source spectrum S,
    smoothed over nf_smooth frequency bins, times an attenuation function of t f alone, which a constant-Q earth
    produces (Margrave et al. 2011). Fitted in the log domain by two alternating passes; A(0) = 1."""
    top = float(amp.max())
    log_amp = np.log(amp + 1e-6 * top + 1e-30)
    tf = np.maximum(centres, 0.0)[:, None] * freqs[None, :]
    edges = np.linspace(0.0, float(tf.max()) + 1e-12, nbins + 1)
    idx = np.clip(np.searchsorted(edges, tf, side="right") - 1, 0, nbins - 1)
    counts = np.bincount(idx.ravel(), minlength=nbins)
    att = np.zeros_like(log_amp)
    src = np.zeros(freqs.size)
    for _ in range(2):
        src = uniform_filter1d((log_amp - att).mean(axis=0), max(1, nf_smooth), mode="nearest")
        resid = log_amp - src[None, :]
        sums = np.bincount(idx.ravel(), weights=resid.ravel(), minlength=nbins)
        a = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        good = np.flatnonzero(np.isfinite(a))
        a = np.interp(np.arange(nbins), good, a[good]) if good.size else np.zeros(nbins)
        a = uniform_filter1d(a, 3, mode="nearest")
        a -= a[0]
        att = a[idx]
    return np.exp(src[None, :] + att)


def gabor_decon_trace(x: np.ndarray, dt: float, t0: float, half_width: float, spacing: float, smoothing: str,
                      t_smooth: float, f_smooth: float, stab: float, band: np.ndarray | None,
                      phase: str) -> np.ndarray:
    """Gabor deconvolution of one trace; ``band`` is the band-pass response on the rfft frequencies (or None)."""
    nt = x.size
    centres, g = gabor_windows(nt, dt, t0, half_width, spacing)
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    freqs = np.fft.rfftfreq(nfft, dt)
    spec = np.fft.rfft(g * x[None, :], nfft, axis=1)                  # the Gabor transform, one row per window
    amp = np.abs(spec)
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    nf_smooth = max(1, int(round(f_smooth / df)))
    if smoothing == "boxcar":
        nt_smooth = max(1, int(round(t_smooth / spacing)))
        est = uniform_filter(amp, (nt_smooth, nf_smooth), mode="nearest")
    else:
        est = hyperbolic_smooth(amp, centres, freqs, nf_smooth)
    est = np.maximum(est + stab * float(est.max()), np.finfo(float).tiny)
    if phase == "minimum":
        op = minimum_phase(-np.log(est))                              # 1 / wavelet, minimum phase
    else:
        op = 1.0 / est
    if band is not None:
        op = op * band[None, :]
    return np.fft.irfft(spec * op, nfft, axis=1)[:, :nt].sum(axis=0)


@register
class GaborDecon(Process):
    name = "gabor_decon"
    label = "Gabor deconvolution (non-stationary)"
    category = "Deconvolution"
    description = ("Deconvolution of a wavelet that changes with time through attenuation (Margrave, Lamoureux & "
                   "Henley 2011). Overlapping Gaussian windows (a partition of unity) give the Gabor spectrum of each "
                   "trace; its amplitude, smoothed, estimates the propagating wavelet at every time and frequency: "
                   "'hyperbolic' smoothing fits a source spectrum times an attenuation function of time x frequency "
                   "(constant Q), 'boxcar' smooths over a time and a frequency width. Every window is divided by the "
                   "minimum-phase (or zero-phase) wavelet of its time, band-limited by the corners, and the windows "
                   "are summed. The method also restores the amplitude decay with time, so each output trace is "
                   "scaled back to the RMS amplitude of its input.")
    params = (Param("window", "float", 0.2, "Gaussian window half-width", unit="s", minimum=0.01),
              Param("spacing", "float", 0.05, "Window spacing", unit="s", minimum=0.002),
              Param("smoothing", "choice", "hyperbolic", "Wavelet estimate", choices=("hyperbolic", "boxcar")),
              Param("t_smooth", "float", 0.4, "Time smoothing (boxcar)", unit="s", minimum=0.0),
              Param("f_smooth", "float", 10.0, "Frequency smoothing", unit="Hz", minimum=0.0),
              Param("stabilisation", "float", 1.0, "Stabilisation (of the largest amplitude)", unit="%",
                    minimum=0.0),
              Param("band", "floats", "5,10,80,100", "Output band f1,f2,f3,f4", unit="Hz"),
              Param("phase", "choice", "minimum", "Wavelet phase", choices=("minimum", "zero")))

    def run(self, data, context):
        if self["spacing"] > self["window"]:
            raise ValueError("the window spacing must not exceed the window half-width (the windows must overlap)")
        corners = list(self["band"])
        nfft = int(2 ** np.ceil(np.log2(2 * data.nt)))
        freqs = np.fft.rfftfreq(nfft, data.dt)
        band = None
        if corners and any(corners):
            if len(corners) != 4:
                raise ValueError("give four band corners f1,f2,f3,f4 (Hz)")
            band = ormsby_response(freqs, *corners)
        out = np.zeros(data.traces.shape, dtype=np.float64)
        stab = self["stabilisation"] / 100.0
        for i in range(data.ntr):
            x = data.traces[i].astype(np.float64)
            rms_in = float(np.sqrt(np.mean(x * x)))
            if rms_in == 0.0:
                continue
            y = gabor_decon_trace(x, data.dt, data.t0, self["window"], self["spacing"], self["smoothing"],
                                  self["t_smooth"], self["f_smooth"], stab, band, self["phase"])
            rms_out = float(np.sqrt(np.mean(y * y)))
            out[i] = y * (rms_in / rms_out) if rms_out > 0 else 0.0
            if i % 50 == 0:
                context.report(i / max(data.ntr, 1))
        return data.copy(out.astype(np.float32))
