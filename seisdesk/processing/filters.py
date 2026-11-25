"""Frequency and f-k filters (all zero phase)."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from .base import Param, Process, register


def ormsby_response(freqs: np.ndarray, f1: float, f2: float, f3: float, f4: float) -> np.ndarray:
    """Trapezoidal pass band with cosine-squared tapers between f1-f2 and f3-f4."""
    if not (0 <= f1 <= f2 <= f3 <= f4):
        raise ValueError("corner frequencies must satisfy 0 <= f1 <= f2 <= f3 <= f4")
    f = np.abs(freqs)
    h = np.zeros_like(f)
    h[(f >= f2) & (f <= f3)] = 1.0
    if f2 > f1:
        m = (f > f1) & (f < f2)
        h[m] = np.sin(0.5 * np.pi * (f[m] - f1) / (f2 - f1)) ** 2
    if f4 > f3:
        m = (f > f3) & (f < f4)
        h[m] = np.cos(0.5 * np.pi * (f[m] - f3) / (f4 - f3)) ** 2
    return h


def bandpass(traces: np.ndarray, dt: float, corners) -> np.ndarray:
    n = traces.shape[1]
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    freqs = np.fft.rfftfreq(nfft, dt)
    h = ormsby_response(freqs, *corners)
    return np.fft.irfft(np.fft.rfft(traces, nfft, axis=1) * h, nfft, axis=1)[:, :n].astype(np.float32)


@register
class Bandpass(Process):
    name = "bandpass"
    label = "Bandpass (Ormsby)"
    category = "Filtering"
    description = "Zero-phase trapezoidal band-pass filter with corners f1 < f2 < f3 < f4."
    params = (Param("corners", "floats", "5,10,60,80", "Corners f1,f2,f3,f4", unit="Hz"),)

    def run(self, data, context):
        c = self["corners"]
        if len(c) != 4:
            raise ValueError("give four corner frequencies")
        return data.copy(bandpass(data.traces, data.dt, c))


@register
class Butterworth(Process):
    name = "butterworth"
    label = "Butterworth filter"
    category = "Filtering"
    description = "Zero-phase (forward-backward) Butterworth low-cut / high-cut / band-pass."
    params = (Param("low_cut", "float", 5.0, "Low cut (0 = none)", unit="Hz", minimum=0.0),
              Param("high_cut", "float", 80.0, "High cut (0 = none)", unit="Hz", minimum=0.0),
              Param("order", "int", 4, "Order", minimum=1, maximum=12))

    def run(self, data, context):
        nyq = 0.5 / data.dt
        lo, hi = self["low_cut"], self["high_cut"]
        if hi >= nyq:                                  # a high cut at or above Nyquist does nothing
            hi = 0.0
        if lo >= nyq:
            raise ValueError(f"low cut {lo:g} Hz is not below the Nyquist frequency {nyq:g} Hz")
        if lo > 0 and hi > 0:
            sos = signal.butter(self["order"], [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        elif lo > 0:
            sos = signal.butter(self["order"], lo / nyq, btype="highpass", output="sos")
        elif hi > 0:
            sos = signal.butter(self["order"], hi / nyq, btype="lowpass", output="sos")
        else:
            return data.copy()
        return data.copy(signal.sosfiltfilt(sos, data.traces, axis=1).astype(np.float32))


@register
class Notch(Process):
    name = "notch"
    label = "Notch filter"
    category = "Filtering"
    description = "Remove a narrow band around one frequency (e.g. 50 Hz power-line noise)."
    params = (Param("frequency", "float", 50.0, "Frequency", unit="Hz", minimum=0.1),
              Param("quality", "float", 30.0, "Quality factor", minimum=1.0))

    def run(self, data, context):
        b, a = signal.iirnotch(self["frequency"], self["quality"], fs=1.0 / data.dt)
        return data.copy(signal.filtfilt(b, a, data.traces, axis=1).astype(np.float32))


def fk_fan_filter(gather: np.ndarray, dt: float, dx: float, vmin: float, vmax: float, mode: str = "reject",
                  taper: float = 0.1, fmax: float | None = None) -> np.ndarray:
    """Pass or reject apparent velocities |v| between vmin and vmax in one regularly sampled gather (ntr, nt)."""
    ntr, nt = gather.shape
    nx = int(2 ** np.ceil(np.log2(2 * ntr)))
    ntf = int(2 ** np.ceil(np.log2(2 * nt)))
    spec = np.fft.fft2(gather, s=(nx, ntf))
    k = np.fft.fftfreq(nx, dx)[:, None]
    f = np.fft.fftfreq(ntf, dt)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        vapp = np.abs(f / np.where(k == 0, 1e-30, k))
    inside = np.clip((vapp - vmin * (1 - taper)) / max(vmin * taper, 1e-9), 0, 1) * \
        np.clip((vmax * (1 + taper) - vapp) / max(vmax * taper, 1e-9), 0, 1)
    inside[np.broadcast_to(k == 0, inside.shape)] = 0.0
    weight = 1.0 - inside if mode == "reject" else inside
    if fmax is not None:
        weight = weight * (np.abs(f) <= fmax)
    out = np.real(np.fft.ifft2(spec * weight))[:ntr, :nt]
    return out.astype(np.float32)


@register
class FKFilter(Process):
    name = "fk_filter"
    label = "f-k fan filter"
    category = "Filtering"
    description = ("Pass or reject a fan of apparent velocities (e.g. ground roll, direct waves, linear noise) "
                   "gather by gather; traces are taken in order of the position key with their median spacing.")
    params = (Param("gather_key", "key", "fldr", "Gather key"),
              Param("position_key", "key", "offset", "Trace position key"),
              Param("vmin", "float", 0.0, "Minimum |apparent velocity|", unit="m/s", minimum=0.0),
              Param("vmax", "float", 1800.0, "Maximum |apparent velocity|", unit="m/s", minimum=1.0),
              Param("mode", "choice", "reject", "Mode", choices=("reject", "pass")),
              Param("taper", "float", 0.1, "Velocity taper (fraction)", minimum=0.0, maximum=0.9))

    def run(self, data, context):
        out = data.traces.copy()
        values = list(data.iter_gathers(self["gather_key"], secondary=self["position_key"]))
        for n, (_, idx) in enumerate(values):
            if idx.size < 4:
                continue
            pos = data.header(self["position_key"])[idx]
            dx = float(np.median(np.abs(np.diff(pos)))) or 1.0
            out[idx] = fk_fan_filter(data.traces[idx], data.dt, dx, self["vmin"], self["vmax"], self["mode"],
                                     self["taper"])
            context.report((n + 1) / len(values))
        return data.copy(out)


def parse_tv_windows(text: str) -> list[tuple[float, list[float]]]:
    """'0.0: 8,12,80,100; 1.0: 6,10,60,80' -> [(0.0, [8, 12, 80, 100]), (1.0, [6, 10, 60, 80])], sorted by time."""
    out = []
    for part in str(text).split(";"):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"time-variant band '{part.strip()}' needs the form 'time: f1,f2,f3,f4'")
        t, corners = part.split(":", 1)
        c = [float(v) for v in corners.replace(" ", "").split(",") if v]
        if len(c) != 4 or not (0 <= c[0] <= c[1] <= c[2] <= c[3]):
            raise ValueError(f"band at {t.strip()} s needs four increasing corners f1,f2,f3,f4")
        out.append((float(t), c))
    if not out:
        raise ValueError("give at least one 'time: f1,f2,f3,f4' entry")
    return sorted(out)


@register
class TimeVariantBandpass(Process):
    name = "tv_bandpass"
    label = "Time-variant band-pass"
    category = "Filtering"
    description = ("Band-pass whose pass band changes with time: wide in the shallow part, narrower at depth where "
                   "the high frequencies have been absorbed. Each 'time: f1,f2,f3,f4' entry (separated by ';') is "
                   "the band at that time; the filtered versions are blended linearly between the times.")
    params = (Param("windows", "str", "0.0: 8,12,80,100; 1.0: 6,10,60,80; 2.5: 4,8,40,55", "Times (s): corners (Hz)"),)

    def run(self, data, context):
        bands = parse_tv_windows(self["windows"])
        centres = np.array([t for t, _ in bands])
        out = np.zeros(data.traces.shape, dtype=np.float64)
        for k, (_, corners) in enumerate(bands):
            weight = np.interp(data.times, centres, np.eye(len(bands))[k])
            out += bandpass(data.traces, data.dt, corners) * weight[None, :]
        return data.copy(out.astype(np.float32))


@register
class SpectralWhitening(Process):
    name = "spectral_whitening"
    label = "Spectral whitening"
    category = "Filtering"
    description = ("Flattens the amplitude spectrum of each trace inside a band by dividing by its smoothed amplitude "
                   "spectrum (phase kept), widening the band like spiking deconvolution without a minimum-phase "
                   "assumption. Each trace keeps its RMS amplitude.")
    params = (Param("corners", "floats", "5,10,70,90", "Band corners f1,f2,f3,f4", unit="Hz"),
              Param("smoothing", "float", 10.0, "Spectral smoothing width", unit="Hz", minimum=0.1),
              Param("stabilisation", "float", 1.0, "Stabilisation", unit="%", minimum=0.0))

    def run(self, data, context):
        c = self["corners"]
        if len(c) != 4:
            raise ValueError("give four corner frequencies")
        n = data.nt
        nfft = int(2 ** np.ceil(np.log2(2 * n)))
        f = np.fft.rfftfreq(nfft, data.dt)
        spec = np.fft.rfft(data.traces.astype(np.float64), nfft, axis=1)
        amp = np.abs(spec)
        width = max(1, int(round(self["smoothing"] / (f[1] - f[0]))))
        smooth = uniform_filter1d(amp, width, axis=1, mode="nearest")
        stab = self["stabilisation"] / 100.0 * smooth.max(axis=1, keepdims=True) + 1e-30
        out = np.fft.irfft(spec / (smooth + stab) * ormsby_response(f, *c)[None, :], nfft, axis=1)[:, :n]
        rms_in = np.sqrt(np.mean(data.traces.astype(np.float64) ** 2, axis=1, keepdims=True))
        rms_out = np.sqrt(np.mean(out ** 2, axis=1, keepdims=True))
        return data.copy((out * np.where(rms_out > 0, rms_in / np.maximum(rms_out, 1e-30), 0.0)).astype(np.float32))
