"""Wavelet processing: constant phase rotation, zero- and minimum-phase conversion, Wiener shaping filters,
signature (deterministic) deconvolution with a known source wavelet, and homomorphic (cepstral) deconvolution."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from ..core.synthetic import ricker
from .base import Param, Process, register
from .filters import ormsby_response
from .surface import minimum_phase


def _nfft(n: int) -> int:
    return int(2 ** np.ceil(np.log2(max(2, n))))


# ---------------------------------------------------------------------------------------------- wavelet helpers
def read_wavelet(path: str, dt: float, file_dt: float = 0.0) -> tuple[np.ndarray, float]:
    """A wavelet from a text file, resampled to the data interval dt; returns (samples, time of the first sample).

    One number per line: amplitudes sampled at file_dt (the data interval when file_dt is 0), first sample at t = 0.
    Two columns: time and amplitude; times in seconds, or in milliseconds when the largest |time| exceeds 5. Lines
    that do not start with a number (headers, comments) are skipped; commas and semicolons count as spaces."""
    if not path:
        raise ValueError("no wavelet file given")
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.replace(",", " ").replace(";", " ").split()
            if not parts:
                continue
            try:
                rows.append([float(p) for p in parts[:2]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no numbers in the wavelet file {path}")
    if all(len(r) == 2 for r in rows):
        arr = np.array(rows, dtype=np.float64)
        t, a = arr[:, 0], arr[:, 1]
        if np.max(np.abs(t)) > 5.0:
            t = t / 1000.0
        order = np.argsort(t, kind="stable")
        t, a = t[order], a[order]
        k0 = int(np.ceil(t[0] / dt - 1e-6))
        k1 = int(np.floor(t[-1] / dt + 1e-6))
        if k1 < k0:
            raise ValueError(f"the wavelet in {path} is shorter than one sample")
        tt = np.arange(k0, k1 + 1) * dt
        return np.interp(tt, t, a), float(tt[0])
    a = np.array([r[0] for r in rows], dtype=np.float64)
    step = file_dt if file_dt > 0 else dt
    if abs(step - dt) > 1e-9 * dt and a.size > 1:
        t = np.arange(a.size) * step
        a = np.interp(np.arange(0.0, t[-1] + 1e-12, dt), t, a)
    return a, 0.0


def wavelet_spectrum(w: np.ndarray, t_first: float, dt: float, nfft: int) -> np.ndarray:
    """Spectrum (rfft layout) of a wavelet whose first sample is at time t_first (negative times wrap to the end of
    the buffer, so the spectrum carries the wavelet's true time origin)."""
    w = np.asarray(w, dtype=np.float64)[:nfft]
    buf = np.zeros(nfft)
    idx = (int(round(t_first / dt)) + np.arange(w.size)) % nfft
    np.add.at(buf, idx, w)
    return np.fft.rfft(buf)


def design_amplitude(traces: np.ndarray, times: np.ndarray, dt: float, tmin: float, tmax: float, nfft: int,
                     smooth_hz: float) -> np.ndarray:
    """Average amplitude spectrum of the traces in a design window (the square root of the mean power spectrum,
    i.e. of the transform of the average autocorrelation), each trace normalised to unit energy, smoothed."""
    sel = (times >= tmin) & ((times <= tmax) if tmax > tmin else np.ones(times.size, bool))
    if sel.sum() < 16:
        sel[:] = True
    seg = traces[:, sel].astype(np.float64) * np.hanning(int(sel.sum()))[None, :]
    power = np.abs(np.fft.rfft(seg, nfft, axis=1)) ** 2
    energy = power.sum(axis=1)
    live = energy > 0
    if not live.any():
        raise ValueError("the design window holds no signal")
    amp = np.sqrt(np.mean(power[live] / energy[live, None], axis=0))
    if smooth_hz > 0:
        df = 1.0 / (nfft * dt)
        amp = uniform_filter1d(amp, max(1, int(round(smooth_hz / df))), mode="nearest")
    return amp


def min_phase_of(amp: np.ndarray, water_level: float) -> np.ndarray:
    """Minimum-phase spectrum whose amplitude is amp plus a water level (fraction of the peak)."""
    return minimum_phase(np.log(amp + water_level * float(np.max(amp)) + 1e-30))


def apply_spectrum(traces: np.ndarray, h: np.ndarray, nfft: int) -> np.ndarray:
    nt = traces.shape[1]
    return np.fft.irfft(np.fft.rfft(traces.astype(np.float64), nfft, axis=1) * h[None, :], nfft, axis=1)[:, :nt]


def apply_filter(traces: np.ndarray, f: np.ndarray, advance: int = 0) -> np.ndarray:
    """Convolve every trace with the filter f and advance the result by ``advance`` samples (negative: delay)."""
    nt = traces.shape[1]
    nfft = _nfft(nt + f.size + abs(advance))
    full = np.fft.irfft(np.fft.rfft(traces.astype(np.float64), nfft, axis=1) * np.fft.rfft(f, nfft), nfft, axis=1)
    return np.roll(full, -advance, axis=1)[:, :nt]


def rotate_phase(traces: np.ndarray, angle_deg: float) -> np.ndarray:
    """Constant phase rotation: Re{analytic signal x exp(i angle)} = x cos(a) - H(x) sin(a)."""
    x = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    nt = x.shape[1]
    analytic = hilbert(x, N=_nfft(2 * nt), axis=1)[:, :nt]
    return np.real(analytic * np.exp(1j * np.deg2rad(angle_deg)))


def zero_phase_wavelet(corners, dt: float, half: int) -> tuple[np.ndarray, int]:
    """Zero-phase Ormsby band-pass wavelet of 2 half + 1 samples, peak at index half, unit peak."""
    nfft = _nfft(4 * half + 4)
    w = np.fft.irfft(ormsby_response(np.fft.rfftfreq(nfft, dt), *corners), nfft)
    out = np.concatenate([w[nfft - half:], w[:half + 1]])
    peak = float(np.max(np.abs(out)))
    return (out / peak if peak > 0 else out), half


# ---------------------------------------------------------------------------------------------- phase rotation
@register
class PhaseRotation(Process):
    name = "phase_rotation"
    label = "Phase rotation (constant phase)"
    category = "Wavelet"
    description = ("Rotates the phase of every trace by a constant angle at all frequencies: out = in cos(a) - H(in) "
                   "sin(a), with H the Hilbert transform. 180 degrees reverses the polarity; +90 or -90 degrees turns "
                   "a zero-phase wavelet into an antisymmetric one. Use it to match the phase of two surveys, or of "
                   "the data and a well synthetic.")
    params = (Param("angle", "float", 90.0, "Phase angle", unit="deg", minimum=-360.0, maximum=360.0),)

    def run(self, data, context):
        return data.copy(rotate_phase(data.traces, self["angle"]).astype(np.float32))


# ---------------------------------------------------------------------------------------------- phase conversion
PHASE_MODES = ("data wavelet", "wavelet file", "per trace")
PHASE_PARAMS = (Param("mode", "choice", "data wavelet", "Wavelet", choices=PHASE_MODES),
                Param("tmin", "float", 0.2, "Design window start", unit="s"),
                Param("tmax", "float", 2.0, "Design window end", unit="s"),
                Param("smoothing", "float", 5.0, "Spectral smoothing", unit="Hz", minimum=0.0),
                Param("wavelet_file", "infile", "", "Wavelet file (text)"),
                Param("file_dt", "float", 0.0, "Sample interval of a one-column file (0 = data)", unit="ms",
                      minimum=0.0),
                Param("water_level", "float", 0.1, "Stabilisation (water level)", unit="%", minimum=0.0))


def convert_phase(data, target: str, mode: str, tmin: float, tmax: float, smoothing: float, wavelet_file: str,
                  file_dt: float, water_level: float) -> np.ndarray:
    """Zero-phase (target 'zero') or minimum-phase (target 'minimum') equivalent of the data wavelet.

    data wavelet: amplitude spectrum from the design window; the wavelet is taken as minimum phase when converting to
    zero phase and as zero phase when converting to minimum phase, so the filter is all-pass (amplitudes unchanged).
    wavelet file: the known wavelet W gives the filter T conj(W) / (|W|^2 + e), T its zero- or minimum-phase
    equivalent. per trace: every trace is treated as a wavelet; a zero-phase result is centred at the envelope peak,
    a minimum-phase result starts where the envelope first reaches 10 % of its peak."""
    x = data.traces.astype(np.float64)
    ntr, nt = x.shape
    nfft = _nfft(2 * nt)
    f = np.fft.rfftfreq(nfft, data.dt)
    wl = water_level / 100.0
    if mode == "per trace":
        out = np.zeros_like(x)
        spec = np.fft.rfft(x, nfft, axis=1)
        env = np.abs(hilbert(x, N=nfft, axis=1)[:, :nt])
        for i in range(ntr):
            amp = np.abs(spec[i])
            if amp.max() <= 0:
                continue
            if target == "zero":
                k = int(np.argmax(env[i]))
                s = amp
            else:
                k = int(np.argmax(env[i] >= 0.1 * env[i].max()))
                s = min_phase_of(amp, wl)
            out[i] = np.fft.irfft(s * np.exp(-2j * np.pi * f * k * data.dt), nfft)[:nt]
        return out
    if mode == "data wavelet":
        amp = design_amplitude(x, data.times, data.dt, tmin, tmax, nfft, smoothing)
        phase = np.angle(min_phase_of(amp, max(wl, 1e-4)))
        h = np.exp(-1j * phase) if target == "zero" else np.exp(1j * phase)
    elif mode == "wavelet file":
        w, t_first = read_wavelet(wavelet_file, data.dt, file_dt / 1000.0)
        spec_w = wavelet_spectrum(w, t_first, data.dt, nfft)
        amp = np.abs(spec_w)
        tgt = amp if target == "zero" else min_phase_of(amp, max(wl, 1e-4))
        h = tgt * np.conj(spec_w) / (amp ** 2 + wl * float(amp.max()) ** 2 + 1e-30)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return apply_spectrum(x, h, nfft)


@register
class ZeroPhase(Process):
    name = "zero_phase"
    label = "Zero-phase conversion"
    category = "Wavelet"
    description = ("Replaces the phase of the wavelet by zero phase and keeps its amplitude spectrum, so reflections "
                   "become symmetric peaks centred on the reflection time (best for interpretation). 'data wavelet' "
                   "takes the wavelet as minimum phase (dynamite, deconvolved data) with the amplitude spectrum of the "
                   "design window; 'wavelet file' uses a known wavelet (a text file); 'per trace' converts every "
                   "trace on its own (for wavelet or signature traces, not for reflection data).")
    params = PHASE_PARAMS

    def run(self, data, context):
        out = convert_phase(data, "zero", self["mode"], self["tmin"], self["tmax"], self["smoothing"],
                            self["wavelet_file"], self["file_dt"], self["water_level"])
        return data.copy(out.astype(np.float32))


@register
class MinPhase(Process):
    name = "min_phase"
    label = "Minimum-phase conversion"
    category = "Wavelet"
    description = ("Converts the wavelet to its minimum-phase equivalent (same amplitude spectrum, energy at the "
                   "front), which predictive deconvolution assumes. 'data wavelet' takes the data wavelet as zero "
                   "phase (correlated vibroseis, zero-phase processed data) with the amplitude spectrum of the design "
                   "window; 'wavelet file' uses a known wavelet; 'per trace' converts every trace on its own (for "
                   "wavelet traces).")
    params = PHASE_PARAMS

    def run(self, data, context):
        out = convert_phase(data, "minimum", self["mode"], self["tmin"], self["tmax"], self["smoothing"],
                            self["wavelet_file"], self["file_dt"], self["water_level"])
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- shaping filter
def wiener_shaping(b: np.ndarray, d: np.ndarray, length: int, prewhitening: float) -> np.ndarray:
    """Least-squares filter f of ``length`` samples with b * f ~ d (Robinson & Treitel 1980): Toeplitz normal
    equations R f = g, R the autocorrelation of b, g the crosscorrelation of d with b."""
    b = np.asarray(b, dtype=np.float64)
    nb = b.size
    r_full = np.correlate(b, b, "full")
    r = np.zeros(length)
    m = min(length, nb)
    r[:m] = r_full[nb - 1:nb - 1 + m]
    if r[0] <= 0:
        raise ValueError("the input wavelet is zero")
    r[0] *= 1.0 + prewhitening / 100.0
    dd = np.zeros(max(d.size, nb + length))
    dd[:d.size] = d
    g = np.correlate(dd, b, "full")[nb - 1:nb - 1 + length]
    return solve_toeplitz(r, g)


@register
class ShapingFilter(Process):
    name = "shaping_filter"
    label = "Wiener shaping filter"
    category = "Wavelet"
    description = ("Designs the least-squares (Wiener) filter that turns the input wavelet into a desired output "
                   "wavelet and applies it to every trace. Input wavelet: estimated from the data (minimum phase, "
                   "from the average autocorrelation in the design window) or read from a file. Desired output: a "
                   "zero-phase Ricker, a zero-phase Ormsby band-pass wavelet or a file. The output delay lets the "
                   "filter reach before the input onset; it is removed again, so a zero-phase output is centred on "
                   "the onset time of the input wavelet.")
    params = (Param("source", "choice", "estimated from the data", "Input wavelet",
                    choices=("estimated from the data", "wavelet file")),
              Param("tmin", "float", 0.2, "Design window start", unit="s"),
              Param("tmax", "float", 2.0, "Design window end", unit="s"),
              Param("smoothing", "float", 5.0, "Spectral smoothing", unit="Hz", minimum=0.0),
              Param("wavelet_file", "infile", "", "Input wavelet file (text)"),
              Param("desired", "choice", "zero-phase Ricker", "Desired output",
                    choices=("zero-phase Ricker", "zero-phase band-pass (Ormsby)", "file")),
              Param("ricker_f0", "float", 30.0, "Ricker peak frequency", unit="Hz", minimum=1.0),
              Param("corners", "floats", "8,12,60,80", "Band-pass corners f1,f2,f3,f4", unit="Hz"),
              Param("desired_file", "infile", "", "Desired wavelet file (text)"),
              Param("file_dt", "float", 0.0, "Sample interval of one-column files (0 = data)", unit="ms", minimum=0.0),
              Param("operator_length", "float", 0.2, "Operator length", unit="s", minimum=0.01),
              Param("prewhitening", "float", 0.1, "Pre-whitening", unit="%", minimum=0.0),
              Param("delay", "float", -1.0, "Output delay (-1 = half the operator)", unit="s"))

    def run(self, data, context):
        dt = data.dt
        length = max(4, int(round(self["operator_length"] / dt)))
        if self["source"] == "wavelet file":
            b, t_b = read_wavelet(self["wavelet_file"], dt, self["file_dt"] / 1000.0)
        else:
            nfft = _nfft(2 * data.nt)
            amp = design_amplitude(data.traces, data.times, dt, self["tmin"], self["tmax"], nfft, self["smoothing"])
            nb = min(nfft // 2, max(length, int(round(0.25 / dt))))
            b = np.fft.irfft(min_phase_of(amp, 1e-3), nfft)[:nb]
            t_b = 0.0
        peak = float(np.max(np.abs(b)))
        if peak <= 0:
            raise ValueError("the input wavelet is zero")
        b = b / peak
        if self["desired"] == "zero-phase Ricker":
            q, cq = ricker(self["ricker_f0"], dt)
        elif self["desired"] == "file":
            q, t_q = read_wavelet(self["desired_file"], dt, self["file_dt"] / 1000.0)
            cq = int(round(-t_q / dt))
        else:
            corners = self["corners"]
            if len(corners) != 4:
                raise ValueError("give four band-pass corners f1,f2,f3,f4")
            q, cq = zero_phase_wavelet(corners, dt, max(8, min(length, int(round(0.2 / dt)))))
        k_d = length // 2 if self["delay"] < 0 else int(round(self["delay"] / dt))
        d = np.zeros(b.size + length - 1)
        for j in range(q.size):                              # desired wavelet with its time zero at sample k_d
            n = k_d - cq + j
            if 0 <= n < d.size:
                d[n] = q[j]
        norm_d = float(np.linalg.norm(d))
        if norm_d <= 0:
            raise ValueError("the desired wavelet falls outside the operator; reduce the output delay")
        d *= float(np.linalg.norm(b)) / norm_d
        f = wiener_shaping(b, d, length, self["prewhitening"])
        advance = k_d + int(round(t_b / dt))
        out = apply_filter(data.traces, f, advance)
        context.results["shaping_filter"] = {"filter": f.tolist(), "input_wavelet": b.tolist(),
                                             "input_t0": t_b, "desired": d.tolist(), "dt": dt}
        actual = np.convolve(b, f)[:d.size]
        err = float(np.sum((actual - d) ** 2) / max(np.sum(d ** 2), 1e-30))
        context.log(f"    shaping filter {length} samples; relative error of the shaped wavelet {100 * err:.1f} %")
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- signature decon
DESIRED_PARAMS = (Param("output", "choice", "zero-phase band-pass", "Desired output",
                        choices=("spike", "zero-phase band-pass")),
                  Param("corners", "floats", "5,10,60,80", "Band-pass corners f1,f2,f3,f4", unit="Hz"))


def _desired(values, f: np.ndarray) -> np.ndarray:
    if values["output"] == "spike":
        return np.ones(f.size)
    corners = values["corners"]
    if len(corners) != 4:
        raise ValueError("give four band-pass corners f1,f2,f3,f4")
    return ormsby_response(f, *corners)


def stabilised_inverse(spec_w: np.ndarray, water_level: float) -> np.ndarray:
    """conj(W) / (|W|^2 + e) with e = water level x the peak power of W."""
    power = np.abs(spec_w) ** 2
    return np.conj(spec_w) / (power + water_level * float(power.max()) + 1e-30)


@register
class SignatureDecon(Process):
    name = "signature_decon"
    label = "Signature (deterministic) deconvolution"
    category = "Deconvolution"
    description = ("Removes a known source wavelet (a measured or modelled air-gun signature, a vibroseis or "
                   "explosive wavelet): every trace is divided in the frequency domain by the wavelet spectrum, "
                   "stabilised by a water level, and the result is shaped to a spike or a zero-phase band-pass "
                   "wavelet. The wavelet comes from a text file (its time origin is honoured) or from a trace in the "
                   "data selected by a header value (e.g. trid 9 for a near-field signature); with a gather key each "
                   "gather uses its own signature trace.")
    params = (Param("source", "choice", "wavelet file", "Wavelet", choices=("wavelet file", "trace in the data")),
              Param("wavelet_file", "infile", "", "Wavelet file (text)"),
              Param("file_dt", "float", 0.0, "Sample interval of a one-column file (0 = data)", unit="ms", minimum=0.0),
              Param("trace_key", "key", "trid", "Signature trace header"),
              Param("trace_value", "float", 9.0, "Signature trace header value"),
              Param("gather_key", "key", "fldr", "One signature per gather of (empty = first only)"),
              Param("water_level", "float", 1.0, "Stabilisation (water level)", unit="%", minimum=0.0)) + DESIRED_PARAMS

    def run(self, data, context):
        x = data.traces.astype(np.float64)
        nt = data.nt
        wl = self["water_level"] / 100.0
        if self["source"] == "wavelet file":
            w, t_first = read_wavelet(self["wavelet_file"], data.dt, self["file_dt"] / 1000.0)
            nfft = _nfft(2 * nt + w.size)
            f = np.fft.rfftfreq(nfft, data.dt)
            h = _desired(self.values, f) * stabilised_inverse(wavelet_spectrum(w, t_first, data.dt, nfft), wl)
            return data.copy(apply_spectrum(x, h, nfft).astype(np.float32))
        key, value = self["trace_key"], self["trace_value"]
        sig = np.flatnonzero(np.abs(data.header(key) - value) <= 1e-6)
        if sig.size == 0:
            raise ValueError(f"no signature trace with {key} = {value:g}")
        nfft = _nfft(2 * nt)
        f = np.fft.rfftfreq(nfft, data.dt)
        desired = _desired(self.values, f)
        use = np.zeros(data.ntr, dtype=np.int64)
        gk = self["gather_key"]
        if gk and sig.size > 1:
            g = data.header(gk)
            first = {}
            for n, k in enumerate(sig):
                first.setdefault(float(g[k]), n)
            use = np.array([first.get(float(v), 0) for v in g], dtype=np.int64)
        out = x.copy()
        for n, k in enumerate(sig):
            rows = np.flatnonzero(use == n)
            if rows.size == 0:
                continue
            spec_w = wavelet_spectrum(x[k], data.t0, data.dt, nfft)
            out[rows] = apply_spectrum(x[rows], desired * stabilised_inverse(spec_w, wl), nfft)
        context.log(f"    {sig.size} signature trace(s) with {key} = {value:g}")
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- homomorphic decon
def cepstral_wavelet(traces: np.ndarray, nfft: int, cutoff: int, water_level: float) -> np.ndarray:
    """Wavelet spectrum (rfft layout) from the low-quefrency part of the averaged complex cepstrum (Ulrych 1971).

    For every trace: log amplitude (with a water level) plus i x unwrapped phase, minus the best-fitting linear
    phase (a delay, weighted by the power spectrum). The complex log spectra are averaged over the traces — the
    reflectivity parts are different from trace to trace and average out, the wavelet part is common — and the
    cepstrum is liftered to quefrencies |n| <= cutoff."""
    spec = np.fft.rfft(traces, nfft, axis=1)
    amp = np.abs(spec)
    level = amp.max(axis=1)
    live = np.flatnonzero(level > 0)
    if live.size == 0:
        raise ValueError("the design window holds no signal")
    k = np.arange(spec.shape[1], dtype=np.float64)
    acc = np.zeros(spec.shape[1], dtype=np.complex128)
    for i in live:
        logamp = np.log(amp[i] + water_level * level[i])
        phase = np.unwrap(np.angle(spec[i]))
        weight = amp[i] ** 2
        slope = float(np.sum(weight * k * (phase - phase[0])) / max(np.sum(weight * k * k), 1e-30))
        phase = phase - slope * k
        phase -= 2.0 * np.pi * np.round(phase[0] / (2.0 * np.pi))
        acc += logamp + 1j * phase
    acc /= live.size
    acc[0] = acc[0].real
    acc[-1] = acc[-1].real
    cep = np.fft.irfft(acc, nfft)
    lifter = np.zeros(nfft)
    nc = int(max(1, min(cutoff, nfft // 2 - 1)))
    lifter[:nc + 1] = 1.0
    lifter[nfft - nc:] = 1.0
    return np.exp(np.fft.rfft(cep * lifter, nfft))


@register
class HomomorphicDecon(Process):
    name = "homomorphic_decon"
    label = "Homomorphic (cepstral) deconvolution"
    category = "Deconvolution"
    description = ("Estimates a mixed-phase wavelet — no minimum-phase assumption — from the low-quefrency part of "
                   "the complex cepstrum averaged over all traces (Ulrych 1971), then deconvolves every trace with it "
                   "(water-level stabilised) to a spike or a zero-phase band-pass wavelet. The estimate is stored in "
                   "the results (its amplitude spectrum is reliable, its phase approximate); the method needs many "
                   "traces and a wavelet shorter than the cepstral cut-off.")
    params = (Param("tmin", "float", 0.2, "Design window start", unit="s"),
              Param("tmax", "float", 2.0, "Design window end", unit="s"),
              Param("cutoff", "float", 0.03, "Cepstral cut-off (quefrency)", unit="s", minimum=0.002),
              Param("water_level", "float", 1.0, "Stabilisation (water level)", unit="%", minimum=0.0)) + DESIRED_PARAMS

    def run(self, data, context):
        x = data.traces.astype(np.float64)
        nt = data.nt
        nfft = _nfft(2 * nt)
        t = data.times
        sel = (t >= self["tmin"]) & ((t <= self["tmax"]) if self["tmax"] > self["tmin"] else np.ones(nt, bool))
        if sel.sum() < 32:
            raise ValueError("the design window is too short")
        seg = np.zeros_like(x)
        seg[:, sel] = x[:, sel] * np.hanning(int(sel.sum()))[None, :]
        wl = self["water_level"] / 100.0
        spec_w = cepstral_wavelet(seg, nfft, int(round(self["cutoff"] / data.dt)), max(wl, 1e-4))
        f = np.fft.rfftfreq(nfft, data.dt)
        out = apply_spectrum(x, _desired(self.values, f) * stabilised_inverse(spec_w, wl), nfft)
        w = np.fft.irfft(spec_w, nfft)
        half = min(nfft // 2, int(round(0.15 / data.dt)))
        centred = np.concatenate([w[nfft - half:], w[:half + 1]])
        context.results["homomorphic_decon"] = {"wavelet": centred.tolist(), "t_first": -half * data.dt,
                                                "dt": data.dt}
        return data.copy(out.astype(np.float32))
