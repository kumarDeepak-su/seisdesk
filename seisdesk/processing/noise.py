"""Noise attenuation: f-x deconvolution, median filtering, linear tau-p (slant-stack) filtering, trace mixing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate1d, median_filter

from .base import Param, Process, register


def _groups(data, key: str):
    """Trace index groups: one per gather (ordered by offset when present), or the whole dataset."""
    if not key:
        return [np.arange(data.ntr)]
    secondary = "offset" if data.has_header("offset") else None
    return [idx for _, idx in data.iter_gathers(key, secondary)]


# ---------------------------------------------------------------------------------------------- f-x deconvolution
def fx_predict(d: np.ndarray, length: int, damping: float) -> np.ndarray:
    """Forward-backward complex Wiener prediction of a spatial sequence (one frequency across the traces)."""
    n = d.size
    if n < 2 * length + 2:
        return d
    a_mat = np.stack([d[length - k: n - k] for k in range(1, length + 1)], axis=1)        # rows j = L..n-1
    b_mat = np.stack([d[k: n - length + k] for k in range(1, length + 1)], axis=1)        # rows j = 0..n-L-1
    out_f, out_b = d.copy(), d.copy()
    for mat, target, sl, store in ((a_mat, d[length:], slice(length, n), out_f),
                                   (b_mat, d[: n - length], slice(0, n - length), out_b)):
        r = mat.conj().T @ mat
        mu = damping * float(np.real(np.trace(r))) / length + 1e-30
        coef = np.linalg.solve(r + mu * np.eye(length), mat.conj().T @ target)
        store[sl] = mat @ coef
    out = 0.5 * (out_f + out_b)
    out[:length] = out_b[:length]
    out[n - length:] = out_f[n - length:]
    return out


def fx_decon_gather(g: np.ndarray, dt: float, window: float, ntraces: int, length: int, damping: float,
                    fmin: float, fmax: float, keep_outside: bool = False) -> np.ndarray:
    """f-x deconvolution of one gather (ntr, nt) in overlapping time and trace windows (weighted overlap-add).

    Frequencies above fmax are tapered off over 20 % of fmax and those below fmin removed, unless keep_outside."""
    ntr, nt = g.shape
    nw = max(8, min(nt, int(round(window / dt))))
    nx = max(2 * length + 2, min(ntr, ntraces))
    t_starts = list(range(0, max(nt - nw, 0) + 1, max(1, nw // 2)))
    if t_starts[-1] + nw < nt:
        t_starts.append(nt - nw)
    x_starts = list(range(0, max(ntr - nx, 0) + 1, max(1, nx // 2)))
    if x_starts[-1] + nx < ntr:
        x_starts.append(ntr - nx)
    acc = np.zeros((ntr, nt))
    wsum = np.zeros((ntr, nt))
    wt = np.hanning(nw) + 0.02
    wx = np.hanning(nx) + 0.02 if nx < ntr else np.ones(nx)
    nfft = int(2 ** np.ceil(np.log2(nw)))
    f = np.fft.rfftfreq(nfft, dt)
    band = np.flatnonzero((f >= fmin) & (f <= 1.2 * fmax))
    taper = np.where(f < fmin, 0.0, np.clip((1.2 * fmax - f) / (0.2 * fmax), 0.0, 1.0))
    for x0 in x_starts:
        for t0 in t_starts:
            block = g[x0:x0 + nx, t0:t0 + nw].astype(np.float64)
            spec = np.fft.rfft(block, nfft, axis=1)
            for j in band:
                spec[:, j] = fx_predict(spec[:, j], length, damping)
            if not keep_outside:
                spec *= taper[None, :]
            filt = np.fft.irfft(spec, nfft, axis=1)[:, :nw]
            w = wx[:, None] * wt[None, :]
            acc[x0:x0 + nx, t0:t0 + nw] += filt * w
            wsum[x0:x0 + nx, t0:t0 + nw] += w
    return acc / np.maximum(wsum, 1e-30)


@register
class FXDecon(Process):
    name = "fx_decon"
    label = "f-x deconvolution (random noise)"
    category = "Filtering"
    description = ("Random-noise attenuation: in overlapping time and trace windows, every frequency is predicted "
                   "across the traces with a complex Wiener filter (forward and backward). Laterally coherent "
                   "(linear) events are predictable and kept; random noise is not. Gather by gather, or on a whole "
                   "stacked section (empty gather key). The output is limited to the band fmin-fmax (tapered over "
                   "20 % above fmax) unless 'outside the band' is set to keep.")
    params = (Param("gather_key", "key", "", "Gather key (empty = whole section)"),
              Param("window", "float", 0.5, "Time window", unit="s", minimum=0.05),
              Param("traces", "int", 30, "Trace window", minimum=8),
              Param("filter_length", "int", 4, "Prediction filter length (traces)", minimum=1, maximum=20),
              Param("damping", "float", 1.0, "Damping", unit="%", minimum=0.0),
              Param("fmin", "float", 2.0, "Lowest frequency", unit="Hz", minimum=0.0),
              Param("fmax", "float", 80.0, "Highest frequency", unit="Hz", minimum=1.0),
              Param("outside", "choice", "remove", "Outside the band", choices=("remove", "keep")))

    def run(self, data, context):
        out = data.traces.astype(np.float64).copy()
        groups = _groups(data, self["gather_key"])
        for n, idx in enumerate(groups):
            if idx.size < 2 * self["filter_length"] + 2:
                continue
            out[idx] = fx_decon_gather(data.traces[idx], data.dt, self["window"], self["traces"],
                                       self["filter_length"], self["damping"] / 100.0, self["fmin"], self["fmax"],
                                       self["outside"] == "keep")
            context.report((n + 1) / len(groups))
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- median filter
@register
class MedianFilter(Process):
    name = "median_filter"
    label = "Median filter"
    category = "Filtering"
    description = ("Running median across the traces (at each time) or along each trace. 'filter' outputs the median: "
                   "removes spikes and random noise and keeps flat events. 'subtract' outputs the data minus the "
                   "median: removes what is flat across the traces (for example a flattened multiple or a "
                   "horizontal noise train) and keeps the rest.")
    params = (Param("direction", "choice", "across traces", "Direction", choices=("across traces", "along trace")),
              Param("length", "int", 5, "Length (traces or samples, odd)", minimum=3),
              Param("output", "choice", "filter", "Output", choices=("filter", "subtract")),
              Param("gather_key", "key", "", "Gather key (empty = whole dataset in file order)"))

    def run(self, data, context):
        n = self["length"] if self["length"] % 2 else self["length"] + 1
        size = (n, 1) if self["direction"] == "across traces" else (1, n)
        out = data.traces.astype(np.float64).copy()
        for idx in _groups(data, self["gather_key"]):
            g = out[idx]
            med = median_filter(g, size=size, mode="nearest")
            out[idx] = med if self["output"] == "filter" else g - med
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- linear tau-p
def linear_radon(gather: np.ndarray, x: np.ndarray, dt: float, p: np.ndarray, damping: float, fmin: float,
                 fmax: float):
    """Least-squares linear Radon (slant stack) t = tau + p x in the frequency domain: (model, operator, nfft, band)."""
    ntr, nt = gather.shape
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    freqs = np.fft.rfftfreq(nfft, dt)
    data_f = np.fft.rfft(gather.astype(np.float64), nfft, axis=1)
    band = np.flatnonzero((freqs >= fmin) & (freqs <= fmax))
    model = np.zeros((p.size, freqs.size), dtype=np.complex128)
    xx = np.asarray(x, dtype=np.float64)[:, None]

    def operator(j):
        return np.exp(-2j * np.pi * freqs[j] * xx * p[None, :])

    for j in band:
        op = operator(j)
        lhl = op.conj().T @ op
        mu = damping * float(np.real(np.trace(lhl))) / p.size + 1e-30
        model[:, j] = np.linalg.solve(lhl + mu * np.eye(p.size), op.conj().T @ data_f[:, j])
    return model, operator, nfft, band


@register
class TauPFilter(Process):
    name = "taup_filter"
    label = "Linear tau-p (slant-stack) filter"
    category = "Filtering"
    description = ("Least-squares linear Radon transform (tau-p) of each gather. Events slower than the cut velocity "
                   "(ground roll, guided and direct waves, other linear noise) are modelled and subtracted, or kept "
                   "alone with 'pass slow'. Signed offsets are used, so the two sides of a split spread are handled "
                   "correctly; irregular offsets are no problem.")
    params = (Param("gather_key", "key", "fldr", "Gather key"),
              Param("v_cut", "float", 1200.0, "Apparent velocity cut", unit="m/s", minimum=10.0),
              Param("v_min", "float", 200.0, "Slowest velocity modelled", unit="m/s", minimum=10.0),
              Param("mode", "choice", "reject slow", "Mode", choices=("reject slow", "pass slow")),
              Param("np", "int", 121, "Number of p values", minimum=11),
              Param("damping", "float", 0.05, "Damping (fraction of mean diagonal)", minimum=0.0),
              Param("fmin", "float", 2.0, "Lowest frequency", unit="Hz", minimum=0.0),
              Param("fmax", "float", 80.0, "Highest frequency", unit="Hz", minimum=1.0))

    def run(self, data, context):
        if self["v_min"] >= self["v_cut"]:
            raise ValueError("the slowest velocity modelled must be below the cut velocity")
        p = np.linspace(-1.0 / self["v_min"], 1.0 / self["v_min"], self["np"])
        slow = np.abs(p) >= 1.0 / self["v_cut"]
        out = data.traces.astype(np.float64).copy()
        groups = _groups(data, self["gather_key"])
        for n, idx in enumerate(groups):
            if idx.size < 4:
                continue
            g = data.traces[idx]
            model, op, nfft, band = linear_radon(g, data.header("offset")[idx], data.dt, p, self["damping"],
                                                 self["fmin"], self["fmax"])
            noise_f = np.zeros((idx.size, model.shape[1]), dtype=np.complex128)
            for j in band:
                noise_f[:, j] = op(j)[:, slow] @ model[slow, j]
            noise = np.fft.irfft(noise_f, nfft, axis=1)[:, :data.nt]
            out[idx] = (g - noise) if self["mode"] == "reject slow" else noise
            context.report((n + 1) / len(groups))
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- trace mixing
@register
class TraceMix(Process):
    name = "trace_mix"
    label = "Trace mixing (spatial smoothing)"
    category = "Filtering"
    description = ("Weighted running average of neighbouring traces (e.g. weights 1,2,1), gather by gather or along a "
                   "section: raises lateral coherence and suppresses random noise, but smears steep dips.")
    params = (Param("weights", "floats", "1,2,1", "Weights (odd count)"),
              Param("gather_key", "key", "", "Gather key (empty = whole section)"))

    def run(self, data, context):
        w = np.asarray(self["weights"], dtype=np.float64)
        if w.size % 2 == 0 or w.sum() == 0:
            raise ValueError("give an odd number of weights with a non-zero sum, e.g. 1,2,1")
        w = w / w.sum()
        out = data.traces.astype(np.float64).copy()
        for idx in _groups(data, self["gather_key"]):
            out[idx] = correlate1d(out[idx], w, axis=0, mode="nearest")
        return data.copy(out.astype(np.float32))
