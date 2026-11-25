"""Surface-consistent processing: amplitude correction and deconvolution by least-squares decomposition into
source, receiver, offset (and CMP) terms."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.sparse import coo_matrix

from ..core.qc import trace_rms
from .base import Param, Process, register


def surface_terms(data, rows: np.ndarray, offset_class: float, use_cdp: bool):
    """Sparse design matrix (one row per selected trace) with one column per source position, receiver position,
    offset class and, optionally, CMP. Returns the matrix and the column index ranges of each term."""
    sx = np.round(data.header("sx")[rows], 2)
    gx = np.round(data.header("gx")[rows], 2)
    oc = np.floor(np.abs(data.header("offset")[rows]) / offset_class)
    keys = [("source", sx), ("receiver", gx), ("offset", oc)]
    if use_cdp:
        keys.append(("cdp", data.header("cdp")[rows]))
    cols, ranges, start = [], {}, 0
    for name, values in keys:
        u, inv = np.unique(values, return_inverse=True)
        cols.append(inv + start)
        ranges[name] = (start, start + u.size, u, inv)
        start += u.size
    n = rows.size
    r = np.tile(np.arange(n), len(cols))
    a = coo_matrix((np.ones(r.size), (r, np.concatenate(cols))), shape=(n, start)).tocsr()
    return a, ranges


def solve_terms(a, y: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Damped least squares for one or many right-hand sides (columns of y)."""
    ata = (a.T @ a).toarray()
    lam = damping * float(np.trace(ata)) / ata.shape[0]
    return np.linalg.solve(ata + lam * np.eye(ata.shape[0]), a.T @ y)


@register
class SCAmplitude(Process):
    name = "sc_amplitude"
    label = "Surface-consistent amplitude correction"
    category = "Gain"
    description = ("Decomposes the log RMS amplitude of every trace (in a window) into source, receiver, offset and "
                   "CMP terms by least squares, and removes the source and receiver terms (charge size, coupling, "
                   "geophone plants), so amplitudes vary only with offset and geology.")
    params = (Param("tmin", "float", 0.2, "Window start", unit="s"),
              Param("tmax", "float", 0.0, "Window end (0 = trace end)", unit="s"),
              Param("offset_class", "float", 100.0, "Offset class width", unit="m", minimum=1.0),
              Param("use_cdp", "bool", True, "Include a CMP term"))

    def run(self, data, context):
        rms = trace_rms(data, self["tmin"], self["tmax"])
        live = np.flatnonzero(rms > 0)
        if live.size < 4:
            raise ValueError("too few live traces")
        a, ranges = surface_terms(data, live, self["offset_class"], self["use_cdp"])
        y = np.log(rms[live])
        sol = solve_terms(a, y - y.mean())
        corr = np.zeros(data.ntr)
        terms = {}
        for name in ("source", "receiver"):
            s0, s1, u, inv = ranges[name]
            term = sol[s0:s1] - sol[s0:s1].mean()
            corr[live] += term[inv]
            terms[name] = dict(zip(u.tolist(), (20.0 / np.log(10.0) * term).tolist()))
        out = data.copy((data.traces * np.exp(-corr)[:, None]).astype(np.float32))
        context.results["sc_amplitude"] = terms
        spread = 20.0 / np.log(10.0) * np.ptp(corr[live])
        context.log(f"    source and receiver terms removed (range {spread:.1f} dB)")
        return out


def minimum_phase(log_amp: np.ndarray) -> np.ndarray:
    """Minimum-phase spectrum (rfft layout, last axis) with the given log amplitude, by the folded real cepstrum."""
    nf = log_amp.shape[-1]
    nfft = 2 * (nf - 1)
    cep = np.fft.irfft(log_amp, nfft, axis=-1)
    fold = np.zeros(nfft)
    fold[0] = 1.0
    fold[1:nfft // 2] = 2.0
    fold[nfft // 2] = 1.0
    return np.exp(np.fft.rfft(cep * fold, nfft, axis=-1))


@register
class SCDecon(Process):
    name = "sc_decon"
    label = "Surface-consistent deconvolution"
    category = "Deconvolution"
    description = ("Decomposes the log amplitude spectra of all traces (design window) into source, receiver and "
                   "offset terms by least squares; each trace is deconvolved with the minimum-phase inverse of its "
                   "average x source x receiver spectrum. Wavelets become consistent along the line and noisy "
                   "traces do not spoil their own operators.")
    params = (Param("tmin", "float", 0.2, "Design window start", unit="s"),
              Param("tmax", "float", 2.0, "Design window end", unit="s"),
              Param("offset_class", "float", 100.0, "Offset class width", unit="m", minimum=1.0),
              Param("smoothing", "float", 5.0, "Spectral smoothing", unit="Hz", minimum=0.0),
              Param("prewhitening", "float", 1.0, "Pre-whitening", unit="%", minimum=0.0),
              Param("fmax", "float", 0.0, "Highest frequency kept (0 = Nyquist)", unit="Hz", minimum=0.0))

    def run(self, data, context):
        t = data.times
        sel = np.flatnonzero((t >= self["tmin"]) & (t <= self["tmax"]))
        if sel.size < 16:
            raise ValueError("the design window is too short")
        seg = data.traces[:, sel].astype(np.float64) * np.hanning(sel.size)[None, :]
        nw = int(2 ** np.ceil(np.log2(sel.size)))
        fw = np.fft.rfftfreq(nw, data.dt)
        amp = np.abs(np.fft.rfft(seg, nw, axis=1))
        if self["smoothing"] > 0:
            amp = uniform_filter1d(amp, max(1, int(round(self["smoothing"] / (fw[1] - fw[0])))), axis=1,
                                   mode="nearest")
        level = amp.max(axis=1)
        live = np.flatnonzero(level > 0)
        logamp = np.log(amp[live] + 1e-6 * level[live, None])
        a, ranges = surface_terms(data, live, self["offset_class"], False)
        mean = logamp.mean(axis=0)
        sol = solve_terms(a, logamp - mean[None, :])
        log_op = np.tile(mean, (data.ntr, 1))
        for name in ("source", "receiver"):
            s0, s1, _u, inv = ranges[name]
            term = sol[s0:s1] - sol[s0:s1].mean(axis=0, keepdims=True)
            log_op[live] += term[inv]
        nfft = int(2 ** np.ceil(np.log2(2 * data.nt)))
        f = np.fft.rfftfreq(nfft, data.dt)
        out = data.traces.astype(np.float64).copy()
        band = np.ones(f.size)
        if self["fmax"] > 0:
            band = np.clip((self["fmax"] - f) / max(0.1 * self["fmax"], 1e-6), 0.0, 1.0)
        pw = self["prewhitening"] / 100.0
        for i in live:
            wav = np.exp(np.interp(f, fw, log_op[i]))
            inv_amp = 1.0 / (wav + pw * wav.max())
            spec = minimum_phase(np.log(inv_amp)) * band
            y = np.fft.irfft(np.fft.rfft(data.traces[i].astype(np.float64), nfft) * spec, nfft)[:data.nt]
            r_in = np.sqrt(np.mean(data.traces[i, sel].astype(np.float64) ** 2))
            r_out = np.sqrt(np.mean(y[sel] ** 2))
            out[i] = y * (r_in / r_out if r_out > 0 else 0.0)
            if i % 200 == 0:
                context.report(i / data.ntr)
        context.log(f"    {len(ranges['source'][2])} source and {len(ranges['receiver'][2])} receiver spectra, "
                    f"{len(ranges['offset'][2])} offset classes")
        return data.copy(out.astype(np.float32))
