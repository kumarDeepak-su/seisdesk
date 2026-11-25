"""Separation of coherent signal in windows of traces: eigenimage (SVD) filtering (Freire & Ulrych 1988) and
signal/noise separation by nonlinear shrinkage of local slant stacks (after Harlan, Claerbout & Rocca 1984)."""

from __future__ import annotations

import numpy as np

from .base import Param, Process, register
from .noise import _groups


def _window_starts(n: int, size: int) -> list[int]:
    """Start indices of windows of ``size`` covering n samples with 50 % overlap."""
    size = max(1, min(size, n))
    starts = list(range(0, max(n - size, 0) + 1, max(1, size // 2)))
    if starts[-1] + size < n:
        starts.append(n - size)
    return starts


def overlap_add(g: np.ndarray, nx: int, nw: int, fn) -> np.ndarray:
    """Apply fn to overlapping blocks of nx traces by nw samples of g (ntr, nt) and blend them with Hann weights."""
    ntr, nt = g.shape
    nx = max(1, min(nx, ntr))
    nw = max(1, min(nw, nt))
    acc = np.zeros((ntr, nt))
    wsum = np.zeros((ntr, nt))
    wt = np.hanning(nw) + 0.02 if nw < nt else np.ones(nw)
    wx = np.hanning(nx) + 0.02 if nx < ntr else np.ones(nx)
    w = wx[:, None] * wt[None, :]
    for x0 in _window_starts(ntr, nx):
        for t0 in _window_starts(nt, nw):
            acc[x0:x0 + nx, t0:t0 + nw] += fn(g[x0:x0 + nx, t0:t0 + nw]) * w
            wsum[x0:x0 + nx, t0:t0 + nw] += w
    return acc / np.maximum(wsum, 1e-30)


def _shift_length(nt: int, shifts: np.ndarray, dt: float) -> int:
    smax = float(np.max(np.abs(shifts))) if shifts.size else 0.0
    return int(2 ** np.ceil(np.log2(nt + int(np.ceil(smax / dt)) + 2)))


def flatten(g: np.ndarray, shifts: np.ndarray, dt: float) -> np.ndarray:
    """Advance every trace by shifts (s) with an FFT phase shift on a zero-padded copy; the padded length is kept, so
    unflatten() restores the input exactly (nothing is cropped in between)."""
    n = _shift_length(g.shape[1], shifts, dt)
    f = np.fft.rfftfreq(n, dt)
    spec = np.fft.rfft(g, n, axis=1) * np.exp(2j * np.pi * f[None, :] * shifts[:, None])
    return np.fft.irfft(spec, n, axis=1)


def unflatten(flat: np.ndarray, shifts: np.ndarray, dt: float, nt: int) -> np.ndarray:
    n = flat.shape[1]
    f = np.fft.rfftfreq(n, dt)
    spec = np.fft.rfft(flat, n, axis=1) * np.exp(-2j * np.pi * f[None, :] * shifts[:, None])
    return np.fft.irfft(spec, n, axis=1)[:, :nt]


def eigenimage(block: np.ndarray, rank: int, mode: str = "pass") -> np.ndarray:
    """Sum of the first ``rank`` eigenimages of a block (traces x samples) for 'pass', the rest for 'reject'."""
    u, s, vt = np.linalg.svd(block, full_matrices=False)
    k = min(int(rank), s.size)
    low = (u[:, :k] * s[:k]) @ vt[:k]
    return low if mode == "pass" else block - low


@register
class EigenFilter(Process):
    name = "eigen_filter"
    label = "Eigenimage (SVD) filter"
    category = "Filtering"
    description = ("Splits windows of traces into eigenimages by singular value decomposition. Events that are flat "
                   "across a window (after the optional flattening by a linear moveout velocity or a dip) go into the "
                   "first eigenimages. 'pass' keeps the first k: random noise and dipping events are removed. "
                   "'reject' removes them: flat events are removed and everything else is kept. Use small trace "
                   "windows on sections with dipping or curved events.")
    params = (Param("rank", "int", 1, "Number of eigenimages (k)", minimum=1),
              Param("mode", "choice", "pass", "Mode", choices=("pass", "reject")),
              Param("gather_key", "key", "", "Gather key (empty = whole section)"),
              Param("traces", "int", 0, "Trace window (0 = whole gather)", minimum=0),
              Param("window", "float", 0.0, "Time window (0 = whole trace)", unit="s", minimum=0.0),
              Param("lmo_velocity", "float", 0.0, "Flatten by linear moveout |offset| / v (0 = off)", unit="m/s",
                    minimum=0.0),
              Param("dip", "float", 0.0, "Flatten by a dip (0 = off)", unit="ms/trace"))

    def run(self, data, context):
        out = data.traces.astype(np.float64).copy()
        groups = _groups(data, self["gather_key"])
        offsets = np.abs(data.header("offset"))
        for n, idx in enumerate(groups):
            g = out[idx]
            shifts = np.zeros(idx.size)
            if self["lmo_velocity"] > 0:
                shifts += offsets[idx] / self["lmo_velocity"]
            if self["dip"] != 0:
                shifts += 1e-3 * self["dip"] * np.arange(idx.size)
            moved = bool(np.any(shifts != 0))
            work = flatten(g, shifts, data.dt) if moved else g
            nx = self["traces"] if self["traces"] > 0 else idx.size
            nw = int(round(self["window"] / data.dt)) if self["window"] > 0 else work.shape[1]
            filt = overlap_add(work, nx, nw, lambda b: eigenimage(b, self["rank"], self["mode"]))
            out[idx] = unflatten(filt, shifts, data.dt, data.nt) if moved else filt
            context.report((n + 1) / len(groups))
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- Harlan separation
def slant_operators(nx: int, nw: int, dt: float, p: np.ndarray, fmax: float, damping: float):
    """Local slant-stack operators for nx traces at unit spacing (p in s per trace), t = tau + p x: for every rfft bin
    up to fmax of an nw-sample window padded to twice its length, the forward operator L (nx, np) and the damped
    least-squares inverse (np, nx). Returns (nfft, band, L (nb, nx, np), inverse (nb, np, nx))."""
    nfft = int(2 ** np.ceil(np.log2(max(2 * nw, 2))))
    freqs = np.fft.rfftfreq(nfft, dt)
    band = np.flatnonzero(freqs <= fmax)
    x = np.arange(nx) - 0.5 * (nx - 1)
    fwd = np.exp(-2j * np.pi * freqs[band][:, None, None] * x[None, :, None] * p[None, None, :])   # (nb, nx, np)
    inv = np.zeros((band.size, p.size, nx), dtype=np.complex128)
    eye = np.eye(p.size)
    for j in range(band.size):
        lop = fwd[j]
        lhl = lop.conj().T @ lop
        mu = damping * float(np.real(np.trace(lhl))) / p.size + 1e-30
        inv[j] = np.linalg.solve(lhl + mu * eye, lop.conj().T)
    return nfft, band, fwd, inv


def harlan_block(block: np.ndarray, ops, threshold: float, shrinkage: str = "soft") -> np.ndarray:
    """Signal estimate of one window: local slant stack, shrinkage of the tau-p panel at threshold x the noise level
    (median absolute deviation of the panel within the window's time range), inverse slant stack."""
    nfft, band, fwd, inv = ops
    nx, nw = block.shape
    spec = np.fft.rfft(block, nfft, axis=1)                                  # (nx, nf)
    model = np.zeros((inv.shape[1], spec.shape[1]), dtype=np.complex128)     # (np, nf)
    model[:, band] = np.einsum("jpx,xj->pj", inv, spec[:, band])
    panel = np.fft.irfft(model, nfft, axis=1)                                # tau-p panel (np, nfft)
    sigma = float(np.median(np.abs(panel[:, :nw]))) / 0.6745
    thr = threshold * sigma
    if shrinkage == "hard":
        panel = np.where(np.abs(panel) > thr, panel, 0.0)
    else:
        panel = np.sign(panel) * np.maximum(np.abs(panel) - thr, 0.0)
    shrunk = np.fft.rfft(panel, nfft, axis=1)
    sig = np.zeros_like(spec)
    sig[:, band] = np.einsum("jxp,pj->xj", fwd, shrunk[:, band])
    return np.fft.irfft(sig, nfft, axis=1)[:, :nw]


@register
class HarlanSeparation(Process):
    name = "harlan"
    label = "Signal/noise separation (Harlan, local slant stacks)"
    category = "Filtering"
    description = ("Separates laterally coherent signal from random noise. In overlapping windows of traces, the "
                   "data are slant-stacked (least-squares local tau-p, dips up to the largest dip in ms per trace): "
                   "linear events focus into a few strong values, random noise spreads evenly. The noise level is "
                   "the median absolute deviation of the tau-p panel; values below threshold x noise level are "
                   "shrunk to zero (soft) or cut (hard) and the rest transformed back. Output the signal, or the "
                   "noise (data minus signal) for QC. The signal is limited to frequencies up to fmax.")
    params = (Param("output", "choice", "signal", "Output", choices=("signal", "noise")),
              Param("gather_key", "key", "", "Gather key (empty = whole section)"),
              Param("traces", "int", 16, "Trace window", minimum=4),
              Param("window", "float", 0.3, "Time window", unit="s", minimum=0.02),
              Param("max_dip", "float", 8.0, "Largest dip", unit="ms/trace", minimum=0.0),
              Param("threshold", "float", 2.0, "Threshold (x noise level)", minimum=0.0),
              Param("shrinkage", "choice", "soft", "Shrinkage", choices=("soft", "hard")),
              Param("fmax", "float", 100.0, "Highest frequency", unit="Hz", minimum=1.0),
              Param("damping", "float", 0.01, "Damping (fraction of mean diagonal)", minimum=0.0))

    def run(self, data, context):
        out = data.traces.astype(np.float64).copy()
        groups = _groups(data, self["gather_key"])
        fmax = min(self["fmax"], 0.5 / data.dt)
        nw = max(4, min(int(round(self["window"] / data.dt)), data.nt))
        pmax = 1e-3 * self["max_dip"]
        cache = {}
        for n, idx in enumerate(groups):
            nx = min(self["traces"], idx.size)
            if nx < 3:
                continue
            if nx not in cache:
                npv = max(11, int(np.ceil(2.4 * pmax * fmax * nx)) + 1) if pmax > 0 else 1
                p = np.linspace(-pmax, pmax, npv) if pmax > 0 else np.zeros(1)
                cache[nx] = slant_operators(nx, nw, data.dt, p, fmax, self["damping"])
            ops = cache[nx]
            g = out[idx]
            signal = overlap_add(g, nx, nw, lambda b: harlan_block(b, ops, self["threshold"], self["shrinkage"]))
            out[idx] = signal if self["output"] == "signal" else g - signal
            context.report((n + 1) / len(groups))
        return data.copy(out.astype(np.float32))
