"""Multichannel analysis of surface waves (MASW): the phase-shift dispersion image of Park, Miller & Xia (1998),
automatic picking of the fundamental-mode ridge, an approximate Vs profile by the wavelength rule, and a synthetic
dispersive surface-wave record for tests and demonstrations."""

from __future__ import annotations

import numpy as np


def dispersion_image(traces: np.ndarray, offsets: np.ndarray, dt: float, fmin: float = 2.0, fmax: float = 60.0,
                     vmin: float = 50.0, vmax: float = 1000.0, nv: int = 300):
    """Phase-shift dispersion image (Park, Miller & Xia 1998).

    The spectrum of every trace is normalised to unit amplitude (only its phase is kept); for every frequency f and
    trial phase velocity c the traces are phase-shifted by exp(+i 2 pi f x / c), x being the distance from the source,
    and summed. A wave travelling at c adds up coherently. Velocities below f x dx (dx: median trace spacing) are
    blanked: there the image only holds the spatial-aliasing (grating) lobes of faster waves. Returns the
    frequencies f (nf,), the velocities v (nv,) and the image (nv, nf): rows are velocities, columns frequencies,
    every column normalised to a maximum of 1."""
    tr = np.asarray(traces, dtype=np.float64)
    x = np.abs(np.asarray(offsets, dtype=np.float64)).reshape(-1)
    if tr.ndim != 2 or tr.shape[0] != x.size or tr.shape[0] < 2:
        raise ValueError("the dispersion image needs at least two traces with their offsets")
    if not (vmax > vmin > 0):
        raise ValueError("the velocity range must satisfy 0 < vmin < vmax")
    nt = tr.shape[1]
    nfft = int(2 ** np.ceil(np.log2(max(nt, int(round(2.0 / dt)), 2))))     # frequency step 0.5 Hz or finer
    freqs = np.fft.rfftfreq(nfft, dt)
    sel = np.flatnonzero((freqs >= fmin) & (freqs <= fmax) & (freqs > 0))
    if sel.size == 0:
        raise ValueError("no frequencies between fmin and fmax")
    spec = np.fft.rfft(tr, nfft, axis=1)[:, sel]
    mag = np.abs(spec)
    unit = spec / np.where(mag > 0, mag, 1.0)
    v = np.linspace(vmin, vmax, max(int(nv), 2))
    steps = np.diff(np.unique(np.round(x, 6)))
    dx = float(np.median(steps)) if steps.size else 0.0
    image = np.empty((v.size, sel.size))
    for j, f in enumerate(freqs[sel]):
        steer = np.exp(2j * np.pi * f * x[None, :] / v[:, None])             # (nv, ntr)
        image[:, j] = np.abs(steer @ unit[:, j])
        image[v < f * dx, j] = 0.0
    image /= np.maximum(image.max(axis=0, keepdims=True), 1e-30)
    return freqs[sel], v, image


def _peaks(column: np.ndarray, threshold: float) -> np.ndarray:
    """Indices of the local maxima of a column at or above threshold x its maximum."""
    c = np.concatenate([[-np.inf], column, [-np.inf]])
    is_peak = (c[1:-1] >= c[:-2]) & (c[1:-1] > c[2:])
    return np.flatnonzero(is_peak & (column >= threshold * column.max()))


def _refine(column: np.ndarray, i: int, v: np.ndarray) -> float:
    """Velocity of a peak refined by a parabola through the peak sample and its neighbours."""
    if 0 < i < column.size - 1:
        a, b, c = column[i - 1], column[i], column[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return float(v[i] + 0.5 * (a - c) / den * (v[1] - v[0]))
    return float(v[i])


def pick_curve(image: np.ndarray, f: np.ndarray, v: np.ndarray, threshold: float = 0.8, max_jump: float = 0.15,
               start: float | None = None):
    """Automatic fundamental-mode picks on a dispersion image (nv, nf).

    The ridge starts at the frequency ``start`` (Hz) or, by default, at the frequency where the image is most
    sharply peaked; there the lowest-velocity peak above threshold x the column maximum is taken (the fundamental mode
    is the slowest). The ridge is then followed to lower and to higher frequencies, every pick within max_jump
    (fraction) of the previous velocity; tracking stops where the ridge fades. Returns picked frequencies and
    velocities, sorted by frequency."""
    image = np.asarray(image, dtype=np.float64)
    nv, nf = image.shape
    if nf == 0 or nv < 2:
        return np.zeros(0), np.zeros(0)
    if start is None:
        colmax = image.max(axis=0)
        sharp = np.where(colmax > 0, image.mean(axis=0) / np.maximum(colmax, 1e-30), np.inf)
        j0 = int(np.argmin(sharp))
    else:
        j0 = int(np.argmin(np.abs(np.asarray(f) - start)))
    cand = _peaks(image[:, j0], threshold)
    if cand.size == 0:
        return np.zeros(0), np.zeros(0)
    picks = {j0: _refine(image[:, j0], int(cand[0]), v)}
    for direction in (1, -1):
        prev = picks[j0]
        j = j0 + direction
        while 0 <= j < nf:
            col = image[:, j]
            near = np.flatnonzero(np.abs(v - prev) <= max_jump * prev)
            if near.size == 0:
                break
            cand = [i for i in _peaks(col, 0.5 * threshold) if abs(v[i] - prev) <= max_jump * prev]
            if cand:
                i = min(cand, key=lambda k: abs(v[k] - prev))
            else:
                i = int(near[np.argmax(col[near])])
                if col[i] < 0.5 * threshold * col.max() or i in (near[0], near[-1]):
                    break
            prev = _refine(col, int(i), v)
            picks[j] = prev
            j += direction
    cols = np.array(sorted(picks))
    return np.asarray(f, dtype=np.float64)[cols], np.array([picks[c] for c in cols])


def approximate_vs(f: np.ndarray, vph: np.ndarray, depth_factor: float = 1.0 / 3.0, vs_factor: float = 1.09):
    """Rule-of-thumb Vs profile from a fundamental-mode dispersion curve: a wavelength lambda = vph / f senses the
    ground down to about depth_factor x lambda, where Vs is about vs_factor x vph (the Rayleigh-wave velocity is
    about 0.92 Vs). An approximation for a first look, not an inversion. Returns depth (increasing) and Vs."""
    f = np.asarray(f, dtype=np.float64)
    vph = np.asarray(vph, dtype=np.float64)
    ok = (f > 0) & (vph > 0)
    depth = depth_factor * vph[ok] / f[ok]
    vs = vs_factor * vph[ok]
    order = np.argsort(depth, kind="stable")
    return depth[order], vs[order]


def surface_wave_record(offsets: np.ndarray, nt: int, dt: float, curve_f, curve_v, f0: float = 15.0,
                        t_delay: float = 0.0, spreading: bool = True) -> np.ndarray:
    """A synthetic fundamental-mode surface-wave record (ntr, nt): every frequency travels at its phase velocity
    from the dispersion curve (linear interpolation, constant beyond its ends), with the amplitude spectrum of a
    Ricker wavelet of peak frequency f0 and, optionally, 1/sqrt(x) geometric spreading."""
    x = np.abs(np.asarray(offsets, dtype=np.float64)).reshape(-1)
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    f = np.fft.rfftfreq(nfft, dt)
    c = np.interp(f, np.asarray(curve_f, dtype=np.float64), np.asarray(curve_v, dtype=np.float64))
    amp = (f / f0) ** 2 * np.exp(1.0 - (f / f0) ** 2)
    spec = amp[None, :] * np.exp(-2j * np.pi * f[None, :] * (t_delay + x[:, None] / c[None, :]))
    if spreading:
        spec /= np.sqrt(np.maximum(x, 1.0))[:, None]
    return np.fft.irfft(spec, nfft, axis=1)[:, :nt]
