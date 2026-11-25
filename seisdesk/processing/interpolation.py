"""Trace interpolation: f-x interpolation of regularly sampled gathers (Spitz 1991), which also handles spatially
aliased dips, and reconstruction of missing or dead traces by projection onto convex sets (POCS) with a decreasing
threshold in the f-k domain (Abma & Kabir 2006)."""

from __future__ import annotations

import numpy as np

from .base import Param, Process, register

# headers that vary smoothly along a gather and are interpolated for new traces; all others are copied
POSITION_KEYS = ("offset", "sx", "sy", "gx", "gy", "cdpx", "cdpy", "selev", "gelev", "sdepth", "sstat", "gstat",
                 "cdp")


def _nfft(n: int) -> int:
    return int(2 ** np.ceil(np.log2(max(2, n))))


def _groups(data, key: str, order_key: str):
    """Trace index groups (one per gather, or the whole dataset), each ordered by order_key."""
    secondary = order_key or None
    if not key:
        idx = np.arange(data.ntr)
        if secondary:
            idx = idx[np.argsort(data.header(secondary), kind="stable")]
        return [idx]
    return [idx for _, idx in data.iter_gathers(key, secondary)]


# ---------------------------------------------------------------------------------------------- Spitz f-x
def prediction_filter(x: np.ndarray, length: int, damping: float) -> np.ndarray | None:
    """Coefficients a with x[n] ~ sum_k a[k] x[n-k] (k = 1..length), least squares over the forward and the
    (conjugate) backward predictions of a complex spatial sequence; None if the sequence is too short."""
    n = x.size
    if n < length + 2:
        return None
    fwd = np.stack([x[length - k:n - k] for k in range(1, length + 1)], axis=1)
    bwd = np.stack([np.conj(x[k:n - length + k]) for k in range(1, length + 1)], axis=1)
    mat = np.vstack([fwd, bwd])
    rhs = np.concatenate([x[length:], np.conj(x[:n - length])])
    r = mat.conj().T @ mat
    mu = damping * float(np.real(np.trace(r))) / length + 1e-30
    return np.linalg.solve(r + mu * np.eye(length), mat.conj().T @ rhs)


def fill_odd(known: np.ndarray, a: np.ndarray, damping: float) -> np.ndarray:
    """Values at the odd positions of the refined sequence u (u[0::2] = known) that minimise the forward and
    backward prediction errors of the filter a on the fine grid (Spitz 1991)."""
    n = known.size
    m = 2 * n - 1
    p = a.size
    if m <= p + 1:
        return 0.5 * (known[:-1] + known[1:])
    rows = m - p
    r = np.arange(rows)
    mat = np.zeros((2 * rows, m), dtype=np.complex128)
    mat[r, r + p] = 1.0
    mat[rows + r, r] = 1.0
    for k in range(1, p + 1):
        mat[r, r + p - k] = -a[k - 1]
        mat[rows + r, r + k] = -np.conj(a[k - 1])
    au = mat[:, 1::2]
    rhs = -(mat[:, 0::2] @ known)
    g = au.conj().T @ au
    mu = damping * float(np.real(np.trace(g))) / g.shape[0] + 1e-30
    return np.linalg.solve(g + mu * np.eye(g.shape[0]), au.conj().T @ rhs)


def spitz_interpolate(g: np.ndarray, dt: float, length: int = 4, damping: float = 0.01, fmin: float = 2.0,
                      fmax: float = 80.0, window: float = 0.5, ntraces: int = 24) -> np.ndarray:
    """Spitz (1991) f-x interpolation of a regularly sampled gather (ntr, nt): returns (2 ntr - 1, nt) with the input
    traces unchanged at the even rows and the new traces at the odd rows.

    In overlapping time and trace windows, the prediction filter of the coarse traces at frequency f/2 is also the
    prediction filter of the refined gather at frequency f (linear events: the phase step over the coarse spacing at
    f/2 equals the step over half the spacing at f); the new traces minimise its prediction error. Frequencies
    outside fmin-fmax are interpolated linearly."""
    g = np.asarray(g, dtype=np.float64)
    ntr, nt = g.shape
    if ntr < 2:
        return g.copy()
    m = 2 * ntr - 1
    lin = np.empty((m, nt))
    lin[0::2] = g
    lin[1::2] = 0.5 * (g[:-1] + g[1:])
    if ntr < length + 3:
        return lin
    nw = max(16, min(nt, int(round(window / dt))))
    nx = max(length + 3, min(ntr, ntraces))
    t_starts = list(range(0, max(nt - nw, 0) + 1, max(1, nw // 2)))
    if t_starts[-1] + nw < nt:
        t_starts.append(nt - nw)
    x_starts = list(range(0, max(ntr - nx, 0) + 1, max(1, nx // 2)))
    if x_starts[-1] + nx < ntr:
        x_starts.append(ntr - nx)
    mx = 2 * nx - 1
    wt = np.hanning(nw) + 0.02 if nw < nt else np.ones(nw)
    wx = np.hanning(mx) + 0.02 if nx < ntr else np.ones(mx)
    nfft = _nfft(2 * nw)
    f = np.fft.rfftfreq(nfft, dt)
    band = np.flatnonzero((f > 0) & (f >= fmin) & (f <= fmax))
    acc = np.zeros((m, nt))
    wsum = np.zeros((m, nt))
    for x0 in x_starts:
        for t0 in t_starts:
            block = g[x0:x0 + nx, t0:t0 + nw]
            spec = np.fft.rfft(block, nfft, axis=1)
            half = np.fft.rfft(block, 2 * nfft, axis=1)          # bin j of this transform is frequency f_j / 2
            fine = np.empty((mx, spec.shape[1]), dtype=np.complex128)
            fine[0::2] = spec
            fine[1::2] = 0.5 * (spec[:-1] + spec[1:])
            for j in band:
                a = prediction_filter(half[:, j], length, damping)
                if a is not None:
                    fine[1::2, j] = fill_odd(spec[:, j], a, damping)
            out = np.fft.irfft(fine, nfft, axis=1)[:, :nw]
            w = wx[:, None] * wt[None, :]
            acc[2 * x0:2 * x0 + mx, t0:t0 + nw] += out * w
            wsum[2 * x0:2 * x0 + mx, t0:t0 + nw] += w
    res = acc / np.maximum(wsum, 1e-30)
    res[0::2] = g
    return res


def refined_headers(headers: dict, idx: np.ndarray) -> dict:
    """Headers of a gather refined by two: the originals at even rows, new traces between neighbours at odd rows
    (position headers interpolated, the others copied from the left neighbour)."""
    out = {}
    for key, values in headers.items():
        v = values[idx]
        fine = np.empty(2 * idx.size - 1)
        fine[0::2] = v
        fine[1::2] = 0.5 * (v[:-1] + v[1:]) if key in POSITION_KEYS else v[:-1]
        out[key] = fine
    return out


@register
class FXInterpolation(Process):
    name = "fx_interpolation"
    label = "Trace interpolation (f-x, Spitz)"
    category = "Interpolation"
    description = ("Inserts a new trace midway between every pair of neighbouring traces of each gather (factor 2; "
                   "factor 4 repeats it), by f-x prediction filters (Spitz 1991). Unlike simple averaging this keeps "
                   "steep, spatially aliased events (ground roll, steep dips, refractions) continuous. The traces "
                   "must be regularly spaced along the order key. New traces get interpolated positions (offset, "
                   "source/receiver and CMP x), header interp = 1 and renumbered tracf; run the geometry step "
                   "again afterwards to recompute CMP numbers on the finer grid.")
    params = (Param("gather_key", "key", "fldr", "Gather key (empty = whole dataset)"),
              Param("order_key", "key", "offset", "Order traces by"),
              Param("factor", "choice", "2", "Interpolation factor", choices=("2", "4")),
              Param("filter_length", "int", 4, "Prediction filter length (traces)", minimum=1, maximum=20),
              Param("damping", "float", 1.0, "Damping", unit="%", minimum=0.0),
              Param("window", "float", 0.5, "Time window", unit="s", minimum=0.05),
              Param("traces", "int", 24, "Trace window", minimum=6),
              Param("fmin", "float", 2.0, "Lowest frequency", unit="Hz", minimum=0.0),
              Param("fmax", "float", 80.0, "Highest frequency", unit="Hz", minimum=1.0))

    def run(self, data, context):
        out = data
        for _ in range(2 if self["factor"] == "4" else 1):
            out = self._refine(out, context)
        return out

    def _refine(self, data, context):
        if "interp" not in data.headers:
            data = data.copy()
            data.set_header("interp", np.zeros(data.ntr))
        groups = _groups(data, self["gather_key"], self["order_key"])
        blocks, heads = [], []
        irregular = 0
        for n, idx in enumerate(groups):
            if idx.size < 2:
                blocks.append(data.traces[idx].astype(np.float64))
                heads.append({k: v[idx].copy() for k, v in data.headers.items()})
                continue
            if self["order_key"]:
                step = np.diff(data.header(self["order_key"])[idx])
                if np.ptp(step) > 0.1 * max(float(np.median(np.abs(step))), 1e-9):
                    irregular += 1
            blocks.append(spitz_interpolate(data.traces[idx], data.dt, self["filter_length"],
                                            self["damping"] / 100.0, self["fmin"], self["fmax"], self["window"],
                                            self["traces"]))
            h = refined_headers(data.headers, idx)
            h["interp"][1::2] = 1.0
            h["tracf"] = np.arange(1, 2 * idx.size, dtype=np.float64)
            heads.append(h)
            context.report((n + 1) / len(groups))
        if irregular:
            context.log(f"    warning: {irregular} gather(s) are not regularly spaced along {self['order_key']}; "
                        f"f-x interpolation assumes a regular spacing")
        keys = list(data.headers) + [k for k in ("tracf",) if k not in data.headers]
        headers = {k: np.concatenate([h[k] if k in h else np.zeros(b.shape[0]) for h, b in zip(heads, blocks)])
                   for k in keys}
        traces = np.concatenate(blocks, axis=0)
        headers["tracl"] = np.arange(1, traces.shape[0] + 1, dtype=np.float64)
        context.log(f"    {data.ntr} -> {traces.shape[0]} traces")
        return data.copy(traces.astype(np.float32), headers=headers)


# ---------------------------------------------------------------------------------------------- POCS
def pocs_reconstruct(d: np.ndarray, known: np.ndarray, niter: int = 50, final: float = 0.005) -> np.ndarray:
    """Abma & Kabir (2006): fill the rows of d (nx, nt) where known is False. Each iteration keeps the 2D Fourier
    (f-k) coefficients above a threshold that falls exponentially from the largest coefficient to final x largest,
    transforms back and re-inserts the recorded traces."""
    obs = np.asarray(d, dtype=np.float64)
    nx, nt = obs.shape
    if known.all() or not known.any():
        return obs.copy()
    nfft = _nfft(nt + nt // 2)
    obs = np.where(known[:, None], obs, 0.0)
    cur = obs.copy()
    amax = float(np.abs(np.fft.fft(np.fft.rfft(obs, nfft, axis=1), axis=0)).max())
    if amax <= 0:
        return cur
    for k in range(niter):
        tau = amax * final ** ((k + 1) / niter)
        spec = np.fft.fft(np.fft.rfft(cur, nfft, axis=1), axis=0)
        spec[np.abs(spec) < tau] = 0.0
        rec = np.fft.irfft(np.fft.ifft(spec, axis=0), nfft, axis=1)[:, :nt]
        cur = np.where(known[:, None], obs, rec)
    return cur


@register
class POCSFill(Process):
    name = "pocs_fill"
    label = "Fill missing or dead traces (POCS)"
    category = "Interpolation"
    description = ("Puts each gather on a regular grid of the position key (spacing: the median trace spacing, or "
                   "given), inserts empty traces where positions are missing — and, if asked, treats dead (all-zero) "
                   "traces as missing — and rebuilds them by POCS: iterative thresholding of the f-k spectrum with "
                   "re-insertion of the recorded traces (Abma & Kabir 2006). Recorded traces are not changed; filled "
                   "traces get header interp = 1. Works best for scattered gaps of a few traces.")
    params = (Param("gather_key", "key", "fldr", "Gather key (empty = whole dataset)"),
              Param("position_key", "key", "offset", "Position header of the regular grid"),
              Param("spacing", "float", 0.0, "Grid spacing (0 = median spacing)", minimum=0.0),
              Param("fill_dead", "bool", True, "Also rebuild dead (all-zero) traces"),
              Param("iterations", "int", 50, "Iterations", minimum=1, maximum=1000),
              Param("final_threshold", "float", 0.5, "Final threshold", unit="% of max", minimum=1e-4,
                    maximum=100.0))

    def run(self, data, context):
        pkey = self["position_key"]
        if not pkey:
            raise ValueError("give the position header of the regular grid")
        groups = _groups(data, self["gather_key"], pkey)
        blocks, heads = [], []
        filled_total = merged_total = 0
        base_interp = data.header("interp") if "interp" in data.headers else np.zeros(data.ntr)
        for n, idx in enumerate(groups):
            pos = data.header(pkey)[idx]
            upos = np.unique(pos)
            step = np.diff(upos)
            dx = self["spacing"] if self["spacing"] > 0 else (float(np.median(step)) if step.size else 0.0)
            gi = np.rint((pos - pos.min()) / dx).astype(np.int64) if dx > 0 else np.zeros(idx.size, np.int64)
            ngrid = int(gi.max()) + 1
            if dx <= 0 or ngrid > 4 * idx.size + 16:
                if dx > 0:
                    context.log(f"    gather {n + 1}: positions too irregular for a grid; left unchanged")
                blocks.append(data.traces[idx].astype(np.float64))
                h = {k: v[idx].copy() for k, v in data.headers.items()}
                h["interp"] = base_interp[idx].copy()
                heads.append(h)
                continue
            grid = np.zeros((ngrid, data.nt))
            count = np.zeros(ngrid)
            first = np.full(ngrid, -1, dtype=np.int64)
            for j, row in enumerate(gi):
                grid[row] += data.traces[idx[j]]
                count[row] += 1
                if first[row] < 0:
                    first[row] = idx[j]
            merged_total += int(np.sum(count[count > 1] - 1))
            have = count > 0
            grid[have] /= count[have, None]
            known = have.copy()
            if self["fill_dead"]:
                known &= np.any(grid != 0.0, axis=1)
            rec = pocs_reconstruct(grid, known, self["iterations"], self["final_threshold"] / 100.0)
            filled = ~known
            filled_total += int(filled.sum())
            rows = np.arange(ngrid)
            src = rows[have]
            nearest = src[np.clip(np.searchsorted(src, rows), 0, src.size - 1)]
            left = src[np.clip(np.searchsorted(src, rows) - 1, 0, src.size - 1)]
            nearest = np.where(np.abs(left - rows) < np.abs(nearest - rows), left, nearest)
            h = {}
            for key, values in data.headers.items():
                v = values[first[nearest]]
                if key in POSITION_KEYS or key == "tracf":
                    v = np.interp(rows, src, values[first[src]])
                    if key == "tracf":
                        v = np.rint(v)
                h[key] = v.astype(np.float64)
            h[pkey] = np.where(have, data.header(pkey)[np.maximum(first, 0)], pos.min() + rows * dx)
            interp = base_interp[first[nearest]].copy()
            interp[filled] = 1.0
            h["interp"] = interp
            if "trid" in h:
                h["trid"] = np.where(filled, 1.0, h["trid"])
            blocks.append(rec)
            heads.append(h)
            context.report((n + 1) / len(groups))
        keys = list(data.headers) + ["interp"]
        keys = list(dict.fromkeys(keys))
        headers = {k: np.concatenate([h[k] for h in heads]) for k in keys}
        traces = np.concatenate(blocks, axis=0)
        headers["tracl"] = np.arange(1, traces.shape[0] + 1, dtype=np.float64)
        msg = f"    {filled_total} trace(s) filled, {data.ntr} -> {traces.shape[0]} traces"
        if merged_total:
            msg += f"; {merged_total} trace(s) sharing a grid position were averaged"
        context.log(msg)
        return data.copy(traces.astype(np.float32), headers=headers)
