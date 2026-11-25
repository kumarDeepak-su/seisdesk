"""Statics: elevation statics, applying statics from headers, surface-consistent residual statics.

Sign convention: a positive static (ms) moves the data later (down) on the trace; corrections that remove travel time
above a datum are therefore negative.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .base import Param, Process, cdp_values, register


def shift_traces(traces: np.ndarray, dt: float, shifts_s: np.ndarray) -> np.ndarray:
    """Delay each trace by shifts_s[i] seconds (fractional shifts, frequency domain, zero padded)."""
    ntr, nt = traces.shape
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    w = 2 * np.pi * np.fft.rfftfreq(nfft, dt)
    spec = np.fft.rfft(traces, nfft, axis=1) * np.exp(-1j * w[None, :] * shifts_s[:, None])
    return np.fft.irfft(spec, nfft, axis=1)[:, :nt].astype(np.float32)


@register
class ElevationStatics(Process):
    name = "elevation_statics"
    label = "Elevation statics (to datum)"
    category = "Statics"
    description = ("Source and receiver statics that move the data to a flat datum with a replacement velocity; "
                   "written to sstat/gstat (ms) and applied unless 'apply' is off.")
    params = (Param("datum", "float", 0.0, "Datum elevation", unit="m"),
              Param("replacement_velocity", "float", 2000.0, "Replacement velocity", unit="m/s", minimum=1.0),
              Param("apply", "bool", True, "Apply now"))

    def run(self, data, context):
        vr = self["replacement_velocity"]
        s = -(data.header("selev") - data.header("sdepth") - self["datum"]) / vr * 1000.0
        g = -(data.header("gelev") - self["datum"]) / vr * 1000.0
        out = data.copy()
        out.set_header("sstat", s)
        out.set_header("gstat", g)
        if self["apply"]:
            out.traces = shift_traces(out.traces, data.dt, (s + g) / 1000.0)
            out.set_header("tstat", out.header("tstat") + s + g)
        return out


@register
class ApplyStatics(Process):
    name = "apply_statics"
    label = "Apply statics from headers"
    category = "Statics"
    description = "Shift each trace by the sum of the chosen header values (ms), e.g. 'sstat,gstat'."
    params = (Param("keys", "keys", "sstat,gstat", "Header keys (ms)"),
              Param("sign", "float", 1.0, "Multiply by (+1 / -1)"))

    def run(self, data, context):
        total = np.zeros(data.ntr)
        for k in self["keys"]:
            total += data.header(k)
        total *= self["sign"]
        out = data.copy(shift_traces(data.traces, data.dt, total / 1000.0))
        out.set_header("tstat", data.header("tstat") + total)
        return out


@register
class TableStatics(Process):
    name = "table_statics"
    label = "Apply statics from an x-static table"
    category = "Statics"
    description = ("Interpolate a two-column table 'x static_ms' (e.g. refraction statics from tomography) at every "
                   "source and receiver position and shift each trace by the sum. 'remove' shifts the trace EARLIER "
                   "by the table values (the usual case for delay times); 'add' shifts it later.")
    params = (Param("file", "infile", "statics.txt", "Statics table (x, ms)"),
              Param("mode", "choice", "remove", "Mode", choices=("remove", "add")))

    def run(self, data, context):
        import os
        from ..core.geometry import read_xy
        path = self["file"] if os.path.isabs(self["file"]) else os.path.join(context.work_dir, self["file"])
        x, st = read_xy(path)
        sign = -1.0 if self["mode"] == "remove" else 1.0
        s = sign * np.interp(data.header("sx"), x, st)
        g = sign * np.interp(data.header("gx"), x, st)
        out = data.copy(shift_traces(data.traces, data.dt, (s + g) / 1000.0))
        out.set_header("sstat", data.header("sstat") + s)
        out.set_header("gstat", data.header("gstat") + g)
        out.set_header("tstat", data.header("tstat") + s + g)
        return out


def _xcorr_lag(a: np.ndarray, b: np.ndarray, maxlag: int) -> float:
    """Lag (samples, fractional) maximising the cross-correlation of a against b; positive = a is late."""
    n = a.size
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    lags = np.concatenate([np.arange(0, maxlag + 1), np.arange(-maxlag, 0)])
    vals = np.concatenate([c[: maxlag + 1], c[-maxlag:]])
    k = int(np.argmax(vals))
    best = lags[k]
    if abs(best) < maxlag:
        i = best % nfft
        y0, y1, y2 = c[(i - 1) % nfft], c[i], c[(i + 1) % nfft]
        den = y0 - 2 * y1 + y2
        if den < 0:
            best = best + 0.5 * (y0 - y2) / den
    return float(best)


@register
class ResidualStatics(Process):
    name = "residual_statics"
    label = "Residual statics (surface consistent)"
    category = "Statics"
    description = ("On NMO-corrected data: cross-correlate every trace with its CMP pilot stack in a window, split the "
                   "time shifts into source and receiver terms by least squares, apply, iterate. Totals go to "
                   "sstat/gstat (ms).")
    params = (Param("source_key", "key", "sx", "Source position key"),
              Param("receiver_key", "key", "gx", "Receiver position key"),
              Param("tmin", "float", 0.3, "Window start", unit="s"),
              Param("tmax", "float", 2.5, "Window end", unit="s"),
              Param("max_shift", "float", 0.024, "Maximum shift", unit="s", minimum=0.001),
              Param("iterations", "int", 3, "Iterations", minimum=1, maximum=20))

    def run(self, data, context):
        cdp_values(data)                                  # CMP numbers must be set (pilot traces are CMP stacks)
        src = np.round(data.header(self["source_key"]), 3)
        rec = np.round(data.header(self["receiver_key"]), 3)
        su, si = np.unique(src, return_inverse=True)
        ru, ri = np.unique(rec, return_inverse=True)
        t = data.times
        win = (t >= self["tmin"]) & (t <= self["tmax"])
        maxlag = max(1, int(round(self["max_shift"] / data.dt)))
        total_s = np.zeros(su.size)
        total_r = np.zeros(ru.size)
        current = data.traces.copy()
        groups = [idx for _, idx in data.iter_gathers("cdp", secondary=None)]
        for it in range(self["iterations"]):
            lag = np.zeros(data.ntr)
            for idx in groups:
                if idx.size < 2:
                    continue
                pilot = current[idx][:, win].mean(axis=0)
                for i in idx:
                    lag[i] = _xcorr_lag(current[i, win], pilot, maxlag) * data.dt
            n = data.ntr
            rows = np.concatenate([np.arange(n), np.arange(n)])
            cols = np.concatenate([si, su.size + ri])
            A = coo_matrix((np.ones(2 * n), (rows, cols)), shape=(n, su.size + ru.size)).tocsr()
            sol = lsqr(A, lag, damp=1e-3)[0]
            s_corr, r_corr = sol[: su.size], sol[su.size:]
            s_corr -= s_corr.mean()
            r_corr -= r_corr.mean()
            total_s -= s_corr
            total_r -= r_corr
            current = shift_traces(data.traces, data.dt, total_s[si] + total_r[ri])
            context.log(f"    residual statics iteration {it + 1}: rms update {np.sqrt(np.mean(sol ** 2)) * 1000:.2f} ms")
            context.report((it + 1) / self["iterations"])
        out = data.copy(current)
        out.set_header("sstat", data.header("sstat") + total_s[si] * 1000.0)
        out.set_header("gstat", data.header("gstat") + total_r[ri] * 1000.0)
        context.results["residual_statics"] = {"source": dict(zip(su.tolist(), (total_s * 1000).tolist())),
                                               "receiver": dict(zip(ru.tolist(), (total_r * 1000).tolist()))}
        return out
