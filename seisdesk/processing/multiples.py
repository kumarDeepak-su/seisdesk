"""Multiple attenuation: hyperbolic Radon demultiple, tau-p deconvolution, 2D SRME and adaptive subtraction."""

from __future__ import annotations

import os

import numpy as np
from scipy import fft as sfft
from scipy.sparse import csr_matrix

from ..core.dataset import SeismicData
from ..core.segy_io import read_any
from .base import Param, Process, register, resolve_velocity, taper_window
from .decon import apply_filters, prediction_error_filters
from .noise import _groups, linear_radon


# ---------------------------------------------------------------------------------------------- hyperbolic Radon
def hyperbolic_operator(offsets: np.ndarray, times: np.ndarray, t0: float, dt: float, nt: int,
                        slowness2: np.ndarray) -> csr_matrix:
    """Time-domain hyperbolic Radon operator (Thorson & Claerbout 1985) as a sparse matrix.

    Maps a model m(s2, tau), tau = the data times, s2 = 1 / v^2, to a gather d(x, t) with t = sqrt(tau^2 + x^2 s2),
    by linear interpolation in time, so that its transpose is the exact adjoint (the velocity stack).
    Shape (ntr * nt, nv * ntau); rows are trace-major, columns velocity-major."""
    x2 = np.asarray(offsets, dtype=np.float64) ** 2
    tau = np.asarray(times, dtype=np.float64)
    s2 = np.asarray(slowness2, dtype=np.float64)
    ntau = tau.size
    rows, cols, vals = [], [], []
    for i in range(x2.size):
        t = np.sqrt(tau[None, :] ** 2 + x2[i] * s2[:, None])                     # (nv, ntau)
        pos = (t - t0) / dt
        jv, kt = np.nonzero((pos >= 0) & (pos <= nt - 1))
        p = pos[jv, kt]
        i0 = np.floor(p).astype(np.int64)
        fr = p - i0
        col = jv * ntau + kt
        upper = i0 < nt - 1                         # a sample exactly at the last time has no upper neighbour
        rows += [i * nt + i0, i * nt + i0[upper] + 1]
        cols += [col, col[upper]]
        vals += [1.0 - fr, fr[upper]]
    shape = (x2.size * nt, s2.size * ntau)
    if not rows:
        return csr_matrix(shape, dtype=np.float32)
    return csr_matrix((np.concatenate(vals).astype(np.float32),
                       (np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32))), shape=shape)


def largest_eigenvalue(op, iterations: int = 5, seed: int = 0) -> float:
    """Largest eigenvalue of op^T op by power iteration (sets the damping scale)."""
    x = np.random.default_rng(seed).standard_normal(op.shape[1])
    lam = 0.0
    for _ in range(iterations):
        y = op.T @ (op @ x)
        norm = float(np.linalg.norm(y))
        if norm == 0.0:
            return 0.0
        lam = norm / max(float(np.linalg.norm(x)), 1e-30)
        x = y / norm
    return lam


def cgls(op, data: np.ndarray, iterations: int, damping: float, weights: np.ndarray | None = None) -> np.ndarray:
    """Conjugate-gradient least squares for min |d - L W z|^2 + damping |z|^2, returning m = W z (Paige & Saunders
    1982). Unit weights give damped least squares; weights from the previous model give the iteratively reweighted
    (sparse, high-resolution) solution."""
    w = np.ones(op.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64)
    z = np.zeros(op.shape[1])
    r = np.asarray(data, dtype=np.float64).copy()
    s = w * (op.T @ r)
    p = s.copy()
    gamma = float(s @ s)
    gamma0 = gamma
    for _ in range(int(iterations)):
        if gamma <= 1e-24 * max(gamma0, 1e-300):
            break
        q = op @ (w * p)
        delta = float(q @ q) + damping * float(p @ p)
        if delta <= 0.0:
            break
        alpha = gamma / delta
        z += alpha * p
        r -= alpha * q
        s = w * (op.T @ r) - damping * z
        gamma_new = float(s @ s)
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
    return w * z


def hyperbolic_radon(gather: np.ndarray, offsets: np.ndarray, times: np.ndarray, t0: float, dt: float,
                     slowness2: np.ndarray, iterations: int = 20, damping: float = 0.01, irls: int = 0,
                     eps: float = 0.01, op=None, lam: float | None = None):
    """Hyperbolic Radon model (nv, ntau) of one gather by CGLS; irls > 0 adds sparse reweighting passes (Trad,
    Ulrych & Sacchi 2003). Returns (model, operator, largest eigenvalue) so a caller can reuse the operator."""
    nt = gather.shape[1]
    if op is None:
        op = hyperbolic_operator(offsets, times, t0, dt, nt, slowness2)
    if lam is None:
        lam = largest_eigenvalue(op)
    d = np.asarray(gather, dtype=np.float64).ravel()
    mu = damping * lam
    m = cgls(op, d, iterations, mu)
    for _ in range(int(irls)):
        a = np.abs(m)
        m = cgls(op, d, iterations, mu, np.sqrt(a / max(float(a.max()), 1e-30) + eps))
    return m.reshape(np.asarray(slowness2).size, np.asarray(times).size), op, lam


@register
class HyperbolicRadon(Process):
    name = "hyperbolic_radon"
    label = "Hyperbolic Radon demultiple (velocity stack)"
    category = "Multiples"
    description = ("On CMP gathers WITHOUT NMO: every gather is decomposed into hyperbolas t = sqrt(tau^2 + x^2/v^2) "
                   "by a least-squares (conjugate-gradient) time-domain Radon transform; 'high resolution' adds "
                   "sparse reweighting passes that focus the velocity panel. Multiples travel slower than the "
                   "primaries at the same time, so the part of the panel below a cut velocity (a constant, a fraction "
                   "of the picked stacking velocity, or a velocity band) is modelled and subtracted. Output the "
                   "primaries, the multiples or the velocity panels.")
    params = (Param("vmin", "float", 1400.0, "Lowest velocity", unit="m/s", minimum=100.0),
              Param("vmax", "float", 4500.0, "Highest velocity", unit="m/s", minimum=100.0),
              Param("nv", "int", 60, "Number of velocities", minimum=8, maximum=400),
              Param("cut", "choice", "constant velocity", "Multiples are",
                    choices=("constant velocity", "fraction of the picked velocity", "velocity band")),
              Param("v_cut", "float", 1900.0, "Slower than (constant velocity)", unit="m/s", minimum=100.0),
              Param("fraction", "float", 0.92, "Slower than this fraction of the picked velocity", minimum=0.1,
                    maximum=1.0),
              Param("velocity", "velocity", "project", "Picked velocity (fraction mode)",
                    "project picks, a constant, 't:v,...' or a file"),
              Param("band_low", "float", 1450.0, "Band from (velocity band)", unit="m/s", minimum=100.0),
              Param("band_high", "float", 1700.0, "Band to (velocity band)", unit="m/s", minimum=100.0),
              Param("iterations", "int", 20, "Conjugate-gradient iterations", minimum=1, maximum=500),
              Param("damping", "float", 0.01, "Damping (fraction of the largest eigenvalue)", minimum=0.0),
              Param("high_resolution", "bool", False, "High resolution (sparse reweighting)"),
              Param("irls", "int", 3, "Sparse passes (high resolution)", minimum=1, maximum=20),
              Param("output", "choice", "primaries", "Output", choices=("primaries", "multiples", "velocity panel")))

    def run(self, data, context):
        if self["vmin"] >= self["vmax"]:
            raise ValueError("the lowest velocity must be below the highest")
        if not data.has_header("cdp"):
            raise ValueError("CMP numbers (cdp) are not set; run a geometry step and sort to CMP gathers")
        s2 = np.linspace(1.0 / self["vmax"] ** 2, 1.0 / self["vmin"] ** 2, self["nv"])  # even steps of moveout
        v = 1.0 / np.sqrt(s2)
        times = data.times
        mode = self["cut"]
        picks = resolve_velocity(self["velocity"], context) if mode == "fraction of the picked velocity" else None
        if mode == "velocity band" and self["band_low"] >= self["band_high"]:
            raise ValueError("the velocity band needs 'from' below 'to'")
        irls = self["irls"] if self["high_resolution"] else 0
        out = data.traces.astype(np.float64).copy()
        panels, heads = [], {"cdp": [], "radon_v": [], "tracf": []}
        cache: dict = {}
        misfit = []
        gathers = list(data.iter_gathers("cdp"))
        for n, (cdp, idx) in enumerate(gathers):
            g = data.traces[idx].astype(np.float64)
            off = data.header("offset")[idx]
            if idx.size < 4 and self["output"] != "velocity panel":
                continue
            key = tuple(np.round(np.abs(off), 1).tolist())
            if key not in cache:
                if len(cache) >= 6:
                    cache.pop(next(iter(cache)))
                op = hyperbolic_operator(off, times, data.t0, data.dt, data.nt, s2)
                cache[key] = (op, largest_eigenvalue(op))
            op, lam = cache[key]
            model, _, _ = hyperbolic_radon(g, off, times, data.t0, data.dt, s2, self["iterations"], self["damping"],
                                           irls, op=op, lam=lam)
            if self["output"] == "velocity panel":
                panels.append(model)
                heads["cdp"] += [cdp] * v.size
                heads["radon_v"] += v.tolist()
                heads["tracf"] += list(range(1, v.size + 1))
            else:
                if mode == "constant velocity":
                    mask = np.broadcast_to((v < self["v_cut"])[:, None], model.shape)
                elif mode == "velocity band":
                    mask = np.broadcast_to(((v >= self["band_low"]) & (v <= self["band_high"]))[:, None],
                                           model.shape)
                else:
                    vpick = picks.function(cdp, times)
                    mask = v[:, None] < self["fraction"] * vpick[None, :]
                mult = (op @ np.where(mask, model, 0.0).ravel()).reshape(idx.size, data.nt)
                fit = (op @ model.ravel()).reshape(idx.size, data.nt)
                energy = float(np.sum(g * g))
                if energy > 0:
                    misfit.append(float(np.sum((g - fit) ** 2)) / energy)
                out[idx] = (g - mult) if self["output"] == "primaries" else mult
            context.report((n + 1) / len(gathers))
        if misfit:
            context.log(f"    hyperbolic Radon: relative misfit of the data fit {100 * np.median(misfit):.1f} % "
                        "(median over the gathers)")
        context.results[self.name] = {"velocities": v.tolist(), "misfit": misfit}
        if self["output"] == "velocity panel":
            if not panels:
                raise ValueError("no CMP gather")
            out_d = SeismicData(np.concatenate(panels).astype(np.float32), data.dt,
                                {k: np.asarray(val, dtype=np.float64) for k, val in heads.items()}, data.t0,
                                data.name + " velocity panels", "time", list(data.history), dict(data.meta))
            out_d.meta["radon_velocities"] = v.tolist()
            return out_d
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- tau-p deconvolution
def multiple_period(trace: np.ndarray, dt: float, min_lag: float = 0.05, threshold: float = 0.15) -> float | None:
    """Period of a reverberation: the lag of the largest |autocorrelation| beyond min_lag (None if it is weak)."""
    x = np.asarray(trace, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    nfft = int(2 ** np.ceil(np.log2(2 * max(n, 2))))
    acf = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2, nfft)[:n]
    if acf[0] <= 0:
        return None
    acf = acf / acf[0]
    i0 = max(1, int(round(min_lag / dt)))
    if i0 >= n // 2:
        return None
    seg = np.abs(acf[i0:n // 2])
    k = int(np.argmax(seg))
    return None if seg[k] < threshold else (i0 + k) * dt


@register
class TauPDecon(Process):
    name = "taup_decon"
    label = "Tau-p deconvolution"
    category = "Multiples"
    description = ("Water-layer reverberations and short-period multiples are periodic in time only at vertical "
                   "incidence; in the tau-p domain they stay periodic at every slowness p, with the period shrinking "
                   "as cos(theta) = sqrt(1 - p^2 v_water^2). Each shot or CMP gather is slant-stacked (least-squares "
                   "linear Radon), every p trace is deconvolved with a predictive filter whose gap follows that "
                   "period, and the removed part is transformed back and subtracted from the data (Taner 1980; "
                   "Alam & Austin 1981). Give the period at p = 0 (the water-bottom two-way time) or let it be found "
                   "from the autocorrelation.")
    params = (Param("gather_key", "key", "fldr", "Gather key (shots or cdp)"),
              Param("period", "float", 0.0, "Period at p = 0 (0 = from the autocorrelation)", unit="s", minimum=0.0),
              Param("water_velocity", "float", 1500.0, "Water velocity", unit="m/s", minimum=100.0),
              Param("operator", "float", 0.12, "Operator length", unit="s", minimum=0.004),
              Param("lead", "float", 0.02, "Start the operator this much before the period", unit="s",
                    minimum=0.0),
              Param("prewhitening", "float", 0.5, "Pre-whitening", unit="%", minimum=0.0),
              Param("v_min", "float", 1400.0, "Slowest apparent velocity (p range)", unit="m/s", minimum=10.0),
              Param("np", "int", 121, "Number of p values", minimum=11),
              Param("damping", "float", 0.05, "Damping (fraction of mean diagonal)", minimum=0.0),
              Param("fmin", "float", 3.0, "Lowest frequency", unit="Hz", minimum=0.0),
              Param("fmax", "float", 80.0, "Highest frequency", unit="Hz", minimum=1.0),
              Param("output", "choice", "deconvolved", "Output", choices=("deconvolved", "removed multiples")))

    def run(self, data, context):
        dt, nt = data.dt, data.nt
        p = np.linspace(-1.0 / self["v_min"], 1.0 / self["v_min"], self["np"])
        arg = 1.0 - (p * self["water_velocity"]) ** 2
        cosine = np.sqrt(np.clip(arg, 0.0, None))
        valid = arg > 0.01                        # p traces beyond the critical slowness are left untouched
        j0 = int(np.argmin(np.abs(p)))
        out = data.traces.astype(np.float64).copy()
        groups = _groups(data, self["gather_key"])
        periods = []
        for n, idx in enumerate(groups):
            if idx.size < 4:
                continue
            g = data.traces[idx].astype(np.float64)
            model, op, nfft, band = linear_radon(g, data.header("offset")[idx], dt, p, self["damping"], self["fmin"],
                                                 self["fmax"])
            mt = np.fft.irfft(model, nfft, axis=1)[:, :nt]
            period = self["period"] if self["period"] > 0 else multiple_period(mt[j0], dt)
            if period is None:
                context.log(f"    gather {n + 1}: no clear reverberation period in the autocorrelation; skipped "
                            "(give the period)")
                continue
            periods.append(period)
            dec = mt.copy()
            for j in np.flatnonzero(valid):
                gap = max(dt, period * cosine[j] - self["lead"])
                try:
                    filt = prediction_error_filters(mt[j:j + 1], dt, self["operator"], gap, self["prewhitening"],
                                                    0.0, 0.0, 0.0)
                except ValueError:                # trace too short for this gap and operator
                    continue
                dec[j] = apply_filters(mt[j:j + 1], filt)[0]
            dm = np.fft.rfft(mt - dec, nfft, axis=1)
            removed_f = np.zeros((idx.size, dm.shape[1]), dtype=np.complex128)
            for j in band:
                removed_f[:, j] = op(j) @ dm[:, j]
            removed = np.fft.irfft(removed_f, nfft, axis=1)[:, :nt]
            out[idx] = (g - removed) if self["output"] == "deconvolved" else removed
            context.report((n + 1) / len(groups))
        if periods:
            context.log(f"    tau-p deconvolution: period at p = 0 {1000 * np.median(periods):.0f} ms "
                        f"({len(periods)} gathers)")
        context.results[self.name] = {"periods": periods}
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- adaptive subtraction
def _starts(n: int, w: int) -> list[int]:
    step = max(1, w // 2)
    starts = list(range(0, max(n - w, 0) + 1, step))
    if starts[-1] + w < n:
        starts.append(n - w)
    return starts


def adaptive_subtract(d: np.ndarray, m: np.ndarray, dt: float, filter_length: float = 0.06, window_time: float = 0.5,
                      window_traces: int = 20, norm: str = "L2", iterations: int = 5,
                      damping: float = 1e-3) -> np.ndarray:
    """Model matched to the data: one two-sided convolution filter per overlapping time and trace window, by damped
    least squares (L2; Verschuur et al. 1992) or iteratively reweighted least squares (L1, which leaves strong
    primaries out of the fit; Guitton & Verschuur 2004). Windows are blended with Hanning weights. Subtract the
    result from the data to remove the multiples."""
    d = np.asarray(d, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    ntr, nt = d.shape
    half = max(1, int(round(0.5 * filter_length / dt)))
    lags = np.arange(-half, half + 1)
    nw = int(min(nt, max(4 * half + 2, int(round(window_time / dt)))))
    nx = int(min(ntr, max(1, int(window_traces))))
    wt = np.hanning(nw) + 0.02 if nw < nt else np.ones(nw)
    wx = np.hanning(nx) + 0.02 if nx < ntr else np.ones(nx)
    weight = wx[:, None] * wt[None, :]
    mp = np.pad(m, ((0, 0), (half, half)))
    acc = np.zeros((ntr, nt))
    wsum = np.zeros((ntr, nt))
    for x0 in _starts(ntr, nx):
        for t0 in _starts(nt, nw):
            dw = d[x0:x0 + nx, t0:t0 + nw].ravel()
            # column for lag l: the model delayed by l samples, m(t - l)
            a = np.stack([mp[x0:x0 + nx, t0 + half - lag:t0 + half - lag + nw].ravel() for lag in lags], axis=1)
            ata = a.T @ a
            tr = float(np.trace(ata))
            if tr > 0:
                eps = damping * tr / lags.size
                f = np.linalg.solve(ata + eps * np.eye(lags.size), a.T @ dw)
                if norm == "L1":
                    floor = 1e-3 * max(float(np.max(np.abs(dw))), 1e-30)
                    for _ in range(int(iterations)):
                        wgt = 1.0 / np.maximum(np.abs(dw - a @ f), floor)
                        aw = a * wgt[:, None]
                        lhs = a.T @ aw
                        f = np.linalg.solve(lhs + damping * float(np.trace(lhs)) / lags.size * np.eye(lags.size),
                                            aw.T @ dw)
                matched = (a @ f).reshape(nx, nw)
                acc[x0:x0 + nx, t0:t0 + nw] += matched * weight
            wsum[x0:x0 + nx, t0:t0 + nw] += weight
    return acc / np.maximum(wsum, 1e-30)


def _model_on(data: SeismicData, model: SeismicData) -> np.ndarray:
    """The traces of a model dataset on the time axis of the data (interpolated when dt or t0 differ)."""
    if model.ntr != data.ntr:
        raise ValueError(f"the model has {model.ntr} traces, the data {data.ntr}: they must match trace by trace")
    if abs(model.dt - data.dt) < 1e-9 and abs(model.t0 - data.t0) < 1e-9 and model.nt >= data.nt:
        return model.traces[:, :data.nt].astype(np.float64)
    mt = model.times
    return np.stack([np.interp(data.times, mt, tr, left=0.0, right=0.0) for tr in model.traces.astype(np.float64)])


SUBTRACTION_PARAMS = (Param("filter_length", "float", 0.06, "Matching filter length", unit="s", minimum=0.004),
                      Param("window_time", "float", 0.5, "Window length", unit="s", minimum=0.05),
                      Param("window_traces", "int", 20, "Window width (traces)", minimum=1),
                      Param("norm", "choice", "L2", "Norm (L1 protects strong primaries)", choices=("L2", "L1")))


@register
class AdaptiveSubtraction(Process):
    name = "adaptive_subtraction"
    label = "Adaptive subtraction of a multiple model"
    category = "Multiples"
    description = ("Subtracts a multiple model (for example from SRME or another program, as a SEG-Y or SU file "
                   "with the same traces) after matching it to the data: in overlapping windows of time and traces a "
                   "short two-sided filter corrects the model's amplitude, wavelet and small time shifts by least "
                   "squares (L2) or by the more robust L1 norm, which keeps strong primaries out of the fit.")
    params = (Param("model_file", "infile", "", "Multiple model (SEG-Y or SU file)"),
              Param("gather_key", "key", "fldr", "Gather key (windows stay inside a gather)")) + SUBTRACTION_PARAMS + (
              Param("output", "choice", "data minus matched model", "Output",
                    choices=("data minus matched model", "matched model")),)

    def run(self, data, context):
        path = self["model_file"]
        if not path:
            raise ValueError("give the multiple model file (SEG-Y or SU)")
        if not os.path.isabs(path):
            path = os.path.join(context.work_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"multiple model {path}")
        m = _model_on(data, read_any(path))
        d = data.traces.astype(np.float64)
        out = d.copy()
        groups = _groups(data, self["gather_key"])
        for n, idx in enumerate(groups):
            matched = adaptive_subtract(d[idx], m[idx], data.dt, self["filter_length"], self["window_time"],
                                        self["window_traces"], self["norm"])
            out[idx] = (d[idx] - matched) if self["output"] == "data minus matched model" else matched
            context.report((n + 1) / len(groups))
        return data.copy(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------- SRME
def nmo_shift_trace(trace: np.ndarray, times: np.ndarray, x_from: float, x_to: float, v: float) -> np.ndarray:
    """A trace moved from offset x_from to offset x_to with hyperbolic moveout of velocity v (simple near-offset
    extrapolation): out(t) = in(sqrt(t^2 + (x_from^2 - x_to^2) / v^2))."""
    t_in = np.sqrt(np.clip(times ** 2 + (x_from ** 2 - x_to ** 2) / v ** 2, 0.0, None))
    return np.interp(t_in, times, trace, left=0.0, right=0.0)


def srme_predict(data: SeismicData, grid: float = 0.0, reciprocity: bool = True, max_shift: int = 4,
                 v_extrap: float = 0.0, near_traces: int = 0, fmin: float = 0.0, fmax: float = 0.0,
                 taper: float = 0.1, progress=None) -> tuple[np.ndarray, dict]:
    """2D surface-related multiple prediction M(s, r) = - sum_k D(s, k) * D(k, r) (time convolution; Verschuur,
    Berkhout & Wapenaar 1992) for every input trace.

    The traces are placed on a regular surface grid of sources and receivers; missing ones are filled by reciprocity
    D(s, r) = D(r, s), by copying the trace of a neighbouring shot at the same offset (up to max_shift grid steps;
    this also fills shot positions between shots), and for the near offsets that no receiver recorded by hyperbolic
    moveout from the nearest recorded offset (velocity v_extrap). Returns the model (ntr, nt) and fill statistics."""
    if data.domain != "time" or abs(data.t0) > 1e-9:
        raise ValueError("SRME requires time traces starting at zero; align the recording axis first")
    if np.ptp(np.r_[data.header("sy"), data.header("gy")]) > 1e-3:
        raise ValueError("this SRME operator is 2D; project the survey onto a straight line first")
    sx, gx = data.header("sx"), data.header("gx")
    if not (np.any(sx) or np.any(gx)):
        raise ValueError("SRME needs source and receiver positions (sx, gx): run a geometry step")
    nt, dt = data.nt, data.dt
    if grid <= 0:
        ux = np.unique(np.round(gx, 3))
        steps = np.diff(ux)
        steps = steps[steps > 1e-6]
        if steps.size == 0:
            raise ValueError("cannot find the receiver spacing; give the grid spacing")
        grid = float(np.median(steps))
    tol = 0.3 * grid
    x0 = float(min(sx.min(), gx.min()))
    n = int(round((float(max(sx.max(), gx.max())) - x0) / grid)) + 1
    need = n * n * nt * 4 * 3
    if need > 3e9:
        raise ValueError(f"SRME on a {n} x {n} surface grid needs about {need / 1e9:.1f} GB: split the line or use a "
                         "coarser grid")
    ks = np.rint((sx - x0) / grid).astype(np.int64)
    kr = np.rint((gx - x0) / grid).astype(np.int64)
    ok = (np.abs(sx - (x0 + ks * grid)) <= tol) & (np.abs(gx - (x0 + kr * grid)) <= tol)
    if not np.all(ok):
        raise ValueError("source/receiver positions do not lie on the SRME grid; regularise the geometry first")
    if np.unique(np.column_stack([ks, kr]), axis=0).shape[0] != data.ntr:
        raise ValueError("SRME needs one trace per source/receiver pair; select one component and merge repeats first")
    cube = np.zeros((n, n, nt), dtype=np.float32)
    have = np.zeros((n, n), dtype=bool)
    cube[ks[ok], kr[ok]] = data.traces[ok]
    have[ks[ok], kr[ok]] = True
    stats = {"grid": grid, "positions": n, "recorded": int(have.sum())}
    if reciprocity:
        miss = ~have & have.T
        if miss.any():
            ci, cj = np.nonzero(miss)
            cube[ci, cj] = cube[cj, ci]
            have[ci, cj] = True
    stats["after_reciprocity"] = int(have.sum())
    for step in range(1, int(max_shift) + 1):
        for sh in (step, -step):
            a = slice(max(0, -sh), n - max(0, sh))           # target shot and receiver range
            b = slice(max(0, sh), n - max(0, -sh))           # the same offsets, shot moved by sh
            fill = ~have[a, a] & have[b, b]
            if fill.any():
                target = cube[a, a]
                target[fill] = cube[b, b][fill]
                have[a, a] |= fill
    stats["after_translation"] = int(have.sum())
    if v_extrap > 0 and near_traces > 0:
        times = data.times
        for k in range(n):
            recorded = np.flatnonzero(have[k])
            if recorded.size == 0:
                continue
            for r in range(max(0, k - near_traces), min(n, k + near_traces + 1)):
                if have[k, r]:
                    continue
                side = recorded[np.sign(recorded - k) == np.sign(r - k)] if r != k else recorded
                if side.size == 0:
                    side = recorded
                src = int(side[np.argmin(np.abs(np.abs(side - k) - abs(r - k)))])
                cube[k, r] = nmo_shift_trace(cube[k, src].astype(np.float64), times, (src - k) * grid,
                                             (r - k) * grid, v_extrap)
                have[k, r] = True
    stats["after_extrapolation"] = int(have.sum())
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    freqs = np.fft.rfftfreq(nfft, dt)
    top = freqs[-1] if fmax <= 0 else fmax
    band = np.flatnonzero((freqs >= fmin) & (freqs <= top))
    spec = sfft.rfft(cube, nfft, axis=2)
    del cube
    w = (taper_window(n, taper) if taper > 0 else np.ones(n)).astype(np.float32)
    keep = np.zeros(freqs.size, dtype=bool)
    keep[band] = True
    spec[:, :, ~keep] = 0.0
    for c, j in enumerate(band):
        dj = spec[:, :, j].copy()
        spec[:, :, j] = -(dj * w[None, :]) @ dj * np.float32(grid)
        if progress is not None and c % 16 == 0:
            progress(0.9 * c / max(band.size, 1))
    model_cube = sfft.irfft(spec, nfft, axis=2)[:, :, :nt]
    model = np.zeros((data.ntr, nt))
    model[ok] = model_cube[ks[ok], kr[ok]]
    return model, stats


@register
class SRME(Process):
    name = "srme"
    label = "SRME (surface-related multiple prediction)"
    category = "Multiples"
    description = ("Data-driven prediction of all surface-related multiples (Verschuur, Berkhout & Wapenaar 1992): "
                   "every multiple is a primary convolved with the data at the free surface, M = -D * D summed over "
                   "the surface positions, so no velocity or water depth is needed. The shot gathers are placed on a "
                   "regular grid of co-located sources and receivers; gaps are filled by reciprocity, by neighbouring "
                   "shots at the same offset and, for the missing near offsets, by moveout from the nearest recorded "
                   "offset. The predicted model (wavelet squared, amplitudes approximate) is matched to the data by "
                   "adaptive subtraction, or output for inspection.")
    params = (Param("grid", "float", 0.0, "Surface grid spacing (0 = receiver spacing)", unit="m", minimum=0.0),
              Param("reciprocity", "bool", True, "Fill missing traces by reciprocity"),
              Param("max_shift", "int", 4, "Fill from neighbouring shots up to (grid steps)", minimum=0, maximum=50),
              Param("v_extrap", "float", 1500.0, "Near-offset extrapolation velocity (0 = off)", unit="m/s",
                    minimum=0.0),
              Param("near_traces", "int", 8, "Near-offset grid steps to extrapolate", minimum=0, maximum=200),
              Param("fmin", "float", 3.0, "Lowest frequency", unit="Hz", minimum=0.0),
              Param("fmax", "float", 80.0, "Highest frequency", unit="Hz", minimum=1.0),
              Param("taper", "float", 0.1, "Aperture taper at each line end (fraction)", minimum=0.0, maximum=0.5),
              Param("shot_key", "key", "fldr", "Shot header (adaptive subtraction per shot)"),
              Param("output", "choice", "data minus multiples", "Output",
                    choices=("data minus multiples", "multiple model"))) + SUBTRACTION_PARAMS

    def run(self, data, context):
        model, stats = srme_predict(data, self["grid"], self["reciprocity"], self["max_shift"], self["v_extrap"],
                                    self["near_traces"], self["fmin"], self["fmax"], self["taper"], context.report)
        d = data.traces.astype(np.float64)
        mm = float(np.sum(model * model))
        scale = float(np.sum(d * model)) / mm if mm > 0 else 0.0
        stats["scale"] = scale
        context.results[self.name] = stats
        context.log(f"    SRME: {stats['positions']} surface positions every {stats['grid']:g} m; traces on the grid "
                    f"{stats['recorded']} recorded, {stats['after_extrapolation']} after filling")
        if self["output"] == "multiple model":
            # Prediction must remain visible even when its events do not overlap the input primaries.
            return data.copy(model.astype(np.float32))
        out = d.copy()
        groups = _groups(data, self["shot_key"] if data.has_header(self["shot_key"]) else "")
        for n, idx in enumerate(groups):
            matched = adaptive_subtract(d[idx], model[idx], data.dt, self["filter_length"], self["window_time"],
                                        self["window_traces"], self["norm"])
            out[idx] = d[idx] - matched
            context.report(0.9 + 0.1 * (n + 1) / len(groups))
        return data.copy(out.astype(np.float32))
