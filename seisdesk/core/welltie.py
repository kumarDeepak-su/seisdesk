"""Seismic-to-well tie: wavelet extraction at the well (statistical and deterministic), the constant-phase scan,
stretch and squeeze by dynamic time warping or manual tie points, the tied time-depth relation, a multi-trace tie
panel, and angle-dependent (AVO) reflection coefficients and synthetics from logs.

Units: time in seconds, velocity in m/s, density in kg/m3, angles in degrees."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from .dataset import SeismicData


def _nfft(n: int) -> int:
    return int(2 ** np.ceil(np.log2(max(int(n), 2))))


def _corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b)) / den if den > 0 else 0.0


def quadrature(x) -> np.ndarray:
    """Hilbert transform of a trace (zero-padded to limit wrap-around)."""
    x = np.asarray(x, dtype=np.float64)
    return np.imag(hilbert(x, N=_nfft(2 * x.size))[:x.size])


def rotate(w, angle_deg: float) -> np.ndarray:
    """Constant phase rotation Re{analytic(w) exp(i a)} = w cos a - H(w) sin a (the convention of the Wavelet steps)."""
    w = np.asarray(w, dtype=np.float64)
    a = np.deg2rad(angle_deg)
    return w * np.cos(a) - quadrature(w) * np.sin(a)


# ---------------------------------------------------------------------------------------------- wavelet extraction
def statistical_wavelet(traces, dt: float, length: float) -> tuple[np.ndarray, int]:
    """Zero-phase wavelet with the amplitude spectrum of the data (statistical wavelet estimation, White 1980): the
    power spectra of the Hann-tapered traces are averaged, smoothed by a triangular window on the autocorrelation lags
    (half the wavelet length), and their square root is transformed back with zero phase. Returns the wavelet (peak
    1, ``length`` seconds long, tapered) and the index of its time zero."""
    x = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    x = x[np.any(x != 0, axis=1)]
    if x.size == 0:
        raise ValueError("the wavelet window holds no signal")
    n = x.shape[1]
    half = max(2, int(round(0.5 * length / dt)))
    if 2 * half + 1 > n:
        raise ValueError(f"the window ({n} samples) is shorter than the wavelet ({2 * half + 1} samples)")
    nfft = _nfft(2 * n)
    seg = (x - x.mean(axis=1, keepdims=True)) * np.hanning(n)[None, :]
    power = np.mean(np.abs(np.fft.rfft(seg, nfft, axis=1)) ** 2, axis=0)
    acf = np.fft.irfft(power, nfft)
    lag = np.zeros(nfft)
    tri = 1.0 - np.arange(half + 1) / (half + 1.0)                 # Bartlett: keeps the spectrum non-negative
    lag[:half + 1] = tri
    lag[nfft - half:] = tri[1:][::-1]
    smooth = np.maximum(np.fft.rfft(acf * lag, nfft).real, 0.0)
    w = np.fft.irfft(np.sqrt(smooth), nfft)
    out = np.concatenate([w[nfft - half:], w[:half + 1]]) * np.hanning(2 * half + 3)[1:-1]
    peak = float(np.max(np.abs(out)))
    return (out / peak if peak > 0 else out), half


def constant_phase_scan(synthetic, seismic, step: float = 1.0) -> dict:
    """Best constant phase rotation of a synthetic (on the samples of the seismic): the correlation of
    s cos a - H(s) sin a with the seismic for a in [-180, 180). The Hilbert transform commutes with the convolution, so
    this equals rotating the wavelet by a. Returns the best angle, its correlation and the whole curve."""
    s = np.asarray(synthetic, dtype=np.float64)
    d = np.asarray(seismic, dtype=np.float64)
    hs = quadrature(s)
    angles = np.arange(-180.0, 180.0, step)
    a = np.deg2rad(angles)
    sd, hd, dd = float(s @ d), float(hs @ d), float(d @ d)
    ss, hh, sh = float(s @ s), float(hs @ hs), float(s @ hs)
    num = np.cos(a) * sd - np.sin(a) * hd
    norm = np.cos(a) ** 2 * ss + np.sin(a) ** 2 * hh - 2.0 * np.sin(a) * np.cos(a) * sh
    curve = num / np.sqrt(np.maximum(norm * dd, 1e-300))
    k = int(np.argmax(curve))
    return {"angle": float(angles[k]), "r": float(curve[k]), "angles": angles, "curve": curve}


def deterministic_wavelet(rc_times, rc, seis_times, seis, length: float, prewhitening: float = 0.1,
                          t0: float | None = None, t1: float | None = None) -> dict:
    """Least-squares (Wiener) wavelet that best turns the reflectivity into the seismic trace over [t0, t1] (Walden &
    White 1984): convolution matrix of the reflectivity with the time zero in the middle of the wavelet, normal
    equations damped by ``prewhitening`` percent of their mean diagonal. The reflectivity is resampled onto the seismic
    samples (give it already shifted by the bulk tie shift). Returns the wavelet, its centre index, the predicted
    trace on the seismic samples, the correlation of prediction and data in the window, and the window indices."""
    st = np.asarray(seis_times, dtype=np.float64)
    d_all = np.asarray(seis, dtype=np.float64)
    r = np.interp(st, np.asarray(rc_times, dtype=np.float64), np.asarray(rc, dtype=np.float64), left=0.0, right=0.0)
    dt = float(st[1] - st[0])
    rt = np.asarray(rc_times, dtype=np.float64)
    lo = rt[0] if t0 is None else t0
    hi = rt[-1] if t1 is None else t1
    idx = np.flatnonzero((st >= lo) & (st <= hi))
    half = max(1, int(round(0.5 * length / dt)))
    n = 2 * half + 1
    if idx.size < 2 * n:
        raise ValueError(f"the tie window holds {idx.size} samples: too short for a {n}-sample wavelet")
    mat = np.zeros((idx.size, n))
    for j in range(n):
        src = idx - (j - half)
        ok = (src >= 0) & (src < r.size)
        mat[ok, j] = r[src[ok]]
    d = d_all[idx]
    normal = mat.T @ mat
    eps = prewhitening / 100.0 * float(np.trace(normal)) / n + 1e-30
    w = np.linalg.solve(normal + eps * np.eye(n), mat.T @ d)
    prediction = np.convolve(r, w)[half: half + r.size]
    return {"wavelet": w, "centre": half, "prediction": prediction, "r": _corr(prediction[idx], d), "index": idx,
            "reflectivity": r}


def wavelet_spectrum(w, centre: int, dt: float) -> dict:
    """Amplitude spectrum (normalised to a peak of 1) and phase (degrees, relative to the wavelet's time zero, NaN where
    the amplitude is below 5 % of the peak) of a wavelet."""
    w = np.asarray(w, dtype=np.float64)
    nfft = _nfft(8 * w.size)
    buf = np.zeros(nfft)
    buf[:w.size] = w
    spec = np.fft.rfft(np.roll(buf, -int(centre)))
    f = np.fft.rfftfreq(nfft, dt)
    amp = np.abs(spec)
    peak = float(amp.max()) if amp.size else 0.0
    phase = np.rad2deg(np.angle(spec))
    phase[amp < 0.05 * peak] = np.nan
    return {"freq": f, "amplitude": amp / peak if peak > 0 else amp, "phase": phase}


# ---------------------------------------------------------------------------------------------- stretch and squeeze
def dynamic_warp(f, g, dt: float, max_shift: float, strain: float = 0.1, lag_step: float = 0.5,
                 smooth: bool = True) -> dict:
    """Dynamic time warping of trace f (the synthetic) onto trace g (the seismic) on the same samples (Hale 2013):
    shifts u(t), in steps of ``lag_step`` samples up to ``max_shift`` seconds, minimising sum (f(t) - g(t + u(t)))^2
    with |du/dt| <= strain. Both traces are scaled to unit RMS first; the optimal path is accumulated by dynamic
    programming (a lag may change by one step only every ceil(lag_step / strain) samples), traced back and smoothed.
    Returns the shifts (s) at the samples of f, f warped onto the samples of g, the correlation before and after, and
    the largest strain of the smoothed shifts."""
    f = np.asarray(f, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    n = f.size
    if f.shape != g.shape or dt <= 0 or max_shift < 0 or not 0 < strain < 1 or lag_step <= 0:
        raise ValueError("warping needs equal-length traces, positive sampling/lag step and strain between zero and one")
    if n < 4 or g.size < 4:
        raise ValueError("the traces are too short to warp")
    fr, gr = float(np.sqrt(np.mean(f * f))), float(np.sqrt(np.mean(g * g)))
    if fr == 0 or gr == 0:
        raise ValueError("a trace to warp is empty")
    fn, gn = f / fr, g / gr
    nl_half = int(np.floor(max_shift / (lag_step * dt) + 1e-9))
    lags = np.arange(-nl_half, nl_half + 1) * lag_step                     # in samples
    nl = lags.size
    pos = np.arange(n)[:, None] + lags[None, :]
    gi = np.interp(pos, np.arange(g.size), gn)
    e = (fn[:, None] - gi) ** 2
    b = max(1, int(np.ceil(lag_step / max(strain, 1e-6) - 1e-9)))
    ce = np.vstack([np.zeros((1, nl)), np.cumsum(e, axis=0)])              # ce[k] = sum of e[:k]
    acc = np.empty_like(e)
    acc[0] = e[0]
    for k in range(1, n):
        best = acc[k - 1].copy()
        if k - b >= 0 and nl > 1:
            part = ce[k] - ce[k - b + 1]                                   # e[k-b+1 .. k-1]
            step = np.full(nl, np.inf)
            step[1:] = acc[k - b, :-1] + part[:-1]                         # from lag l - 1
            step[:-1] = np.minimum(step[:-1], acc[k - b, 1:] + part[1:])   # from lag l + 1
            best = np.minimum(best, step)
        acc[k] = e[k] + best
    u = np.empty(n)
    lag = int(np.argmin(acc[n - 1]))
    k = n - 1
    while k >= 0:
        u[k] = lags[lag]
        if k == 0:
            break
        choice, value = 0, acc[k - 1, lag]
        if k - b >= 0:
            part = ce[k] - ce[k - b + 1]
            for move in (-1, 1):
                q = lag + move
                if 0 <= q < nl and acc[k - b, q] + part[q] < value:
                    choice, value = move, acc[k - b, q] + part[q]
        if choice == 0:
            k -= 1
        else:
            lag += choice
            u[k - b + 1:k] = lags[lag]
            k -= b
    if smooth and b > 1:
        u = uniform_filter1d(u, 2 * b + 1, mode="nearest")
    t = np.arange(n, dtype=np.float64)
    warped = warp_trace(f, t + u, n)
    return {"shifts": u * dt, "warped": warped, "r_before": _corr(f, g[:n]), "r_after": _corr(warped, g[:n]),
            "max_strain": float(np.max(np.abs(np.diff(u)))) if n > 1 else 0.0, "lag_block": b}


def warp_trace(f, new_positions, n_out: int) -> np.ndarray:
    """Trace f moved so that its sample k lands at position new_positions[k] (increasing), resampled onto samples
    0..n_out-1; zero where nothing lands."""
    f = np.asarray(f, dtype=np.float64)
    p = np.asarray(new_positions, dtype=np.float64)
    if np.any(np.diff(p) <= 0):
        raise ValueError("tie times must increase; the requested warp folds the time axis")
    grid = np.arange(n_out, dtype=np.float64)
    src = np.interp(grid, p, np.arange(f.size, dtype=np.float64), left=-1.0, right=-1.0)
    out = np.interp(src, np.arange(f.size, dtype=np.float64), f)
    out[(grid < p[0]) | (grid > p[-1])] = 0.0
    return out


def parse_tie_points(text: str) -> list[tuple[float, float]]:
    """Tie points 'synthetic time : seismic time' separated by commas or semicolons, in seconds or (values above 10)
    milliseconds, e.g. '1.100:1.112, 1.300:1.318'."""
    pts = []
    for item in text.replace(";", ",").split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"tie point {item.strip()!r}: write synthetic time : seismic time")
        a, b = (float(v) for v in item.split(":")[:2])
        if a > 10.0 or b > 10.0:
            a, b = a / 1000.0, b / 1000.0
        pts.append((a, b))
    if not pts:
        raise ValueError("no tie points given")
    return sorted(pts)


def tie_point_shifts(times, points) -> np.ndarray:
    """Shifts (s) at the given synthetic times from tie points [(synthetic time, seismic time), ...]: seismic minus
    synthetic time, linear between the points and constant beyond the end ones."""
    pts = sorted(points)
    ts = np.array([p[0] for p in pts], dtype=np.float64)
    tg = np.array([p[1] for p in pts], dtype=np.float64)
    if np.any(np.diff(ts) <= 0):
        raise ValueError("two tie points share the same synthetic time")
    if not np.all(np.isfinite(pts)) or np.any(np.diff(tg) <= 0):
        raise ValueError("tie points must be finite and both time axes must increase")
    return np.interp(np.asarray(times, dtype=np.float64), ts, tg - ts)


def strain_of(times, shifts) -> float:
    """Largest |d shift / dt| of a shift function."""
    t, s = np.asarray(times, dtype=np.float64), np.asarray(shifts, dtype=np.float64)
    if t.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(s) / np.diff(t))))


# ---------------------------------------------------------------------------------------------- time-depth
def tied_time_depth(depth, twt, shift: float = 0.0, warp_times=None, warp_shifts=None) -> np.ndarray:
    """Two-way times of the log depths after the tie: the bulk shift, then the stretch-and-squeeze shifts (given at
    times of the bulk-shifted synthetic) interpolated at every shifted time and held constant beyond their ends."""
    t = np.asarray(twt, dtype=np.float64) + float(shift)
    if warp_times is not None and np.size(warp_times) > 1:
        t = t + np.interp(t, np.asarray(warp_times, dtype=np.float64), np.asarray(warp_shifts, dtype=np.float64))
    return t


def velocity_change(depth, twt_before, twt_after, window: int = 11) -> float:
    """Largest relative change (%) of the interval velocity 2 dz / dt implied by a new time-depth relation, over
    intervals of ``window`` samples."""
    z = np.asarray(depth, dtype=np.float64)
    t0, t1 = np.asarray(twt_before, dtype=np.float64), np.asarray(twt_after, dtype=np.float64)
    k = max(1, min(int(window), z.size - 1))
    dz = z[k:] - z[:-k]
    d0, d1 = t0[k:] - t0[:-k], t1[k:] - t1[:-k]
    ok = (d0 > 0) & (d1 > 0) & (dz > 0)
    if not ok.any():
        return 0.0
    return float(100.0 * np.max(np.abs(d0[ok] / d1[ok] - 1.0)))


# ---------------------------------------------------------------------------------------------- tie panel
def tie_panel(seis: SeismicData, index: int, syn_times, synthetic, n_side: int = 10, n_repeat: int = 5,
              gap: int = 1) -> SeismicData:
    """A section for the visual tie: up to n_side traces of the seismic on each side of the well trace (the well trace
    on the left side), and between them the synthetic (on the seismic samples, scaled to the RMS of the seismic over the
    synthetic's span) repeated n_repeat times, framed by ``gap`` blank traces. Header tie_trace is 1 for synthetic
    traces, -1 for blanks and 0 for seismic; cdp numbers the panel's traces."""
    syn = np.interp(seis.times, np.asarray(syn_times, dtype=np.float64), np.asarray(synthetic, dtype=np.float64),
                    left=0.0, right=0.0)
    span = (seis.times >= float(np.min(syn_times))) & (seis.times <= float(np.max(syn_times)))
    i0, i1 = max(0, int(index) - n_side), min(seis.ntr, int(index) + n_side + 1)
    left = seis.traces[i0:int(index) + 1].astype(np.float64)
    right = seis.traces[int(index) + 1:i1].astype(np.float64)
    near = seis.traces[i0:i1].astype(np.float64)
    rms_seis = float(np.sqrt(np.mean(near[:, span] ** 2))) if span.any() else float(np.sqrt(np.mean(near ** 2)))
    rms_syn = float(np.sqrt(np.mean(syn[span] ** 2))) if span.any() else 0.0
    syn = syn * (rms_seis / rms_syn if rms_syn > 0 else 0.0)
    blank = np.zeros((gap, seis.nt))
    rows = [left, blank, np.tile(syn, (n_repeat, 1)), blank, right]
    flags = [np.zeros(left.shape[0]), -np.ones(gap), np.ones(n_repeat), -np.ones(gap), np.zeros(right.shape[0])]
    traces = np.vstack(rows)
    flag = np.concatenate(flags)
    heads = {"cdp": np.arange(1.0, traces.shape[0] + 1.0), "tie_trace": flag,
             "tracf": np.arange(1.0, traces.shape[0] + 1.0)}
    return SeismicData(traces.astype(np.float32), seis.dt, heads, seis.t0, name="well tie panel")


# ---------------------------------------------------------------------------------------------- AVO
def _avg_props(vp1, vs1, rho1, vp2, vs2, rho2):
    a = [np.asarray(v, dtype=np.float64) for v in (vp1, vs1, rho1, vp2, vs2, rho2)]
    return a


def zoeppritz_pp(vp1, vs1, rho1, vp2, vs2, rho2, theta_deg) -> np.ndarray:
    """Exact P-P reflection coefficient of a plane interface for a P wave incident at theta (Zoeppritz equations in the
    explicit form of Aki & Richards 1980, eq. 5.40). Complex beyond the critical angle: the real part is returned."""
    vp1, vs1, rho1, vp2, vs2, rho2 = _avg_props(vp1, vs1, rho1, vp2, vs2, rho2)
    th = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    p = np.sin(th) / vp1
    ci1 = np.cos(th) + 0j
    ci2 = np.sqrt(1.0 - (p * vp2) ** 2 + 0j)
    cj1 = np.sqrt(1.0 - (p * vs1) ** 2 + 0j)
    cj2 = np.sqrt(1.0 - (p * vs2) ** 2 + 0j)
    a = rho2 * (1.0 - 2.0 * vs2 ** 2 * p ** 2) - rho1 * (1.0 - 2.0 * vs1 ** 2 * p ** 2)
    b = rho2 * (1.0 - 2.0 * vs2 ** 2 * p ** 2) + 2.0 * rho1 * vs1 ** 2 * p ** 2
    c = rho1 * (1.0 - 2.0 * vs1 ** 2 * p ** 2) + 2.0 * rho2 * vs2 ** 2 * p ** 2
    d = 2.0 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)
    e = b * ci1 / vp1 + c * ci2 / vp2
    f = b * cj1 / vs1 + c * cj2 / vs2
    g = a - d * (ci1 / vp1) * (cj2 / vs2)
    h = a - d * (ci2 / vp2) * (cj1 / vs1)
    den = e * f + g * h * p ** 2
    rpp = ((b * ci1 / vp1 - c * ci2 / vp2) * f - (a + d * (ci1 / vp1) * (cj2 / vs2)) * h * p ** 2) / den
    return np.real(rpp)


def aki_richards(vp1, vs1, rho1, vp2, vs2, rho2, theta_deg) -> np.ndarray:
    """Linearised P-P reflection coefficient (Aki & Richards 1980): average properties of the two layers, the angle
    the mean of the incidence and transmission angles."""
    vp1, vs1, rho1, vp2, vs2, rho2 = _avg_props(vp1, vs1, rho1, vp2, vs2, rho2)
    th1 = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    p = np.sin(th1) / vp1
    th2 = np.arcsin(np.clip(p * vp2, -1.0, 1.0))
    th = 0.5 * (th1 + th2)
    vp, vs, rho = 0.5 * (vp1 + vp2), 0.5 * (vs1 + vs2), 0.5 * (rho1 + rho2)
    dvp, dvs, drho = vp2 - vp1, vs2 - vs1, rho2 - rho1
    k = (vs / vp) ** 2
    s2 = np.sin(th) ** 2
    return (0.5 * (1.0 - 4.0 * k * s2) * drho / rho + 0.5 / np.cos(th) ** 2 * dvp / vp
            - 4.0 * k * s2 * dvs / vs)


def intercept_gradient(vp1, vs1, rho1, vp2, vs2, rho2):
    """Shuey's (1985) two-term intercept A and gradient B: R(theta) = A + B sin^2(theta)."""
    vp1, vs1, rho1, vp2, vs2, rho2 = _avg_props(vp1, vs1, rho1, vp2, vs2, rho2)
    vp, vs, rho = 0.5 * (vp1 + vp2), 0.5 * (vs1 + vs2), 0.5 * (rho1 + rho2)
    dvp, dvs, drho = vp2 - vp1, vs2 - vs1, rho2 - rho1
    a = 0.5 * (dvp / vp + drho / rho)
    b = 0.5 * dvp / vp - 2.0 * (vs / vp) ** 2 * (drho / rho + 2.0 * dvs / vs)
    return a, b


def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta_deg) -> np.ndarray:
    a, b = intercept_gradient(vp1, vs1, rho1, vp2, vs2, rho2)
    return a + b * np.sin(np.deg2rad(np.asarray(theta_deg, dtype=np.float64))) ** 2


AVO_METHODS = {"Zoeppritz": zoeppritz_pp, "Aki-Richards": aki_richards, "Shuey": shuey}


def avo_class(a: float, b: float, threshold: float = 0.02) -> str:
    """Approximate AVO class of an intercept and gradient (Rutherford & Williams 1989, Castagna & Swan 1997):
    I for a clearly positive intercept, II (IIp with a polarity reversal before 30 degrees) for an intercept near zero,
    III for a negative intercept with a negative gradient, IV for a negative intercept with a positive gradient."""
    if a >= threshold:
        return "I"
    if a > -threshold:
        return "IIp" if a > 0 and a + b * np.sin(np.deg2rad(30.0)) ** 2 < 0 else "II"
    return "III" if b < 0 else "IV"


def _place(tm, coeff, dt: float, t0: float, nt: int) -> np.ndarray:
    """Coefficients at times tm shared linearly between the two nearest samples of t0 + k dt."""
    out = np.zeros(nt)
    pos = (np.asarray(tm, dtype=np.float64) - t0) / dt
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    for shift, weight in ((0, 1.0 - frac), (1, frac)):
        idx = i0 + shift
        ok = (idx >= 0) & (idx < nt) & np.isfinite(coeff)
        np.add.at(out, idx[ok], (coeff * weight)[ok])
    return out


def avo_synthetic(twt, vp, vs, rho, angles_deg, dt: float, wavelet, centre: int, method: str = "Aki-Richards",
                  t0: float | None = None, nt: int | None = None) -> dict:
    """Angle-gather synthetic of a well: at every angle the reflection coefficients between successive log samples
    (the chosen method, the angle taken as the incidence angle at each interface) placed at their two-way times and
    convolved with the wavelet. Returns times, angles and the gather (one trace per angle)."""
    if method not in AVO_METHODS:
        raise ValueError(f"unknown AVO method {method!r}")
    t = np.asarray(twt, dtype=np.float64)
    p, s, d = (np.asarray(v, dtype=np.float64) for v in (vp, vs, rho))
    tm = 0.5 * (t[1:] + t[:-1])
    if t0 is None:
        t0 = float(np.floor(t[0] / dt + 1e-9)) * dt
    if nt is None:
        nt = int(np.ceil((t[-1] - t0) / dt)) + 2
    w = np.asarray(wavelet, dtype=np.float64)
    angles = np.asarray(angles_deg, dtype=np.float64)
    gather = np.zeros((angles.size, nt))
    fn = AVO_METHODS[method]
    for k, ang in enumerate(angles):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = fn(p[:-1], s[:-1], d[:-1], p[1:], s[1:], d[1:], ang)
        r = np.where(np.isfinite(r), r, 0.0)
        series = _place(tm, r, dt, t0, nt)
        gather[k] = np.convolve(series, w)[int(centre): int(centre) + nt]
    return {"times": t0 + np.arange(nt) * dt, "angles": angles, "gather": gather}


def layer_contrast(depth, vp, vs, rho, z: float, window: float = 5.0):
    """Mean properties in ``window`` metres above and below depth z: (vp1, vs1, rho1, vp2, vs2, rho2)."""
    zz = np.asarray(depth, dtype=np.float64)
    up = (zz >= z - window) & (zz < z)
    down = (zz >= z) & (zz < z + window)
    if not up.any() or not down.any():
        raise ValueError(f"depth {z:g} m is not inside the logs (with {window:g} m above and below)")
    props = []
    for sel in (up, down):
        for v in (vp, vs, rho):
            props.append(float(np.nanmean(np.asarray(v, dtype=np.float64)[sel])))
    return tuple(props)
