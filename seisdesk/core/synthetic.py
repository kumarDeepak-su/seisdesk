"""Synthetic shot records for testing, demonstration and training.

``marine_line`` builds a towed-streamer 2D line over flat reflectors with hyperbolic moveout (optionally a dipping
reflector, a point diffractor, water-bottom multiples, a direct wave and noise). ``land_refraction_line`` builds a
land line whose first arrivals are a direct wave followed by refracted branches, with surface-consistent statics.
Both return a ``SeismicData`` with full geometry headers and CMP numbers.
"""

from __future__ import annotations

import numpy as np

from .dataset import SeismicData
from .geometry import geometry_2d


def ricker(f0: float, dt: float, half_length: float | None = None) -> tuple[np.ndarray, int]:
    """Zero-phase Ricker wavelet and the index of its peak."""
    half = half_length if half_length is not None else 1.5 / f0
    n = int(round(half / dt))
    t = np.arange(-n, n + 1) * dt
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a), n


def causal_wavelet(f0: float, dt: float) -> tuple[np.ndarray, int]:
    """A causal (onset-at-zero) wavelet t^2 exp(-1.5 pi f0 t) sin(2 pi f0 t), peak-normalised; index of t = 0 is 0.
    Recorded first arrivals start at their travel time, which makes picked onsets comparable with the true times."""
    t = np.arange(int(round(3.0 / (f0 * dt))) + 1) * dt
    w = t ** 2 * np.exp(-1.5 * np.pi * f0 * t) * np.sin(2 * np.pi * f0 * t)
    return w / np.abs(w).max(), 0


def events_to_traces(times: np.ndarray, amps: np.ndarray, nt: int, dt: float, wavelet: np.ndarray,
                     centre: int) -> np.ndarray:
    """Place events (ntr, nev) as interpolated spikes and convolve every trace with the wavelet (FFT)."""
    ntr = times.shape[0]
    spikes = np.zeros((ntr, nt))
    pos = times / dt
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    rows = np.repeat(np.arange(ntr)[:, None], times.shape[1], axis=1)
    for shift, weight in ((0, 1.0 - frac), (1, frac)):
        idx = i0 + shift
        ok = (idx >= 0) & (idx < nt) & np.isfinite(times)
        np.add.at(spikes, (rows[ok], idx[ok]), (amps * weight)[ok])
    nfft = int(2 ** np.ceil(np.log2(nt + wavelet.size)))
    w = np.fft.rfft(np.roll(np.pad(wavelet, (0, nfft - wavelet.size)), -centre))
    return np.fft.irfft(np.fft.rfft(spikes, nfft, axis=1) * w, nfft, axis=1)[:, :nt]


def marine_line(nshots: int = 40, shot_interval: float = 50.0, ngroups: int = 60, group_interval: float = 25.0,
                near_offset: float = 100.0, nt: int = 1001, dt: float = 0.004, f0: float = 25.0,
                reflectors=((0.6, 1.0, 1700.0), (1.1, -0.7, 2100.0), (1.6, 0.8, 2500.0), (2.2, 0.6, 2900.0)),
                water_bottom: tuple | None = (0.6, 1500.0), multiple_orders: int = 2, multiple_coeff: float = -0.6,
                direct_wave: float | None = 1500.0, dipping: tuple | None = None, diffractor: tuple | None = None,
                noise: float = 0.0, first_shot_x: float = 2000.0, seed: int = 0) -> SeismicData:
    """A 2D streamer line (receivers towed behind the source).

    reflectors: (t0 [s], reflection coefficient, v_rms [m/s]) with hyperbolic moveout.
    water_bottom: (t0 [s], v [m/s]) of the reflector whose surface multiples are added (orders 2..multiple_orders+1),
        each scaled by ``multiple_coeff`` per bounce.
    dipping: (t0 at x = first_shot_x [s], dip [s/km of midpoint], v [m/s], coeff) plane reflector in constant velocity.
    diffractor: (x [m], depth [m], v [m/s], amplitude) point scatterer in constant velocity.
    """
    rng = np.random.default_rng(seed)
    shot = np.repeat(np.arange(nshots), ngroups)
    chan = np.tile(np.arange(1, ngroups + 1), nshots)
    sx = first_shot_x + shot * shot_interval
    off = near_offset + (chan - 1) * group_interval
    gx = sx - off
    x = gx - sx
    ev_t, ev_a = [], []
    for t0, coeff, v in reflectors:
        ev_t.append(np.sqrt(t0 ** 2 + (x / v) ** 2))
        ev_a.append(np.full(x.shape, coeff) * t0 / np.maximum(ev_t[-1], 1e-6))
    if water_bottom is not None:
        twb, vwb = water_bottom
        for order in range(2, multiple_orders + 2):
            tm = np.sqrt((order * twb) ** 2 + (x / vwb) ** 2)
            ev_t.append(tm)
            ev_a.append(np.full(x.shape, multiple_coeff ** (order - 1)) * reflectors[0][1] * order * twb / tm)
    if direct_wave:
        ev_t.append(np.abs(x) / direct_wave + 0.02)
        ev_a.append(np.full(x.shape, 0.8))
    if dipping is not None:
        t00, dip_s_per_km, v, coeff = dipping
        p = dip_s_per_km / 1000.0                             # s/m of midpoint
        sin_theta = np.clip(v * p / 2.0, -0.95, 0.95)
        xm = 0.5 * (sx + gx) - first_shot_x
        tz = t00 + p * xm
        ev_t.append(np.sqrt(tz ** 2 + (x * np.sqrt(1 - sin_theta ** 2) / v) ** 2))
        ev_a.append(np.full(x.shape, coeff))
    if diffractor is not None:
        xd, zd, v, amp = diffractor
        td = (np.hypot(zd, sx - xd) + np.hypot(zd, gx - xd)) / v
        ev_t.append(td)
        ev_a.append(np.full(x.shape, amp) * (2 * zd / v) / td)
    times = np.stack(ev_t, axis=1)
    amps = np.stack(ev_a, axis=1)
    w, c = ricker(f0, dt)
    traces = events_to_traces(times, amps, nt, dt, w, c)
    if noise > 0:
        traces = traces + noise * rng.standard_normal(traces.shape)
    headers = {"fldr": shot + 1001, "ep": shot + 1, "tracf": chan, "sx": sx, "gx": gx, "sy": 0.0, "gy": 0.0,
               "offset": x, "trid": 1, "sdepth": 6.0, "gelev": -8.0}
    data = SeismicData(traces.astype(np.float32), dt, headers, name="synthetic marine line")
    data = geometry_2d(data, group_interval / 2.0)
    data.meta["synthetic"] = {"reflectors": [list(r) for r in reflectors], "water_bottom": water_bottom,
                              "dipping": dipping, "diffractor": diffractor, "f0": f0}
    return data


def land_refraction_line(nshots: int = 12, shot_interval: float = 100.0, nrec: int = 96, rec_interval: float = 10.0,
                         nt: int = 751, dt: float = 0.002, f0: float = 30.0, v1: float = 800.0,
                         refractors=((2000.0, 0.040), (3500.0, 0.090)), reflectors=((0.45, 0.5, 2200.0),),
                         source_statics_ms: float = 0.0, receiver_statics_ms: float = 0.0, noise: float = 0.0,
                         seed: int = 1, wavelet: str = "causal") -> SeismicData:
    """A split-spread land line. First arrivals: direct wave x/v1 and refracted branches x/v + intercept time.

    refractors: (refractor velocity [m/s], intercept time [s]). Surface-consistent statics (ms, uniform random within
    ±value) are added per source position and per receiver station and stored as the true values in meta.
    """
    rng = np.random.default_rng(seed)
    length = (nrec - 1) * rec_interval
    rec_x = np.arange(nrec) * rec_interval
    shot_x = np.linspace(0.0, length, nshots) if nshots > 1 else np.array([length / 2])
    shot = np.repeat(np.arange(nshots), nrec)
    station = np.tile(np.arange(nrec), nshots)
    sx = shot_x[shot]
    gx = rec_x[station]
    x = np.abs(gx - sx)
    branches = [x / v1] + [x / v + ti for v, ti in refractors]
    first = np.min(np.stack(branches), axis=0)
    ev_t = [first] + [np.sqrt(t0 ** 2 + (x / v) ** 2) for t0, _, v in reflectors]
    ev_a = [np.full(x.shape, 1.0)] + [np.full(x.shape, r) for _, r, _ in reflectors]
    s_stat = rng.uniform(-source_statics_ms, source_statics_ms, nshots) / 1000.0 if source_statics_ms else np.zeros(nshots)
    r_stat = rng.uniform(-receiver_statics_ms, receiver_statics_ms, nrec) / 1000.0 if receiver_statics_ms else np.zeros(nrec)
    shift = s_stat[shot] + r_stat[station]
    times = np.stack(ev_t, axis=1) + shift[:, None]
    amps = np.stack(ev_a, axis=1)
    w, c = causal_wavelet(f0, dt) if wavelet == "causal" else ricker(f0, dt)
    traces = events_to_traces(times, amps, nt, dt, w, c)
    if noise > 0:
        traces = traces + noise * rng.standard_normal(traces.shape)
    headers = {"fldr": shot + 1, "ep": shot + 1, "tracf": station + 1, "sx": sx, "gx": gx, "sy": 0.0, "gy": 0.0,
               "offset": gx - sx, "trid": 1}
    data = SeismicData(traces.astype(np.float32), dt, headers, name="synthetic land refraction line")
    data = geometry_2d(data, rec_interval / 2.0)
    data.meta["synthetic"] = {"v1": v1, "refractors": [list(r) for r in refractors], "first_break": first + shift,
                              "source_statics_s": s_stat, "receiver_statics_s": r_stat}
    return data
