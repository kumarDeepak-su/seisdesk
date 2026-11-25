"""Near-surface tools: linear moveout (reduced time) with one velocity or frequency by frequency from a dispersion
curve, and the refraction convolution section of Palmer (2001)."""

from __future__ import annotations

import numpy as np

from ..core.dataset import SeismicData
from .base import Param, Process, register, resolve_picks
from .firstbreak import mer_picks
from .refstatics import pick_geometry


def parse_curve(text: str) -> tuple[np.ndarray, np.ndarray]:
    """A dispersion curve 'f:v, f:v, ...' (Hz, m/s) as frequency and velocity arrays, sorted by frequency."""
    pairs = [p.split(":") for p in str(text).replace(";", ",").split(",") if p.strip()]
    try:
        f = np.array([float(a) for a, _ in pairs])
        v = np.array([float(b) for _, b in pairs])
    except ValueError as exc:
        raise ValueError(f"cannot read the dispersion curve {text!r}: give 'f:v, f:v, ...' (Hz, m/s)") from exc
    if f.size == 0:
        raise ValueError("give the dispersion curve as 'f:v, f:v, ...' (Hz, m/s)")
    if np.any(v <= 0):
        raise ValueError("the velocities of the dispersion curve must be positive")
    order = np.argsort(f)
    return f[order], v[order]


def apply_delays(traces: np.ndarray, dt: float, delays, max_shift: float, chunk: int = 512) -> np.ndarray:
    """Delay every trace in the frequency domain; ``delays(w, rows)`` gives the delay in seconds (negative = earlier)
    for angular frequencies w and the trace rows, broadcastable to (rows, w). The transform is padded by the largest
    shift, so nothing wraps around: samples moved outside the record are lost."""
    ntr, nt = traces.shape
    pad = int(np.ceil(abs(max_shift) / dt)) + 1
    nfft = int(2 ** np.ceil(np.log2(2 * (nt + pad))))
    w = 2 * np.pi * np.fft.rfftfreq(nfft, dt)
    out = np.empty((ntr, nt))
    for start in range(0, ntr, chunk):
        rows = np.arange(start, min(ntr, start + chunk))
        spec = np.fft.rfft(traces[rows].astype(np.float64), nfft, axis=1)
        spec *= np.exp(-1j * w[None, :] * delays(w, rows))
        out[rows] = np.fft.irfft(spec, nfft, axis=1)[:, :nt]
    return out


@register
class LinearMoveout(Process):
    name = "lmo"
    label = "Linear moveout (reduced time)"
    category = "Moveout"
    description = ("Reduced time t' = t - |offset| / v + shift. Events travelling at the reduction velocity become "
                   "flat: a refracted branch at its refractor velocity (its intercept time is then read directly), "
                   "or linear noise before f-k or tau-p work. 'dynamic' moves every frequency by its own phase "
                   "velocity from a dispersion curve 'f:v, ...': a dispersed surface-wave train collapses to a short "
                   "pulse at the shift time (Seismic Unix sudlmo). 'inverse' undoes the same correction.")
    params = (Param("velocity", "float", 2000.0, "Reduction velocity", unit="m/s", minimum=1.0),
              Param("shift", "float", 0.0, "Time shift added", unit="s"),
              Param("direction", "choice", "forward", "Direction", choices=("forward", "inverse")),
              Param("mode", "choice", "one velocity", "Mode", choices=("one velocity", "dynamic (dispersion curve)")),
              Param("curve", "str", "5:400, 10:300, 20:220, 40:180", "Dispersion curve f:v (dynamic mode)",
                    "frequency (Hz) : phase velocity (m/s) pairs"),
              Param("offset_key", "key", "offset", "Offset header"))

    def run(self, data, context):
        if data.domain != "time":
            raise ValueError("linear moveout needs time data")
        x = np.abs(data.header(self["offset_key"]))
        sign = 1.0 if self["direction"] == "forward" else -1.0
        shift = self["shift"]
        if self["mode"] == "one velocity":
            delays = sign * (shift - x / self["velocity"])
            out = apply_delays(data.traces, data.dt, lambda w, rows: delays[rows, None],
                               float(np.max(np.abs(delays), initial=0.0)))
        else:
            fc, vc = parse_curve(self["curve"])

            def delays(w, rows):
                c = np.interp(w / (2 * np.pi), fc, vc)
                return sign * (shift - x[rows, None] / c[None, :])

            out = apply_delays(data.traces, data.dt, delays, float(x.max(initial=0.0) / vc.min() + abs(shift)))
        res = data.copy(out.astype(np.float32))
        if self["direction"] == "forward":
            res.meta["reduced_time"] = {"velocity": self["velocity"], "shift": shift, "mode": self["mode"]}
        else:
            res.meta.pop("reduced_time", None)
        return res


# ---------------------------------------------------------------------------------------------- convolution section
def _shot_table(data: SeismicData, key: str) -> dict:
    """Shot value -> (source x, trace indices) for every shot (gather of ``key``)."""
    sx = data.header("sx")
    return {value: (float(np.median(sx[idx])), idx) for value, idx in data.iter_gathers(key, None)}


def _trace_at(data: SeismicData, idx: np.ndarray, x: float, tol: float) -> int | None:
    """Index of the trace of a shot recorded nearest to receiver position x (None if farther than tol)."""
    gx = data.header("gx")[idx]
    j = int(np.argmin(np.abs(gx - x)))
    return int(idx[j]) if abs(gx[j] - x) <= tol else None


def _pick_lookup(ps, data: SeismicData, phase: str, key: str) -> dict:
    """Shot value -> (receiver x, pick time), sorted by x, from the picks of one phase."""
    _sx, gx, idx, t = pick_geometry(ps, data, phase)
    shots = data.header(key)[idx]
    table = {}
    for value in np.unique(shots):
        m = shots == value
        order = np.argsort(gx[m])
        table[float(value)] = (gx[m][order], t[m][order])
    return table


def _picked_time(table: dict, shot: float, x: float, tol: float) -> float | None:
    entry = table.get(float(shot))
    if entry is None or entry[0].size == 0:
        return None
    px, pt = entry
    if x < px[0] - tol or x > px[-1] + tol:
        return None
    return float(np.interp(x, px, pt))


def _keep_window(trace: np.ndarray, times: np.ndarray, t_fb: float, keep: float, taper: float = 0.01) -> np.ndarray:
    """The trace from shortly before its first break to ``keep`` seconds after it, cosine-tapered at both ends."""
    w = np.ones(times.size)
    lo, hi = t_fb - taper, t_fb + keep
    rise = np.clip((times - (lo - taper)) / taper, 0.0, 1.0)
    fall = np.clip(((hi + taper) - times) / taper, 0.0, 1.0)
    w *= 0.5 * (1 - np.cos(np.pi * rise)) * 0.5 * (1 - np.cos(np.pi * fall))
    return trace * w


@register
class RefractionConvolution(Process):
    name = "refraction_convolution"
    label = "Refraction convolution section"
    category = "Statics"
    description = ("Palmer's refraction convolution section (2001). At every receiver the trace of a forward shot "
                   "and the trace of a reverse shot are convolved, which adds their travel times, and the "
                   "reciprocal time (forward shot to reverse shot) is subtracted: the refractor appears at twice "
                   "its time-depth (the GRM time-depth, i.e. the two-way delay time under the receiver), and the "
                   "product of the amplitudes cancels most of the geometric spreading. 'automatic' pairs every "
                   "receiver with the nearest shot on each side at least the minimum offset away (beyond the "
                   "crossover distance, so that the first arrival is the refracted wave); 'two shots' uses one "
                   "forward and one reverse FFID for the receivers between them. The reciprocal time is measured "
                   "on the traces (first breaks of each shot at the other shot's position), read from the picks of "
                   "the refracted phase, or given.")
    params = (Param("pairing", "choice", "automatic", "Shot pairs", choices=("automatic", "two shots")),
              Param("forward_ffid", "float", 0.0, "Forward shot (two shots)"),
              Param("reverse_ffid", "float", 0.0, "Reverse shot (two shots)"),
              Param("min_offset", "float", 100.0, "Minimum offset (automatic pairs)", unit="m", minimum=0.0),
              Param("reciprocal", "choice", "first breaks of the traces", "Reciprocal time from",
                    choices=("first breaks of the traces", "picks", "value")),
              Param("reciprocal_time", "float", 0.0, "Reciprocal time (value)", unit="s", minimum=0.0),
              Param("pickset", "picks", "first breaks", "Pick set (picks)"),
              Param("phase", "str", "refractor 1", "Refracted phase (picks)"),
              Param("fb_window", "float", 0.02, "First-break energy window", unit="s", minimum=0.002),
              Param("keep", "float", 0.0, "Keep after the first break (0 = whole traces)", unit="s", minimum=0.0),
              Param("tmax", "float", 0.0, "Output length (0 = input length)", unit="s", minimum=0.0),
              Param("shot_key", "key", "fldr", "Shot header"))

    def run(self, data, context):
        if data.domain != "time":
            raise ValueError("the refraction convolution section needs time data")
        key = self["shot_key"]
        if not data.has_header(key) or not (data.has_header("sx") or data.has_header("gx")):
            raise ValueError(f"needs shot numbers ('{key}') and source and receiver positions (sx, gx): "
                             "run a geometry step")
        shots = _shot_table(data, key)
        gx_all = data.header("gx")
        stations = np.unique(np.round(gx_all, 3))
        steps = np.diff(stations)
        station = float(np.median(steps[steps > 1e-6])) if np.any(steps > 1e-6) else 1.0
        tol = 0.51 * station
        times = data.times
        mode = self["reciprocal"]
        picks = None
        if mode == "picks":
            picks = _pick_lookup(resolve_picks(self["pickset"], context), data, self["phase"], key)
        # receiver station -> trace index, per shot (stations are the receiver positions rounded to the mm)
        at_station = {s: dict(zip(np.round(gx_all[idx], 3).tolist(), idx.tolist()))
                      for s, (_x, idx) in shots.items()}

        def first_break(i: int) -> float:
            # the earliest strong onset: a direct wave later in the trace can be stronger than the refraction
            return float(mer_picks(data.traces[i:i + 1], times, self["fb_window"], np.array([times[0]]),
                                   np.array([times[-1]]), first_fraction=0.1)[0])

        cache: dict = {}

        def reciprocal(a: float, b: float) -> float | None:
            if (a, b) in cache:
                return cache[(a, b)]
            xa, ia = shots[a]
            xb, ib = shots[b]
            value = None
            if mode == "value":
                value = self["reciprocal_time"]
            else:
                found = []
                for shot, idx, x_other in ((a, ia, xb), (b, ib, xa)):
                    if mode == "picks":
                        t = _picked_time(picks, shot, x_other, 1.5 * station)
                    else:
                        i = _trace_at(data, idx, x_other, 1.5 * station)
                        t = None if i is None else first_break(i)
                    if t is not None:
                        found.append(t)
                value = float(np.mean(found)) if found else None
            cache[(a, b)] = value
            return value

        # forward (smaller x) and reverse shot for every receiver station
        order = sorted(shots, key=lambda s: shots[s][0])
        pairs = []
        for g in stations:
            if self["pairing"] == "two shots":
                fa, fb = float(self["forward_ffid"]), float(self["reverse_ffid"])
                if fa not in shots or fb not in shots:
                    raise ValueError(f"shots {fa:g} and {fb:g} are not both in the dataset ('{key}')")
                a, b = (fa, fb) if shots[fa][0] <= shots[fb][0] else (fb, fa)
                if not (shots[a][0] < g < shots[b][0]):
                    continue
            else:
                gmin = max(self["min_offset"], 1e-6)
                left = [s for s in order if g - shots[s][0] >= gmin and g in at_station[s]]
                right = [s for s in order if shots[s][0] - g >= gmin and g in at_station[s]]
                if not left or not right:
                    continue
                a, b = left[-1], right[0]
            ia = at_station[a].get(g)
            ib = at_station[b].get(g)
            if ia is None or ib is None:
                continue
            pairs.append((float(g), a, b, ia, ib))
        if not pairs:
            raise ValueError("no receiver has a forward and a reverse shot (check the pairing, the minimum offset "
                             "and that the shots record across each other)")
        n_out = int(round(self["tmax"] / data.dt)) + 1 if self["tmax"] > 0 else data.nt
        tau = np.arange(n_out) * data.dt
        nfft = int(2 ** np.ceil(np.log2(2 * data.nt)))
        out = np.zeros((len(pairs), n_out))
        rows = []
        for n, (g, a, b, ia, ib) in enumerate(pairs):
            t_ab = reciprocal(a, b)
            if t_ab is None:
                context.log(f"    receiver {g:g}: no reciprocal time for shots {a:g} and {b:g}, skipped")
                continue
            tr_a = data.traces[ia].astype(np.float64)
            tr_b = data.traces[ib].astype(np.float64)
            if self["keep"] > 0:
                for name, tr_i, shot in (("a", ia, a), ("b", ib, b)):
                    t_fb = _picked_time(picks, shot, g, tol) if picks is not None else None
                    if t_fb is None:
                        t_fb = first_break(tr_i)
                    if name == "a":
                        tr_a = _keep_window(tr_a, times, t_fb, self["keep"])
                    else:
                        tr_b = _keep_window(tr_b, times, t_fb, self["keep"])
            conv = np.fft.irfft(np.fft.rfft(tr_a, nfft) * np.fft.rfft(tr_b, nfft), nfft)[:2 * data.nt - 1]
            pos = (tau + t_ab - 2.0 * data.t0) / data.dt          # sample k of the convolution is at 2 t0 + k dt
            out[n] = np.interp(pos, np.arange(conv.size), conv, left=0.0, right=0.0)
            rows.append((n, g, a, b, t_ab, ia, ib))
            context.report((n + 1) / len(pairs))
        if not rows:
            raise ValueError("no reciprocal time could be found for any shot pair")
        keep_rows = np.array([r[0] for r in rows])
        gx = np.array([r[1] for r in rows])
        gelev = np.array([0.5 * (data.header("gelev")[r[5]] + data.header("gelev")[r[6]]) for r in rows])
        station_no = np.rint((gx - stations[0]) / station).astype(np.int64) + 1
        heads = {"gx": gx, "sx": gx, "cdpx": gx, "offset": 0.0, "cdp": station_no, "tracf": station_no,
                 "fldr": 1.0, "gelev": gelev, "selev": gelev, "rcs_fwd": [r[2] for r in rows],
                 "rcs_rev": [r[3] for r in rows], "rcs_trec": [r[4] * 1000.0 for r in rows]}
        res = SeismicData(out[keep_rows].astype(np.float32), data.dt, heads, 0.0,
                          data.name + " refraction convolution", "time", list(data.history), dict(data.meta))
        res.meta["refraction_convolution"] = {"time": "twice the time-depth (two-way delay) under each receiver"}
        context.results["refraction_convolution"] = {
            "x": gx.tolist(), "forward": [r[2] for r in rows], "reverse": [r[3] for r in rows],
            "reciprocal_ms": [r[4] * 1000.0 for r in rows]}
        context.log(f"    {len(rows)} receivers, {len({(r[2], r[3]) for r in rows})} shot pairs; output time = twice "
                    f"the time-depth")
        return res
