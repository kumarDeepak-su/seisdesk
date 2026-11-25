"""Refraction statics by delay-time inversion of first-break picks (the least-squares generalisation of the
plus-minus method): surface-consistent delay times, refractor velocity, weathering thickness and statics to datum."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .base import Param, Process, register, resolve_picks
from .statics import shift_traces


def pick_geometry(ps, data, phase: str):
    """Source x, receiver x, trace index and time of every pick of one phase, matched to the traces."""
    if ps.gather_key not in data.headers or ps.x_key not in data.headers:
        raise ValueError(f"the dataset has no '{ps.gather_key}' or '{ps.x_key}' header for these picks")
    rows = [(i, t) for _, _, _, t, i in ps.matched_records(data, phase)]
    if len(rows) < 4:
        raise ValueError(f"fewer than four picks of phase {phase!r} match traces of this dataset")
    idx = np.array([r[0] for r in rows], dtype=np.int64)
    t = np.array([r[1] for r in rows], dtype=np.float64)
    return data.header("sx")[idx], data.header("gx")[idx], idx, t


def delay_time_inversion(sx, gx, t, station: float = 0.0, v2: float = 0.0, smoothing: float = 0.1):
    """Least-squares delay times at surface stations (shared by sources and receivers) and refractor slowness from
    t = delay(source) + delay(receiver) + |gx - sx| / V2, with a second-difference smoothing of the delays.

    Returns station x, delays (s), V2 (m/s), rms misfit (s) and the number of picks per station."""
    pos = np.concatenate([sx, gx])
    if station <= 0:
        u = np.unique(np.round(gx, 3))
        steps = np.diff(u)
        steps = steps[steps > 1e-6]
        station = float(np.median(steps)) if steps.size else 1.0
    origin = float(pos.min())
    s_idx = np.rint((sx - origin) / station).astype(np.int64)
    r_idx = np.rint((gx - origin) / station).astype(np.int64)
    used = np.unique(np.concatenate([s_idx, r_idx]))
    ns = int(used.max()) + 1
    n = t.size
    x = np.abs(gx - sx)
    solve_v = v2 <= 0
    ncol = ns + (1 if solve_v else 0)
    rows = [np.arange(n), np.arange(n)]
    cols = [s_idx, r_idx]
    vals = [np.ones(n), np.ones(n)]
    rhs = t - (0.0 if solve_v else x / v2)
    if solve_v:
        rows.append(np.arange(n))
        cols.append(np.full(n, ns))
        vals.append(x / max(float(x.max()), 1.0))               # scaled slowness column
    lam = smoothing * np.sqrt(n / max(ns, 1))
    if lam > 0 and ns > 2:
        k = np.arange(1, ns - 1)
        base = n + k - 1
        for off, wgt in ((-1, 1.0), (0, -2.0), (1, 1.0)):
            rows.append(base)
            cols.append(k + off)
            vals.append(np.full(k.size, lam * wgt))
        rhs = np.concatenate([rhs, np.zeros(k.size)])
    nrow = rhs.size
    a = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrow, ncol)).tocsr()
    sol = lsqr(a, rhs, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    delays = sol[:ns]
    if solve_v:
        slowness = sol[ns] / max(float(x.max()), 1.0)
        if not slowness > 0:
            raise ValueError("the refractor slowness came out negative: check the phase picks")
        v2 = 1.0 / slowness
    pred = delays[s_idx] + delays[r_idx] + x / v2
    rms = float(np.sqrt(np.mean((t - pred) ** 2)))
    counts = np.bincount(np.concatenate([s_idx, r_idx]), minlength=ns)
    return origin + np.arange(ns) * station, delays, float(v2), rms, counts, s_idx, r_idx


@register
class RefractionStatics(Process):
    name = "refraction_statics"
    label = "Refraction statics (delay-time inversion)"
    category = "Statics"
    description = ("From the first-break picks of one refracted branch, solves t = delay(source) + delay(receiver) + "
                   "|offset| / V2 by least squares over all shots (delay times at surface stations, smoothed along "
                   "the line). Delays become weathering thickness with V1 (from the direct-wave phase or given), and "
                   "source and receiver statics to a flat datum follow with the replacement velocity. Writes "
                   "sstat/gstat (ms), applies them if asked, and keeps the station table (elevation, base of "
                   "weathering, statics) for display and export.")
    params = (Param("pickset", "picks", "first breaks", "Pick set"),
              Param("phase", "str", "refractor 1", "Refracted branch (phase)"),
              Param("v1", "float", 0.0, "Weathering velocity V1 (0 = from the direct-wave phase)", unit="m/s",
                    minimum=0.0),
              Param("direct_phase", "str", "direct", "Direct-wave phase (for V1)"),
              Param("v2", "float", 0.0, "Refractor velocity V2 (0 = solve)", unit="m/s", minimum=0.0),
              Param("datum", "float", 0.0, "Datum elevation", unit="m"),
              Param("vr", "float", 0.0, "Replacement velocity (0 = V2)", unit="m/s", minimum=0.0),
              Param("station", "float", 0.0, "Station spacing (0 = receiver spacing)", unit="m", minimum=0.0),
              Param("smoothing", "float", 0.1, "Smoothing of the delays", minimum=0.0),
              Param("apply", "bool", True, "Apply the statics"))

    def run(self, data, context):
        ps = resolve_picks(self["pickset"], context)
        sx, gx, _idx, t = pick_geometry(ps, data, self["phase"])
        sta_x, delays, v2, rms, counts, _s, _r = delay_time_inversion(sx, gx, t, self["station"], self["v2"],
                                                                      self["smoothing"])
        v1 = self["v1"]
        if v1 <= 0:
            v1 = float(ps.fit_line(None, self["direct_phase"])["velocity"])
        if not v2 > v1:
            raise ValueError(f"the refractor velocity ({v2:.0f} m/s) must exceed the weathering velocity ({v1:.0f})")
        vr = self["vr"] if self["vr"] > 0 else v2
        thick = np.maximum(delays, 0.0) * v1 * v2 / np.sqrt(v2 * v2 - v1 * v1)
        # surface elevation at the stations from the receivers (sources where there is no receiver)
        station = sta_x[1] - sta_x[0] if sta_x.size > 1 else 1.0
        elev = np.full(sta_x.size, np.nan)
        for xs, zs in ((data.header("sx"), data.header("selev")), (data.header("gx"), data.header("gelev"))):
            k = np.clip(np.rint((xs - sta_x[0]) / station).astype(np.int64), 0, sta_x.size - 1)
            tot = np.bincount(k, weights=zs, minlength=sta_x.size)
            cnt = np.bincount(k, minlength=sta_x.size)
            elev = np.where(cnt > 0, tot / np.maximum(cnt, 1), elev)
        known = np.isfinite(elev)
        elev = np.interp(sta_x, sta_x[known], elev[known]) if known.any() else np.zeros(sta_x.size)
        base = elev - thick
        static = -(thick / v1 + (base - self["datum"]) / vr) * 1000.0           # ms, to the datum
        # per trace: source and receiver station statics (a buried source skips the weathering above it)
        k_s = np.clip(np.rint((data.header("sx") - sta_x[0]) / station).astype(np.int64), 0, sta_x.size - 1)
        k_r = np.clip(np.rint((data.header("gx") - sta_x[0]) / station).astype(np.int64), 0, sta_x.size - 1)
        depth = data.header("sdepth")
        th_s = np.maximum(thick[k_s] - depth, 0.0)
        s_stat = -(th_s / v1 + (elev[k_s] - depth - th_s - self["datum"]) / vr) * 1000.0
        g_stat = static[k_r]
        out = data.copy()
        out.set_header("sstat", s_stat)
        out.set_header("gstat", g_stat)
        if self["apply"]:
            out = out.copy(shift_traces(out.traces, data.dt, (s_stat + g_stat) / 1000.0))
            out.set_header("tstat", data.header("tstat") + s_stat + g_stat)
        context.results["refraction_statics"] = {
            "x": sta_x.tolist(), "elevation": elev.tolist(), "delay_ms": (delays * 1000.0).tolist(),
            "thickness": thick.tolist(), "base_elevation": base.tolist(), "static_ms": static.tolist(),
            "picks": counts.tolist(), "v1": v1, "v2": v2, "vr": vr, "datum": self["datum"], "rms_ms": rms * 1000.0}
        context.log(f"    V1 {v1:.0f} m/s, V2 {v2:.0f} m/s, misfit {rms * 1000:.2f} ms; weathering "
                    f"{thick.min():.1f}-{thick.max():.1f} m; statics {static.min():.1f} to {static.max():.1f} ms")
        return out
