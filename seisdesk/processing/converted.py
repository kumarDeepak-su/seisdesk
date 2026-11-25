"""Converted-wave (P-SV) processing: common-conversion-point binning, PS moveout correction, PS dip moveout by
migration to zero offset, Vp/Vs from PP and PS horizon times, and ray-traced PS synthetics.

A P wave that goes down and comes back up as an S wave (PS or C wave) reflects at a conversion point that lies closer
to the receiver than the midpoint, and that approaches its asymmetric deep limit with depth. gamma = Vp/Vs (often 1.6-2.5 in
sediments) controls this asymmetry, the moveout and the relation between the times of one reflector on PP and PS
sections: t_PS = t_PP (1 + gamma) / 2. The radial component (horizontal, positive away from the source) records PS
reflections with one polarity on both sides of the source.
"""

from __future__ import annotations

import numpy as np

from ..core.dataset import SeismicData
from ..core.synthetic import events_to_traces, ricker
from .base import Param, Process, register, resolve_picks, resolve_velocity
from .migration_time import rho_filter
from .moveout import sample_at
from .nmo import velocity_per_trace


def _check_gamma(gamma: float) -> float:
    g = float(gamma)
    if not g > 1.0:
        raise ValueError("gamma = Vp/Vs must be greater than 1 for converted waves")
    return g


def _receiver_spacing(data: SeismicData, default: float = 25.0) -> float:
    gx = np.unique(np.round(data.header("gx"), 3))
    steps = np.diff(gx)
    steps = steps[steps > 1e-6]
    return float(np.median(steps)) if steps.size else default


# ---------------------------------------------------------------------------------------------- conversion points
def asymptotic_ccp(sx, gx, gamma: float) -> np.ndarray:
    """Asymptotic conversion point (Fromm, Krey & Wiest 1985): x_s + (x_g - x_s) gamma / (1 + gamma), the limit of the
    true conversion point at depths large compared with the offset."""
    sx = np.asarray(sx, dtype=np.float64)
    return sx + (np.asarray(gx, dtype=np.float64) - sx) * gamma / (1.0 + gamma)


def conversion_distance(offset, depth, vp: float, vs: float, iterations: int = 50) -> np.ndarray:
    """Horizontal distance from the source to the conversion point of a P-SV reflection from depth z in a homogeneous
    layer (Tessmer & Behle 1988): Snell's law x_p / (vp sqrt(x_p^2 + z^2)) = (x - x_p) / (vs sqrt((x - x_p)^2 + z^2))
    solved by bisection on 0 <= x_p <= |x|; signed like the offset. It tends to x gamma / (1 + gamma) at depth and to
    the receiver position near the surface."""
    off = np.asarray(offset, dtype=np.float64)
    x, z = np.broadcast_arrays(np.abs(off), np.asarray(depth, dtype=np.float64))
    lo = np.zeros(x.shape)
    hi = np.array(x, dtype=np.float64)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f = mid / (vp * np.hypot(mid, z) + 1e-30) - (x - mid) / (vs * np.hypot(x - mid, z) + 1e-30)
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    return np.sign(off) * 0.5 * (lo + hi)


@register
class PSBinning(Process):
    name = "ps_binning"
    label = "Converted-wave CCP binning (asymptotic or depth-variant)"
    category = "Converted waves"
    description = ("Common-conversion-point binning of PS data. 'asymptotic' moves every trace to its asymptotic "
                   "conversion point x_s + offset gamma / (1 + gamma) and writes the bin number (cdp), the bin centre "
                   "(cdpx) and the exact point (ccpx); sort by cdp, apply PS NMO and stack as usual. 'depth-variant' "
                   "(Tessmer & Behle 1988) needs PS-NMO-corrected input: every sample goes to the bin of its true "
                   "conversion point, found by Snell's law at the depth z = Vp t / (1 + gamma), and the samples are "
                   "stacked into a CCP section (one trace per bin, header nhs = fold).")
    params = (Param("method", "choice", "asymptotic", "Method", choices=("asymptotic", "depth-variant (CCP stack)")),
              Param("gamma", "float", 2.0, "Vp/Vs (gamma)", minimum=1.01, maximum=10.0),
              Param("vp", "float", 2000.0, "P velocity for the depth (depth-variant)", unit="m/s", minimum=10.0),
              Param("bin_size", "float", 0.0, "Bin size (0 = half the receiver spacing)", unit="m", minimum=0.0),
              Param("origin", "float", -1e30, "First bin edge (default: smallest conversion point)", unit="m"))

    def run(self, data, context):
        gamma = _check_gamma(self["gamma"])
        sx, gx = data.header("sx"), data.header("gx")
        if np.ptp(np.concatenate([sx, gx])) <= 0:
            raise ValueError("source and receiver positions (sx, gx) are not set: run a geometry step first")
        size = self["bin_size"] if self["bin_size"] > 0 else 0.5 * _receiver_spacing(data)
        ccp = asymptotic_ccp(sx, gx, gamma)
        origin = float(ccp.min()) - 0.5 * size if self["origin"] <= -1e29 else self["origin"]
        if self["method"] == "asymptotic":
            out = data.copy()
            nbin = np.floor((ccp - origin) / size).astype(np.int64)
            out.set_header("ccpx", ccp)
            out.set_header("cdp", nbin + 1)
            out.set_header("cdpx", origin + (nbin + 0.5) * size)
            out.meta["geometry"] = dict(out.meta.get("geometry", {}), bin_size=size, ps_gamma=gamma)
            context.log(f"    {np.unique(nbin).size} CCP bins of {size:g} m (asymptotic, gamma {gamma:g})")
            return out
        vp = self["vp"]
        vs = vp / gamma
        times = data.times
        depth = np.maximum(times, 0.0) * vp / (1.0 + gamma)
        offset = gx - sx
        # the deepest sample gives the bin range: conversion points move towards the asymptotic ones with depth
        lo = min(float(np.min(np.minimum(sx, gx))), float(ccp.min()))
        hi = max(float(np.max(np.maximum(sx, gx))), float(ccp.max()))
        if self["origin"] > -1e29:
            lo = self["origin"]
        else:
            lo -= 0.5 * size
        nbins = int(np.floor((hi - lo) / size)) + 1
        stack = np.zeros((nbins, data.nt))
        fold = np.zeros((nbins, data.nt))
        cols = np.arange(data.nt)
        for i in range(data.ntr):
            xc = sx[i] + conversion_distance(offset[i], depth, vp, vs)
            b = np.floor((xc - lo) / size).astype(np.int64)
            tr = data.traces[i].astype(np.float64)
            ok = (b >= 0) & (b < nbins) & (tr != 0)
            np.add.at(stack, (b[ok], cols[ok]), tr[ok])
            np.add.at(fold, (b[ok], cols[ok]), 1.0)
            if i % 200 == 0:
                context.report(i / data.ntr)
        live = fold.max(axis=1) > 0
        stack = np.where(fold > 0, stack / np.maximum(fold, 1.0), 0.0)[live]
        idx = np.flatnonzero(live)
        centres = lo + (idx + 0.5) * size
        heads = {"cdp": idx + 1.0, "cdpx": centres, "ccpx": centres, "sx": centres, "gx": centres,
                 "offset": np.zeros(idx.size), "nhs": fold.max(axis=1)[live]}
        out = SeismicData(stack.astype(np.float32), data.dt, heads, data.t0, data.name + " CCP stack", "time",
                          list(data.history), dict(data.meta))
        out.meta["geometry"] = dict(data.meta.get("geometry", {}), bin_size=size, ps_gamma=gamma)
        context.log(f"    depth-variant CCP stack: {idx.size} bins of {size:g} m, Vp {vp:g} m/s, gamma {gamma:g}")
        return out


# ---------------------------------------------------------------------------------------------- PS moveout
def ps_moveout_time(t0, x, vc, gamma) -> np.ndarray:
    """Converted-wave reflection time (Tsvankin & Thomsen 1994; Thomsen 1999): t^2 = t0^2 + x^2 / vc^2 +
    A4 x^4 / (1 + A5 x^2), A4 = -(gamma - 1)^2 / (4 gamma t0^2 vc^4), A5 = (gamma - 1) / (4 t0^2 vc^2), with vc^2 =
    vp vs the C-wave stacking velocity and gamma = vp / vs. Written without 1/t0^2 so that t0 = 0 is safe:
    A4 x^4 / (1 + A5 x^2) = -(gamma - 1)^2 x^4 / (gamma vc^2 (4 t0^2 vc^2 + (gamma - 1) x^2)). It tends to x / vp at
    large offsets and is exact to the fourth order in offset for a homogeneous layer."""
    t0 = np.asarray(t0, dtype=np.float64)
    x2 = np.asarray(x, dtype=np.float64) ** 2
    v2 = np.asarray(vc, dtype=np.float64) ** 2
    g = np.asarray(gamma, dtype=np.float64)
    t2 = t0 ** 2 + x2 / v2 - (g - 1.0) ** 2 * x2 * x2 / (g * v2 * (4.0 * t0 ** 2 * v2 + (g - 1.0) * x2) + 1e-30)
    return np.sqrt(np.maximum(t2, 0.0))


def ps_nmo_correct(traces: np.ndarray, times: np.ndarray, offsets: np.ndarray, vc_field, gamma: float,
                   stretch_limit: float = 0.5) -> np.ndarray:
    """out(t0) = in(t_PS(t0, x)) per trace; vc_field (ntr, nt) or scalar; stretch above the limit is muted."""
    dt = times[1] - times[0]
    t0 = np.broadcast_to(times[None, :], traces.shape)
    tt = ps_moveout_time(t0, np.asarray(offsets, dtype=np.float64)[:, None], vc_field, gamma)
    vals, valid = sample_at(traces.astype(np.float64), tt, times[0], dt)
    if stretch_limit > 0:
        dtt = np.gradient(tt, dt, axis=1)
        valid &= dtt >= 1.0 / (1.0 + stretch_limit)
    return np.where(valid, vals, 0.0).astype(np.float32)


def ps_inverse_nmo(traces: np.ndarray, times: np.ndarray, offsets: np.ndarray, vc_field, gamma: float) -> np.ndarray:
    """Undo PS NMO by inverting the monotone moveout curve of every trace."""
    dt = times[1] - times[0]
    t0 = np.broadcast_to(times[None, :], traces.shape)
    tt = ps_moveout_time(t0, np.asarray(offsets, dtype=np.float64)[:, None], vc_field, gamma)
    out = np.zeros(traces.shape)
    for i in range(traces.shape[0]):
        t0_of_t = np.interp(times, tt[i], times, left=np.nan, right=np.nan)
        pos = (t0_of_t - times[0]) / dt
        ok = np.isfinite(pos)
        out[i, ok] = np.interp(pos[ok], np.arange(times.size), traces[i])
    return out.astype(np.float32)


@register
class PSNMO(Process):
    name = "ps_nmo"
    label = "PS NMO (converted-wave moveout)"
    category = "Converted waves"
    description = ("Moveout correction of converted (PS) reflections with the non-hyperbolic equation of Tsvankin & "
                   "Thomsen (1994): t^2 = t0^2 + x^2 / Vc^2 + A4 x^4 / (1 + A5 x^2), where Vc is the C-wave stacking "
                   "velocity (sqrt(Vp Vs) for one layer; pick it with the ordinary velocity analysis on CCP gathers) "
                   "and gamma = Vp/Vs sets the non-hyperbolic terms. With gamma = 1 it is ordinary hyperbolic NMO. "
                   "Run it on CCP gathers (after 'Converted-wave CCP binning'); samples stretched beyond the limit "
                   "are muted. 'inverse' undoes it.")
    params = (Param("velocity", "velocity", "project", "C-wave velocity Vc", "project picks, a constant, 't:v,...' or "
                    "a file"),
              Param("gamma", "float", 2.0, "Vp/Vs (gamma)", minimum=1.0, maximum=10.0),
              Param("stretch_limit", "float", 0.5, "Stretch mute limit (0.5 = 50 %)", minimum=0.0),
              Param("inverse", "bool", False, "Inverse PS NMO"))

    def run(self, data, context):
        picks = resolve_velocity(self["velocity"], context)
        if data.has_header("cdp"):
            vf = velocity_per_trace(data, picks)
        else:
            vf = np.broadcast_to(picks.function(float(picks.cdps[0]), data.times)[None, :], data.traces.shape)
        off = data.header("offset")
        out = np.empty_like(data.traces)
        block = 2000
        for s in range(0, data.ntr, block):
            sl = slice(s, s + block)
            if self["inverse"]:
                out[sl] = ps_inverse_nmo(data.traces[sl], data.times, off[sl], vf[sl], self["gamma"])
            else:
                out[sl] = ps_nmo_correct(data.traces[sl], data.times, off[sl], vf[sl], self["gamma"],
                                         self["stretch_limit"])
            context.report(min(1.0, (s + block) / data.ntr))
        return data.copy(out)


# ---------------------------------------------------------------------------------------------- PS DMO (MZO)
def ps_migrate_offset(traces: np.ndarray, t0: float, dt: float, sx: np.ndarray, gx: np.ndarray, x_out: np.ndarray,
                      tau: np.ndarray, vp: float, gamma: float, aperture: float, max_angle_deg: float = 60.0,
                      progress=None) -> np.ndarray:
    """Kirchhoff time migration of PS traces to an image in PS vertical time tau (ntau, nx_out), constant Vp and
    gamma: the image point at depth z = vp tau / (1 + gamma) receives every trace at t = sqrt(z^2 + (x - s)^2) / vp +
    sqrt(z^2 + (x - g)^2) / vs, after the anti-causal half derivative of 2D migration."""
    vs = vp / gamma
    filt = rho_filter(traces.astype(np.float64), dt)
    z = tau[:, None] * vp / (1.0 + gamma)
    nt = traces.shape[1]
    image = np.zeros((tau.size, x_out.size))
    tan_max = np.tan(np.deg2rad(max_angle_deg))
    ccp = asymptotic_ccp(sx, gx, gamma)
    for i in range(traces.shape[0]):
        cols = np.flatnonzero(np.abs(x_out - ccp[i]) <= aperture)
        if cols.size == 0:
            continue
        xs = x_out[cols][None, :]
        t = np.hypot(z, xs - sx[i]) / vp + np.hypot(z, xs - gx[i]) / vs
        pos = (t - t0) / dt
        ok = (pos >= 0) & (pos < nt - 1) & (np.abs(xs - ccp[i]) <= np.maximum(z * tan_max, 1e-6))
        i0 = np.clip(np.floor(np.where(ok, pos, 0.0)).astype(np.int64), 0, nt - 2)
        fr = np.where(ok, pos - i0, 0.0)
        val = (1.0 - fr) * filt[i, i0] + fr * filt[i, i0 + 1]
        wgt = np.where(t > 0, tau[:, None] / np.maximum(t, 1e-9), 0.0)
        image[:, cols] += np.where(ok, val * wgt, 0.0)
        if progress is not None and i % 100 == 0:
            progress(i / traces.shape[0])
    return image


def ps_demigrate_zero_offset(image: np.ndarray, tau: np.ndarray, x: np.ndarray, vp: float, gamma: float,
                             aperture: float, t0: float, dt: float, nt: int) -> np.ndarray:
    """Zero-offset PS Kirchhoff modelling of a PS time image (ntau, nx) into traces (nx, nt): every image sample is
    spread along t = sqrt(z^2 + d^2) (1 + gamma) / vp (source and receiver at the same point, lateral distance d),
    then the causal half derivative of 2D modelling restores the wavelet."""
    ntau, nx = image.shape
    out = np.zeros(nt * nx)
    z = tau[:, None] * vp / (1.0 + gamma)
    slow = (1.0 + gamma) / vp
    steps = np.abs(np.diff(np.asarray(x, dtype=np.float64)))
    steps = steps[steps > 0]
    kmax = int(min(max(nx - 1, 0), np.ceil(aperture / float(steps.min())))) if steps.size else 0
    for k in range(-kmax, kmax + 1):
        j = np.arange(max(0, -k), min(nx, nx - k))
        if j.size == 0:
            continue
        jo = j + k
        d = np.abs(x[jo] - x[j])[None, :]
        t = np.hypot(z, d) * slow
        pos = (t - t0) / dt
        ok = (pos >= 0) & (pos < nt - 1) & (d <= aperture)
        if not np.any(ok):
            continue
        i0 = np.floor(np.where(ok, pos, 0.0)).astype(np.int64)
        fr = np.where(ok, pos - i0, 0.0)
        val = image[:, j] * np.where(t > 0, tau[:, None] / np.maximum(t, 1e-9), 0.0)
        flat = (i0 * nx + jo[None, :])[ok]
        out += np.bincount(flat, weights=((1.0 - fr) * val)[ok], minlength=nt * nx)
        out += np.bincount(flat + nx, weights=(fr * val)[ok], minlength=nt * nx)
    return rho_filter(out.reshape(nt, nx).T, dt, modelling=True)


@register
class PSDMO(Process):
    name = "ps_dmo"
    label = "PS DMO (converted-wave migration to zero offset)"
    category = "Converted waves"
    description = ("Converted-wave dip moveout as a migration to zero offset (the kinematic equivalent of PS NMO + "
                   "DMO + true CCP binning; Alfaro & Corrigan 1996 give an f-k form of the same operator): every "
                   "offset class of NMO-uncorrected PS data is migrated with constant Vp and gamma (Kirchhoff, "
                   "conversion-point traveltimes) and modelled back to zero offset. The output has one trace per "
                   "position of a regular grid and offset class (headers cdp, cdpx, offset), with flat reflections "
                   "at their zero-offset PS times: stack it by cdp. Slow on long lines; for small 2D lines.")
    params = (Param("vp", "float", 2000.0, "P velocity", unit="m/s", minimum=10.0),
              Param("gamma", "float", 2.0, "Vp/Vs (gamma)", minimum=1.01, maximum=10.0),
              Param("offset_class", "float", 0.0, "Offset class width (0 = 4 receiver spacings)", unit="m",
                    minimum=0.0),
              Param("bin_size", "float", 0.0, "Output spacing (0 = half the receiver spacing)", unit="m", minimum=0.0),
              Param("aperture", "float", 1000.0, "Aperture (half width)", unit="m", minimum=10.0),
              Param("max_angle", "float", 60.0, "Maximum angle", unit="deg", minimum=1.0, maximum=89.0))

    def run(self, data, context):
        gamma = _check_gamma(self["gamma"])
        vp = self["vp"]
        sx, gx = data.header("sx"), data.header("gx")
        if np.ptp(np.concatenate([sx, gx])) <= 0:
            raise ValueError("source and receiver positions (sx, gx) are not set: run a geometry step first")
        spacing = _receiver_spacing(data)
        size = self["bin_size"] if self["bin_size"] > 0 else 0.5 * spacing
        width = self["offset_class"] if self["offset_class"] > 0 else 4.0 * spacing
        ccp = asymptotic_ccp(sx, gx, gamma)
        x0 = float(np.floor(ccp.min() / size) * size)
        x_out = x0 + np.arange(int(np.ceil((ccp.max() - x0) / size)) + 1) * size
        off = np.abs(gx - sx)
        cls = np.floor(off / width).astype(np.int64)
        classes = np.unique(cls)
        tau = data.times
        blocks, heads = [], {"cdp": [], "cdpx": [], "offset": []}
        for n, c in enumerate(classes):
            idx = np.flatnonzero(cls == c)

            def report(f, n=n):
                context.report((n + 0.5 * f) / classes.size)

            image = ps_migrate_offset(data.traces[idx], data.t0, data.dt, sx[idx], gx[idx], x_out, tau, vp, gamma,
                                      self["aperture"], self["max_angle"], report)
            zero = ps_demigrate_zero_offset(image, tau, x_out, vp, gamma, self["aperture"], data.t0, data.dt, data.nt)
            rms_in = float(np.sqrt(np.mean(data.traces[idx].astype(np.float64) ** 2)))
            rms_out = float(np.sqrt(np.mean(zero ** 2)))
            if rms_out > 0:
                zero *= rms_in / rms_out
            live = np.flatnonzero(np.any(zero != 0, axis=1))
            blocks.append(zero[live])
            heads["cdp"].append(live + 1.0)
            heads["cdpx"].append(x_out[live])
            heads["offset"].append(np.full(live.size, (c + 0.5) * width))
            context.report((n + 1) / classes.size)
        traces = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, data.nt))
        if traces.shape[0] == 0:
            raise ValueError("no output: check Vp, gamma, the aperture and the geometry")
        h = {k: np.concatenate(v) for k, v in heads.items()}
        h.update({"sx": h["cdpx"], "gx": h["cdpx"], "ccpx": h["cdpx"]})
        out = SeismicData(traces.astype(np.float32), data.dt, h, data.t0, data.name + " PS DMO", "time",
                          list(data.history), dict(data.meta))
        out = out.sort(["cdp", "offset"])
        out.meta["geometry"] = dict(data.meta.get("geometry", {}), bin_size=size, ps_gamma=gamma)
        context.log(f"    PS migration to zero offset: {classes.size} offset classes of {width:g} m, {x_out.size} "
                    f"positions of {size:g} m")
        return out


# ---------------------------------------------------------------------------------------------- Vp/Vs analysis
def gamma_from_times(t_pp: np.ndarray, t_ps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average and interval Vp/Vs from the PP and PS two-way times of horizons (rows: positions, columns: horizons in
    depth order): gamma_avg = 2 t_PS / t_PP - 1 and, between horizons, gamma_int = 2 dt_PS / dt_PP - 1 (Gaiser 1996);
    NaN where the times do not increase."""
    t_pp = np.asarray(t_pp, dtype=np.float64)
    t_ps = np.asarray(t_ps, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(t_pp > 0, 2.0 * t_ps / t_pp - 1.0, np.nan)
        dpp = np.diff(np.concatenate([np.zeros((t_pp.shape[0], 1)), t_pp], axis=1), axis=1)
        dps = np.diff(np.concatenate([np.zeros((t_ps.shape[0], 1)), t_ps], axis=1), axis=1)
        inter = np.where((dpp > 0) & (dps > 0), 2.0 * dps / dpp - 1.0, np.nan)
    return avg, inter


def _horizon_times(ps, data: SeismicData, phases) -> np.ndarray:
    """Times (ntr, nphases) of picked horizons at the traces of the dataset (x from the pick set's key)."""
    if ps.x_key not in data.headers and ps.x_key != "tracl":
        raise ValueError(f"the dataset has no '{ps.x_key}' header for the pick set '{ps.name}'")
    xv = data.header(ps.x_key)
    gv = np.zeros(data.ntr) if ps.gather_key in ("none", "") else data.header(ps.gather_key)
    return np.stack([ps.interpolate(gv, xv, ph, use_abs_x=False) for ph in phases], axis=1)


@register
class VpVsAnalysis(Process):
    name = "vpvs_analysis"
    label = "Vp/Vs from PP and PS horizon times"
    category = "Converted waves"
    description = ("Vp/Vs (gamma) from the same horizons picked on a PP section and on a PS section: average gamma = "
                   "2 t_PS / t_PP - 1 and interval gamma = 2 dt_PS / dt_PP - 1 between horizons (Gaiser 1996). Pick the "
                   "horizons with the picking tool on the two stacks (one phase per horizon, same order or same names "
                   "in both pick sets). The output is either the interval-gamma section (on the PP time axis) or the "
                   "input PS section stretched to PP time with those gammas, for direct comparison with the PP stack; "
                   "the table goes to the results.")
    params = (Param("pp_picks", "picks", "PP horizons", "PP horizon picks"),
              Param("ps_picks", "picks", "PS horizons", "PS horizon picks"),
              Param("pairing", "choice", "by order", "Match horizons", choices=("by order", "by name")),
              Param("output", "choice", "interval Vp/Vs section", "Output",
                    choices=("interval Vp/Vs section", "PS section in PP time")))

    def run(self, data, context):
        pp = resolve_picks(self["pp_picks"], context)
        ps = resolve_picks(self["ps_picks"], context)
        if self["pairing"] == "by name":
            names = [p for p in pp.phase_order if p in ps.phase_order and pp.gathers(p).size and ps.gathers(p).size]
            pairs = [(p, p) for p in names]
        else:
            a = [p for p in pp.phase_order if pp.gathers(p).size]
            b = [p for p in ps.phase_order if ps.gathers(p).size]
            pairs = list(zip(a, b))
        if not pairs:
            raise ValueError("no horizon is picked in both pick sets")
        t_pp = _horizon_times(pp, data, [p for p, _ in pairs])
        t_ps = _horizon_times(ps, data, [q for _, q in pairs])
        order = np.argsort(np.nanmean(t_pp, axis=0))
        t_pp, t_ps = t_pp[:, order], t_ps[:, order]
        pairs = [pairs[k] for k in order]
        avg, inter = gamma_from_times(t_pp, t_ps)
        inter = np.where(np.isfinite(inter), inter, avg)
        times = data.times
        out = np.zeros(data.traces.shape)
        for i in range(data.ntr):
            ok = np.isfinite(t_pp[i]) & np.isfinite(t_ps[i]) & np.isfinite(inter[i])
            if not np.any(ok):
                continue
            tpp, tps, g = t_pp[i, ok], t_ps[i, ok], inter[i, ok]
            if self["output"] == "interval Vp/Vs section":
                k = np.clip(np.searchsorted(tpp, times, side="left"), 0, tpp.size - 1)
                out[i] = g[k]
            else:
                # PS time of every PP output time: piecewise linear through the horizons, last gamma below them
                xp = np.concatenate([[0.0], tpp])
                fp = np.concatenate([[0.0], tps])
                t_src = np.interp(times, xp, fp)
                below = times > xp[-1]
                t_src[below] = fp[-1] + (times[below] - xp[-1]) * (1.0 + g[-1]) / 2.0
                vals, valid = sample_at(data.traces[i:i + 1].astype(np.float64), t_src[None, :], data.t0, data.dt)
                out[i] = np.where(valid[0], vals[0], 0.0)
        name = data.name + (" interval Vp/Vs" if self["output"].startswith("interval") else " in PP time")
        res = data.copy(out.astype(np.float32), name=name)
        context.results["vpvs_analysis"] = {
            "horizons": [f"{a} / {b}" for a, b in pairs], "x_key": pp.x_key,
            "x": data.header(pp.x_key).tolist(), "t_pp": t_pp.tolist(), "t_ps": t_ps.tolist(),
            "gamma_avg": avg.tolist(), "gamma_int": inter.tolist()}
        med = np.nanmedian(inter, axis=0)
        context.log("    interval Vp/Vs (median along the line): " +
                    ", ".join(f"{a}: {m:.2f}" for (a, _), m in zip(pairs, med)))
        return res


# ---------------------------------------------------------------------------------------------- PS synthetics
def ps_reflection_coefficient(vp1, vs1, rho1, vp2, vs2, rho2, p) -> np.ndarray:
    """Linearised P-to-SV reflection coefficient at ray parameter p (s/m), Aki & Richards (1980):
    R = -(p a / (2 cos j)) [(1 - 2 b^2 p^2 + 2 b^2 cos i cos j / (a b)) drho / rho - (4 b^2 p^2 - 4 b^2 cos i cos j /
    (a b)) db / b], with a, b, rho the averages across the interface, i the P angle and j the S angle."""
    a, b, r = 0.5 * (vp1 + vp2), 0.5 * (vs1 + vs2), 0.5 * (rho1 + rho2)
    p = np.asarray(p, dtype=np.float64)
    if b <= 0:
        return np.zeros(p.shape)
    ci = np.sqrt(1.0 - np.clip(p * a, 0.0, 0.9999) ** 2)
    cj = np.sqrt(1.0 - np.clip(p * b, 0.0, 0.9999) ** 2)
    k = ci * cj / (a * b)
    t_rho = 1.0 - 2.0 * b * b * p * p + 2.0 * b * b * k
    t_vs = 4.0 * b * b * p * p - 4.0 * b * b * k
    return -(p * a / (2.0 * cj)) * (t_rho * (rho2 - rho1) / r - t_vs * (vs2 - vs1) / b)


def _ray_parameter_two(down, vd, up, vu, offset, vmax, iterations=64):
    """Ray parameter (s/m) per row for the horizontal distance ``offset`` of a ray with down-going legs (thickness
    down, velocities vd) and up-going legs (up, vu): bisection on p vmax in [0, 1) of the increasing X(p)."""
    lo = np.zeros(offset.size)
    hi = np.full(offset.size, 1.0 - 1e-12)
    for _ in range(iterations):
        u = 0.5 * (lo + hi)
        p = (u / vmax)[:, None]
        sd = np.where(down > 0, p * vd[None, :], 0.0)
        su = np.where(up > 0, p * vu[None, :], 0.0)
        x = (down * sd / np.sqrt(1.0 - sd ** 2)).sum(axis=1) + (up * su / np.sqrt(1.0 - su ** 2)).sum(axis=1)
        far = x > offset
        hi = np.where(far, u, hi)
        lo = np.where(far, lo, u)
    return 0.5 * (lo + hi) / vmax


def ps_flat_reflections(model, sx, sz, gx, gz, component: str = "radial"):
    """Times and amplitudes (ntr, ninterfaces) of P-SV reflections in flat layers by exact two-point ray tracing: P
    down from the source, S up to the receiver, one ray parameter (Snell's law). Amplitude: the linearised PS
    coefficient times the radial (cos) or vertical (sin) part of the S motion at the receiver, over a spherical
    divergence v_rms^2 t / vp_source. NaN where a station lies below the interface or an S leg crosses a fluid."""
    vp, vs, rho = model.properties()
    tops = np.array([la.top_z[0] for la in model.layers])
    n = tops.size
    sx, sz, gx, gz = (np.asarray(a, dtype=np.float64) for a in (sx, sz, gx, gz))
    off = np.abs(gx - sx)
    ntr = off.size
    times = np.full((ntr, max(n - 1, 0)), np.nan)
    amps = np.zeros((ntr, max(n - 1, 0)))
    src_layer = np.clip(np.searchsorted(tops, sz + 1e-6, side="right") - 1, 0, n - 1)
    rec_layer = np.clip(np.searchsorted(tops, gz + 1e-6, side="right") - 1, 0, n - 1)
    for k in range(1, n):
        zk = tops[k]
        lo, hi = tops[:k], tops[1:k + 1]
        down = np.clip(np.minimum(hi[None, :], zk) - np.maximum(lo[None, :], sz[:, None]), 0.0, None)
        up = np.clip(np.minimum(hi[None, :], zk) - np.maximum(lo[None, :], gz[:, None]), 0.0, None)
        vd, vu = vp[:k], vs[:k]
        fluid = np.any((up > 0) & (vu[None, :] <= 0), axis=1)
        vmax = np.maximum(np.max(np.where(down > 0, vd[None, :], 0.0), axis=1),
                          np.max(np.where(up > 0, vu[None, :], 0.0), axis=1))
        ok = (sz < zk) & (gz < zk) & (vmax > 0) & ~fluid & (vs[k - 1] > 0)
        if not np.any(ok):
            continue
        vu_safe = np.where(vu > 0, vu, 1.0)
        p = _ray_parameter_two(down, vd, np.where(vu[None, :] > 0, up, 0.0), vu_safe, off,
                               np.where(vmax > 0, vmax, 1.0))
        sd = np.where(down > 0, p[:, None] * vd[None, :], 0.0)
        su = np.where(up > 0, p[:, None] * vu_safe[None, :], 0.0)
        td = down / (vd[None, :] * np.sqrt(1.0 - sd ** 2))
        tu = np.where(up > 0, up / (vu_safe[None, :] * np.sqrt(1.0 - su ** 2)), 0.0)
        t = td.sum(axis=1) + tu.sum(axis=1)
        vrms2 = ((td * vd[None, :] ** 2).sum(axis=1) + (tu * vu_safe[None, :] ** 2).sum(axis=1)) / np.maximum(t, 1e-12)
        spread = vrms2 * t / vp[src_layer]
        r = ps_reflection_coefficient(vp[k - 1], vs[k - 1], rho[k - 1], vp[k], vs[k], rho[k], p)
        sin_rec = np.clip(p * vs[rec_layer], 0.0, 0.9999)
        part = np.sqrt(1.0 - sin_rec ** 2) if component == "radial" else sin_rec
        times[:, k - 1] = np.where(ok, t, np.nan)
        amps[:, k - 1] = np.where(ok, r * part / np.maximum(spread, 1e-9), 0.0)
    return times, amps


@register
class PSSynthetics(Process):
    name = "ps_synthetics"
    label = "Converted-wave ray synthetics (layered)"
    category = "Modelling"
    description = ("P-SV reflections of a flat layered model with the geometry of the dataset: exact two-point ray "
                   "tracing (P down, S up, Snell's law), linearised Aki-Richards PS coefficients, the radial or "
                   "vertical part of the S motion at the receiver, spherical divergence and a Ricker wavelet. The "
                   "radial component is positive away from the source, so a PS reflection keeps its polarity on both "
                   "sides of the source (it vanishes at zero offset). S velocities come from the model (vp / 1.73 "
                   "when not given). The event times are kept in the results.")
    params = (Param("model", "str", "2000, 2400, 2800 / 300, 700", "Model",
                    "flat layers 'vp1, vp2, ... / z1, z2, ...' (optionally '/ vs1, ... / rho1, ...') or a model file "
                    "with flat layers"),
              Param("component", "choice", "radial", "Component", choices=("radial", "vertical")),
              Param("receiver_depth", "float", 0.0, "Receiver depth below the surface", unit="m", minimum=0.0),
              Param("f0", "float", 20.0, "Ricker peak frequency", unit="Hz", minimum=1.0),
              Param("tmax", "float", 0.0, "Record length (0 = as input)", unit="s", minimum=0.0),
              Param("dt_ms", "float", 0.0, "Sample interval (0 = as input)", unit="ms", minimum=0.0))

    def run(self, data, context):
        from ..core.models import LayeredModel, model_from_text
        from .modelling import _finish, _stations, _time_axis
        model = model_from_text(self["model"], context.work_dir, context.depth_model)
        if not isinstance(model, LayeredModel) or not model.is_flat or len(model.layers) < 2:
            raise ValueError("PS synthetics need flat layers (at least two) given as text or in a model file")
        model, sx, sz, gx, gz = _stations(data, model, "model surface", self["receiver_depth"])
        nt, dt = _time_axis(data, self["tmax"], self["dt_ms"])
        times, amps = ps_flat_reflections(model, sx, sz, gx, gz, self["component"])
        good = np.isfinite(times)
        w, c = ricker(self["f0"], dt)
        traces = events_to_traces(np.where(good, times, -1.0), np.where(good, amps, 0.0), nt, dt, w, c)
        context.results[self.name] = {"times": times, "amplitudes": amps}
        return _finish(data, traces, dt, f"PS synthetics {self['component']}", self.name)
