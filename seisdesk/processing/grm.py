"""Palmer's generalised reciprocal method for a selected pair of refractor branches.

The velocity and time-depth functions use receiver pairs X,Y centred at a station:
tV=(tAY-tBX+tAB)/2; tG=(tAY+tBX-tAB-XY/Vr)/2. Depth conversion assumes a locally
horizontal interface and one overburden velocity. XY selection is a diagnostic,
not a unique geological solution; inspect every candidate in the results.
"""
import numpy as np

from .base import Param, Process, register, resolve_picks
from .refstatics import pick_geometry
from .statics import shift_traces


def grm_functions(xa, ta, xb, tb, reciprocal, separations, v1, spacing=0.0):
    xa, ta, xb, tb = map(lambda a: np.asarray(a, dtype=float), (xa, ta, xb, tb))
    ia, ib = np.argsort(xa), np.argsort(xb)
    xa, ta, xb, tb = xa[ia], ta[ia], xb[ib], tb[ib]
    if len(np.unique(xa)) != len(xa) or len(np.unique(xb)) != len(xb):
        raise ValueError("GRM needs one receiver pick per position in each shot")
    xy = np.asarray(separations, dtype=float)
    if not xy.size or np.any(xy < 0) or v1 <= 0 or reciprocal <= 0:
        raise ValueError("give nonnegative XY values, positive reciprocal time and overburden velocity")
    step = spacing or float(np.median(np.diff(np.unique(np.r_[xa, xb]))))
    lo = max(xa.min(), xb.min()) + xy.max() / 2
    hi = min(xa.max(), xb.max()) - xy.max() / 2
    if step <= 0 or hi - lo < 3 * step:
        raise ValueError("forward/reverse picks do not have enough common aperture for the requested XY values")
    x = np.arange(lo, hi + 0.1 * step, step)
    candidates = []
    for distance in xy:
        ay, bx = np.interp(x + distance / 2, xa, ta), np.interp(x - distance / 2, xb, tb)
        tv = (ay - bx + reciprocal) / 2
        slope, intercept = np.polyfit(x - x[0], tv, 1)
        velocity = 1 / slope if slope > 0 else np.inf
        if not np.isfinite(velocity) or velocity <= v1:
            continue
        tg = (ay + bx - reciprocal - distance / velocity) / 2
        residual = tv - (slope * (x - x[0]) + intercept)
        score = float(np.sqrt(np.mean(residual ** 2)) + 0.25 * np.sqrt(np.mean(np.diff(tg, n=2) ** 2)))
        depth = tg * v1 * velocity / np.sqrt(velocity ** 2 - v1 ** 2)
        candidates.append(dict(xy=float(distance), velocity=float(velocity), tv=tv, tg=tg, depth=depth, score=score))
    if not candidates:
        raise ValueError("no GRM candidate gives a refractor velocity above the overburden velocity")
    best = min(candidates, key=lambda c: c["score"])
    return {"x": x, "candidates": candidates, "best": best, "reciprocal_time": reciprocal, "v1": v1}


@register
class GRM(Process):
    name, label, category = "grm", "Generalised reciprocal method (refractor branches)", "Statics"
    description = ("Forward/reverse refractor picks give velocity and time-depth functions for candidate XY "
                   "separations. Reports all candidates; depth assumes locally flat layers and one overburden "
                   "velocity. Restrict picks to the same critically refracted branch in both directions.")
    params = (
        Param("pickset", "picks", "first breaks", "Pick set"),
        Param("phase", "str", "refractor 1", "Refractor phase"),
        Param("forward", "float", -1.0, "Forward gather (-1 = leftmost source)"),
        Param("reverse", "float", -1.0, "Reverse gather (-1 = rightmost source)"),
        Param("xy", "floats", [0, 10, 20, 30], "Candidate XY separations", unit="m"),
        Param("reciprocal_time", "float", 0.0, "Reciprocal time (0 = pick at opposing source)", unit="s", minimum=0.0),
        Param("v1", "float", 500.0, "Overburden velocity", unit="m/s", minimum=50.0),
        Param("apply", "bool", False, "Apply replacement statics"),
        Param("datum", "float", 0.0, "Datum elevation", unit="m"),
    )

    def run(self, data, context):
        if any(np.ptp(data.header(k)) > 1e-4 for k in ("sy", "gy")):
            raise ValueError("GRM currently requires a 2D line in x with constant y")
        ps = resolve_picks(self["pickset"], context)
        if ps.phase_type(self["phase"]) not in ("refraction", "unclassified"):
            raise ValueError("GRM requires a refractor phase")
        sx, gx, idx, times = pick_geometry(ps, data, self["phase"])
        gathers = data.header(ps.gather_key)[idx]
        left = float(gathers[np.argmin(sx)]) if self["forward"] < 0 else self["forward"]
        right = float(gathers[np.argmax(sx)]) if self["reverse"] < 0 else self["reverse"]
        a, b = gathers == left, gathers == right
        if left == right or a.sum() < 4 or b.sum() < 4:
            raise ValueError("choose distinct forward and reverse shots with at least four refractor picks each")
        if np.ptp(sx[a]) > 1e-4 or np.ptp(sx[b]) > 1e-4 or sx[a][0] >= sx[b][0]:
            raise ValueError("forward source must lie left of reverse source, with consistent shot geometry")
        reciprocal = self["reciprocal_time"]
        if reciprocal == 0:
            xa, xb = sx[a][0], sx[b][0]
            if not (gx[a].min() <= xb <= gx[a].max() and gx[b].min() <= xa <= gx[b].max()):
                raise ValueError("opposing source positions are outside picked receiver ranges; enter reciprocal time")
            ia, ib = np.argsort(gx[a]), np.argsort(gx[b])
            reciprocal = 0.5 * (np.interp(xb, gx[a][ia], times[a][ia]) + np.interp(xa, gx[b][ib], times[b][ib]))
        result = grm_functions(gx[a], times[a], gx[b], times[b], reciprocal, self["xy"], self["v1"])
        best = result["best"]
        context.log(f"    GRM: XY {best['xy']:g} m, refractor velocity {best['velocity']:.1f} m/s")
        out = data.copy()
        if self["apply"]:
            if np.any(best["depth"] < 0):
                raise ValueError("negative GRM depths: check reciprocal time/phase selection before applying statics")
            shifts = []
            for position, elevation in ((data.header("sx"), data.header("selev") - data.header("sdepth")),
                                        (data.header("gx"), data.header("gelev"))):
                if position.min() < result["x"][0] or position.max() > result["x"][-1]:
                    raise ValueError("GRM statics cannot extrapolate beyond the reciprocal overlap; select traces within it")
                depth = np.interp(position, result["x"], best["depth"])
                shifts.append(-(depth / self["v1"] + (elevation - depth - self["datum"]) / best["velocity"]))
            ss, gs = shifts
            out.traces = shift_traces(out.traces, data.dt, ss + gs)
            out.set_header("sstat", ss * 1000); out.set_header("gstat", gs * 1000)
            out.set_header("tstat", data.header("tstat") + 1000 * (ss + gs))
        context.results["grm"] = result
        return out
