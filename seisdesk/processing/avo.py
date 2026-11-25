"""AVO analysis: incidence angles of NMO-corrected CMP gathers, angle mutes and angle gathers, angle (range-limited)
stacks, intercept-gradient attributes, and two-layer AVO modelling with exact Zoeppritz or Aki-Richards reflection
coefficients."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..core.dataset import SeismicData
from ..core.synthetic import events_to_traces, ricker
from ..core.velocity import dix
from .base import Param, Process, cdp_values, register, resolve_velocity

# ---------------------------------------------------------------------------------------------- reflection coefficients


def zoeppritz_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta_deg) -> np.ndarray:
    """Exact P-P reflection coefficient of a plane interface (the explicit solution of the Zoeppritz equations,
    Aki & Richards 1980, eq. 5.40) for incidence angles theta_deg (degrees) in the upper medium. Complex beyond the
    critical angle; the shear velocities must be positive."""
    a1, b1, r1, a2, b2, r2 = (float(v) for v in (vp1, vs1, rho1, vp2, vs2, rho2))
    if min(a1, b1, r1, a2, b2, r2) <= 0:
        raise ValueError("velocities and densities must be positive (the exact solution needs Vs > 0)")
    th = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    p = np.sin(th) / a1
    p2 = p * p

    def cos_of(v):
        return np.sqrt((1.0 - p2 * v * v).astype(np.complex128))

    ci1, ci2, cj1, cj2 = cos_of(a1), cos_of(a2), cos_of(b1), cos_of(b2)
    a = r2 * (1.0 - 2.0 * b2 * b2 * p2) - r1 * (1.0 - 2.0 * b1 * b1 * p2)
    b = r2 * (1.0 - 2.0 * b2 * b2 * p2) + 2.0 * r1 * b1 * b1 * p2
    c = r1 * (1.0 - 2.0 * b1 * b1 * p2) + 2.0 * r2 * b2 * b2 * p2
    d = 2.0 * (r2 * b2 * b2 - r1 * b1 * b1)
    e = b * ci1 / a1 + c * ci2 / a2
    f = b * cj1 / b1 + c * cj2 / b2
    g = a - d * (ci1 / a1) * (cj2 / b2)
    h = a - d * (ci2 / a2) * (cj1 / b1)
    den = e * f + g * h * p2
    return ((b * ci1 / a1 - c * ci2 / a2) * f - (a + d * (ci1 / a1) * (cj2 / b2)) * h * p2) / den


def aki_richards_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta_deg) -> np.ndarray:
    """Linearised P-P reflection coefficient for small contrasts (Aki & Richards 1980): average properties, the
    angle the average of the incidence and transmission angles. Valid well below the critical angle."""
    a1, b1, r1, a2, b2, r2 = (float(v) for v in (vp1, vs1, rho1, vp2, vs2, rho2))
    th1 = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    p = np.sin(th1) / a1
    th2 = np.arcsin(np.clip(p * a2, -1.0, 1.0))
    th = 0.5 * (th1 + th2)
    a, b, r = 0.5 * (a1 + a2), 0.5 * (b1 + b2), 0.5 * (r1 + r2)
    da, db, dr = a2 - a1, b2 - b1, r2 - r1
    q = 4.0 * b * b * p * p
    return 0.5 * (1.0 - q) * dr / r + da / (2.0 * a * np.cos(th) ** 2) - q * db / b


def shuey_terms(vp1, vs1, rho1, vp2, vs2, rho2) -> tuple[float, float, float]:
    """Intercept A, gradient B and curvature C of R(theta) = A + B sin^2 + C sin^2 tan^2 (Shuey 1985, in the form
    rearranged from Aki & Richards): A = (dVp/Vp + drho/rho) / 2, B = dVp/(2Vp) - 2 (Vs/Vp)^2 (drho/rho + 2 dVs/Vs),
    C = dVp/(2Vp), with average properties."""
    a1, b1, r1, a2, b2, r2 = (float(v) for v in (vp1, vs1, rho1, vp2, vs2, rho2))
    a, b, r = 0.5 * (a1 + a2), 0.5 * (b1 + b2), 0.5 * (r1 + r2)
    da, db, dr = (a2 - a1) / a, (b2 - b1) / b, (r2 - r1) / r
    k2 = (b / a) ** 2
    return 0.5 * (da + dr), 0.5 * da - 2.0 * k2 * (dr + 2.0 * db), 0.5 * da


def avo_class(a: float, b: float, theta_max_deg: float = 35.0, zero: float = 0.02) -> str:
    """AVO class of an interface from its intercept and gradient (Rutherford & Williams 1989; class IIp after Ross &
    Kinman 1995, class IV after Castagna & Swan 1997): I high-impedance (A > zero), II near-zero intercept (|A| <=
    zero; IIp when the small positive intercept reverses polarity within theta_max), III low-impedance with the
    amplitude growing with angle (A < -zero, B < 0), IV low-impedance with the amplitude falling (A < -zero, B >= 0)."""
    if a > zero:
        return "I"
    if a >= -zero:
        reverses = a > 0 and a + b * np.sin(np.deg2rad(theta_max_deg)) ** 2 < 0
        return "IIp" if reverses else "II"
    return "III" if b < 0 else "IV"


# ---------------------------------------------------------------------------------------------- incidence angles
def incidence_angles(times: np.ndarray, offsets: np.ndarray, vrms: np.ndarray, vint: np.ndarray) -> np.ndarray:
    """Incidence angle (degrees) at every sample of an NMO-corrected CMP gather: straight-ray sin(theta) =
    x v_int / (t v_rms^2) with t = sqrt(t0^2 + x^2 / v_rms^2) the traveltime before NMO (Walden 1991). vrms and vint
    are functions of the zero-offset time t0; (ntr, nt) result, NaN where the ray cannot reach (sin >= 1)."""
    t0 = np.asarray(times, dtype=np.float64)[None, :]
    x = np.abs(np.asarray(offsets, dtype=np.float64))[:, None]
    vr = np.asarray(vrms, dtype=np.float64)[None, :]
    vi = np.asarray(vint, dtype=np.float64)[None, :]
    t = np.sqrt(t0 * t0 + (x / vr) ** 2)
    s = np.divide(x * vi, t * vr * vr, out=np.zeros(np.broadcast(t, x).shape), where=t > 0)
    with np.errstate(invalid="ignore"):
        return np.where(s < 1.0, np.degrees(np.arcsin(np.minimum(s, 1.0))), np.nan)


class _AngleField:
    """Velocity functions per CMP from the picks: v_rms(t0) and the (smoothed) Dix interval velocity."""

    def __init__(self, data: SeismicData, picks, smooth: float):
        if data.domain != "time":
            raise ValueError("AVO steps need time data (NMO-corrected CMP gathers)")
        if not data.has_header("offset"):
            raise ValueError("AVO steps need source-receiver offsets: run a geometry step")
        self.times = data.times
        self.cdps = np.unique(cdp_values(data))
        self.vrms = picks.field(self.cdps, self.times)
        n = int(round(smooth / data.dt))
        self.vint = np.empty_like(self.vrms)
        for j in range(self.cdps.size):
            vi = dix(self.times, self.vrms[j])                 # absolute two-way times, also for a late first sample
            self.vint[j] = uniform_filter1d(vi, n, mode="nearest") if n > 1 else vi

    def angles(self, cdp: float, offsets: np.ndarray) -> np.ndarray:
        j = int(np.clip(np.searchsorted(self.cdps, cdp), 0, self.cdps.size - 1))
        return incidence_angles(self.times, offsets, self.vrms[j], self.vint[j])


def _warn_without_nmo(data: SeismicData, context) -> None:
    if not any("nmo" in str(h).lower() for h in data.history):
        context.log("    note: no NMO step in the history of this dataset; AVO steps expect NMO-corrected gathers")


def _cmp_headers(data: SeismicData, groups) -> dict:
    """One header row per CMP (the first trace of each), offsets zero, positions at the CMP."""
    heads = {k: np.array([data.headers[k][idx[0]] for _, idx in groups]) for k in data.headers}
    heads["cdp"] = np.array([v for v, _ in groups], dtype=np.float64)
    heads["offset"] = np.zeros(len(groups))
    if "cdpx" in heads and np.any(heads["cdpx"]):
        heads["sx"] = heads["cdpx"].copy()
        heads["gx"] = heads["cdpx"].copy()
    return heads


def angle_weight(theta: np.ndarray, theta_min: float, theta_max: float, taper: float) -> np.ndarray:
    """1 inside [theta_min, theta_max], a cosine taper of width `taper` degrees outside, 0 beyond and where NaN."""
    th = np.nan_to_num(theta, nan=1e9)
    w = ((th >= theta_min) & (th <= theta_max)).astype(np.float64)
    if taper > 0:
        lo = (th < theta_min) & (th > theta_min - taper)
        hi = (th > theta_max) & (th < theta_max + taper)
        w[lo] = 0.5 * (1.0 + np.cos(np.pi * (theta_min - th[lo]) / taper))
        w[hi] = 0.5 * (1.0 + np.cos(np.pi * (th[hi] - theta_max) / taper))
    return w


def parse_ranges(text: str) -> list[tuple[float, float]]:
    """'0-12; 12-24; 24-36' -> [(0, 12), (12, 24), (24, 36)] (degrees)."""
    out = []
    for part in str(text).replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if not sep:
            raise ValueError(f"angle range {part!r}: write it as 'from-to' in degrees")
        a, b = float(lo), float(hi)
        if not (0.0 <= a < b <= 90.0):
            raise ValueError(f"angle range {part!r} must satisfy 0 <= from < to <= 90")
        out.append((a, b))
    if not out:
        raise ValueError("give at least one angle range, e.g. '0-12; 12-24; 24-36'")
    return out


VELOCITY = Param("velocity", "velocity", "project", "Velocity (RMS)", "project picks, a constant, 't:v,...' or a file")
SMOOTH = Param("smooth", "float", 0.1, "Interval-velocity smoothing", unit="s", minimum=0.0)


# ---------------------------------------------------------------------------------------------- steps
@register
class OffsetToAngle(Process):
    name = "offset_to_angle"
    label = "Angle gathers (offset to incidence angle)"
    category = "AVO"
    description = ("For NMO-corrected CMP gathers: the incidence angle of every sample from the RMS velocities and the "
                   "Dix interval velocity (straight rays, Walden 1991). 'angle mute' keeps the samples between the two "
                   "angles (cosine taper outside); 'angle gathers' averages the samples into angle bins, one trace per "
                   "bin with the bin centre in header 'angle'.")
    params = (VELOCITY,
              Param("output", "choice", "angle mute", "Output", choices=("angle mute", "angle gathers")),
              Param("theta_min", "float", 0.0, "Smallest angle", unit="deg", minimum=0.0, maximum=89.0),
              Param("theta_max", "float", 35.0, "Largest angle", unit="deg", minimum=1.0, maximum=89.0),
              Param("taper", "float", 2.0, "Mute taper", unit="deg", minimum=0.0),
              Param("angle_step", "float", 5.0, "Angle bin width (angle gathers)", unit="deg", minimum=0.5),
              SMOOTH)

    def run(self, data, context):
        if self["theta_max"] <= self["theta_min"]:
            raise ValueError("the largest angle must exceed the smallest")
        _warn_without_nmo(data, context)
        field = _AngleField(data, resolve_velocity(self["velocity"], context), self["smooth"])
        off = data.header("offset")
        keep_angles = data.ntr * data.nt <= 5_000_000
        all_angles = np.full(data.traces.shape, np.nan, np.float32) if keep_angles else None
        groups = list(data.iter_gathers("cdp", "offset"))
        if self["output"] == "angle mute":
            out = data.traces.astype(np.float64).copy()
            for n, (c, idx) in enumerate(groups):
                th = field.angles(c, off[idx])
                out[idx] *= angle_weight(th, self["theta_min"], self["theta_max"], self["taper"])
                if all_angles is not None:
                    all_angles[idx] = th
                context.report((n + 1) / len(groups))
            result = data.copy(out.astype(np.float32))
        else:
            edges = np.arange(self["theta_min"], self["theta_max"] + 1e-9, self["angle_step"])
            if edges[-1] < self["theta_max"] - 1e-6:          # a last, narrower bin up to the largest angle
                edges = np.append(edges, self["theta_max"])
            centres = 0.5 * (edges[:-1] + edges[1:])
            nb = centres.size
            traces = np.zeros((len(groups) * nb, data.nt))
            fold = np.zeros(len(groups) * nb)
            for n, (c, idx) in enumerate(groups):
                g = data.traces[idx].astype(np.float64)
                th = np.nan_to_num(field.angles(c, off[idx]), nan=1e9)
                if all_angles is not None:
                    all_angles[idx] = np.where(th < 1e8, th, np.nan)
                live = g != 0
                for k in range(nb):
                    m = live & (th >= edges[k]) & (th < edges[k + 1])
                    cnt = m.sum(axis=0)
                    traces[n * nb + k] = np.where(cnt > 0, (g * m).sum(axis=0) / np.maximum(cnt, 1), 0.0)
                    fold[n * nb + k] = cnt.max()
                context.report((n + 1) / len(groups))
            base = _cmp_headers(data, groups)
            heads = {k: np.repeat(v, nb) for k, v in base.items()}
            heads["angle"] = np.tile(centres, len(groups))
            heads["tracf"] = np.tile(np.arange(1, nb + 1, dtype=np.float64), len(groups))
            heads["nhs"] = fold
            heads["tracl"] = np.arange(1, traces.shape[0] + 1, dtype=np.float64)
            result = SeismicData(traces.astype(np.float32), data.dt, heads, data.t0, data.name + " angle gathers",
                                 data.domain, list(data.history), dict(data.meta))
        context.results[self.name] = {"output": self["output"], "theta_min": self["theta_min"],
                                      "theta_max": self["theta_max"], "angles": all_angles,
                                      "cdp": field.cdps.tolist()}
        if all_angles is None:
            context.log("    (angles of every sample not kept in the results: dataset too large)")
        return result


@register
class AngleStack(Process):
    name = "angle_stack"
    label = "Angle (range-limited) stack"
    category = "AVO"
    description = ("Stacks NMO-corrected CMP gathers within incidence-angle ranges (near, mid, far, ...): one section "
                   "per range, one trace per CMP, the sections one after the other with the range centre in header "
                   "'angle' (display them with gather key 'angle'). Only live (non-zero) samples are averaged.")
    params = (VELOCITY,
              Param("ranges", "str", "0-12; 12-24; 24-36", "Angle ranges (degrees)", "from-to, separated by ';'"),
              SMOOTH)

    def run(self, data, context):
        ranges = parse_ranges(self["ranges"])
        _warn_without_nmo(data, context)
        field = _AngleField(data, resolve_velocity(self["velocity"], context), self["smooth"])
        off = data.header("offset")
        groups = list(data.iter_gathers("cdp", "offset"))
        nr, nc = len(ranges), len(groups)
        traces = np.zeros((nr, nc, data.nt))
        fold = np.zeros((nr, nc))
        for j, (c, idx) in enumerate(groups):
            g = data.traces[idx].astype(np.float64)
            th = np.nan_to_num(field.angles(c, off[idx]), nan=1e9)
            live = g != 0
            for r, (lo, hi) in enumerate(ranges):
                m = live & (th >= lo) & (th < hi)
                cnt = m.sum(axis=0)
                traces[r, j] = np.where(cnt > 0, (g * m).sum(axis=0) / np.maximum(cnt, 1), 0.0)
                fold[r, j] = cnt.max()
            context.report((j + 1) / nc)
        base = _cmp_headers(data, groups)
        heads = {k: np.tile(v, nr) for k, v in base.items()}
        heads["angle"] = np.repeat([0.5 * (lo + hi) for lo, hi in ranges], nc)
        heads["angle_min"] = np.repeat([lo for lo, _ in ranges], nc)
        heads["angle_max"] = np.repeat([hi for _, hi in ranges], nc)
        heads["nhs"] = fold.reshape(-1)
        heads["tracl"] = np.arange(1, nr * nc + 1, dtype=np.float64)
        out = SeismicData(traces.reshape(nr * nc, data.nt).astype(np.float32), data.dt, heads, data.t0,
                          data.name + " angle stacks", data.domain, list(data.history), dict(data.meta))
        out.meta["stacked"] = True
        context.results[self.name] = {"ranges": ranges, "cdps": int(nc)}
        return out


ATTRIBUTES = ("intercept A", "gradient B", "curvature C (three-term)", "A x B", "pseudo-Poisson A + B",
              "fluid factor", "chi projection (EEI)", "fit error")


def fit_avo(g: np.ndarray, theta: np.ndarray, theta_min: float, theta_max: float, three_term: bool,
            min_traces: int = 3):
    """Least-squares fit per time sample of an NMO-corrected gather g (ntr, nt) with incidence angles theta
    (degrees): g = A + B sin^2 (+ C sin^2 tan^2) over the live samples with theta_min <= theta <= theta_max.
    Returns A, B, C (zeros for two terms) and the relative RMS misfit, each (nt,); zero where the fit is not
    determined (too few traces or too narrow an angle range)."""
    th = np.nan_to_num(theta, nan=1e9)
    w = ((g != 0) & (th >= theta_min) & (th <= theta_max)).astype(np.float64)
    rad = np.deg2rad(np.where(w > 0, th, 0.0))
    s2 = np.sin(rad) ** 2
    f = [np.ones_like(s2), s2, s2 * np.tan(rad) ** 2]
    m = 3 if three_term else 2
    nt = g.shape[1]
    n_mat = np.empty((nt, m, m))
    rhs = np.empty((nt, m))
    for k in range(m):
        rhs[:, k] = (w * f[k] * g).sum(axis=0)
        for l in range(k, m):
            n_mat[:, k, l] = n_mat[:, l, k] = (w * f[k] * f[l]).sum(axis=0)
    count = w.sum(axis=0)
    scale = np.maximum(np.einsum("tkk->t", n_mat) / m, 1e-30)
    det = np.linalg.det(n_mat / scale[:, None, None])
    ok = (count >= max(min_traces, m)) & (det > 1e-9)
    coef = np.zeros((nt, m))
    if ok.any():
        reg = n_mat[ok] + 1e-12 * scale[ok][:, None, None] * np.eye(m)[None, :, :]
        coef[ok] = np.linalg.solve(reg, rhs[ok][:, :, None])[:, :, 0]
    pred = sum(coef[None, :, k] * f[k] for k in range(m))
    num = (w * (g - pred) ** 2).sum(axis=0)
    den = (w * g * g).sum(axis=0)
    err = np.where(ok & (den > 0), np.sqrt(num / np.maximum(den, 1e-30)), 0.0)
    a, b = coef[:, 0], coef[:, 1]
    c = coef[:, 2] if three_term else np.zeros(nt)
    return a, b, c, err


def fluid_factor(a: np.ndarray, b: np.ndarray, vp_vs: float = 2.0, mudrock_slope: float = 1.16) -> np.ndarray:
    """Smith & Gidlow (1987) fluid factor dF = dVp/Vp - m (Vs/Vp) dVs/Vs, with dVp/Vp and dVs/Vs from intercept and
    gradient under Gardner's density relation (drho/rho = dVp/(4 Vp)) and a background Vp/Vs ratio; m is the slope
    of the mudrock line (Castagna et al. 1985)."""
    k = 1.0 / vp_vs
    dvp = 1.6 * a
    dvs = ((0.5 - 0.5 * k * k) * dvp - b) / (4.0 * k * k)
    return dvp - mudrock_slope * k * dvs


@register
class AVOAttributes(Process):
    name = "avo_attributes"
    label = "AVO attributes (intercept and gradient)"
    category = "AVO"
    description = ("Least-squares fit of every sample of NMO-corrected CMP gathers against the incidence angle: "
                   "R = A + B sin^2 (Shuey 1985 two-term) or with the third term C sin^2 tan^2. Output one attribute "
                   "section (one trace per CMP): intercept A, gradient B, curvature C, A x B, pseudo-Poisson "
                   "reflectivity A + B, the Smith & Gidlow (1987) fluid factor, the chi-angle projection A cos(chi) + "
                   "B sin(chi) (extended elastic impedance reflectivity, Whitcombe et al. 2002) or the relative fit "
                   "error. Intercept and gradient of the whole line are kept for the AVO crossplot.")
    params = (VELOCITY,
              Param("terms", "choice", "two-term", "Fit", choices=("two-term", "three-term")),
              Param("output", "choice", "intercept A", "Output attribute", choices=ATTRIBUTES),
              Param("theta_min", "float", 0.0, "Smallest angle used", unit="deg", minimum=0.0, maximum=80.0),
              Param("theta_max", "float", 35.0, "Largest angle used", unit="deg", minimum=5.0, maximum=80.0),
              Param("min_traces", "int", 4, "Fewest live traces for a fit", minimum=2),
              Param("vp_vs", "float", 2.0, "Background Vp/Vs (fluid factor)", minimum=1.2, maximum=5.0),
              Param("mudrock_slope", "float", 1.16, "Mudrock-line slope (fluid factor)", minimum=0.1),
              Param("chi", "float", 20.0, "Chi angle (projection)", unit="deg", minimum=-90.0, maximum=90.0),
              SMOOTH)

    def run(self, data, context):
        if self["theta_max"] <= self["theta_min"]:
            raise ValueError("the largest angle must exceed the smallest")
        _warn_without_nmo(data, context)
        field = _AngleField(data, resolve_velocity(self["velocity"], context), self["smooth"])
        off = data.header("offset")
        groups = list(data.iter_gathers("cdp", "offset"))
        three = self["terms"] == "three-term"
        nc = len(groups)
        a_all = np.zeros((nc, data.nt), np.float32)
        b_all = np.zeros((nc, data.nt), np.float32)
        out = np.zeros((nc, data.nt))
        for j, (c, idx) in enumerate(groups):
            g = data.traces[idx].astype(np.float64)
            a, b, cc, err = fit_avo(g, field.angles(c, off[idx]), self["theta_min"], self["theta_max"], three,
                                    self["min_traces"])
            a_all[j], b_all[j] = a, b
            out[j] = self._attribute(a, b, cc, err)
            context.report((j + 1) / nc)
        heads = _cmp_headers(data, groups)
        heads["nhs"] = np.array([idx.size for _, idx in groups], dtype=np.float64)
        heads["tracl"] = np.arange(1, nc + 1, dtype=np.float64)
        res = SeismicData(out.astype(np.float32), data.dt, heads, data.t0, f"{data.name} AVO {self['output']}",
                          data.domain, list(data.history), dict(data.meta))
        res.meta["stacked"] = True
        context.results[self.name] = {"output": self["output"], "cdp": heads["cdp"].tolist(), "A": a_all, "B": b_all,
                                      "t0": data.t0, "dt": data.dt, "terms": self["terms"]}
        return res

    def _attribute(self, a, b, c, err):
        name = self["output"]
        if name == "intercept A":
            return a
        if name == "gradient B":
            return b
        if name.startswith("curvature"):
            return c
        if name == "A x B":
            return a * b
        if name.startswith("pseudo-Poisson"):
            return a + b
        if name == "fluid factor":
            return fluid_factor(a, b, self["vp_vs"], self["mudrock_slope"])
        if name.startswith("chi"):
            chi = np.deg2rad(self["chi"])
            return a * np.cos(chi) + b * np.sin(chi)
        return err


@register
class AVOModelling(Process):
    name = "avo_modelling"
    label = "AVO modelling (two-layer interface)"
    category = "AVO"
    description = ("A synthetic CMP gather of one interface between two layers: reflection coefficients from the exact "
                   "Zoeppritz solution or the Aki-Richards approximation, incidence angles by straight rays in the "
                   "upper layer (tan theta = x / (Vp1 t0)), a Ricker wavelet. The events are flat (NMO-corrected) or "
                   "on their hyperbola. Offsets come from the input traces, or from 0 to the maximum offset when it is "
                   "set. Intercept, gradient and the AVO class of the interface go to the results.")
    params = (Param("vp1", "float", 2190.0, "Upper layer Vp", unit="m/s", minimum=100.0),
              Param("vs1", "float", 820.0, "Upper layer Vs", unit="m/s", minimum=10.0),
              Param("rho1", "float", 2.16, "Upper layer density", unit="g/cc", minimum=0.1),
              Param("vp2", "float", 1950.0, "Lower layer Vp", unit="m/s", minimum=100.0),
              Param("vs2", "float", 1150.0, "Lower layer Vs", unit="m/s", minimum=10.0),
              Param("rho2", "float", 1.95, "Lower layer density", unit="g/cc", minimum=0.1),
              Param("method", "choice", "Zoeppritz (exact)", "Reflection coefficients",
                    choices=("Zoeppritz (exact)", "Aki-Richards")),
              Param("t0", "float", 0.5, "Zero-offset time of the interface", unit="s", minimum=0.0),
              Param("f0", "float", 30.0, "Ricker peak frequency", unit="Hz", minimum=1.0),
              Param("moveout", "choice", "flat (NMO-corrected)", "Events",
                    choices=("flat (NMO-corrected)", "hyperbolic")),
              Param("max_offset", "float", 0.0, "Largest offset (0 = offsets of the input)", unit="m", minimum=0.0),
              Param("offset_step", "float", 50.0, "Offset step (with a largest offset)", unit="m", minimum=1.0),
              Param("tmax", "float", 0.0, "Record length (0 = as the input)", unit="s", minimum=0.0))

    def run(self, data, context):
        dt = data.dt
        if self["max_offset"] > 0:
            x = np.arange(0.0, self["max_offset"] + 0.5 * self["offset_step"], self["offset_step"])
            heads = {"fldr": 1.0, "cdp": 1.0, "offset": x, "sx": -0.5 * x, "gx": 0.5 * x,
                     "tracf": np.arange(1, x.size + 1, dtype=np.float64)}
            t0_data = 0.0
        else:
            x = np.abs(data.header("offset"))
            heads = {k: v.copy() for k, v in data.headers.items()}
            t0_data = data.t0
        nt = int(round(self["tmax"] / dt)) + 1 if self["tmax"] > 0 else data.nt
        props = (self["vp1"], self["vs1"], self["rho1"], self["vp2"], self["vs2"], self["rho2"])
        t0 = self["t0"]
        theta = np.degrees(np.arctan2(x, self["vp1"] * t0)) if t0 > 0 else np.where(x > 0, 90.0, 0.0)
        if self["method"].startswith("Zoeppritz"):
            r_complex = zoeppritz_rpp(*props, theta)
            r = np.real(r_complex)
        else:
            r = aki_richards_rpp(*props, theta)
        times = np.full(x.size, t0) if self["moveout"].startswith("flat") else np.sqrt(t0 ** 2 + (x / self["vp1"]) ** 2)
        w, c = ricker(self["f0"], dt)
        traces = events_to_traces((times - t0_data)[:, None], r[:, None], nt, dt, w, c)
        a, b, cc = shuey_terms(*props)
        crit = np.degrees(np.arcsin(self["vp1"] / self["vp2"])) if self["vp2"] > self["vp1"] else None
        klass = avo_class(a, b, float(theta.max()) if theta.size else 35.0)
        context.results[self.name] = {"A": a, "B": b, "C": cc, "class": klass, "angles": theta.tolist(),
                                      "rpp": r.tolist(), "critical_angle": crit}
        context.log(f"    intercept {a:+.4f}, gradient {b:+.4f}: class {klass}"
                    + (f"; critical angle {crit:.1f} deg" if crit else ""))
        return SeismicData(traces.astype(np.float32), dt, heads, t0_data, f"AVO model (class {klass})", "time",
                           list(data.history), {})
