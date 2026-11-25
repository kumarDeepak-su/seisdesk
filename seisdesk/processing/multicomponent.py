"""Multicomponent processing: horizontal rotation of three-component (3C) recordings to radial and transverse,
Alford rotation of four-component (4C) shear data, polarisation analysis and polarisation filtering.

Components are told apart by a header, by default the SEG-Y trace identification code trid (12 vertical, 13
crossline, 14 inline; after rotation 15 vertical, 16 transverse, 17 radial). The traces recorded together at one
station for one shot are matched by other headers (fldr and tracf by default; fldr,gx when the channels are numbered
per component)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .base import Param, Process, register

VERTICAL, CROSSLINE, INLINE = 12, 13, 14
ROTATED_VERTICAL, TRANSVERSE, RADIAL = 15, 16, 17
ATTRIBUTES = ("rectilinearity", "planarity", "azimuth", "incidence", "largest eigenvalue")
MATCH_LABEL = "Headers matching the components of one station"


def match_components(data, component_key: str, match_keys, codes) -> list[dict]:
    """Traces recorded together (equal values of every match key) as {component code: trace index}, in the order of
    their first trace; only groups holding every code in ``codes`` are returned."""
    comp = np.rint(data.header(component_key)).astype(np.int64)
    keys = [data.header(k) for k in match_keys]
    wanted = {int(c) for c in codes}
    groups, order = {}, []
    for i in range(data.ntr):
        c = int(comp[i])
        if c not in wanted:
            continue
        tag = tuple(round(float(k[i]), 6) for k in keys)
        g = groups.get(tag)
        if g is None:
            g = groups[tag] = {}
            order.append(tag)
        if c in g:
            raise ValueError(f"component code {c} occurs twice for station {tag}; use more specific matching headers")
        g[c] = i
    return [groups[t] for t in order if wanted.issubset(groups[t])]


def _require(groups, what: str, component_key: str, match_keys) -> None:
    if not groups:
        raise ValueError(f"no complete {what} groups: check the component header {component_key!r} and its codes, and "
                         f"the headers that match the components of one station ({','.join(match_keys) or 'none'}; "
                         "try fldr,gx when the channels are numbered per component)")


def _codes(values, n: int, what: str) -> list[int]:
    codes = [int(round(v)) for v in values]
    if len(codes) != n or len(set(codes)) != n:
        raise ValueError(f"give {n} different component codes ({what})")
    return codes


def _window(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    sel = times >= tmin
    if tmax > 0:
        sel &= times <= tmax
    if not sel.any():
        raise ValueError("the time window holds no samples")
    return sel


# ---------------------------------------------------------------------------------------------- horizontal rotation
def rotate_horizontal(inline, crossline, alpha):
    """Radial and transverse traces from the inline and crossline traces; alpha (radians) is the angle from the
    inline axis to the radial direction, anticlockwise (towards the crossline axis)."""
    c, s = np.cos(alpha), np.sin(alpha)
    return c * inline + s * crossline, -s * inline + c * crossline


@register
class HorizontalRotation(Process):
    name = "hrot"
    label = "Horizontal rotation (3C)"
    category = "Multicomponent"
    description = ("Rotates the two horizontal components of three-component recordings to radial (along the "
                   "source-to-receiver direction, positive away from the source) and transverse (90 degrees "
                   "anticlockwise from radial, seen from above), with the azimuth from the source and receiver "
                   "coordinates (sx, sy, gx, gy) or by one fixed angle. The inline component points along the sensor "
                   "azimuth (degrees anticlockwise from the survey x axis) and the crossline component 90 degrees "
                   "anticlockwise from it. Rotated traces get the codes 17 (radial) and 16 (transverse) and the "
                   "angle used in header hrot_az; vertical traces pass unchanged.")
    params = (Param("mode", "choice", "source-receiver azimuth", "Rotation",
                    choices=("source-receiver azimuth", "fixed angle")),
              Param("angle", "float", 0.0, "Fixed angle (inline axis to radial, anticlockwise)", unit="deg"),
              Param("sensor_azimuth", "float", 0.0, "Sensor azimuth (inline axis from the x axis)", unit="deg"),
              Param("component_key", "key", "trid", "Component header"),
              Param("inline_code", "int", INLINE, "Code of the inline component"),
              Param("crossline_code", "int", CROSSLINE, "Code of the crossline component"),
              Param("match_keys", "keys", "fldr,tracf", MATCH_LABEL))

    def run(self, data, context):
        key, ci, cx = self["component_key"], self["inline_code"], self["crossline_code"]
        if ci == cx:
            raise ValueError("the inline and crossline codes must differ")
        groups = match_components(data, key, self["match_keys"], (ci, cx))
        _require(groups, "inline/crossline", key, self["match_keys"])
        ii = np.array([g[ci] for g in groups], dtype=np.int64)
        ix = np.array([g[cx] for g in groups], dtype=np.int64)
        coincident = 0
        if self["mode"] == "fixed angle":
            alpha = np.full(ii.size, np.deg2rad(self["angle"]))
        else:
            dx = data.header("gx")[ii] - data.header("sx")[ii]
            dy = data.header("gy")[ii] - data.header("sy")[ii]
            same = np.hypot(dx, dy) < 1e-6
            coincident = int(np.count_nonzero(same))
            alpha = np.where(same, 0.0, np.arctan2(dy, dx)) - np.deg2rad(self["sensor_azimuth"])
            if coincident:
                context.log(f"    {coincident} station(s) at the source position are rotated by the sensor azimuth only")
        tr = data.traces.astype(np.float64)
        radial, transverse = rotate_horizontal(tr[ii], tr[ix], alpha[:, None])
        tr[ii], tr[ix] = radial, transverse
        out = data.copy(tr.astype(np.float32))
        comp = out.header(key).copy()
        comp[ii], comp[ix] = RADIAL, TRANSVERSE
        out.set_header(key, comp)
        az = out.header("hrot_az").copy()
        az[ii] = az[ix] = np.rad2deg(alpha)
        out.set_header("hrot_az", az)
        context.results["hrot"] = {"stations": int(ii.size), "coincident": coincident}
        context.log(f"    {ii.size} station(s) rotated to radial (code {RADIAL}) and transverse (code {TRANSVERSE})")
        return out


# ---------------------------------------------------------------------------------------------- Alford rotation
def alford_angle(xx, xy, yx, yy) -> float:
    """Angle (radians, 0 to pi/2) of the rotation D' = R D R^T that minimises the off-diagonal energy of the shear
    data matrix D = [[XX, XY], [YX, YY]] summed over the samples given (Alford 1986). With P = XY + YX and
    Q = YY - XX the off-diagonal energy is a constant plus sum((cos 2t P + sin 2t Q)^2) / 2, whose minimum has a
    closed form."""
    p = (np.asarray(xy, np.float64) + yx).ravel()
    q = (np.asarray(yy, np.float64) - xx).ravel()
    a, b, d = float(p @ p), float(p @ q), float(q @ q)
    if a + d <= 0:
        return 0.0
    psi = np.arctan2(b, 0.5 * (a - d))
    return float((psi + np.pi) / 4.0)


def alford_rotate(xx, xy, yx, yy, theta: float):
    """The four components rotated by theta: D' = R D R^T with R = [[cos, sin], [-sin, cos]] (rows: source X, Y;
    columns: receiver x, y)."""
    c, s = np.cos(theta), np.sin(theta)
    cc, ss, cs = c * c, s * s, c * s
    return (cc * xx + cs * (xy + yx) + ss * yy,
            cc * xy - ss * yx + cs * (yy - xx),
            cc * yx - ss * xy + cs * (yy - xx),
            ss * xx - cs * (xy + yx) + cc * yy)


@register
class LinearTransformRotation(Process):
    name, label, category = 'ltt_rotation', 'Shear-wave rotation (sliding linear transform)', 'Multicomponent'
    description = "Sample-dependent 4C rotation from windowed XX/XY/YX/YY covariance. Chooses the continuous local diagonalisation angle modulo 90 degrees; fast/slow identification still needs geological interpretation."
    params = (Param('component_key', 'key', 'comp', 'Component header'),
              Param('codes', 'floats', '1,2,3,4', 'Codes of XX, XY, YX, YY'),
              Param('match_keys', 'keys', 'sx,gx', MATCH_LABEL),
              Param('window', 'float', .08, 'Covariance time window', unit='s', minimum=.001))
    def run(self, data, context):
        codes = _codes(self['codes'], 4, 'XX, XY, YX, YY')
        groups = match_components(data, self['component_key'], self['match_keys'], codes)
        _require(groups, 'four-component', self['component_key'], self['match_keys'])
        window = max(3, int(round(self['window']/data.dt)) | 1)
        out = data.traces.copy(); angles, trace_indices = [], []
        for g in groups:
            rows = [g[c] for c in codes]; xx, xy, yx, yy = data.traces[rows].astype(float)
            p, q = xy+yx, yy-xx
            a = uniform_filter1d(p*p, window); b = uniform_filter1d(p*q, window); d = uniform_filter1d(q*q, window)
            theta = .25*(np.unwrap(np.arctan2(2*b, a-d))+np.pi)
            live = a+d > max(float((a+d).max())*1e-6, 1e-30)
            if np.any(live):
                theta = np.interp(np.arange(data.nt), np.flatnonzero(live), theta[live])
            else:
                theta[:] = 0
            out[rows] = np.stack(alford_rotate(xx, xy, yx, yy, theta))
            angles.append(np.rad2deg(theta)); trace_indices.append(rows)
        context.results[self.name] = dict(angles=np.asarray(angles), trace_indices=np.asarray(trace_indices), times=data.times)
        return data.copy(out)


def _lag(a, b) -> int:
    """Lag (samples) of b behind a at the largest absolute value of their cross-correlation, summed over rows."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    n = a.shape[1]
    if n < 2:
        return 0
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.sum(np.conj(np.fft.rfft(a, nfft, axis=1)) * np.fft.rfft(b, nfft, axis=1), axis=0)
    c = np.fft.irfft(spec, nfft)
    c = np.concatenate([c[nfft - (n - 1):], c[:n]])                 # lags -(n-1) .. n-1
    return int(np.argmax(np.abs(c))) - (n - 1)


@register
class AlfordRotation(Process):
    name = "alford"
    label = "Alford rotation (4C shear)"
    category = "Multicomponent"
    description = ("Rotates four-component shear data (sources X and Y recorded by receivers x and y: XX, XY, YX, YY, "
                   "first letter the source, second the receiver) to the angle that minimises the energy of the "
                   "off-diagonal components XY and YX in a time window (Alford 1986). In an azimuthally anisotropic "
                   "(fractured) layer the rotated XX and YY are the fast and slow shear waves; with 'fast first' the "
                   "earlier of the two is put on XX. The angle (degrees anticlockwise from the acquisition x axis to "
                   "the new x axis) is written to header alford_az and listed in the log; one angle per station, or "
                   "one per gather of the chosen key.")
    params = (Param("component_key", "key", "comp", "Component header"),
              Param("codes", "floats", "1,2,3,4", "Codes of XX, XY, YX, YY"),
              Param("match_keys", "keys", "sx,gx", "Headers matching the four components of one station"),
              Param("tmin", "float", 0.0, "Window start", unit="s", minimum=0.0),
              Param("tmax", "float", 0.0, "Window end (0 = end of trace)", unit="s", minimum=0.0),
              Param("gather_key", "key", "", "One angle per gather of (empty = each station)"),
              Param("fast_first", "bool", True, "Put the fast shear wave on XX"))

    def run(self, data, context):
        key = self["component_key"]
        codes = _codes(self["codes"], 4, "XX, XY, YX, YY")
        groups = match_components(data, key, self["match_keys"], codes)
        _require(groups, "four-component", key, self["match_keys"])
        idx = np.array([[g[c] for c in codes] for g in groups], dtype=np.int64)          # (stations, 4)
        win = _window(data.times, self["tmin"], self["tmax"])
        tr = data.traces.astype(np.float64)
        labels = (data.header(self["gather_key"])[idx[:, 0]] if self["gather_key"]
                  else np.arange(idx.shape[0], dtype=np.float64))
        out = tr.copy()
        theta_of = np.zeros(idx.shape[0])
        before = after = total = 0.0
        ulabels = np.unique(labels)
        for n, lab in enumerate(ulabels):
            members = labels == lab
            rows = idx[members]
            part = [tr[rows[:, k]][:, win] for k in range(4)]
            theta = alford_angle(*part)
            rot = alford_rotate(*part, theta)
            if self["fast_first"] and _lag(rot[0], rot[3]) < 0:              # YY' leads: swap fast and slow
                theta += 0.5 * np.pi
                rot = alford_rotate(*part, theta)
            theta = (theta + 0.5 * np.pi) % np.pi - 0.5 * np.pi               # the rotation repeats every 180 degrees
            full = alford_rotate(*(tr[rows[:, k]] for k in range(4)), theta)
            for k in range(4):
                out[rows[:, k]] = full[k]
            theta_of[members] = theta
            total += sum(float(np.sum(p * p)) for p in part)
            before += float(np.sum(part[1] ** 2) + np.sum(part[2] ** 2))
            after += float(np.sum(rot[1] ** 2) + np.sum(rot[2] ** 2))
            context.report((n + 1) / ulabels.size)
        res = data.copy(out.astype(np.float32))
        az = res.header("alford_az").copy()
        for k in range(4):
            az[idx[:, k]] = np.rad2deg(theta_of)
        res.set_header("alford_az", az)
        total = max(total, 1e-30)
        deg = np.rad2deg(theta_of)
        context.results["alford"] = {"x": data.header("gx")[idx[:, 0]].tolist(), "angle_deg": deg.tolist(),
                                     "offdiag_before": before / total, "offdiag_after": after / total}
        context.log(f"    Alford rotation of {idx.shape[0]} station(s): angle {deg.min():.1f} to {deg.max():.1f} deg; "
                    f"off-diagonal energy {100 * before / total:.1f} % -> {100 * after / total:.1f} %")
        return res


# ---------------------------------------------------------------------------------------------- polarisation
def polarisation_attributes(z, x, y, nwin: int) -> dict:
    """Polarisation attributes at every sample from the covariance matrix of the vertical (z), inline (x) and
    crossline (y) traces in a sliding window of nwin samples (Jurkevics 1988).

    With the eigenvalues l1 >= l2 >= l3 and the main eigenvector u: rectilinearity 1 - (l2 + l3) / (2 l1),
    planarity 1 - 2 l3 / (l1 + l2), azimuth of u (degrees from the inline axis towards the crossline axis, u taken
    with an upward vertical part), incidence of u from the vertical (degrees) and l1 (mean energy). Samples without
    energy get zero."""
    comps = [np.asarray(c, dtype=np.float64) for c in (z, x, y)]
    nt = comps[0].size
    n = max(1, int(nwin))
    cov = np.empty((nt, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            cov[:, i, j] = cov[:, j, i] = uniform_filter1d(comps[i] * comps[j], n, mode="constant")
    w, v = np.linalg.eigh(cov)                                   # ascending eigenvalues, eigenvectors in columns
    l1 = np.maximum(w[:, 2], 0.0)
    l2 = np.maximum(w[:, 1], 0.0)
    l3 = np.maximum(w[:, 0], 0.0)
    top = float(l1.max()) if nt else 0.0
    ok = l1 > 1e-12 * top if top > 0 else np.zeros(nt, dtype=bool)
    safe = np.where(ok, l1, 1.0)
    rect = np.where(ok, 1.0 - (l2 + l3) / (2.0 * safe), 0.0)
    plan = np.where(ok, 1.0 - 2.0 * l3 / np.maximum(l1 + l2, 1e-300), 0.0)
    u = v[:, :, 2]
    u = np.where(u[:, :1] < 0, -u, u)
    az = np.degrees(np.arctan2(u[:, 2], u[:, 1]))
    inc = np.degrees(np.arccos(np.clip(u[:, 0], -1.0, 1.0)))
    return {"rectilinearity": np.clip(rect, 0.0, 1.0), "planarity": np.clip(plan, 0.0, 1.0),
            "azimuth": np.where(ok, az, 0.0), "incidence": np.where(ok, inc, 0.0), "largest eigenvalue": l1}


def _three_component(step, data):
    codes = _codes(step["codes"], 3, "vertical, inline, crossline")
    groups = match_components(data, step["component_key"], step["match_keys"], codes)
    _require(groups, "three-component", step["component_key"], step["match_keys"])
    return codes, groups, max(3, int(round(step["window"] / data.dt)))


COMMON_3C = (Param("component_key", "key", "trid", "Component header"),
             Param("codes", "floats", "12,14,13", "Codes of vertical, inline, crossline"),
             Param("match_keys", "keys", "fldr,tracf", MATCH_LABEL))


@register
class PolarisationAnalysis(Process):
    name = "polarisation"
    label = "Polarisation analysis (3C)"
    category = "Multicomponent"
    description = ("Polarisation attributes from the covariance matrix of the vertical, inline and crossline "
                   "components in a sliding time window (Jurkevics 1988): rectilinearity 1 - (l2 + l3) / (2 l1), "
                   "near 1 for linear motion such as body waves and about 0.5 for circular motion such as ground "
                   "roll; planarity 1 - 2 l3 / (l1 + l2); azimuth of the main motion (degrees from the inline axis "
                   "towards the crossline axis); incidence from the vertical (degrees); or the largest eigenvalue "
                   "(energy). Output: one attribute trace per station, with the headers of its vertical trace.")
    params = (Param("attribute", "choice", "rectilinearity", "Attribute", choices=ATTRIBUTES),
              Param("window", "float", 0.05, "Window", unit="s", minimum=0.002)) + COMMON_3C

    def run(self, data, context):
        codes, groups, nwin = _three_component(self, data)
        out = np.zeros((len(groups), data.nt), dtype=np.float32)
        for n, g in enumerate(groups):
            att = polarisation_attributes(data.traces[g[codes[0]]], data.traces[g[codes[1]]],
                                          data.traces[g[codes[2]]], nwin)
            out[n] = att[self["attribute"]]
            if n % 50 == 0:
                context.report(n / len(groups))
        iz = np.array([g[codes[0]] for g in groups], dtype=np.int64)
        res = data.subset(iz).copy(out, name=f"{data.name} {self['attribute']}")
        res.meta["attribute"] = self["attribute"]
        return res


@register
class PolarisationFilter(Process):
    name = "polar_filter"
    label = "Polarisation filter (3C)"
    category = "Multicomponent"
    description = ("Weights every sample of the three components by the rectilinearity raised to a power, so that "
                   "linearly polarised body waves pass and elliptically polarised ground roll is attenuated. "
                   "Optionally the weight also passes only near-vertical incidence (P reflections) or only "
                   "near-horizontal motion, with a 15-degree taper beyond the incidence limit. The weights come from "
                   "the covariance matrix in a sliding window (Jurkevics 1988) and are smoothed over the same window; "
                   "traces without all three components pass unchanged.")
    params = (Param("window", "float", 0.05, "Window", unit="s", minimum=0.002),
              Param("power", "float", 2.0, "Rectilinearity power", minimum=0.0),
              Param("direction", "choice", "any", "Direction", choices=("any", "near-vertical", "near-horizontal")),
              Param("max_incidence", "float", 35.0, "Incidence limit", unit="deg", minimum=1.0,
                    maximum=89.0)) + COMMON_3C

    def run(self, data, context):
        codes, groups, nwin = _three_component(self, data)
        out = data.traces.astype(np.float64)
        lim = self["max_incidence"]
        means = []
        for n, g in enumerate(groups):
            att = polarisation_attributes(out[g[codes[0]]], out[g[codes[1]]], out[g[codes[2]]], nwin)
            w = att["rectilinearity"] ** self["power"]
            if self["direction"] != "any":
                ramp = np.clip((att["incidence"] - lim) / 15.0, 0.0, 1.0)
                steep = 0.5 * (1.0 + np.cos(np.pi * ramp))          # 1 up to the limit, 0 from 15 degrees beyond it
                w = w * (steep if self["direction"] == "near-vertical" else 1.0 - steep)
            w = uniform_filter1d(w, nwin, mode="nearest")
            for c in codes:
                out[g[c]] *= w
            means.append(float(np.mean(w)))
            if n % 50 == 0:
                context.report(n / len(groups))
        context.results["polar_filter"] = {"stations": len(groups), "mean_weight": float(np.mean(means))}
        return data.copy(out.astype(np.float32))
