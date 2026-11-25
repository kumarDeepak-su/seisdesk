"""Well logs: LAS 2.0 reading, sonic and density conversion, two-way time from the integrated sonic (with checkshot
correction), reflectivity and synthetic seismograms, Backus averaging with Thomsen parameters, Gassmann fluid
substitution and the correlation of a synthetic with a seismic trace (well tie).

Units inside: depth in metres, velocity in m/s, density in kg/m3, moduli in Pa, time in seconds."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d

from .synthetic import ricker

NULL_DEFAULT = -999.25
FT = 0.3048

# usual mnemonics of the curves the tools need, most common first
ALIASES = {
    "sonic": ("DT", "DTC", "DTCO", "DTP", "DT4P", "AC", "DTCOMP", "SONIC", "DT24"),
    "shear sonic": ("DTS", "DTSM", "DTSH", "DT4S", "DTSD", "DTSHEAR"),
    "density": ("RHOB", "RHOZ", "DEN", "ZDEN", "RHO", "DENS", "DENSITY"),
    "vp": ("VP", "VEL", "VELP", "PVEL"),
    "vs": ("VS", "VELS", "SVEL"),
    "porosity": ("PHIE", "PHIT", "PHI", "NPHI", "POR", "POROSITY"),
    "gamma ray": ("GR", "GRC", "SGR", "CGR"),
}


@dataclass
class WellLog:
    depth: np.ndarray                                   # increasing, in depth_unit
    curves: dict = field(default_factory=dict)          # mnemonic -> values (NaN where null); the depth curve first
    units: dict = field(default_factory=dict)
    descriptions: dict = field(default_factory=dict)
    well: dict = field(default_factory=dict)            # ~W items: mnemonic -> (value, unit, description)
    params: dict = field(default_factory=dict)          # ~P items, same form
    depth_unit: str = "m"
    name: str = "well"
    version: str = "2.0"

    @property
    def names(self) -> list[str]:
        return list(self.curves)

    @property
    def depth_m(self) -> np.ndarray:
        return self.depth * (FT if self.depth_unit == "ft" else 1.0)

    def curve(self, mnemonic: str) -> np.ndarray:
        if mnemonic not in self.curves:
            raise KeyError(f"no curve {mnemonic!r} in well {self.name} (curves: {', '.join(self.curves)})")
        return self.curves[mnemonic]


# ---------------------------------------------------------------------------------------------- LAS 2.0
def _header_line(line: str):
    """(mnemonic, unit, value, description) of a LAS header line 'MNEM.UNIT  VALUE : DESCRIPTION': the first dot ends
    the mnemonic, the first space after it ends the unit and the last colon starts the description."""
    dot = line.find(".")
    if dot < 0:
        return None
    mnem = line[:dot].strip()
    rest = line[dot + 1:]
    space = re.search(r"\s", rest)
    unit = rest[:space.start()] if space else rest
    remainder = rest[space.start():] if space else ""
    colon = remainder.rfind(":")
    if colon < 0:
        return mnem, unit.strip(), remainder.strip(), ""
    return mnem, unit.strip(), remainder[:colon].strip(), remainder[colon + 1:].strip()


def _float(text, default: float) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_las(text: str, name: str = "well") -> WellLog:
    """A LAS 2.0 (or 1.2) log from its text: ~Version, ~Well, ~Curve, ~Parameter and ~ASCII sections, NULL values as
    NaN, depth unit m or ft. Wrapped files (WRAP YES) are not supported."""
    section = ""
    version, well, params, curve_defs, tokens = {}, {}, {}, [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            section = line[1:2].upper()
            continue
        if section == "A":
            tokens.extend(line.replace(",", " ").split())
            continue
        item = _header_line(line)
        if item is None:
            continue
        mnem, unit, value, desc = item
        if section == "V":
            version[mnem.upper()] = value
        elif section == "W":
            well[mnem.upper()] = (value, unit, desc)
        elif section == "C":
            curve_defs.append((mnem, unit, desc))
        elif section == "P":
            params[mnem.upper()] = (value, unit, desc)
    if version.get("WRAP", "NO").strip().upper().startswith("Y"):
        raise ValueError("wrapped LAS files (WRAP YES) are not supported: save the log unwrapped (WRAP NO)")
    if not curve_defs:
        raise ValueError("the LAS file has no ~Curve section")
    ncurve = len(curve_defs)
    try:
        values = np.array([float(t) for t in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"non-numeric value in the ~ASCII section ({exc})") from None
    if values.size == 0 or values.size % ncurve:
        raise ValueError(f"the ~ASCII section holds {values.size} values, not a multiple of the {ncurve} curves")
    table = values.reshape(-1, ncurve)
    null = _float(well.get("NULL", ("",))[0], NULL_DEFAULT)
    table[np.abs(table - null) <= 1e-6 * max(1.0, abs(null))] = np.nan
    table = table[np.isfinite(table[:, 0])]
    if table.shape[0] > 1 and np.any(np.diff(table[:, 0]) < 0):
        table = table[np.argsort(table[:, 0], kind="stable")]
    curves, units, descs = {}, {}, {}
    for k, (mnem, unit, desc) in enumerate(curve_defs):
        key = base = mnem or f"CURVE{k + 1}"
        n = 1
        while key in curves:
            n += 1
            key = f"{base}:{n}"
        curves[key], units[key], descs[key] = table[:, k], unit, desc
    depth_key = next(iter(curves))
    unit = (units[depth_key] or well.get("STRT", ("", ""))[1]).strip().upper()
    well_name = well.get("WELL", ("",))[0].strip()
    return WellLog(curves[depth_key], curves, units, descs, well, params, "ft" if unit.startswith("F") else "m",
                   well_name or name, version.get("VERS", "2.0"))


def read_las(path: str) -> WellLog:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return parse_las(text, os.path.splitext(os.path.basename(path))[0])


def find_curve(log: WellLog, kind: str) -> str | None:
    """The first curve whose mnemonic is a usual name for ``kind`` (a key of ALIASES), or None."""
    names = {n.upper().split(":")[0]: n for n in reversed(log.names[1:])}
    for alias in ALIASES.get(kind, ()):
        if alias in names:
            return names[alias]
    return None


# ---------------------------------------------------------------------------------------------- conversions
def sonic_to_velocity(dt, unit: str = "") -> np.ndarray:
    """Velocity (m/s) from a slowness log in microseconds per foot or per metre: from the unit, else from the size of
    the values (a median above 250 is taken as us/m)."""
    dt = np.asarray(dt, dtype=np.float64)
    u = unit.strip().lower().replace("usec", "us")
    if "f" in u:
        per_m = False
    elif "m" in u:
        per_m = True
    else:
        per_m = bool(np.any(np.isfinite(dt))) and float(np.nanmedian(dt)) > 250.0
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (1e6 if per_m else 1e6 * FT) / dt
    v[~np.isfinite(v) | (v <= 0)] = np.nan
    return v


def density_to_si(rho, unit: str = "") -> np.ndarray:
    """Density in kg/m3 from g/cc (g/cm3) or kg/m3: from the unit, else from the size of the values."""
    rho = np.asarray(rho, dtype=np.float64)
    u = unit.strip().lower()
    if "kg" in u:
        scale = 1.0
    elif "g" in u:
        scale = 1000.0
    else:
        scale = 1000.0 if bool(np.any(np.isfinite(rho))) and float(np.nanmedian(rho)) < 10.0 else 1.0
    out = rho * scale
    out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out


def mudrock_vs(vp) -> np.ndarray:
    """Shear velocity from the mudrock line Vp = 1.16 Vs + 1360 m/s (Castagna et al. 1985), at least 10 % of Vp."""
    vp = np.asarray(vp, dtype=np.float64)
    return np.maximum((vp - 1360.0) / 1.16, 0.1 * vp)


def gardner_density(vp) -> np.ndarray:
    """Density (kg/m3) from Vp (m/s) by Gardner et al. (1974): 310 Vp^0.25."""
    return 310.0 * np.asarray(vp, dtype=np.float64) ** 0.25


def fill_gaps(depth, values) -> np.ndarray:
    """Values with interior gaps (NaN) filled by linear interpolation in depth; beyond the first and last valid
    samples they stay NaN."""
    v = np.asarray(values, dtype=np.float64).copy()
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return v
    first, last = np.flatnonzero(ok)[[0, -1]]
    inner = np.arange(first, last + 1)
    z = np.asarray(depth, dtype=np.float64)
    v[inner] = np.interp(z[inner], z[ok], v[ok])
    return v


def valid_span(*curves) -> slice:
    """Slice from the first to the last sample where every curve has a value."""
    ok = np.ones(np.asarray(curves[0]).shape, dtype=bool)
    for c in curves:
        ok &= np.isfinite(np.asarray(c, dtype=np.float64))
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        raise ValueError("the logs share fewer than two valid samples")
    return slice(int(idx[0]), int(idx[-1]) + 1)


# ---------------------------------------------------------------------------------------------- time-depth
def two_way_time(depth_m, vp, t_top: float | None = None, v_above: float = 2000.0) -> np.ndarray:
    """Two-way time (s) at every log depth from the integrated slowness (trapezoid rule). The top of the log is at
    t_top, or at 2 z_top / v_above when t_top is not given (the unlogged overburden at v_above)."""
    z = np.asarray(depth_m, dtype=np.float64)
    s = 1.0 / np.asarray(vp, dtype=np.float64)
    if t_top is None:
        t_top = 2.0 * z[0] / v_above
    steps = np.diff(z) * (s[:-1] + s[1:])                    # 2 dz (s1 + s2) / 2
    return t_top + np.concatenate([[0.0], np.cumsum(steps)])


def apply_checkshots(depth_m, twt, cs_depth_m, cs_twt) -> np.ndarray:
    """Sonic two-way times corrected to checkshot times: the drift (checkshot minus sonic time) at the checkshots
    inside the logged depths, interpolated linearly in depth and held constant beyond the end ones, is added."""
    z, t = np.asarray(depth_m, dtype=np.float64), np.asarray(twt, dtype=np.float64)
    cz, ct = np.asarray(cs_depth_m, dtype=np.float64), np.asarray(cs_twt, dtype=np.float64)
    inside = (cz >= z[0]) & (cz <= z[-1])
    if not inside.any():
        raise ValueError("no checkshot lies within the logged depths")
    order = np.argsort(cz[inside])
    cz, ct = cz[inside][order], ct[inside][order]
    drift = ct - np.interp(cz, z, t)
    return t + np.interp(z, cz, drift)


def read_checkshots(path: str, one_way: bool = True, time_unit: str = "ms", depth_unit: str = "m"):
    """A checkshot table: depth and time in the first two columns (spaces, commas or semicolons; header lines are
    skipped). Returns depth (m) and two-way time (s)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.replace(",", " ").replace(";", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no depth-time pairs in {path}")
    a = np.array(rows, dtype=np.float64)
    z = a[:, 0] * (FT if depth_unit == "ft" else 1.0)
    t = a[:, 1] / (1000.0 if time_unit == "ms" else 1.0)
    return z, t * (2.0 if one_way else 1.0)


def time_depth_table(depth_m, twt, step: int = 1) -> str:
    lines = ["depth_m,twt_s,owt_ms"]
    for z, t in zip(np.asarray(depth_m)[::step], np.asarray(twt)[::step]):
        lines.append(f"{z:.2f},{t:.5f},{500.0 * t:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- synthetics
def reflectivity_series(twt, impedance, dt: float, t0: float | None = None, nt: int | None = None):
    """Reflection coefficients (Z2 - Z1) / (Z2 + Z1) between successive log samples, placed at the two-way time of
    each interface and shared linearly between the two nearest samples of the time axis t0 + k dt (t0 a multiple of
    dt before the first log time by default). Returns (times, reflectivity)."""
    t, z = np.asarray(twt, dtype=np.float64), np.asarray(impedance, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (z[1:] - z[:-1]) / (z[1:] + z[:-1])
    tm = 0.5 * (t[1:] + t[:-1])
    if t0 is None:
        t0 = float(np.floor(t[0] / dt + 1e-9)) * dt
    if nt is None:
        nt = int(np.ceil((t[-1] - t0) / dt)) + 2
    out = np.zeros(max(int(nt), 1))
    pos = (tm - t0) / dt
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    for shift, weight in ((0, 1.0 - frac), (1, frac)):
        idx = i0 + shift
        ok = (idx >= 0) & (idx < out.size) & np.isfinite(r)
        np.add.at(out, idx[ok], (r * weight)[ok])
    return t0 + np.arange(out.size) * dt, out


def convolve_wavelet(series, wavelet, centre: int) -> np.ndarray:
    """``series`` convolved with ``wavelet`` whose time zero is its sample ``centre``; same length as the series."""
    s = np.asarray(series, dtype=np.float64)
    full = np.convolve(s, np.asarray(wavelet, dtype=np.float64))
    return full[centre: centre + s.size]


def read_wavelet(path: str, dt: float):
    """A wavelet from a text file: one amplitude per line (sampled at dt, time zero at the largest absolute value)
    or two columns time (s) and amplitude (resampled to dt, time zero at t = 0). Returns (wavelet, centre)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(p) for p in parts[:2]])
            except ValueError:
                continue
    rows = [r for r in rows if r]
    if len(rows) < 2:
        raise ValueError(f"no wavelet samples in {path}")
    if all(len(r) == 2 for r in rows):
        a = np.array(rows, dtype=np.float64)
        a = a[np.argsort(a[:, 0])]
        k0, k1 = int(np.ceil(a[0, 0] / dt - 1e-9)), int(np.floor(a[-1, 0] / dt + 1e-9))
        tt = np.arange(k0, k1 + 1) * dt
        w = np.interp(tt, a[:, 0], a[:, 1])
        centre = -k0 if k0 <= 0 <= k1 else int(np.argmax(np.abs(w)))
        return w, centre
    w = np.array([r[0] for r in rows], dtype=np.float64)
    return w, int(np.argmax(np.abs(w)))


def synthetic_seismogram(depth_m, vp, rho, dt: float, f0: float = 30.0, wavelet=None, centre: int = 0, twt=None,
                         v_above: float = 2000.0, t0: float | None = None, nt: int | None = None) -> dict:
    """Synthetic seismogram of a well: impedance vp * rho, two-way time (given, or from the sonic with the overburden
    at v_above), reflectivity on the time axis, convolved with a zero-phase Ricker wavelet of peak frequency f0 (or
    the wavelet given, time zero at its sample ``centre``). Gaps are interpolated and the logs cut to their common
    valid span; ``twt``, if given, has one value per depth sample."""
    z = np.asarray(depth_m, dtype=np.float64)
    v, r = fill_gaps(z, vp), fill_gaps(z, rho)
    sl = valid_span(v, r)
    z, v, r = z[sl], v[sl], r[sl]
    t = two_way_time(z, v, v_above=v_above) if twt is None else np.asarray(twt, dtype=np.float64)[sl]
    imp = v * r
    times, rc = reflectivity_series(t, imp, dt, t0, nt)
    if wavelet is None:
        w, c = ricker(f0, dt)
    else:
        w, c = np.asarray(wavelet, dtype=np.float64), int(centre)
    return {"times": times, "reflectivity": rc, "synthetic": convolve_wavelet(rc, w, c), "depth": z, "twt": t,
            "impedance": imp, "vp": v, "rho": r}


def well_tie(syn_times, synthetic, seis_times, seismic, max_shift: float = 0.1) -> dict:
    """Normalised cross-correlation of a synthetic with a seismic trace over the time span of the synthetic, for
    shifts up to max_shift (s). Returns the shift to add to the synthetic times (s; positive when the seismic events
    come later), the correlation at that shift, the most negative correlation and the correlation curve."""
    st, sy = np.asarray(syn_times, dtype=np.float64), np.asarray(synthetic, dtype=np.float64)
    tt, tr = np.asarray(seis_times, dtype=np.float64), np.asarray(seismic, dtype=np.float64)
    dt = float(tt[1] - tt[0]) if tt.size > 1 else 1.0
    idx = np.flatnonzero((tt >= st[0]) & (tt <= st[-1]))
    if idx.size < 4:
        raise ValueError("the synthetic and the seismic trace do not overlap in time")
    a_all = np.interp(tt[idx], st, sy)
    nmax = int(round(max_shift / dt))
    lags = np.arange(-nmax, nmax + 1)
    curve = np.zeros(lags.size)
    for k, lag in enumerate(lags):
        j = idx + lag
        ok = (j >= 0) & (j < tr.size)
        if ok.sum() < 4:
            continue
        a, b = a_all[ok], tr[j[ok]]
        den = np.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
        curve[k] = float(np.sum(a * b)) / den if den > 0 else 0.0
    best = int(np.argmax(curve))
    return {"shift": float(lags[best] * dt), "r": float(curve[best]), "r_negative": float(curve.min()),
            "lags": lags * dt, "curve": curve}


# ---------------------------------------------------------------------------------------------- rock physics
def backus_average(depth_m, vp, vs, rho, length_m: float) -> dict:
    """Backus (1962) average of finely layered isotropic logs over a moving window of ``length_m`` (counted in
    samples of the median depth step): stiffnesses C11, C13, C33, C44, C66 of the equivalent VTI medium, its
    density, vertical velocities vp0 and vs0, and the Thomsen (1986) parameters epsilon, delta and gamma at every
    depth (NaN outside the common valid span of the logs). Shear velocities below 1 m/s are raised to 1 m/s."""
    z = np.asarray(depth_m, dtype=np.float64)
    vp, vs, rho = (fill_gaps(z, c) for c in (vp, vs, rho))
    sl = valid_span(vp, vs, rho)
    step = float(np.median(np.diff(z[sl])))
    n = max(1, int(round(length_m / step))) if step > 0 else 1
    p, s, d = vp[sl], np.maximum(vs[sl], 1.0), rho[sl]
    mu = d * s * s
    m = d * p * p
    lam = m - 2.0 * mu

    def avg(x):
        return uniform_filter1d(x, n, mode="nearest")

    c33 = 1.0 / avg(1.0 / m)
    c44 = 1.0 / avg(1.0 / mu)
    c66 = avg(mu)
    lm = avg(lam / m)
    c13 = lm * c33
    c11 = avg(4.0 * mu * (lam + mu) / m) + lm * lm * c33
    rho_e = avg(d)
    part = {"c11": c11, "c13": c13, "c33": c33, "c44": c44, "c66": c66, "rho": rho_e,
            "vp0": np.sqrt(c33 / rho_e), "vs0": np.sqrt(c44 / rho_e),
            "epsilon": (c11 - c33) / (2.0 * c33), "gamma": (c66 - c44) / (2.0 * c44),
            "delta": ((c13 + c44) ** 2 - (c33 - c44) ** 2) / (2.0 * c33 * (c33 - c44))}
    out = {}
    for key, values in part.items():
        full = np.full(z.size, np.nan)
        full[sl] = values
        out[key] = full
    out["window_samples"] = n
    return out


def backus_length(velocity: float, frequency: float, fraction: float = 1.0 / 3.0) -> float:
    """Longest Backus averaging length (m) for waves of the given frequency: a fraction of the shortest wavelength
    velocity / frequency (Liner and Fei 2006 suggest at most a third)."""
    return fraction * float(velocity) / max(float(frequency), 1e-9)


def gassmann(vp, vs, rho, porosity, k_mineral: float, k_fluid1: float, rho_fluid1: float, k_fluid2: float,
             rho_fluid2: float):
    """Gassmann (1951) fluid substitution: Vp, Vs (m/s) and density (kg/m3) after replacing the pore fluid (bulk
    modulus k_fluid1 in Pa, density rho_fluid1 in kg/m3) by another (k_fluid2, rho_fluid2). The dry-rock modulus comes
    from the inverted Gassmann equation with the mineral modulus k_mineral; the shear modulus does not change.
    Samples with porosity up to 1 %, or without a valid result, keep their values."""
    vp, vs, rho = (np.asarray(c, dtype=np.float64) for c in (vp, vs, rho))
    phi = np.broadcast_to(np.asarray(porosity, dtype=np.float64), vp.shape)
    km = float(k_mineral)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = rho * vs * vs
        k1 = rho * vp * vp - 4.0 / 3.0 * mu
        a = phi * km / k_fluid1
        kdry = np.clip((k1 * (a + 1.0 - phi) - km) / (a + k1 / km - 1.0 - phi), 0.0, km)
        k2 = kdry + (1.0 - kdry / km) ** 2 / (phi / k_fluid2 + (1.0 - phi) / km - kdry / km ** 2)
        rho2 = rho + phi * (rho_fluid2 - rho_fluid1)
        vp2 = np.sqrt((k2 + 4.0 / 3.0 * mu) / rho2)
        vs2 = np.sqrt(mu / rho2)
    keep = ~(phi > 0.01) | ~np.isfinite(vp2) | ~np.isfinite(vs2) | ~np.isfinite(rho2) | ~(rho2 > 0)
    return np.where(keep, vp, vp2), np.where(keep, vs, vs2), np.where(keep, rho, rho2)
