"""Deviated wells: deviation surveys (measured depth, inclination, azimuth), the minimum-curvature well path (true
vertical depth and horizontal position), measured depth to true vertical depth, and the position of the well along a
2D seismic line.

Angles in degrees (inclination from the vertical, azimuth clockwise from north), depths in metres."""

from __future__ import annotations

import numpy as np

FT = 0.3048


def parse_deviation(text: str, depth_unit: str = "m"):
    """A deviation survey from text: measured depth, inclination and azimuth in the first three columns (spaces,
    commas, semicolons or tabs; header and comment lines are skipped). Returns MD (m), inclination and azimuth
    (degrees), sorted by MD with repeated depths removed."""
    rows = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        parts = line.replace(",", " ").replace(";", " ").split()
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if len(rows) < 2:
        raise ValueError("a deviation survey needs at least two stations (MD, inclination, azimuth)")
    a = np.array(rows, dtype=np.float64)
    a = a[np.argsort(a[:, 0], kind="stable")]
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
    a = a[keep]
    if np.any(a[:, 1] < 0) or np.any(a[:, 1] > 180):
        raise ValueError("inclinations must lie between 0 and 180 degrees")
    md = a[:, 0] * (FT if depth_unit.lower().startswith("f") else 1.0)
    return md, a[:, 1], a[:, 2] % 360.0


def read_deviation(path: str, depth_unit: str = "m"):
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_deviation(f.read(), depth_unit)


def minimum_curvature(md, inc_deg, azi_deg, x0: float = 0.0, y0: float = 0.0) -> dict:
    """Well path by the minimum-curvature method (exact for circular arcs between stations): true vertical depth,
    east and north displacement from the well head, map position x0 + east, y0 + north, and the dogleg severity
    (degrees per 30 m) of each interval. The hole is taken as vertical from the surface to the first station."""
    md = np.asarray(md, dtype=np.float64)
    inc = np.deg2rad(np.asarray(inc_deg, dtype=np.float64))
    azi = np.deg2rad(np.asarray(azi_deg, dtype=np.float64))
    if md.size < 2 or np.any(np.diff(md) <= 0):
        raise ValueError("measured depths must increase")
    i1, i2, a1, a2 = inc[:-1], inc[1:], azi[:-1], azi[1:]
    dmd = np.diff(md)
    cos_beta = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rf = np.where(beta > 1e-9, 2.0 / beta * np.tan(0.5 * beta), 1.0)
    half = 0.5 * dmd * rf
    d_north = half * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2))
    d_east = half * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2))
    d_vert = half * (np.cos(i1) + np.cos(i2))
    north = np.concatenate([[0.0], np.cumsum(d_north)])
    east = np.concatenate([[0.0], np.cumsum(d_east)])
    tvd = md[0] + np.concatenate([[0.0], np.cumsum(d_vert)])
    return {"md": md, "inc": np.rad2deg(inc), "azi": np.rad2deg(azi), "tvd": tvd, "east": east, "north": north,
            "x": x0 + east, "y": y0 + north, "dogleg": np.rad2deg(beta) / dmd * 30.0,
            "displacement": np.hypot(east, north)}


def _extrapolated(md_query, md, values, slope_end: float, slope_start: float) -> np.ndarray:
    q = np.asarray(md_query, dtype=np.float64)
    out = np.interp(q, md, values)
    below, above = q > md[-1], q < md[0]
    out[below] = values[-1] + (q[below] - md[-1]) * slope_end
    out[above] = values[0] + (q[above] - md[0]) * slope_start
    return out


def md_to_tvd(md_query, path: dict) -> np.ndarray:
    """True vertical depth at the given measured depths: interpolated on the path, continued along the last station's
    inclination below the survey and vertically above it."""
    return _extrapolated(md_query, path["md"], path["tvd"], float(np.cos(np.deg2rad(path["inc"][-1]))), 1.0)


def md_to_east_north(md_query, path: dict):
    """East and north displacement at the given measured depths (held at the end values beyond the survey)."""
    q = np.asarray(md_query, dtype=np.float64)
    return np.interp(q, path["md"], path["east"]), np.interp(q, path["md"], path["north"])


def tvd_to_md(tvd_query, path: dict) -> np.ndarray:
    """Measured depth at the given true vertical depths, for a path whose TVD increases (no horizontal sections)."""
    tvd = path["tvd"]
    if np.any(np.diff(tvd) <= 0):
        raise ValueError("the well path is horizontal or climbs somewhere: TVD to MD is not unique")
    return _extrapolated(tvd_query, tvd, path["md"], 1.0 / max(float(np.cos(np.deg2rad(path["inc"][-1]))), 1e-6),
                         1.0)


def along_line(east, north, line_azimuth_deg: float):
    """Offset of a displacement (east, north) along a straight 2D line of the given azimuth (degrees from north,
    clockwise) and its distance across the line (positive to the right of the line direction)."""
    a = np.deg2rad(line_azimuth_deg)
    e, n = np.asarray(east, dtype=np.float64), np.asarray(north, dtype=np.float64)
    return e * np.sin(a) + n * np.cos(a), e * np.cos(a) - n * np.sin(a)


def deviation_text(path: dict, step: int = 1) -> str:
    """The computed path as CSV text: MD, inclination, azimuth, TVD, east, north."""
    lines = ["md_m,inc_deg,azi_deg,tvd_m,east_m,north_m"]
    for k in range(0, path["md"].size, max(1, step)):
        lines.append(f"{path['md'][k]:.2f},{path['inc'][k]:.3f},{path['azi'][k]:.3f},{path['tvd'][k]:.2f},"
                     f"{path['east'][k]:.2f},{path['north'][k]:.2f}")
    return "\n".join(lines)
