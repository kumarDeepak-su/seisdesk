"""Geometry: offsets, CMP numbers and 3D bins from source and receiver coordinates.

All functions work on a ``SeismicData`` and return a new one with headers set; coordinates are in metres (the readers
have already applied ``scalco``).
"""

from __future__ import annotations

import os

import numpy as np

from .dataset import SeismicData


def _principal_axis(x: np.ndarray, y: np.ndarray):
    """Origin (first point along the axis), unit direction and normal of the best-fit line through (x, y)."""
    pts = np.column_stack([x, y])
    centre = pts.mean(axis=0)
    if np.allclose(pts, centre):
        return centre, np.array([1.0, 0.0]), np.array([0.0, 1.0])
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    u = vt[0] / np.linalg.norm(vt[0])
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):
        u = -u
    along = (pts - centre) @ u
    origin = centre + along.min() * u
    return origin, u, np.array([-u[1], u[0]])


def geometry_2d(data: SeismicData, bin_size: float, origin: float | None = None, crooked: bool = True,
                signed_offset: bool = True) -> SeismicData:
    """Offset and CMP number on a 2D line.

    The midpoint of each trace is measured along the line: along x when every y is the same, otherwise along the best-fit
    line through the midpoints (``crooked=True``). CMP numbers start at 1 at ``origin`` (default: the first midpoint) and
    advance every ``bin_size`` metres; ``cdpx``/``cdpy`` receive the bin centres and ``cdpt`` the trace number in the bin.
    """
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    out = data.copy()
    sx, sy, gx, gy = (data.header(k) for k in ("sx", "sy", "gx", "gy"))
    if not (np.any(sx != 0) or np.any(gx != 0)):
        raise ValueError("source and receiver x coordinates are all zero; set the geometry first (e.g. streamer_geometry)")
    mx, my = 0.5 * (sx + gx), 0.5 * (sy + gy)
    if crooked and np.ptp(my) > 1e-6:
        o, u, _ = _principal_axis(mx, my)
        along = (np.column_stack([mx, my]) - o) @ u
        dist_s = (np.column_stack([sx, sy]) - o) @ u
        dist_g = (np.column_stack([gx, gy]) - o) @ u
    else:
        o, u = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        along, dist_s, dist_g = mx, sx, gx
    start = float(along.min()) if origin is None else float(origin)
    cdp = np.floor((along - start) / bin_size + 0.5).astype(np.int64) + 1
    off = np.hypot(gx - sx, gy - sy)
    if signed_offset:
        off = off * np.sign(dist_g - dist_s + 1e-12)
    out.set_header("offset", off)
    out.set_header("cdp", cdp)
    centre_along = start + (cdp - 1) * bin_size
    out.set_header("cdpx", o[0] + centre_along * u[0] if crooked and np.ptp(my) > 1e-6 else centre_along)
    out.set_header("cdpy", o[1] + centre_along * u[1] if crooked and np.ptp(my) > 1e-6 else np.zeros(data.ntr))
    cdpt = np.zeros(data.ntr)
    for _, idx in out.iter_gathers("cdp"):
        cdpt[idx] = np.arange(1, idx.size + 1)
    out.set_header("cdpt", cdpt)
    out.meta["geometry"] = {"type": "2D", "bin_size": bin_size, "origin": start, "crooked": bool(crooked)}
    out.add_history(f"2D geometry: CMP bin {bin_size:g} m, {np.unique(cdp).size} CMPs")
    return out


def geometry_3d(data: SeismicData, inline_bin: float, xline_bin: float, origin_x: float | None = None,
                origin_y: float | None = None, azimuth_deg: float | None = None) -> SeismicData:
    """Inline and crossline numbers from midpoints in a rotated survey grid.

    The grid's inline direction makes ``azimuth_deg`` with the x axis (default: the principal axis of the midpoints) and
    starts at (origin_x, origin_y) (default: the corner of the midpoint cloud). Bin numbers start at 1; ``cdp`` becomes
    a running bin number and ``cdpx``/``cdpy`` the bin centres.
    """
    out = data.copy()
    sx, sy, gx, gy = (data.header(k) for k in ("sx", "sy", "gx", "gy"))
    mx, my = 0.5 * (sx + gx), 0.5 * (sy + gy)
    if azimuth_deg is None:
        _, u, _ = _principal_axis(mx, my)
        azimuth = float(np.arctan2(u[1], u[0]))
    else:
        azimuth = np.deg2rad(azimuth_deg)
    u = np.array([np.cos(azimuth), np.sin(azimuth)])
    v = np.array([-u[1], u[0]])
    pts = np.column_stack([mx, my])
    a, b = pts @ u, pts @ v
    if origin_x is None or origin_y is None:
        o = a.min() * u + b.min() * v
    else:
        o = np.array([origin_x, origin_y])
    a, b = (pts - o) @ u, (pts - o) @ v
    xl = np.floor(a / inline_bin + 0.5).astype(np.int64) + 1        # position along the inline direction = crossline no.
    il = np.floor(b / xline_bin + 0.5).astype(np.int64) + 1         # position across = inline no.
    out.set_header("iline", il)
    out.set_header("xline", xl)
    nxl = int(xl.max()) if xl.size else 1
    out.set_header("cdp", (il - 1) * nxl + xl)
    centre = o + np.outer((xl - 1) * inline_bin, u) + np.outer((il - 1) * xline_bin, v)
    out.set_header("cdpx", centre[:, 0])
    out.set_header("cdpy", centre[:, 1])
    out.set_header("offset", np.hypot(gx - sx, gy - sy))
    out.meta["geometry"] = {"type": "3D", "inline_bin": inline_bin, "xline_bin": xline_bin,
                            "origin": [float(o[0]), float(o[1])], "azimuth_deg": float(np.rad2deg(azimuth))}
    out.add_history(f"3D binning: {np.unique(il).size} inlines x {np.unique(xl).size} crosslines")
    return out


def streamer_geometry(data: SeismicData, shot_key: str = "fldr", channel_key: str = "tracf", first_shot_x: float = 0.0,
                      shot_interval: float = 25.0, near_offset: float = 100.0, group_interval: float = 25.0,
                      towed_behind: bool = True, source_depth: float = 0.0, receiver_depth: float = 0.0) -> SeismicData:
    """Source and receiver x from shot and channel numbers for a towed streamer (files without coordinates).

    Shots are numbered in ascending ``shot_key`` order and advance by ``shot_interval``; channel 1 is the nearest group
    at ``near_offset`` and further channels follow every ``group_interval``, behind the source (``towed_behind``) or
    ahead of it.
    """
    out = data.copy()
    shots = data.header(shot_key)
    order = {v: i for i, v in enumerate(np.unique(shots))}
    shot_index = np.array([order[v] for v in shots], dtype=np.float64)
    sx = first_shot_x + shot_index * shot_interval
    channel = data.header(channel_key)
    if not np.any(channel):
        raise ValueError(f"channel header {channel_key!r} is empty")
    off = near_offset + (channel - channel.min()) * group_interval
    gx = sx - off if towed_behind else sx + off
    out.set_header("sx", sx)
    out.set_header("gx", gx)
    out.set_header("sy", np.zeros(data.ntr))
    out.set_header("gy", np.zeros(data.ntr))
    out.set_header("offset", gx - sx)
    out.set_header("sdepth", np.full(data.ntr, source_depth))
    out.set_header("gelev", np.full(data.ntr, -receiver_depth))
    out.add_history(f"streamer geometry: shot interval {shot_interval:g} m, near offset {near_offset:g} m, "
                    f"group interval {group_interval:g} m")
    return out


def header_formula(data: SeismicData, key: str, a: float = 0.0, b: float = 0.0, key2: str = "",
                   c: float = 0.0, key3: str = "", d: float = 1.0) -> SeismicData:
    """key = (a + b * key2 + c * key3) / d, as Seismic Unix's suchw."""
    if d == 0:
        raise ValueError("d must not be zero")
    out = data.copy()
    v2 = data.header(key2) if key2 else 0.0
    v3 = data.header(key3) if key3 else 0.0
    out.set_header(key, (a + b * v2 + c * v3) / d)
    out.add_history(f"header {key} = ({a:g} + {b:g}*{key2 or 0} + {c:g}*{key3 or 0}) / {d:g}")
    return out


def read_xy(path: str):
    """Two-column numeric text file (comma, space, tab or semicolon separated) -> x, y sorted by x."""
    xs, ys = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.replace(",", " ").replace(";", " ").replace("\t", " ").split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue                                      # header line
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f"no numeric x y pairs in {path}")
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def shot_table_geometry(data: SeismicData, table_path: str, topo_path: str | None = None,
                        bin_size: float = 0.0) -> SeismicData:
    """Source and receiver positions of a land 2D line from a shot table, elevations from a topography file.

    Table (CSV with a header row; one row per shot, positions in metres along the line):
        ffid,shot_x,shot_z,first_rec_x,rec_dx[,nchan][,file]
    Receiver x = first_rec_x + (channel - 1) * rec_dx (rec_dx negative when channel 1 is at the far end). shot_z may be
    empty; then it, and every receiver elevation, is interpolated from the topography file (two columns "x z").
    CMP numbers follow with bins of half the receiver spacing unless ``bin_size`` is given.
    """
    import csv
    with open(table_path, newline="", encoding="utf-8-sig") as f:
        rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    rows = [r for r in rows if r.get("ffid")]
    need = {"ffid", "shot_x", "first_rec_x", "rec_dx"}
    if not rows or not need.issubset(rows[0]):
        raise ValueError(f"the shot table needs the columns {sorted(need)} (and optionally shot_z, nchan)")
    topo = read_xy(topo_path) if topo_path else None
    fldr = data.header("fldr").copy()
    by_file = data.meta.get("seg2_ffid_by_file", {})
    renumber = {}
    for r in rows:                     # a 'file' column matches records by file name; they take the table's FFID
        f = os.path.basename(r.get("file", "").replace("\\", "/"))
        if f in by_file and abs(by_file[f] - float(r["ffid"])) > 1e-6:
            renumber[by_file[f]] = float(r["ffid"])
    if renumber:
        old = fldr.copy()
        for a, b in renumber.items():
            fldr[old == a] = b
    chan = data.header("tracf")
    if not np.any(chan):
        raise ValueError("channel numbers (tracf) are not set")
    sx, gx = np.full(data.ntr, np.nan), np.full(data.ntr, np.nan)
    selev, gelev = np.zeros(data.ntr), np.zeros(data.ntr)
    spacing = []
    for r in rows:
        sel = np.abs(fldr - float(r["ffid"])) < 1e-6
        if not sel.any():
            continue
        shot_x, x1, dx = float(r["shot_x"]), float(r["first_rec_x"]), float(r["rec_dx"])
        sx[sel] = shot_x
        gx[sel] = x1 + (chan[sel] - 1.0) * dx
        if r.get("shot_z"):
            selev[sel] = float(r["shot_z"])
        elif topo is not None:
            selev[sel] = np.interp(shot_x, *topo)
        spacing.append(abs(dx))
    missing = np.unique(fldr[np.isnan(sx)])
    if missing.size:
        raise ValueError(f"FFIDs missing from the shot table: {', '.join(f'{v:g}' for v in missing[:20])}")
    if topo is not None:
        gelev = np.interp(gx, *topo)
    out = data.copy()
    out.set_header("fldr", fldr)
    for key, values in (("sx", sx), ("gx", gx), ("sy", 0.0), ("gy", 0.0), ("selev", selev), ("gelev", gelev)):
        out.set_header(key, values)
    out.set_header("offset", gx - sx)
    out.add_history(f"shot-table geometry from {os.path.basename(table_path)}"
                    + (f", elevations from {os.path.basename(topo_path)}" if topo_path else ""))
    size = bin_size if bin_size > 0 else 0.5 * float(np.median(spacing))
    return geometry_2d(out, size)
