"""SEG SPS 2.1 fixed-column source/receiver/relation geometry.

Reference: https://seg.org/wp-content/uploads/2025/11/seg_sps_rev2.1.pdf
Coordinates require H201 or an explicit conversion factor. Other lengths are metres.
No map projection, datum transformation or static correction is applied implicitly.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import os
import numpy as np


def _number(line, first, last, default=None):
    raw = line[first-1:last].strip()
    if not raw and default is not None:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"missing/invalid numeric field in columns {first}-{last}") from None
    if not np.isfinite(value):
        raise ValueError(f"non-finite field in columns {first}-{last}")
    return value


def _integer(line, first, last, default=None):
    value = _number(line, first, last, default)
    if value != int(value):
        raise ValueError(f"non-integer field in columns {first}-{last}")
    return int(value)


def _point_key(line, first, point, index):
    values = []
    for col in (first, point):
        try:
            value = Decimal(line[col-1:col+9].strip())*100
            if not value.is_finite() or value != value.to_integral_value():
                raise InvalidOperation
            values.append(int(value))
        except InvalidOperation:
            raise ValueError(f"invalid SPS line/point at column {col}") from None
    idx = _integer(line, index, index, 1)
    if not 1 <= idx <= 9:
        raise ValueError("SPS point index must be 1 through 9")
    return (*values, idx)


def _records(path, kind):
    headers, rows, sequence = [], [], None
    with open(path, encoding="ascii") as stream:
        for lineno, raw in enumerate(stream, 1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if line.startswith("H"):
                headers.append(line)
                if line.startswith("H00 ") and "SPS 2.1" not in line and "SPS 2.0" not in line:
                    raise ValueError(f"{path}:{lineno}: only SPS 2.0/2.1 layouts are supported")
                if line.startswith("H023"):
                    sequence = int(line[32:].strip().rstrip(";"))
                continue
            if line.startswith("C"):
                continue
            if not line.startswith(kind) or len(line) != 80:
                raise ValueError(f"{path}:{lineno}: expected an 80-column {kind} record")
            rows.append((lineno, line, sequence))
    if not rows:
        raise ValueError(f"{path}: no {kind} records")
    return headers, rows


def read_points(path, kind, coordinate_factor=None):
    if kind not in ("S", "R"):
        raise ValueError("point kind must be S or R")
    headers, rows = _records(path, kind)
    if coordinate_factor is None:
        factors = {float(h[32:46].strip().rstrip(";")) for h in headers if h.startswith("H201")}
        if len(factors) != 1:
            raise ValueError("SPS geometry needs one H201 metres conversion factor or an explicit coordinate factor")
        coordinate_factor = factors.pop()
    if not np.isfinite(coordinate_factor) or coordinate_factor <= 0:
        raise ValueError("SPS coordinate factor must be positive and finite")
    points = {}
    for lineno, line, _ in rows:
        try:
            key = _point_key(line, 2, 12, 24)
            if key in points:
                raise ValueError(f"duplicate {kind} line/point/index {key}")
            points[key] = dict(x=_number(line, 47, 55)*coordinate_factor,
                               y=_number(line, 56, 65)*coordinate_factor,
                               elevation=_number(line, 66, 71), depth=_number(line, 31, 34, 0),
                               datum=_number(line, 35, 38, 0), static_ms=_number(line, 27, 30, np.nan),
                               uphole_ms=_number(line, 39, 40, np.nan), water=_number(line, 41, 46, np.nan),
                               code=line[24:26].strip())
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return points, dict(headers=headers, coordinate_to_m=coordinate_factor)


def read_relations(path):
    headers, rows = _records(path, "X")
    relations = []
    for lineno, line, seq in rows:
        try:
            r = dict(tape=line[1:7].strip(), ffid=_integer(line, 8, 15),
                     repeats=_integer(line, 16, 16, 1), source=_point_key(line, 18, 28, 38),
                     first=_integer(line, 39, 43), last=_integer(line, 44, 48),
                     components=_integer(line, 49, 49, 1),
                     receiver_first=_point_key(line, 50, 60, 80),
                     receiver_last=_point_key(line, 50, 70, 80), sequence=seq)
            if r["ffid"] < 0 or not 1 <= r["repeats"] <= 9 or not 1 <= r["components"] <= 9:
                raise ValueError("invalid record/channel increment")
            if r["first"] < 1 or r["last"] < r["first"]:
                raise ValueError("invalid channel range")
            groups = (r["last"]-r["first"])//r["components"]+1
            delta = r["receiver_last"][1]-r["receiver_first"][1]
            if (groups == 1 and delta != 0) or (groups > 1 and (delta == 0 or delta % (groups-1))):
                raise ValueError("receiver endpoints cannot be matched to the channel groups at 0.01 point precision")
            r["point_step"] = delta//(groups-1) if groups > 1 else 0
            relations.append(r)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return relations, headers


def apply_sps(data, source_file, receiver_file, relation_file, coordinate_factor=None,
              tape="", sequence_key="segd_line_sequence", apply_statics=False):
    """Match every trace by FFID/channel, with tape and line sequence disambiguation.

    Acquisition channel numbers may restart by channel set. Select the seismic set
    before applying X relations if those numbers are ambiguous. Multicomponent X
    records map consecutive component channels to the same station. The X record
    increment represents consecutive field records using the same source/spread.
    """
    sources, smeta = read_points(source_file, "S", coordinate_factor)
    receivers, rmeta = read_points(receiver_file, "R", coordinate_factor)
    relations, xheaders = read_relations(relation_file)
    by_record = {}
    for rel in relations:
        if tape and rel["tape"] != tape:
            continue
        for record in range(rel["ffid"], rel["ffid"]+rel["repeats"]):
            by_record.setdefault(record, []).append(rel)
    ffid, chan = data.header("fldr", None), data.header("tracf", None)
    if not np.isfinite(ffid).all() or not np.isfinite(chan).all() or np.any(ffid != np.rint(ffid)) or np.any(chan != np.rint(chan)):
        raise ValueError("SPS matching requires integer FFID and channel headers")
    out = data.copy()
    keys = ("sx", "sy", "gx", "gy", "selev", "gelev", "sdepth", "receiver_depth", "sdel", "gdel", "sps_receiver_surface_elevation",
            "sps_source_line", "sps_source_point", "sps_source_index", "sps_receiver_line", "sps_receiver_point",
            "sps_receiver_index", "sps_component", "sps_source_static_ms", "sps_receiver_static_ms", "sps_live")
    values = {key: np.zeros(data.ntr) for key in keys}
    seq = data.headers.get(sequence_key) if sequence_key else None
    if "segd_channel_set" in data.headers:
        seen = {}
        for i in range(data.ntr):
            key = (ffid[i], chan[i], seq[i] if seq is not None else None)
            cs = data.header("segd_channel_set")[i]
            if key in seen and seen[key] != cs:
                raise ValueError("SEG-D channel numbers repeat across sets; select one set before applying SPS relations")
            seen[key] = cs
    for i, (f, c) in enumerate(zip(ffid.astype(int), chan.astype(int))):
        matches = [r for r in by_record.get(f, []) if r["first"] <= c <= r["last"] and
                   (seq is None or seq[i] == 0 or r["sequence"] is None or r["sequence"] == seq[i])]
        if len(matches) != 1:
            raise ValueError(f"trace {i+1}: FFID {f}, channel {c} has {len(matches)} SPS relations; "
                             "check tape, line sequence, channel numbering and X overlaps")
        r = matches[0]
        group, comp = divmod(c-r["first"], r["components"])
        recv = (r["receiver_first"][0], r["receiver_first"][1]+group*r["point_step"], r["receiver_first"][2])
        try:
            s, g = sources[r["source"]], receivers[recv]
        except KeyError as exc:
            raise ValueError(f"trace {i+1}: SPS relation references missing point {exc.args[0]}") from exc
        for prefix, point, identity in (("source", s, r["source"]), ("receiver", g, recv)):
            for key, v in zip(("line", "point", "index"), (identity[0]/100, identity[1]/100, identity[2])):
                values[f"sps_{prefix}_{key}"][i] = v
            values[f"sps_{prefix}_static_ms"][i] = point["static_ms"]
        for key, value in dict(sx=s["x"], sy=s["y"], gx=g["x"], gy=g["y"], selev=s["elevation"], gelev=g["elevation"]-g["depth"],
                               sps_receiver_surface_elevation=g["elevation"],
                               sdepth=s["depth"], receiver_depth=g["depth"], sdel=s["datum"], gdel=g["datum"],
                               sps_component=comp+1, sps_live=int(s["code"] not in ("KL", "PM") and g["code"] not in ("KL", "PM"))).items():
            values[key][i] = value
    if apply_statics:
        if not np.isfinite(values["sps_source_static_ms"]).all() or not np.isfinite(values["sps_receiver_static_ms"]).all():
            raise ValueError("some SPS statics are blank; cannot copy them into processing static headers")
        # Copy correction headers only; the statics process applies the actual shift.
        values.update(sstat=values["sps_source_static_ms"].copy(), gstat=values["sps_receiver_static_ms"].copy())
    values.update(offset=np.hypot(values["gx"]-values["sx"], values["gy"]-values["sy"]),
                  cdpx=(values["gx"]+values["sx"])/2, cdpy=(values["gy"]+values["sy"])/2,
                  counit=np.ones(data.ntr))
    values["trid"] = np.where(values["sps_live"] > 0, data.header("trid", 1), 2)
    out.headers.update(values)
    files = []
    for path in (source_file, receiver_file, relation_file):
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        files.append(dict(path=os.path.abspath(path), sha256=digest))
    out.meta["sps_geometry"] = dict(format="SPS 2.1", source=smeta, receiver=rmeta, relation_headers=xheaders,
                                     files=files, matched_traces=data.ntr, tape_filter=tape,
                                     offset_convention="unsigned horizontal distance; run 2D geometry for signed offsets",
                                     static_headers_copied=bool(apply_statics), coordinate_transform="none")
    out.add_history(f"SPS geometry: {data.ntr} traces matched by FFID/channel; coordinates in metres")
    return out
