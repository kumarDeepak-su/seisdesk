"""Trace editing, selection, sorting, resampling and geometry steps."""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..core import geometry as geo
from .base import Param, Process, register


def _values(text: str) -> list[float]:
    """'1,5,9-12' -> [1, 5, 9, 10, 11, 12] (integer ranges expanded; single values may be decimals)."""
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part[0] + part[1:].split("-", 1)[0], part[1:].split("-", 1)[1]
            lo, hi = int(float(a)), int(float(b))
            out.extend(range(min(lo, hi), max(lo, hi) + 1))
        else:
            out.append(float(part))
    return out


@register
class KillTraces(Process):
    name = "kill_traces"
    label = "Kill traces"
    category = "Editing"
    description = "Zero (or remove) traces whose header value is in a list or range, e.g. bad channels 12,45-47."
    params = (Param("key", "key", "tracf", "Header key"),
              Param("values", "str", "", "Values (list and ranges)"),
              Param("action", "choice", "zero", "Action", choices=("zero", "remove")))

    def run(self, data, context):
        vals = np.array(_values(self["values"]))
        if vals.size == 0:
            return data.copy()
        hit = np.isin(np.round(data.header(self["key"]), 6), np.round(vals, 6))
        if self["action"] == "remove":
            return data.subset(~hit)
        out = data.copy()
        out.traces[hit] = 0.0
        trid = out.header("trid", 1.0).copy()
        trid[hit] = 2.0
        out.set_header("trid", trid)
        return out


@register
class ReversePolarity(Process):
    name = "reverse_polarity"
    label = "Reverse polarity"
    category = "Editing"
    description = "Multiply selected traces (or all traces when no values are given) by -1."
    params = (Param("key", "key", "tracf", "Header key"), Param("values", "str", "", "Values (empty = all)"))

    def run(self, data, context):
        out = data.copy()
        vals = np.array(_values(self["values"]))
        hit = np.ones(data.ntr, bool) if vals.size == 0 else np.isin(np.round(data.header(self["key"]), 6),
                                                                        np.round(vals, 6))
        out.traces[hit] *= -1.0
        return out


@register
class SelectTraces(Process):
    name = "select_traces"
    label = "Select traces (header range)"
    category = "Editing"
    description = "Keep the traces with min <= header <= max (and optionally every n-th of them)."
    params = (Param("key", "key", "cdp", "Header key"),
              Param("min", "float", -1e30, "Minimum"), Param("max", "float", 1e30, "Maximum"),
              Param("every", "int", 1, "Keep every n-th value", minimum=1))

    def run(self, data, context):
        h = data.header(self["key"])
        keep = (h >= self["min"]) & (h <= self["max"])
        if self["every"] > 1:
            vals = np.unique(h[keep])[:: self["every"]]
            keep &= np.isin(h, vals)
        return data.subset(keep)


@register
class TimeWindow(Process):
    name = "time_window"
    label = "Time window"
    category = "Editing"
    description = "Keep the samples between tmin and tmax."
    params = (Param("tmin", "float", 0.0, "Start", unit="s"), Param("tmax", "float", 4.0, "End", unit="s"))

    def run(self, data, context):
        return data.time_window(self["tmin"], self["tmax"])


@register
class Sort(Process):
    name = "sort"
    label = "Sort"
    category = "Editing"
    description = "Sort traces by header keys, e.g. 'cdp,offset' for CMP gathers or 'fldr,tracf' for shots."
    params = (Param("keys", "keys", "cdp,offset", "Keys (primary first)"),)

    def run(self, data, context):
        return data.sort(self["keys"])


@register
class Resample(Process):
    name = "resample"
    label = "Resample"
    category = "Editing"
    description = "Change the sample interval with an anti-alias polyphase filter."
    params = (Param("dt", "float", 0.004, "New sample interval", unit="s", minimum=1e-5),)

    def run(self, data, context):
        from fractions import Fraction
        ratio = Fraction(data.dt / self["dt"]).limit_denominator(1000)
        up, down = ratio.numerator, ratio.denominator
        out = signal.resample_poly(data.traces.astype(np.float64), up, down, axis=1)
        return data.copy(out.astype(np.float32), dt=data.dt * down / up)


@register
class HeaderFormula(Process):
    name = "header_formula"
    label = "Set header (formula)"
    category = "Geometry"
    description = "key = (a + b*key2 + c*key3) / d, like Seismic Unix suchw."
    params = (Param("key", "key", "cdp", "Header to set"), Param("a", "float", 0.0, "a"),
              Param("b", "float", 1.0, "b"), Param("key2", "key", "", "key2"), Param("c", "float", 0.0, "c"),
              Param("key3", "key", "", "key3"), Param("d", "float", 1.0, "d"))

    def run(self, data, context):
        return geo.header_formula(data, self["key"], self["a"], self["b"], self["key2"], self["c"], self["key3"],
                                  self["d"])


@register
class Geometry2D(Process):
    name = "geometry_2d"
    label = "2D geometry (offset, CMP)"
    category = "Geometry"
    description = "Signed offset and CMP numbers from source/receiver coordinates; crooked lines projected on the best-fit line."
    params = (Param("bin_size", "float", 12.5, "CMP bin size", unit="m", minimum=0.01),
              Param("origin", "float", -1e30, "Origin along line (default: first midpoint)", unit="m"),
              Param("crooked", "bool", True, "Project crooked lines"))

    def run(self, data, context):
        origin = None if self["origin"] <= -1e29 else self["origin"]
        return geo.geometry_2d(data, self["bin_size"], origin, self["crooked"])


@register
class SPSGeometry(Process):
    name = "sps_geometry"
    label = "SPS source / receiver / relation geometry"
    category = "Geometry"
    description = "SPS 2.0/2.1 S/R/X geometry with strict FFID/channel matching, component groups, point indices and unit checks."
    params = (Param("source_file", "infile", "", "S source file"),
              Param("receiver_file", "infile", "", "R receiver file"),
              Param("relation_file", "infile", "", "X relation file"),
              Param("coordinate_factor", "float", 0., "Grid units to metres (0 = H201)", minimum=0.),
              Param("tape", "str", "", "Tape identifier filter (blank = all)"),
              Param("sequence_key", "key", "segd_line_sequence", "Line sequence header, if present"),
              Param("copy_statics", "bool", False, "Copy SPS corrections to static headers (does not shift samples)"))

    def run(self, data, context):
        from ..core.sps import apply_sps
        import os
        paths = [os.path.join(context.work_dir, self[k]) for k in ("source_file", "receiver_file", "relation_file")]
        return apply_sps(data, *paths, coordinate_factor=self["coordinate_factor"] or None, tape=self["tape"],
                         sequence_key=self["sequence_key"], apply_statics=self["copy_statics"])


@register
class Geometry3D(Process):
    name = "geometry_3d"
    label = "3D binning (inline/crossline)"
    category = "Geometry"
    description = "Inline and crossline numbers from midpoints in a rotated grid (defaults estimated from the data)."
    params = (Param("inline_bin", "float", 25.0, "Bin size along inline", unit="m", minimum=0.01),
              Param("xline_bin", "float", 25.0, "Bin size along crossline", unit="m", minimum=0.01),
              Param("origin_x", "float", -1e30, "Origin x (default: auto)", unit="m"),
              Param("origin_y", "float", -1e30, "Origin y (default: auto)", unit="m"),
              Param("azimuth", "float", -1e30, "Inline azimuth from x axis (default: auto)", unit="deg"))

    def run(self, data, context):
        auto = lambda v: None if v <= -1e29 else v
        return geo.geometry_3d(data, self["inline_bin"], self["xline_bin"], auto(self["origin_x"]),
                               auto(self["origin_y"]), auto(self["azimuth"]))


@register
class StreamerGeometry(Process):
    name = "streamer_geometry"
    label = "Streamer geometry (no coordinates)"
    category = "Geometry"
    description = "Source and receiver x from shot and channel numbers for towed-streamer data without coordinates."
    params = (Param("shot_key", "key", "fldr", "Shot key"), Param("channel_key", "key", "tracf", "Channel key"),
              Param("first_shot_x", "float", 0.0, "First shot x", unit="m"),
              Param("shot_interval", "float", 25.0, "Shot interval", unit="m"),
              Param("near_offset", "float", 100.0, "Near offset (channel 1)", unit="m"),
              Param("group_interval", "float", 25.0, "Group interval", unit="m", minimum=0.01),
              Param("towed_behind", "bool", True, "Streamer behind the source"))

    def run(self, data, context):
        return geo.streamer_geometry(data, self["shot_key"], self["channel_key"], self["first_shot_x"],
                                     self["shot_interval"], self["near_offset"], self["group_interval"],
                                     self["towed_behind"])


@register
class ShotTableGeometry(Process):
    name = "shot_table_geometry"
    label = "Land geometry from shot table (CSV)"
    category = "Geometry"
    description = ("Source/receiver x and elevations of a land 2D line from a shot table (columns ffid, shot_x, shot_z, "
                   "first_rec_x, rec_dx; one row per shot) and a topography file (x z), then offsets and CMPs. Same "
                   "table as the SEG-2 conversion script.")
    params = (Param("table", "infile", "geometry.csv", "Shot table (CSV)"),
              Param("topography", "infile", "", "Topography file x z (optional)"),
              Param("bin_size", "float", 0.0, "CMP bin (0 = half the receiver spacing)", unit="m", minimum=0.0))

    def run(self, data, context):
        import os
        table = self["table"] if os.path.isabs(self["table"]) else os.path.join(context.work_dir, self["table"])
        topo = self["topography"].strip()
        if topo and not os.path.isabs(topo):
            topo = os.path.join(context.work_dir, topo)
        return geo.shot_table_geometry(data, table, topo or None, self["bin_size"])
