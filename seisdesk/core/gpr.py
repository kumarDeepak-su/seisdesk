"""Sensors & Software DT1/HD reader (128-byte headers, signed 16-bit samples).

Layout: EKKO View manual, Appendix B, Sensors & Software. The HD time window is
interpreted as the first-to-last sample span; sample_interval_ns can override it
for acquisition software using a different convention. No seismic time scaling
is applied: all internal times are seconds. Per-trace time-zero adjustments are
aligned onto the union of the original axes, retaining both ends of each trace.
"""
from pathlib import Path
import re

import numpy as np

from .dataset import SeismicData


def read_dt1(path, name=None, *, header_path=None, sample_interval_ns=None):
    path = Path(path)
    if header_path is None:
        header = next((p for p in path.parent.iterdir() if p.stem.lower() == path.stem.lower() and p.suffix.lower() == '.hd'), None)
        if header is None:
            raise ValueError("DT1 import needs its matching .HD file in the same folder")
    else:
        header = Path(header_path)
    info = {}
    for line in header.read_text(encoding="latin1").splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            info[' '.join(key.upper().split())] = value.strip()

    def number(key, default=None):
        value = info.get(key)
        if value is None:
            if default is not None:
                return float(default)
            raise ValueError(f"HD header is missing {key}")
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
        if match is None:
            raise ValueError(f"invalid numeric HD value for {key}")
        return float(match.group())

    ns, ntr = int(number("NUMBER OF PTS/TRC")), int(number("NUMBER OF TRACES"))
    if ns < 2 or ntr < 1 or ns*ntr > 100_000_000:
        raise ValueError("DT1 sample/trace counts are invalid or exceed the in-memory import limit")
    dt = (float(sample_interval_ns) if sample_interval_ns is not None else number("TOTAL TIME WINDOW")/(ns-1))*1e-9
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("GPR sampling must be positive")
    record = 128 + 2*ns
    if path.stat().st_size != ntr*record:
        raise ValueError("DT1 file length does not match the HD trace/sample counts (truncated or unsupported format)")
    raw = np.zeros((ntr, ns), np.float32); positions = np.empty(ntr); adjustment = np.empty(ntr)
    dead = np.zeros(ntr, bool)
    with path.open('rb') as f:
        for i in range(ntr):
            h = np.frombuffer(f.read(128), dtype='<f4', count=25)
            if h[2] != ns or h[5] != 2:
                raise ValueError("only fixed-length DT1 traces with 16-bit samples are supported")
            positions[i], adjustment[i], dead[i] = h[1], h[20], h[21] != 0
            raw[i] = np.frombuffer(f.read(2*ns), dtype='<i2')
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(adjustment)):
        raise ValueError("DT1 coordinates and time-zero adjustments must be finite")
    units = info.get("POSITION UNITS", "metres").lower()
    if units not in ("m", "metres", "meters", "metre", "meter", "ft", "feet", "foot"):
        raise ValueError(f"unsupported position unit {units!r}")
    scale = .3048 if units in ('ft', 'feet', 'foot') else 1.
    positions *= scale
    separation = number("ANTENNA SEPARATION", 0)*scale
    starts = -number("TIMEZERO AT POINT", 0) - adjustment
    first = float(np.floor(starts.min())); last = float(np.ceil((starts+ns-1).max()))
    nt = int(last-first)+1
    if nt*ntr > 100_000_000 or nt > ns*4:
        raise ValueError("DT1 time-zero adjustment is outside a plausible record range")
    grid = first + np.arange(nt)
    traces = np.stack([np.interp(grid, starts[i]+np.arange(ns), raw[i], left=0, right=0) for i in range(ntr)])
    heads = {"tracl": np.arange(1, ntr+1), "tracf": np.arange(1, ntr+1), "cdp": np.arange(1, ntr+1),
             "cdpx": positions, "sx": positions-separation/2, "gx": positions+separation/2,
             "offset": separation, "sy": 0, "gy": 0, "trid": np.where(dead, 2, 1),
             "gpr_timezero_samples": adjustment}
    return SeismicData(traces, dt, heads, first*dt, name or path.stem, meta={
        "format": "Sensors & Software DT1", "display_time_unit": "ns", "hd_header": info,
        "sampling_convention": "explicit dt" if sample_interval_ns is not None else "HD time window / (samples - 1)",
        "timezero_alignment": "per-trace axes aligned with zero padding onto their union"})
