"""Sequential SU and SEG-Y rev 0/1 import for nonuniform trace timing/length.

The regular segyio path remains available. This reader handles standard 240-byte
trace headers and codes 1/2/3/5/8; unsupported revisions/layouts fail explicitly.
Reference: SEG-Y rev 1 (2002), binary and trace header tables.
"""
from __future__ import annotations

import os
import struct
import numpy as np

from .dataset import SeismicData
from .headers import SEGY_FIELDS, STANDARD, SU_EXTRA
from .sampling import common_samples, timing_headers


def decode_samples(raw, code, endian):
    formats = {2: "i4", 3: "i2", 5: "f4", 8: "i1"}
    if code in formats:
        return np.frombuffer(raw, dtype=endian+formats[code]).astype(np.float64)
    if code == 1:
        word = np.frombuffer(raw, dtype=endian+"u4").astype(np.uint64)
        return ((1-2*(word >> 31).astype(float)) * (word & 0xffffff)/float(1 << 24) *
                np.power(16., ((word >> 24) & 127).astype(float)-64))
    raise ValueError(f"unsupported SEG-Y sample format {code}; supported: IBM32, int32, int16, IEEE32, int8")


def read_variable(path, name=None, endian=">", su=False, target_dt=None, window="strict", max_mb=1024):
    if not np.isfinite(max_mb) or max_mb <= 0:
        raise ValueError("import memory limit must be positive and finite")
    from .segy_io import _to_real_units
    fields = STANDARD+SU_EXTRA if su else SEGY_FIELDS
    byte_formats = [(key, byte-1, np.dtype(endian+fmt)) for key, byte, fmt, _ in fields]
    rows, heads, intervals, starts = [], {key: [] for key, *_ in fields}, [], []
    text, interval, binary_ns, code, measurement = "", 0, 0, 5, 1
    raw_budget = max_mb*1024**2
    sample_count = 0
    with open(path, "rb") as f:
        if not su:
            rawtext, binary = f.read(3200), f.read(400)
            if len(binary) != 400:
                raise ValueError("truncated SEG-Y file header")
            revision = int.from_bytes(binary[300:302], "little" if endian == "<" else "big")
            if revision not in (0, 0x100):
                raise ValueError("variable-axis SEG-Y import supports revision 0/1 only")
            interval, binary_ns, code = [struct.unpack_from(endian+"H", binary, offset)[0] for offset in (16, 20, 24)]
            measurement = struct.unpack_from(endian+"H", binary, 54)[0]
            ascii_text = rawtext.decode("ascii", "replace")
            ebcdic = rawtext.decode("cp500", "replace")
            text = max((ascii_text, ebcdic), key=lambda s: sum(c.isascii() and c.isprintable() for c in s))
            text = "\n".join(text[i:i+80].rstrip() for i in range(0, 3200, 80))
            ext = struct.unpack_from(endian+"h", binary, 304)[0] if revision else 0
            if ext < -1 or ext > 1024:
                raise ValueError("invalid SEG-Y extended textual header count")
            for _ in range(ext if ext >= 0 else 1024):
                block = f.read(3200)
                if len(block) != 3200:
                    raise ValueError("truncated extended SEG-Y textual header")
                if ext == -1 and any("((SEG: EndText))" in block.decode(enc, "replace") for enc in ("ascii", "cp500")):
                    break
            else:
                if ext == -1:
                    raise ValueError("SEG-Y EndText marker not found within 1024 blocks")
        size = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}.get(code)
        if size is None:
            raise ValueError(f"unsupported SEG-Y format code {code} for variable-axis import")
        file_size = os.fstat(f.fileno()).st_size
        while f.tell() < file_size:
            h = f.read(240)
            if len(h) != 240:
                raise ValueError("truncated seismic trace header")
            values = {key: float(np.frombuffer(h, dtype=fmt, count=1, offset=off)[0]) for key, off, fmt in byte_formats}
            ns = int(values["ns"]) or binary_ns
            dt_us = values["dt"] or interval
            if ns <= 0 or dt_us <= 0 or f.tell()+size*ns > file_size:
                raise ValueError("invalid/truncated seismic trace or missing sample interval")
            sample_count += ns
            if sample_count*8 > raw_budget:
                raise ValueError(f"input samples exceed the {max_mb:g} MiB import limit")
            rows.append(decode_samples(f.read(size*ns), code, endian))
            intervals.append(dt_us*1e-6); starts.append(values["delrt"]*1e-3)
            for key, value in values.items():
                heads[key].append(value)
    traces, dt, t0, timing = common_samples(rows, intervals, starts, target_dt, window, max_mb)
    heads = _to_real_units({key: np.asarray(v) for key, v in heads.items()})
    timing_headers(heads, timing, traces.shape[1], dt, t0)
    out = SeismicData(traces, dt, heads, t0, name or os.path.splitext(os.path.basename(path))[0])
    out.meta.update(source_file=os.path.abspath(path), import_timing=timing, textual_header=text,
                    segy_measurement_system=measurement if not su else None)
    # The data model uses metres. Angular coordinates require a separate explicit projection.
    if not su:
        if np.any(~np.isin(heads["counit"], [0, 1])):
            raise ValueError("angular SEG-Y coordinates require projection to length units before import")
        if measurement == 2:
            from .headers import COORDINATE_KEYS, ELEVATION_KEYS
            for key in (*COORDINATE_KEYS, *ELEVATION_KEYS, "offset"):
                if key in out.headers:
                    out.headers[key] *= .3048
            out.meta["coordinate_conversion"] = "SEG-Y feet to metres, factor 0.3048"
        elif measurement not in (0, 1):
            raise ValueError("unknown SEG-Y measurement system")
    out.add_history(f"read {'SU' if su else 'SEG-Y'}; {window} time window; dt {dt:g} s")
    return out
