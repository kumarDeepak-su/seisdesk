"""SEG-2 input: the shot files written by engineering seismographs (Geometrics, OYO, ABEM, Seistronix, ...).

A SEG-2 file holds one record: a file descriptor block, a table of pointers and, for every trace, a descriptor block
of keyword strings (SAMPLE_INTERVAL, DELAY, CHANNEL_NUMBER, RECEIVER_LOCATION, SOURCE_LOCATION, DESCALING_FACTOR, ...)
followed by the samples. All five standard sample encodings are supported. DELAY is checked per trace;
normalised records start at shot time zero, using exact integer shifts or a windowed-sinc fractional shift.
SKEW is retained separately and is only added to DELAY when explicitly requested for the instrument in use.

Format reference: Pullan (1990), SEG-2 standard, Appendix B and keyword definitions:
https://seg.org/wp-content/uploads/2025/11/seg_2.pdf
"""

from __future__ import annotations

import os
import re
import struct
import warnings

import numpy as np

from .dataset import SeismicData

_FORMATS = {1: "i2", 2: "i4", 4: "f4", 5: "f8"}


def is_seg2(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) in (b"\x55\x3a", b"\x3a\x55")
    except OSError:
        return False


def _strings(buf: bytes, start: int, end: int, endian: str, term: bytes) -> dict:
    """Keyword strings: each begins with a 2-byte offset to the next string and ends with the terminator."""
    out = {}
    pos = start
    while pos + 2 <= end:
        (step,) = struct.unpack(endian + "H", buf[pos:pos + 2])
        if step == 0:
            break
        if step < 3 or pos+step > end:
            raise ValueError("malformed SEG-2 keyword block")
        raw = buf[pos + 2:min(pos + step, end)]
        raw = raw.split(term, 1)[0] if term else raw
        text = raw.split(b"\x00", 1)[0].decode("latin-1", "replace").strip()
        if text:
            parts = text.split(None, 1)
            out[parts[0].upper()] = parts[1].strip() if len(parts) > 1 else ""
        pos += step
    return out


def _first_float(text, default=None):
    try:
        return float(str(text).replace(",", " ").split()[0])
    except (ValueError, IndexError):
        return default


def _number(info, key, default=None):
    value = _first_float(info.get(key), default)
    if (key in info and _first_float(info[key]) is None) or value is None or not np.isfinite(value):
        raise ValueError(f"SEG-2 {key} must be a finite number")
    return value


def _packed20(buf, offset, ns, endian):
    """Four signed one's-complement mantissas and four binary exponents per 10 bytes."""
    if ns % 4:
        raise ValueError("SEG-2 packed 20-bit sample count must be divisible by four")
    words = np.frombuffer(buf, dtype=endian+"u2", count=ns//4*5, offset=offset).reshape(-1, 5)
    raw = words[:, 1:].astype(np.int64)
    mantissa = np.where(raw & 0x8000, -(raw ^ 0xffff), raw)
    exponent = (words[:, :1].astype(np.int64) >> np.arange(0, 16, 4)) & 15
    return (mantissa * (1 << exponent)).ravel().astype(float)


def _align(data, start, dt, nt):
    shift = start/dt
    out = np.zeros(nt)
    if abs(shift-round(shift)) < 1e-8:
        shift = int(round(shift))
        lo, hi = max(0, shift), min(nt, shift+len(data))
        if hi > lo:
            out[lo:hi] = data[lo-shift:hi-shift]
        return out
    # A finite Lanczos sinc kernel avoids rounding sub-sample acquisition delays.
    base = int(np.floor(shift))
    distance = np.arange(-16, 17)-(shift-base)
    kernel = np.sinc(distance)*np.sinc(distance/16)
    kernel[np.abs(distance) >= 16] = 0
    kernel /= kernel.sum()
    filtered = np.convolve(data, kernel)
    index = np.arange(nt)-base+16
    valid = (index >= 0) & (index < len(filtered)) & (np.arange(nt) >= shift) & (np.arange(nt) <= shift+len(data)-1)
    out[valid] = filtered[index[valid]]
    return out


def read_seg2(path: str, name: str | None = None, remove_delay: bool = True, descale: bool = True,
              apply_skew: bool = False, *, target_dt=None, window=None, max_mb=1024) -> SeismicData:
    """One SEG-2 file -> SeismicData (one record: fldr from SHOT_SEQUENCE_NUMBER or the file name)."""
    if not np.isfinite(max_mb) or max_mb <= 0 or os.path.getsize(path) > max_mb*1024**2:
        raise ValueError("SEG-2 file exceeds a valid positive import memory limit")
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 32:
        raise ValueError(f"{os.path.basename(path)}: truncated SEG-2 file descriptor")
    if buf[:2] == b"\x55\x3a":
        e = "<"
    elif buf[:2] == b"\x3a\x55":
        e = ">"
    else:
        raise ValueError(f"{os.path.basename(path)} is not a SEG-2 file")
    rev, m_size, ntr = struct.unpack(e + "HHH", buf[2:8])
    if rev != 1:
        raise ValueError(f"unsupported SEG-2 revision {rev}")
    if ntr == 0 or m_size < 4*ntr or 32+m_size > len(buf):
        raise ValueError("invalid SEG-2 trace pointer table")
    term_size = buf[8]
    if term_size not in (1, 2):
        raise ValueError("invalid SEG-2 string terminator")
    term = buf[9:9 + term_size]
    pointers = [int(p) for p in np.frombuffer(buf, dtype=e + "u4", count=ntr, offset=32)]
    if len(set(pointers)) != ntr or min(pointers) < 32+m_size or max(pointers)+32 > len(buf):
        raise ValueError("invalid or truncated SEG-2 trace pointers")
    ordered = sorted(pointers)
    limits = dict(zip(ordered, ordered[1:]+[len(buf)]))
    file_info = _strings(buf, 32 + m_size, min(pointers) if pointers else len(buf), e, term)
    rows, infos = [], []
    allocated = 0
    for k, p in enumerate(pointers):
        tid, x_size, y_size, ns = struct.unpack(e + "HHII", buf[p:p + 12])
        if tid != 0x4422:
            raise ValueError(f"{os.path.basename(path)}: trace {k + 1} has no SEG-2 trace descriptor")
        code = buf[p + 12]
        if code not in (*_FORMATS, 3):
            raise ValueError(f"unknown SEG-2 data format code {code}")
        if code == 3 and ns % 4:
            raise ValueError("SEG-2 packed 20-bit sample count must be divisible by four")
        needed = ns//4*10 if code == 3 else ns*np.dtype(_FORMATS[code]).itemsize
        if ns == 0 or x_size < 32 or y_size < needed or p+x_size+y_size > limits[p]:
            raise ValueError(f"{os.path.basename(path)}: trace {k+1} has a truncated or invalid data block")
        allocated += ns*8
        if allocated > max_mb*1024**2:
            raise ValueError("decoded SEG-2 samples exceed the import memory limit")
        info = dict(file_info, **_strings(buf, p + 32, p + x_size, e, term))
        data = (_packed20(buf, p+x_size, ns, e) if code == 3 else
                np.frombuffer(buf, dtype=e + _FORMATS[code], count=ns, offset=p + x_size).astype(np.float64))
        if descale and "DESCALING_FACTOR" in info:
            data = data * _number(info, "DESCALING_FACTOR")
        if not np.isfinite(data).all() or np.any(np.abs(data) > np.finfo(np.float32).max):
            raise ValueError(f"{os.path.basename(path)}: trace {k+1} has non-finite or overflowing samples")
        rows.append(data)
        infos.append(info)
    if not rows:
        raise ValueError(f"{os.path.basename(path)} has no traces")
    intervals = np.array([_number(info, "SAMPLE_INTERVAL") for info in infos])
    dt = intervals[0]
    if np.any(intervals <= 0) or (target_dt is None and not np.allclose(intervals, dt, rtol=1e-9, atol=1e-12)):
        raise ValueError("SEG-2 traces have different or invalid sample intervals; resample them before combining")
    delays = np.array([_number(info, "DELAY", 0) for info in infos])
    skews = np.array([_number(info, "SKEW", 0) for info in infos])
    starts = delays + (skews if apply_skew else 0)
    if not apply_skew and np.any(skews != 0):
        warnings.warn("SEG-2 SKEW is retained but not added to DELAY. Check the instrument's timing convention; "
                      "select apply_skew=True only when DELAY excludes the channel scan offset.", UserWarning)
    timing = None
    if target_dt is not None or window is not None:
        from .sampling import common_samples
        traces, dt, t0, timing = common_samples(rows, intervals, starts, target_dt,
                                                window or ("zero" if remove_delay else "intersection"), max_mb)
    elif remove_delay:
        end = min(start+(len(row)-1)*dt for start, row in zip(starts, rows))
        nt = int(np.floor(end/dt+1e-8))+1
        if nt <= 0:
            raise ValueError("SEG-2 records contain no samples at or after shot time zero")
        traces = np.stack([_align(row, start, dt, nt) for row, start in zip(rows, starts)]).astype(np.float32)
        t0 = 0.
    else:
        if not np.allclose(starts, starts[0], rtol=0, atol=dt*1e-8):
            raise ValueError("SEG-2 traces have different start times; enable delay alignment before combining")
        nt, t0 = min(map(len, rows)), float(starts[0])
        traces = np.stack([row[:nt] for row in rows]).astype(np.float32)

    fldr = _first_float(file_info.get("SHOT_SEQUENCE_NUMBER", infos[0].get("SHOT_SEQUENCE_NUMBER")))
    if fldr is None:
        digits = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
        fldr = float(digits[-1]) if digits else 1.0
    n = traces.shape[0]
    chan = np.array([_first_float(i.get("CHANNEL_NUMBER"), k + 1) for k, i in enumerate(infos)])
    heads = {"fldr": np.full(n, fldr), "tracf": chan, "trid": np.ones(n),
             "seg2_delay_s": delays, "seg2_skew_s": skews}
    if timing is not None:
        from .sampling import timing_headers
        timing_headers(heads, timing, traces.shape[1], dt, t0)
    gx = [_first_float(i.get("RECEIVER_LOCATION")) for i in infos]
    sx = [_first_float(i.get("SOURCE_LOCATION", file_info.get("SOURCE_LOCATION"))) for i in infos]
    if all(v is not None for v in gx):
        heads["gx"] = np.array(gx)
    if all(v is not None for v in sx):
        heads["sx"] = np.array(sx)
    if "gx" in heads and "sx" in heads:
        heads["offset"] = heads["gx"] - heads["sx"]
    out = SeismicData(traces, dt, heads, t0,
                      name or os.path.splitext(os.path.basename(path))[0])
    out.meta["source_file"] = os.path.abspath(path)
    out.meta["seg2_file_info"] = file_info
    out.meta["seg2_trace_info"] = infos
    if timing is not None:
        out.meta["import_timing"] = timing
    out.meta["seg2_delay_s"] = float(delays[0])  # Compatibility; per-trace values are in headers.
    out.meta["seg2_timing"] = dict(aligned_to_shot_zero=bool(remove_delay), skew_applied=bool(apply_skew),
                                   fractional_shift="Lanczos sinc, 16-sample half-width",
                                   original_sample_counts=list(map(len, rows)), end_policy="shortest record")
    if timing is not None:
        out.meta["seg2_timing"].update(aligned_to_shot_zero=timing["window"] == "zero",
                                       fractional_shift=timing["filter"], end_policy=timing["window"])
    out.add_history(f"read SEG-2 {os.path.basename(path)} (per-trace DELAY {'aligned' if remove_delay else 'retained'}; "
                    f"SKEW {'added' if apply_skew else 'retained only'}; common end at {out.tmax:g} s)")
    return out


def read_seg2_files(paths, name: str | None = None, remove_delay: bool = True,
                    descale: bool = True, apply_skew: bool = False, *, target_dt=None, window=None,
                    max_mb=1024) -> SeismicData:
    """Several SEG-2 shot files -> one dataset; each file keeps its own FFID (fldr), numbered by file order if the
    files do not give distinct ones. Records are cut to the shortest one."""
    paths = list(paths)
    if not paths:
        raise ValueError("no files")
    records = [read_seg2(p, remove_delay=remove_delay, descale=descale, apply_skew=apply_skew,
                         target_dt=target_dt, window=window, max_mb=max_mb) for p in paths]
    dt = records[0].dt
    for p, r in zip(paths, records):
        if not np.isclose(r.dt, dt, rtol=1e-9, atol=1e-12):
            raise ValueError(f"{os.path.basename(p)}: sample interval {r.dt * 1e3:g} ms differs from "
                             f"{dt * 1e3:g} ms of the first file")
        if target_dt is None and not np.isclose(r.t0, records[0].t0, rtol=0, atol=dt*1e-8):
            raise ValueError("SEG-2 files have different start times; enable delay alignment before combining")
    ffids = [float(r.header("fldr")[0]) for r in records]
    if len(set(ffids)) != len(ffids):
        for k, r in enumerate(records):
            r.set_header("fldr", np.full(r.ntr, float(k + 1)))
    nt = min(r.nt for r in records)
    if sum(r.traces.nbytes for r in records) > max_mb*1024**2:
        raise ValueError("combined SEG-2 records exceed the import memory limit")
    keys = sorted(set().union(*(r.headers for r in records)))
    traces = np.concatenate([r.traces[:, :nt] for r in records])
    heads = {k: np.concatenate([r.header(k) for r in records]) for k in keys}
    t0 = records[0].t0
    combined_timing = None
    if target_dt is not None:
        from .sampling import common_samples
        traces, dt, t0, combined_timing = common_samples(
            [row for r in records for row in r.traces], [r.dt for r in records for _ in range(r.ntr)],
            [r.t0 for r in records for _ in range(r.ntr)], target_dt,
            window or ("zero" if remove_delay else "intersection"), max_mb)
        for key, value in (("ns", traces.shape[1]), ("dt", dt*1e6), ("delrt", t0*1e3)):
            heads[key] = np.full(traces.shape[0], value)
    first = os.path.splitext(os.path.basename(paths[0]))[0]
    out = SeismicData(traces, dt, heads, t0, name or (f"{first} ({len(paths)} shots)" if len(paths) > 1
                                                                  else first))
    out.meta["source_files"] = [os.path.abspath(p) for p in paths]
    out.meta["seg2_ffid_by_file"] = {os.path.basename(p): float(r.header("fldr")[0]) for p, r in zip(paths, records)}
    out.meta["seg2_records"] = [r.meta for r in records]
    if combined_timing is not None:
        out.meta["record_alignment"] = combined_timing
    out.meta["seg2_timing"] = dict(aligned_to_shot_zero=bool(remove_delay), skew_applied=bool(apply_skew),
                                   original_record_samples=[r.nt for r in records], end_policy="shortest record")
    if combined_timing is not None:
        out.meta["seg2_timing"].update(aligned_to_shot_zero=combined_timing["window"] == "zero",
                                       end_policy=combined_timing["window"])
    out.add_history(f"read {len(paths)} SEG-2 files, {out.ntr} traces; common end at {out.tmax:g} s")
    return out


def write_seg2(data: SeismicData, path: str, record: float | None = None) -> None:
    """Write one record as little-endian SEG-2 with float32 samples (mainly for tests and data exchange)."""
    sel = np.ones(data.ntr, bool) if record is None else data.header("fldr") == record
    traces = data.traces[sel]
    chan = data.header("tracf")[sel]
    gx, sx = data.header("gx")[sel], data.header("sx")[sel]
    ntr, nt = traces.shape
    m_size = 4 * ntr
    file_strings = [f"SHOT_SEQUENCE_NUMBER {int(data.header('fldr')[sel][0])}"]

    def block(strings):
        out = b""
        for s in strings:
            body = s.encode("latin-1") + b"\x00"
            out += struct.pack("<H", len(body) + 2) + body
        return out + b"\x00\x00"

    head = struct.pack("<HHHHB2sB2s", 0x3A55, 1, m_size, ntr, 1, b"\x00\x00", 1, b"\n\x00").ljust(32, b"\x00")
    fstr = block(file_strings)
    pos = 32 + m_size + len(fstr)
    descriptors, pointers = [], []
    for k in range(ntr):
        strings = [f"SAMPLE_INTERVAL {data.dt:.9g}", f"DELAY {data.t0:.9g}", f"CHANNEL_NUMBER {int(chan[k])}",
                   f"RECEIVER_LOCATION {gx[k]:.4f}", f"SOURCE_LOCATION {sx[k]:.4f}"]
        sblock = block(strings)
        x_size = 32 + len(sblock)
        x_size += (-x_size) % 4
        tdb = struct.pack("<HHIIB", 0x4422, x_size, 4 * nt, nt, 4).ljust(32, b"\x00") + sblock
        tdb = tdb.ljust(x_size, b"\x00") + traces[k].astype("<f4").tobytes()
        pointers.append(pos)
        descriptors.append(tdb)
        pos += len(tdb)
    with open(path, "wb") as f:
        f.write(head + struct.pack(f"<{ntr}I", *pointers) + fstr + b"".join(descriptors))
