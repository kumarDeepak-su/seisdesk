"""SEG-Y and Seismic Unix (SU) input and output.

SEG-Y goes through segyio (IBM float and IEEE float, any trace count, no geometry assumed). SU files are read and
written directly with NumPy: a 240-byte trace header followed by float32 samples, native byte order, no file headers.
Headers are converted to real units on input (coordinates by ``scalco``, elevations by ``scalel``) and back on output.
"""

from __future__ import annotations

import os

import numpy as np

from .dataset import SeismicData
from .headers import (COORDINATE_KEYS, ELEVATION_KEYS, SEGY_FIELDS, STANDARD, SU_EXTRA, choose_scalar, scale_factor,
                      su_header_dtype)

INT_RANGE = {"i2": (-32768, 32767), "u2": (0, 65535), "i4": (-2**31, 2**31 - 1)}


def _to_real_units(headers: dict) -> dict:
    """Apply scalco to coordinates and scalel to elevations, in place; keep the scalars for reference."""
    co = scale_factor(headers.get("scalco", np.zeros(1)))
    el = scale_factor(headers.get("scalel", np.zeros(1)))
    for key in COORDINATE_KEYS:
        if key in headers:
            headers[key] = headers[key] * co
    for key in ELEVATION_KEYS:
        if key in headers:
            headers[key] = headers[key] * el
    return headers


def _to_file_units(headers: dict, ntr: int) -> dict:
    """Integer header values for writing: coordinates and elevations re-scaled with chosen scalars."""
    out = {k: np.asarray(v, dtype=np.float64).copy() for k, v in headers.items()}
    coords = [out[k] for k in COORDINATE_KEYS if k in out]
    elevs = [out[k] for k in ELEVATION_KEYS if k in out]
    sco = choose_scalar(np.concatenate(coords)) if coords else 1
    sel = choose_scalar(np.concatenate(elevs)) if elevs else 1
    fco = 1.0 / scale_factor(np.array([sco]))[0]
    fel = 1.0 / scale_factor(np.array([sel]))[0]
    for k in COORDINATE_KEYS:
        if k in out:
            out[k] = out[k] * fco
    for k in ELEVATION_KEYS:
        if k in out:
            out[k] = out[k] * fel
    out["scalco"] = np.full(ntr, float(sco))
    out["scalel"] = np.full(ntr, float(sel))
    return out


def _clip_int(values: np.ndarray, fmt: str) -> np.ndarray:
    lo, hi = INT_RANGE[fmt]
    return np.clip(np.rint(values), lo, hi)


def _check_export_timing(data):
    if data.domain != "time":
        return
    interval, delay = data.dt*1e6, data.t0*1e3
    if (not np.isfinite(interval) or not np.isfinite(delay) or not 1 <= interval <= 65535 or
            not -32768 <= delay <= 32767 or abs(interval-round(interval)) > 1e-6 or
            abs(delay-round(delay)) > 1e-6):
        raise ValueError("this writer needs an integer microsecond interval (1–65535) and integer millisecond "
                         "start (-32768–32767); save a project to preserve exact timing or explicitly resample")


# ---------------------------------------------------------------------------------------------- SEG-Y
def _sample_axis(headers: dict, interval_us: float) -> tuple[float, float]:
    """Validate the common sample axis; never silently collapse per-trace timing.

    Zero trace intervals inherit the binary header. Variable timing requires explicit
    resampling/alignment before import, since a dataset has one common sample axis.
    """
    intervals = np.asarray(headers.get("dt", []), dtype=float)
    specified = intervals[intervals != 0]
    if interval_us <= 0 and specified.size:
        interval_us = float(specified[0])
    if not np.isfinite(interval_us) or interval_us <= 0:
        raise ValueError("no positive sample interval in the binary or trace headers")
    if specified.size and (not np.all(np.isfinite(specified)) or not np.all(specified == interval_us)):
        raise ValueError("sample intervals differ between traces or disagree with the binary header; "
                         "resample to a common interval before import")
    delays = np.asarray(headers.get("delrt", [0.0]), dtype=float)
    delay = float(delays[0]) if delays.size else 0.0
    if not np.all(np.isfinite(delays)) or not np.all(delays == delay):
        raise ValueError("recording delays (delrt) vary between traces; align traces to a common "
                         "time origin before import (no timing has been changed)")
    return interval_us * 1e-6, delay * 1e-3


def read_segy(path: str, name: str | None = None, endian: str = "big", *, target_dt=None,
              window="strict", max_mb=1024) -> SeismicData:
    """Read every trace and the standard trace headers of a SEG-Y file (no geometry is assumed)."""
    if target_dt is not None or window != "strict":
        from .field_io import read_variable
        return read_variable(path, name, "<" if endian == "little" else ">", False, target_dt, window, max_mb)
    import segyio

    with segyio.open(path, "r", ignore_geometry=True, endian=endian) as f:
        traces = np.asarray(f.trace.raw[:], dtype=np.float32)
        if traces.ndim == 1:
            traces = traces[None, :]
        interval_us = float(f.bin[segyio.BinField.Interval]) or float(segyio.tools.dt(f))
        measurement = int(f.bin[segyio.BinField.MeasurementSystem])
        headers = {}
        for key, byte, _, _ in SEGY_FIELDS:
            try:
                headers[key] = np.asarray(f.attributes(byte)[:], dtype=np.float64)
            except Exception:                       # pragma: no cover - unusual header layouts
                continue
        text = ""
        try:
            raw = bytes(f.text[0])
            text = "\n".join(raw[i:i + 80].decode("ascii", "replace").rstrip() for i in range(0, len(raw), 80))
        except Exception:                           # pragma: no cover
            pass
    dt, t0 = _sample_axis(headers, interval_us)
    headers = _to_real_units(headers)
    if measurement not in (0, 1, 2):
        raise ValueError("unknown SEG-Y measurement system")
    if measurement == 2:
        length_coords = np.isin(headers.get("counit", np.ones(traces.shape[0])), [0, 1])
        for key in COORDINATE_KEYS:
            if key in headers:
                headers[key][length_coords] *= .3048
        for key in (*ELEVATION_KEYS, "offset"):
            if key in headers:
                headers[key] *= .3048
    data = SeismicData(traces, dt, headers, t0=t0,
                       name=name or os.path.splitext(os.path.basename(path))[0])
    data.meta["source_file"] = os.path.abspath(path)
    data.meta["textual_header"] = text
    data.meta["segy_measurement_system"] = measurement
    if measurement == 2:
        data.meta["coordinate_conversion"] = "SEG-Y length units: feet to metres, factor 0.3048; angular coordinates unchanged"
    data.add_history(f"read SEG-Y {os.path.basename(path)}")
    return data


def write_segy(data: SeismicData, path: str, textual_header: str | None = None) -> None:
    """Write IEEE-float SEG-Y rev 1 with the dataset's headers (coordinates re-scaled, ns/dt/delrt set)."""
    import segyio

    if data.domain == "time" and data.dt < 1e-6:
        raise ValueError("nanosecond sampling cannot be represented in this SEG-Y writer; save the project to preserve GPR timing")
    if data.nt > 65535:
        raise ValueError("this SEG-Y writer supports at most 65535 samples per trace")
    _check_export_timing(data)

    spec = segyio.spec()
    spec.format = 5
    spec.samples = np.arange(data.nt) * data.dt * 1000.0 if data.domain == "time" else np.arange(data.nt) * data.dt
    spec.tracecount = data.ntr
    spec.sorting = None
    heads = _to_file_units(data.headers, data.ntr)
    dt_us = int(round(data.dt * 1e6)) if data.domain == "time" else int(round(data.dt * 1000.0))
    heads["ns"] = np.full(data.ntr, data.nt, dtype=np.float64)
    heads["dt"] = np.full(data.ntr, dt_us, dtype=np.float64)
    heads["delrt"] = np.full(data.ntr, round(data.t0 * 1000.0), dtype=np.float64)
    fields = [(key, byte, fmt) for key, byte, fmt, _ in SEGY_FIELDS if key in heads]
    columns = {key: _clip_int(heads[key], fmt).astype(np.int64) for key, byte, fmt in fields}
    with segyio.create(path, spec) as f:
        lines = (textual_header or data.meta.get("textual_header") or
                 f"C 1 WRITTEN BY SEISDESK  {data.name}\nC 2 {data.ntr} TRACES {data.nt} SAMPLES DT {dt_us} US").splitlines()
        text = "".join(line[:80].ljust(80) for line in lines[:40]).ljust(3200)
        f.text[0] = text.encode("ascii", errors="replace")
        f.bin.update(hns=data.nt, hdt=dt_us, format=5)
        f.bin[segyio.BinField.MeasurementSystem] = 1
        for i in range(data.ntr):
            f.header[i] = {byte: int(columns[key][i]) for key, byte, fmt in fields}
        f.trace = np.ascontiguousarray(data.traces, dtype=np.float32)


# ---------------------------------------------------------------------------------------------- Seismic Unix
def read_su(path: str, name: str | None = None, byteorder: str = "<", *, target_dt=None,
            window="strict", max_mb=1024) -> SeismicData:
    """Read a Seismic Unix file (240-byte headers + float32 samples, little-endian unless ``byteorder='>'``)."""
    if target_dt is not None or window != "strict":
        from .field_io import read_variable
        return read_variable(path, name, byteorder, True, target_dt, window, max_mb)
    hdr = su_header_dtype().newbyteorder(byteorder) if byteorder != "=" else su_header_dtype()
    with open(path, "rb") as fh:
        header_bytes = fh.read(240)
    if len(header_bytes) != 240:
        raise ValueError("empty or truncated SU trace header")
    first = np.frombuffer(header_bytes, dtype=hdr, count=1)
    ns = int(first["ns"][0])
    if ns <= 0:
        raise ValueError("ns is zero in the first trace header (wrong byte order?)")
    rec = np.dtype([("h", hdr), ("d", np.dtype("f4").newbyteorder(byteorder), (ns,))])
    if os.path.getsize(path) % rec.itemsize:
        raise ValueError("SU file is truncated or has variable trace lengths; fixed-length traces are required")
    raw = np.fromfile(path, dtype=rec)
    if not np.all(raw["h"]["ns"] == ns):
        raise ValueError("SU trace sample counts differ; fixed-length traces are required")
    headers = {}
    for key, _, _, _ in STANDARD + SU_EXTRA:
        headers[key] = raw["h"][key].astype(np.float64)
    headers = _to_real_units(headers)
    dt, t0 = _sample_axis(headers, float(raw["h"]["dt"][0]))
    data = SeismicData(raw["d"].astype(np.float32), dt, headers, t0=t0,
                       name=name or os.path.splitext(os.path.basename(path))[0])
    data.meta["source_file"] = os.path.abspath(path)
    data.add_history(f"read SU {os.path.basename(path)}")
    return data


def write_su(data: SeismicData, path: str, byteorder: str = "<") -> None:
    """Write a Seismic Unix file readable by the SU programs (little-endian by default)."""
    if data.dt < 1e-6 or data.dt*1e6 > 65535 or data.nt > 65535:
        raise ValueError("SU export requires a representable microsecond sample interval and at most 65535 samples; save the project for GPR data")
    _check_export_timing(data)
    hdr = su_header_dtype().newbyteorder(byteorder)
    rec = np.dtype([("h", hdr), ("d", np.dtype("f4").newbyteorder(byteorder), (data.nt,))])
    out = np.zeros(data.ntr, dtype=rec)
    heads = _to_file_units(data.headers, data.ntr)
    dt_us = int(round(data.dt * 1e6))
    heads["ns"] = np.full(data.ntr, data.nt, dtype=np.float64)
    heads["dt"] = np.full(data.ntr, dt_us, dtype=np.float64)
    heads["delrt"] = np.full(data.ntr, round(data.t0 * 1000.0), dtype=np.float64)
    for key, _, fmt, _ in STANDARD + SU_EXTRA:
        if key in heads:
            values = heads[key]
            out["h"][key] = values if fmt == "f4" else _clip_int(values, fmt)
    out["d"] = data.traces
    out.tofile(path)


def read_any(path: str, **kwargs) -> SeismicData:
    """Choose the reader: SEG-2 by its file signature, .su -> Seismic Unix, otherwise SEG-Y."""
    from .seg2 import is_seg2, read_seg2
    if str(path).lower().endswith(".dt1"):
        from .gpr import read_dt1
        return read_dt1(path, **kwargs)
    if is_seg2(path):
        return read_seg2(path, **kwargs)
    from .segd import is_segd, read_segd
    if str(path).lower().endswith((".segd", ".sgd")) or is_segd(path):
        return read_segd(path, **kwargs)
    if str(path).lower().endswith(".su"):
        return read_su(path, **kwargs)
    return read_segy(path, **kwargs)
