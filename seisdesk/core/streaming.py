"""Read one shot gather at a time, retaining only the shot index in memory."""
from pathlib import Path

import numpy as np

from .dataset import SeismicData
from .headers import SEGY_FIELDS, STANDARD, SU_EXTRA, su_header_dtype
from .segy_io import _sample_axis, _to_real_units


def _groups(labels):
    order = np.argsort(labels, kind="stable")
    breaks = np.r_[0, np.flatnonzero(np.diff(labels[order]))+1, len(order)]
    for a, b in zip(breaks[:-1], breaks[1:]):
        yield order[a:b]


def iter_file_gathers(path, gather_key="fldr", byteorder="<"):
    """Yield (original trace indices, SeismicData), from SU or SEG-Y.

    Unsorted shots are gathered by their key. The waveform memory requirement is
    one gather. Timing is validated independently for each gather.
    """
    path = Path(path)
    if path.suffix.lower() == ".su":
        header = su_header_dtype().newbyteorder(byteorder)
        if gather_key not in header.names:
            raise ValueError(f"unknown SU gather key {gather_key!r}")
        with path.open("rb") as stream:
            first = stream.read(240)
        if len(first) != 240:
            raise ValueError("empty or truncated SU file")
        ns = int(np.frombuffer(first, header)["ns"][0])
        if ns < 1:
            raise ValueError("invalid SU ns; check byte order")
        record = np.dtype([("h", header), ("d", np.dtype("f4").newbyteorder(byteorder), (ns,))])
        if path.stat().st_size % record.itemsize:
            raise ValueError("SU file is truncated or has variable trace lengths")
        raw = np.memmap(path, mode="r", dtype=record)
        labels = raw["h"][gather_key].copy()
        for indices in _groups(labels):
            part = raw[indices]
            if not np.all(part["h"]["ns"] == ns):
                raise ValueError("SU file has variable trace lengths")
            headers = {key: part["h"][key].astype(float) for key, _, _, _ in STANDARD+SU_EXTRA}
            dt, t0 = _sample_axis(headers, float(headers["dt"][0]))
            data = SeismicData(part["d"].astype(np.float32), dt, _to_real_units(headers), t0=t0,
                               name=f"{path.stem} {gather_key}={labels[indices[0]]:g}")
            data.meta["source_file"] = str(path.resolve())
            yield indices, data
        del raw
    else:
        import segyio
        fields = {k: b for k, b, _, _ in SEGY_FIELDS}
        if gather_key not in fields:
            raise ValueError(f"unknown SEG-Y gather key {gather_key!r}")
        with segyio.open(str(path), ignore_geometry=True) as stream:
            labels = stream.attributes(fields[gather_key])[:]
            for indices in _groups(labels):
                traces = np.stack([stream.trace[int(i)] for i in indices])
                headers = {k: np.array([stream.header[int(i)][byte] for i in indices], float)
                           for k, byte, _, _ in SEGY_FIELDS}
                dt, t0 = _sample_axis(headers, float(stream.bin[segyio.BinField.Interval]))
                data = SeismicData(traces, dt, _to_real_units(headers), t0=t0,
                                   name=f"{path.stem} {gather_key}={labels[indices[0]]:g}")
                data.meta["source_file"] = str(path.resolve())
                yield indices, data
