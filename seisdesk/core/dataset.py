"""The in-memory seismic dataset: a trace matrix, its sample axis and its trace headers.

``SeismicData`` is deliberately simple: ``traces`` is a (ntr, nt) float32 array, ``headers`` a dict of (ntr,) float64
arrays named as in ``seisdesk.core.headers``, ``dt`` the sample interval in seconds (or metres for depth sections) and
``t0`` the time (depth) of the first sample. Every processing step takes one dataset and returns a new one, so a flow
never modifies its input and any intermediate result can be displayed.
"""

from __future__ import annotations

import copy as _copy
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeismicData:
    traces: np.ndarray
    dt: float
    headers: dict = field(default_factory=dict)
    t0: float = 0.0
    name: str = "data"
    domain: str = "time"            # "time" (dt in s) or "depth" (dt in m)
    history: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    trace_ids: np.ndarray | None = None

    def __post_init__(self):
        self.traces = np.ascontiguousarray(np.asarray(self.traces, dtype=np.float32))
        if self.traces.ndim == 1:
            self.traces = self.traces[None, :]
        if self.traces.ndim != 2:
            raise ValueError("traces must be a 2-D (ntr, nt) array")
        if not (self.dt > 0):
            raise ValueError("dt must be positive")
        if self.domain not in ("time", "depth"):
            raise ValueError("domain must be 'time' or 'depth'")
        ntr = self.traces.shape[0]
        if self.trace_ids is None:
            prefix = uuid.uuid4().hex
            self.trace_ids = np.array([f"{prefix}:{i}" for i in range(ntr)])
        else:
            self.trace_ids = np.asarray(self.trace_ids, dtype=str).reshape(-1).copy()
            if self.trace_ids.size != ntr or np.unique(self.trace_ids).size != ntr:
                raise ValueError("trace_ids must contain one unique identifier per trace")
        clean = {}
        for key, value in dict(self.headers).items():
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size == 1 and ntr != 1:
                arr = np.full(ntr, float(arr[0]))
            if arr.size != ntr:
                raise ValueError(f"header {key!r} has {arr.size} values for {ntr} traces")
            clean[key] = arr
        self.headers = clean
        if "tracl" not in self.headers:
            self.headers["tracl"] = np.arange(1, ntr + 1, dtype=np.float64)

    # ------------------------------------------------------------------ shape and axes
    @property
    def ntr(self) -> int:
        return int(self.traces.shape[0])

    @property
    def nt(self) -> int:
        return int(self.traces.shape[1])

    @property
    def times(self) -> np.ndarray:
        """Sample positions: seconds for time data, metres for depth data."""
        return self.t0 + np.arange(self.nt) * self.dt

    @property
    def tmax(self) -> float:
        return float(self.t0 + (self.nt - 1) * self.dt)

    # ------------------------------------------------------------------ headers
    def header(self, key: str, default: float | None = 0.0) -> np.ndarray:
        """Header values for every trace; a missing header returns ``default`` (or raises if default is None)."""
        if key in self.headers:
            return self.headers[key]
        if default is None:
            raise KeyError(f"header {key!r} is not set")
        return np.full(self.ntr, float(default))

    def has_header(self, key: str) -> bool:
        return key in self.headers and bool(np.any(self.headers[key] != 0))

    def set_header(self, key: str, values) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.full(self.ntr, float(arr[0]))
        if arr.size != self.ntr:
            raise ValueError(f"{arr.size} values for {self.ntr} traces")
        self.headers[key] = arr

    # ------------------------------------------------------------------ copies and selections
    def copy(self, traces: np.ndarray | None = None, **changes) -> "SeismicData":
        """A copy with optionally new traces (same number of traces unless headers are also given)."""
        new = SeismicData(
            traces=self.traces.copy() if traces is None else traces,
            dt=changes.pop("dt", self.dt),
            headers=changes.pop("headers", {k: v.copy() for k, v in self.headers.items()}),
            t0=changes.pop("t0", self.t0),
            name=changes.pop("name", self.name),
            domain=changes.pop("domain", self.domain),
            history=list(self.history),
            meta=_copy.deepcopy(self.meta),
            trace_ids=changes.pop("trace_ids", self.trace_ids.copy() if traces is None or
                                  np.asarray(traces).shape[0] == self.ntr else None),
        )
        if changes:
            raise TypeError(f"unknown fields {sorted(changes)}")
        return new

    def subset(self, index) -> "SeismicData":
        """Traces selected by an index array or boolean mask, headers following."""
        idx = np.asarray(index)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return SeismicData(self.traces[idx], self.dt, {k: v[idx] for k, v in self.headers.items()}, self.t0, self.name,
                           self.domain, list(self.history), _copy.deepcopy(self.meta), self.trace_ids[idx])

    def sort(self, keys) -> "SeismicData":
        """Stable sort by one or more header keys (primary key first)."""
        keys = [keys] if isinstance(keys, str) else list(keys)
        order = np.lexsort([self.header(k) for k in reversed(keys)])
        out = self.subset(order)
        out.meta["sorted_by"] = keys
        return out

    def time_window(self, tmin: float | None = None, tmax: float | None = None) -> "SeismicData":
        i0 = 0 if tmin is None else int(max(0, np.floor((tmin - self.t0) / self.dt + 1e-9)))
        i1 = self.nt if tmax is None else int(min(self.nt, np.floor((tmax - self.t0) / self.dt + 1e-9) + 1))
        if i1 <= i0:
            raise ValueError("empty time window")
        return self.copy(self.traces[:, i0:i1].copy(), t0=self.t0 + i0 * self.dt)

    # ------------------------------------------------------------------ gathers
    def gather_values(self, key: str) -> np.ndarray:
        """Sorted distinct values of a header: the list of gathers along that key."""
        return np.unique(self.header(key))

    def gather_index(self, key: str, value, secondary: str | None = "offset") -> np.ndarray:
        """Trace indices of one gather (key == value), ordered by a secondary key (absolute offset by default)."""
        idx = np.flatnonzero(np.abs(self.header(key) - float(value)) <= 1e-6)
        if secondary and idx.size > 1:
            sec = self.header(secondary)[idx]
            if secondary == "offset":
                sec = np.abs(sec) + 1e-6 * np.sign(sec)
            idx = idx[np.argsort(sec, kind="stable")]
        return idx

    def gather(self, key: str, value, secondary: str | None = "offset") -> "SeismicData":
        return self.subset(self.gather_index(key, value, secondary))

    def iter_gathers(self, key: str, secondary: str | None = "offset"):
        """(value, trace indices) for every gather along ``key``, in ascending key order."""
        values = self.header(key)
        order = np.lexsort((self.header(secondary) if secondary else np.zeros(self.ntr), values))
        sorted_vals = values[order]
        bounds = np.flatnonzero(np.diff(sorted_vals)) + 1
        for chunk in np.split(order, bounds):
            if chunk.size:
                yield float(values[chunk[0]]), chunk

    # ------------------------------------------------------------------ bookkeeping
    def add_history(self, text: str) -> None:
        self.history.append(text)

    def summary(self) -> str:
        unit = "ms" if self.domain == "time" else "m"
        scale = 1000.0 if self.domain == "time" else 1.0
        lines = [f"{self.name}: {self.ntr} traces x {self.nt} samples, d{'t' if self.domain == 'time' else 'z'} = "
                 f"{self.dt * scale:g} {unit}, {self.domain} {self.t0 * scale:g}-{self.tmax * scale:g} {unit}"]
        for key in ("fldr", "ep", "cdp", "offset", "sx", "gx", "iline", "xline"):
            if self.has_header(key):
                v = self.header(key)
                lines.append(f"  {key:7s} {v.min():g} .. {v.max():g} ({np.unique(v).size} distinct)")
        return "\n".join(lines)
