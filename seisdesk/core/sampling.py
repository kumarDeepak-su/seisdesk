"""Explicit common-axis import with anti-alias filtering and retained acquisition timing."""
from __future__ import annotations

import numpy as np


def common_samples(rows, intervals, starts, target_dt=None, window="strict", max_mb=1024,
                   cancelled=lambda: False):
    """Return float32 traces, dt, t0, provenance. All time values are seconds.

    strict preserves a rectangular input exactly. Other policies require an explicit
    target interval: intersection crops, union zero-pads, zero starts at shot time
    zero and ends at the earliest record end. Finite Kaiser-windowed sinc resampling
    uses a 0.95 output-Nyquist low-pass when decimating. Padded samples are not picks
    or observations: per-trace original support is retained in the provenance.
    """
    rows = list(rows)
    dt = np.asarray(intervals, dtype=float).reshape(-1)
    start = np.asarray(starts, dtype=float).reshape(-1)
    ns = np.array([len(r) for r in rows], dtype=int)
    if not rows or dt.size != len(rows) or start.size != len(rows) or np.any(ns < 1):
        raise ValueError("one nonempty sample vector, interval and start time are required per trace")
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0) or not np.all(np.isfinite(start)):
        raise ValueError("sample intervals must be positive and timing must be finite")
    if window not in ("strict", "intersection", "union", "zero"):
        raise ValueError("time window must be strict, intersection, union or zero")
    end = start + (ns - 1)*dt
    if window == "strict":
        if not (np.allclose(dt, dt[0], rtol=1e-10, atol=0) and
                np.allclose(start, start[0], rtol=0, atol=dt[0]*1e-8) and np.all(ns == ns[0])):
            raise ValueError("trace intervals, recording delays or lengths differ; select a target interval "
                             "and an alignment window to resample explicitly")
        t0, stop = float(start[0]), float(end[0])
    else:
        if target_dt is None:
            raise ValueError("alignment requires an explicit target sample interval")
        t0 = float(max(start) if window == "intersection" else min(start) if window == "union" else 0)
        stop = float(max(end) if window == "union" else min(end))
    out_dt = float(target_dt if target_dt is not None else dt[0])
    if not np.isfinite(out_dt) or out_dt <= 0 or not np.isfinite(max_mb) or max_mb <= 0:
        raise ValueError("target interval and memory limit must be positive and finite")
    if np.any(out_dt / dt > 64):
        raise ValueError("decimation exceeds 64:1; resample in stages")
    nt = int(np.floor((stop-t0)/out_dt + 1e-8)) + 1
    if nt < 1:
        raise ValueError("the requested time window contains no samples")
    if len(rows)*nt*4 > max_mb*1024**2:
        raise ValueError(f"resampled traces exceed the {max_mb:g} MiB output limit")
    out = np.zeros((len(rows), nt), dtype=np.float32)
    for k, row in enumerate(rows):
        if cancelled():
            raise InterruptedError("sample alignment cancelled")
        row = np.asarray(row, dtype=float)
        if row.ndim != 1 or not np.isfinite(row).all() or np.any(np.abs(row) > np.finfo(np.float32).max):
            raise ValueError(f"trace {k+1} has invalid or overflowing samples")
        shift = (start[k]-t0)/out_dt
        if abs(dt[k]/out_dt-1) < 1e-10 and abs(shift-round(shift)) < 1e-8:
            offset = int(round(shift))
            lo, hi = max(0, offset), min(nt, offset+ns[k])
            if hi > lo:
                out[k, lo:hi] = row[lo-offset:hi-offset]
            continue
        cutoff = min(1., dt[k]/out_dt)
        if cutoff < 1:
            cutoff *= .95
        half = int(np.ceil(32/cutoff))
        taps = np.arange(-half, half+1)
        chunk = max(1, 262144 // len(taps))
        for lo in range(0, nt, chunk):
            if cancelled():
                raise InterruptedError("sample alignment cancelled")
            hi = min(nt, lo+chunk)
            u = (t0 + np.arange(lo, hi)*out_dt - start[k])/dt[k]
            ix = np.floor(u).astype(np.int64)[:, None] + taps
            distance = u[:, None]-ix
            rel = distance/half
            kernel = cutoff*np.sinc(cutoff*distance)*np.i0(8.6*np.sqrt(np.maximum(0., 1-rel**2)))/np.i0(8.6)
            kernel[np.abs(rel) > 1] = 0
            kernel /= kernel.sum(axis=1, keepdims=True)
            valid = (ix >= 0) & (ix < ns[k])
            values = np.sum(kernel*valid*row[np.clip(ix, 0, ns[k]-1)], axis=1)
            values[(u < -1e-8) | (u > ns[k]-1+1e-8)] = 0
            if np.any(np.abs(values) > np.finfo(np.float32).max):
                raise ValueError("resampling overflows float32; scale the input first")
            out[k, lo:hi] = values
    provenance = dict(window=window, target_dt_s=out_dt, target_start_s=t0,
                      original_dt_s=dt.tolist(), original_start_s=start.tolist(),
                      original_samples=ns.tolist(), original_end_s=end.tolist(),
                      filter="Kaiser sinc: 32 lobes, beta 8.6; decimation cutoff 0.95 target Nyquist",
                      outside_record="zero; not observed samples")
    return out, out_dt, t0, provenance


def timing_headers(headers, provenance, nt, dt, t0):
    """Keep original timing in trace-associated headers, update the common axis."""
    for key, source in (("input_dt_s", "original_dt_s"), ("input_t0_s", "original_start_s"),
                        ("input_ns", "original_samples"), ("input_end_s", "original_end_s")):
        headers[key] = np.asarray(provenance[source], dtype=float)
    n = len(provenance["original_dt_s"])
    headers.update(ns=np.full(n, nt), dt=np.full(n, dt*1e6), delrt=np.full(n, t0*1e3))
    return headers
