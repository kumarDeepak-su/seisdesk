"""Interactive picks organised by gather and by phase.

A ``PickSet`` stores, for every gather (e.g. a shot by ``fldr``), any number of named phases, each a list of
(x, t) points where x is a trace header such as offset. A new phase can be started at any time — the "break" the
picker uses to separate a direct wave from refracted branches that take over one after another away from zero offset,
or a top mute from a bottom mute. Phases interpolate across x and across gathers (for mutes) and fit straight lines
(for refraction velocities and intercept times).
"""

from __future__ import annotations

import csv
import json

import numpy as np


class PickSet:
    def __init__(self, gather_key: str = "fldr", x_key: str = "offset", name: str = "picks"):
        self.gather_key = gather_key
        self.x_key = x_key
        self.name = name
        self.data: dict[float, dict[str, list[tuple[float, float]]]] = {}
        self.phase_order: list[str] = []
        self.colors: dict[str, str] = {}
        self.phase_types: dict[str, str] = {}
        self.trace_ids: dict[float, dict[str, list[str | None]]] = {}

    PHASE_TYPES = ("unclassified", "direct", "refraction", "reflection", "multiple", "mute", "horizon")

    def phase_type(self, phase: str) -> str:
        return self.phase_types.get(phase, "unclassified")

    def set_phase_type(self, phase: str, kind: str) -> None:
        if phase not in self.phase_order or kind not in self.PHASE_TYPES:
            raise ValueError("unknown phase or event type")
        self.phase_types[phase] = kind

    def _ids(self, gather: float, phase: str) -> list:
        ids = self.trace_ids.setdefault(float(gather), {}).setdefault(phase, [])
        pts = self.data.get(float(gather), {}).get(phase, [])
        ids.extend([None] * (len(pts) - len(ids)))
        return ids

    def records(self, gather: float, phase: str):
        """(x, time, trace_id) records; legacy coordinate-only picks have trace_id=None."""
        return [(x, t, tid) for (x, t), tid in
                zip(self.data.get(float(gather), {}).get(phase, []), self._ids(gather, phase))]

    def matched_records(self, data, phase: str | None = None):
        """Yield (gather, phase, x, time, trace index) without ambiguous geometry matches.

        Identified picks only match their original trace or a derived copy. Legacy picks
        match a coordinate only when exactly one trace occupies it.
        """
        ids = {str(tid): i for i, tid in enumerate(data.trace_ids)}
        coords = {}
        if self.x_key in data.headers and (self.gather_key == "none" or self.gather_key in data.headers):
            g = np.zeros(data.ntr) if self.gather_key == "none" else data.header(self.gather_key)
            for i, (gv, xv) in enumerate(zip(g, data.header(self.x_key))):
                coords.setdefault((round(float(gv), 6), round(float(xv), 6)), []).append(i)
        for gather in self.gathers(phase):
            for ph in ([phase] if phase else self.phase_order):
                for x, t, tid in self.records(gather, ph):
                    if tid is not None:
                        i = ids.get(tid)
                    else:
                        candidates = coords.get((round(float(gather), 6), round(x, 6)), [])
                        if len(candidates) > 1:
                            raise ValueError(f"legacy pick for {ph!r} at gather {gather:g}, x={x:g} matches "
                                             "multiple traces; repick it to attach a unique trace identity")
                        i = candidates[0] if candidates else None
                    if i is not None:
                        yield float(gather), ph, x, t, i

    # ------------------------------------------------------------------ phases
    _PALETTE = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#17becf")

    def new_phase(self, name: str | None = None, kind: str = "unclassified") -> str:
        """Start a new phase (a break in the picking) and return its name."""
        if name is None:
            k = len(self.phase_order) + 1
            name = f"phase {k}"
            while name in self.phase_order:
                k += 1
                name = f"phase {k}"
        if name not in self.phase_order:
            if kind not in self.PHASE_TYPES:
                raise ValueError("unknown event type")
            self.phase_order.append(name)
            self.colors[name] = self._PALETTE[(len(self.phase_order) - 1) % len(self._PALETTE)]
            self.phase_types[name] = kind
        return name

    def rename_phase(self, old: str, new: str) -> None:
        if old not in self.phase_order or new in self.phase_order:
            raise ValueError("unknown phase or name already used")
        self.phase_order[self.phase_order.index(old)] = new
        self.colors[new] = self.colors.pop(old)
        self.phase_types[new] = self.phase_types.pop(old, "unclassified")
        for phases in self.trace_ids.values():
            if old in phases:
                phases[new] = phases.pop(old)
        for phases in self.data.values():
            if old in phases:
                phases[new] = phases.pop(old)

    def delete_phase(self, name: str) -> None:
        if name in self.phase_order:
            self.phase_order.remove(name)
            self.colors.pop(name, None)
            self.phase_types.pop(name, None)
        for phases in self.trace_ids.values():
            phases.pop(name, None)
        for phases in self.data.values():
            phases.pop(name, None)

    # ------------------------------------------------------------------ points
    def add(self, gather: float, phase: str, x: float, t: float, trace_id: str | None = None) -> None:
        if phase not in self.phase_order:
            self.new_phase(phase)
        pts = self.data.setdefault(float(gather), {}).setdefault(phase, [])
        if not np.all(np.isfinite([gather, x, t])):
            raise ValueError("pick coordinates and time must be finite")
        tid = None if trace_id is None else str(trace_id)
        ids = self._ids(gather, phase)
        rows = [(px, pt, old) for (px, pt), old in zip(pts, ids)
                if not (old == tid and (tid is not None or abs(px - x) <= 1e-9))]
        rows.append((float(x), float(t), tid))
        rows.sort(key=lambda r: (r[0], r[1], r[2] or ""))
        pts[:] = [(px, pt) for px, pt, _ in rows]
        ids[:] = [ref for _, _, ref in rows]

    def remove_nearest(self, gather: float, x: float, t: float, phase: str | None = None, x_scale: float = 50.0,
                       t_scale: float = 0.02, max_distance: float | None = None,
                       trace_id: str | None = None) -> bool:
        """Delete the pick nearest to (x, t), distances scaled by x_scale and t_scale; with ``max_distance`` only a
        pick within that scaled distance is deleted. Returns whether a pick was deleted."""
        phases = self.data.get(float(gather), {})
        best = None
        for name, pts in phases.items():
            if phase is not None and name != phase:
                continue
            for i, (px, pt) in enumerate(pts):
                tid = self._ids(gather, name)[i]
                if trace_id is not None and tid is not None and tid != str(trace_id):
                    continue
                d = np.hypot((px - x) / x_scale, (pt - t) / t_scale)
                if best is None or d < best[0]:
                    best = (d, name, i)
        if best is None or (max_distance is not None and best[0] > max_distance):
            return False
        self._ids(gather, best[1]).pop(best[2])
        phases[best[1]].pop(best[2])
        if not phases[best[1]]:
            phases.pop(best[1])
        return True

    def clear_gather(self, gather: float) -> None:
        self.data.pop(float(gather), None)
        self.trace_ids.pop(float(gather), None)

    def points(self, gather: float, phase: str) -> np.ndarray:
        return np.array(self.data.get(float(gather), {}).get(phase, []), dtype=np.float64).reshape(-1, 2)

    def phases_at(self, gather: float) -> list[str]:
        here = self.data.get(float(gather), {})
        return [p for p in self.phase_order if p in here]

    def gathers(self, phase: str | None = None) -> np.ndarray:
        return np.array(sorted(g for g, ph in self.data.items() if phase is None or phase in ph), dtype=np.float64)

    def count(self) -> int:
        return sum(len(p) for ph in self.data.values() for p in ph.values())

    # ------------------------------------------------------------------ interpolation (mutes)
    def interpolate(self, gather_values: np.ndarray, x_values: np.ndarray, phase: str,
                    use_abs_x: bool = True) -> np.ndarray:
        """Pick time for each (gather, x) pair: linear in x within picked gathers, linear between picked gathers.

        Outside the picked x range the end picks are extended flat; outside the picked gathers the nearest is used."""
        gathers = self.gathers(phase)
        if gathers.size == 0:
            raise ValueError(f"phase {phase!r} has no picks")
        gv = np.asarray(gather_values, dtype=np.float64)
        xv = np.abs(np.asarray(x_values, dtype=np.float64)) if use_abs_x else np.asarray(x_values, dtype=np.float64)
        per_gather = np.empty((gathers.size, xv.size))
        for i, g in enumerate(gathers):
            p = self.points(g, phase)
            px = np.abs(p[:, 0]) if use_abs_x else p[:, 0]
            order = np.argsort(px)
            for value in np.unique(px):
                tt = p[px == value, 1]
                if np.ptp(tt) > 1e-9:
                    raise ValueError("offset-only interpolation is ambiguous: equal offsets have different "
                                     "pick times; use per-trace picks or separate azimuth/side groups")
            per_gather[i] = np.interp(xv, px[order], p[order, 1])
        if gathers.size == 1:
            return per_gather[0]
        out = np.empty(xv.size)
        pos = np.clip(np.searchsorted(gathers, gv), 1, gathers.size - 1)
        g0, g1 = gathers[pos - 1], gathers[pos]
        w = np.clip((gv - g0) / np.where(g1 > g0, g1 - g0, 1.0), 0.0, 1.0)
        rows = np.arange(xv.size)
        out[:] = (1 - w) * per_gather[pos - 1, rows] + w * per_gather[pos, rows]
        return out

    def times_for(self, data, phase: str, use_abs_x: bool = True) -> np.ndarray:
        """Use exact trace picks before interpolating any missing traces."""
        times = np.full(data.ntr, np.nan)
        for _, _, _, t, i in self.matched_records(data, phase):
            times[i] = t
        missing = ~np.isfinite(times)
        if np.any(missing):
            g = np.zeros(data.ntr) if self.gather_key == "none" else data.header(self.gather_key)
            times[missing] = self.interpolate(g[missing], data.header(self.x_key)[missing], phase, use_abs_x)
        return times

    # ------------------------------------------------------------------ line fits (refraction)
    def _xt(self, gather: float | None, phase: str, side: str = "abs"):
        """|x| and t of a phase in one gather (or pooled over all gathers), optionally one side of the shot only."""
        gathers = [gather] if gather is not None else list(self.gathers(phase))
        xs, ts = [], []
        for g in gathers:
            p = self.points(g, phase)
            if p.size == 0:
                continue
            x = p[:, 0]
            if side == "positive":
                keep = x >= 0
            elif side == "negative":
                keep = x <= 0
            else:
                keep = np.ones(x.size, bool)
            xs.append(np.abs(x[keep]))
            ts.append(p[keep, 1])
        x = np.concatenate(xs) if xs else np.zeros(0)
        t = np.concatenate(ts) if ts else np.zeros(0)
        return x, t

    def fit_line(self, gather: float | None, phase: str, side: str = "abs") -> dict:
        """Least-squares straight line t = intercept + slowness * |x| for one phase (refraction branch, direct wave).

        ``gather=None`` pools the phase over all gathers. ``side`` = 'abs' uses |x|, 'positive'/'negative' keep only
        one side of a split spread. Returns velocity (1/slowness), intercept, slowness, rms misfit, number of picks.
        """
        x, t = self._xt(gather, phase, side)
        if x.size < 2 or np.ptp(x) == 0:
            raise ValueError(f"phase {phase!r} needs at least two picks at different x")
        A = np.column_stack([np.ones_like(x), x])
        (intercept, slowness), *_ = np.linalg.lstsq(A, t, rcond=None)
        resid = t - (intercept + slowness * x)
        return {"phase": phase, "kind": "line", "velocity": (1.0 / slowness) if slowness > 0 else np.inf,
                "intercept": float(intercept), "slowness": float(slowness), "rms": float(np.sqrt(np.mean(resid ** 2))),
                "n": int(x.size), "x_range": (float(x.min()), float(x.max()))}

    def fit_hyperbola(self, gather: float | None, phase: str, side: str = "abs") -> dict:
        """Least-squares reflection hyperbola t^2 = t0^2 + x^2 / v^2 for one phase (a reflection on a shot or CMP
        gather). Returns the NMO velocity v, the zero-offset time t0, the rms time misfit and the number of picks."""
        x, t = self._xt(gather, phase, side)
        if x.size < 3 or np.ptp(x) == 0:
            raise ValueError(f"phase {phase!r} needs at least three picks at different x")
        A = np.column_stack([np.ones_like(x), x * x])
        (a, b), *_ = np.linalg.lstsq(A, t * t, rcond=None)
        if not (a > 0 and b > 0):
            raise ValueError(f"the picks of {phase!r} do not form a reflection hyperbola")
        resid = t - np.sqrt(a + b * x * x)
        return {"phase": phase, "kind": "hyperbola", "velocity": float(1.0 / np.sqrt(b)), "t0": float(np.sqrt(a)),
                "intercept": float(np.sqrt(a)), "rms": float(np.sqrt(np.mean(resid ** 2))), "n": int(x.size),
                "x_range": (float(x.min()), float(x.max()))}

    # ------------------------------------------------------------------ files
    def to_dict(self) -> dict:
        return {"gather_key": self.gather_key, "x_key": self.x_key, "name": self.name, "phase_order": self.phase_order,
                "format": 2, "colors": self.colors, "phase_types": self.phase_types,
                "trace_ids": {str(g): phases for g, phases in self.trace_ids.items()},
                "data": {str(g): {ph: pts for ph, pts in phases.items()} for g, phases in self.data.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "PickSet":
        out = cls(d.get("gather_key", "fldr"), d.get("x_key", "offset"), d.get("name", "picks"))
        out.phase_order = list(d.get("phase_order", []))
        out.colors = dict(d.get("colors", {}))
        out.phase_types = dict(d.get("phase_types", {}))
        out.trace_ids = {float(g): {ph: list(ids) for ph, ids in phases.items()}
                         for g, phases in d.get("trace_ids", {}).items()}
        for g, phases in d.get("data", {}).items():
            out.data[float(g)] = {ph: [tuple(p) for p in pts] for ph, pts in phases.items()}
        for ph in out.phase_order:
            out.colors.setdefault(ph, cls._PALETTE[out.phase_order.index(ph) % len(cls._PALETTE)])
            if out.phase_type(ph) not in cls.PHASE_TYPES:
                raise ValueError(f"unknown event type for {ph!r}")
        for g, phases in out.data.items():
            for ph, pts in phases.items():
                if ph in out.trace_ids.get(g, {}) and len(out.trace_ids[g][ph]) != len(pts):
                    raise ValueError("pick trace identities do not match the pick count")
        return out

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "PickSet":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def refraction_layers(fits: list[dict]) -> list[dict]:
    """Horizontal-layer thicknesses from line fits ordered direct wave first (intercept-time method).

    fits[0] is the direct wave (velocity v1); fits[k] (k >= 1) the k-th refracted branch with velocity v_{k+1} and
    intercept time t_k. h_k = (t_k - sum_{j<k} 2 h_j sqrt(v_{k+1}^2 - v_j^2)/(v_j v_{k+1})) v_k v_{k+1} /
    (2 sqrt(v_{k+1}^2 - v_k^2)). Also returns crossover distances between consecutive branches.
    """
    if len(fits) < 2:
        raise ValueError("need the direct wave and at least one refracted branch")
    v = [f["velocity"] for f in fits]
    ti = [f["intercept"] for f in fits]
    layers, h = [], []
    for k in range(1, len(fits)):
        vk, vk1 = v[k - 1], v[k]
        if not (vk1 > vk):
            raise ValueError(f"branch {k} is not faster than the layer above ({vk1:.0f} <= {vk:.0f} m/s)")
        correction = sum(2.0 * h[j] * np.sqrt(vk1 ** 2 - v[j] ** 2) / (v[j] * vk1) for j in range(k - 1))
        hk = (ti[k] - correction) * vk * vk1 / (2.0 * np.sqrt(vk1 ** 2 - vk ** 2))
        h.append(hk)
        xc = (ti[k] - ti[k - 1]) / (fits[k - 1]["slowness"] - fits[k]["slowness"])
        layers.append({"layer": k, "velocity": vk, "thickness": hk, "depth_to_base": float(np.sum(h)),
                       "refractor_velocity": vk1, "crossover_distance": xc})
    return layers


def dix_layers(fits: list[dict]) -> list[dict]:
    """Interval velocity, thickness and depth of the layers between picked reflections (Dix 1955).

    ``fits`` are hyperbola fits (NMO velocity ~ RMS velocity, zero-offset time t0), ordered here by t0; layer k lies
    between reflection k-1 (the surface for k = 1) and reflection k. Flat layers are assumed."""
    out, depth, t_prev, v2t_prev = [], 0.0, 0.0, 0.0
    for k, f in enumerate(sorted(fits, key=lambda f: f["t0"]), start=1):
        t, v = f["t0"], f["velocity"]
        if t - t_prev <= 0:
            continue
        vint2 = (v * v * t - v2t_prev) / (t - t_prev)
        if vint2 <= 0:
            raise ValueError(f"interval velocity above {f['phase']!r} is undefined (RMS velocity falls too fast)")
        vint = float(np.sqrt(vint2))
        h = vint * (t - t_prev) / 2.0
        depth += h
        out.append({"layer": k, "velocity": vint, "thickness": h, "depth_to_base": depth, "t0": t,
                    "reflector": f["phase"]})
        t_prev, v2t_prev = t, v * v * t
    return out


def pick_table(ps: PickSet, data=None) -> list[dict]:
    """One row per pick (gather, phase, x, time in ms). With the dataset, each pick also gets the source and receiver
    positions and elevations of its trace (matched on the gather and x keys): the input of refraction tomography."""
    matched = {} if data is None else {(g, ph, x, t): i for g, ph, x, t, i in ps.matched_records(data)}
    by_id = {} if data is None else {str(tid): i for i, tid in enumerate(data.trace_ids)}
    rows = []
    for gather in sorted(ps.data):
        for phase in ps.phase_order:
            for x, t, tid in ps.records(gather, phase):
                row = {ps.gather_key: gather, "phase": phase, ps.x_key: x, "time_ms": round(t * 1000.0, 3),
                       "trace_id": tid or "", "event_type": ps.phase_type(phase)}
                i = by_id.get(tid) if tid is not None else matched.get((gather, phase, x, t))
                if i is not None:
                    for key in ("sx", "sy", "gx", "gy", "selev", "gelev", "cdp", "iline", "xline"):
                        if key in data.headers and key not in row:
                            row[key] = float(data.headers[key][i])
                rows.append(row)
    return rows


def write_pick_table(rows: list[dict], path: str) -> None:
    keys = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
