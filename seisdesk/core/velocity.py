"""Velocity picks and velocity fields.

``VelocityPicks`` holds the (t0, v_rms[, eta]) picks of a velocity analysis per CMP and interpolates them into a
(CMP, time) field: linearly in time inside each picked CMP (constant beyond the first and last pick) and linearly
between picked CMPs (constant beyond the end ones). ``DepthModel`` is an interval-velocity grid v(z, x) for depth
migration, built from picks by Dix conversion or set directly.
"""

from __future__ import annotations

import json

import numpy as np


class VelocityPicks:
    def __init__(self):
        self._picks: dict[float, list[tuple[float, float, float]]] = {}

    # ------------------------------------------------------------------ editing
    def add(self, cdp: float, t: float, v: float, eta: float = 0.0) -> None:
        if v <= 0:
            raise ValueError("velocity must be positive")
        rows = [p for p in self._picks.get(float(cdp), []) if abs(p[0] - t) > 1e-6]
        rows.append((float(t), float(v), float(eta)))
        self._picks[float(cdp)] = sorted(rows)

    def remove_nearest(self, cdp: float, t: float, v: float | None = None, t_scale: float = 0.05,
                       v_scale: float = 200.0) -> bool:
        rows = self._picks.get(float(cdp), [])
        if not rows:
            return False
        if v is None:
            dist = [abs(r[0] - t) for r in rows]
        else:
            dist = [np.hypot((r[0] - t) / t_scale, (r[1] - v) / v_scale) for r in rows]
        rows.pop(int(np.argmin(dist)))
        if rows:
            self._picks[float(cdp)] = rows
        else:
            self._picks.pop(float(cdp))
        return True

    def clear(self, cdp: float | None = None) -> None:
        if cdp is None:
            self._picks.clear()
        else:
            self._picks.pop(float(cdp), None)

    # ------------------------------------------------------------------ queries
    @property
    def cdps(self) -> np.ndarray:
        return np.array(sorted(self._picks), dtype=np.float64)

    def picks_at(self, cdp: float) -> np.ndarray:
        """(n, 3) array of (t, v, eta) picks at one CMP (empty if none)."""
        rows = self._picks.get(float(cdp), [])
        return np.array(rows, dtype=np.float64).reshape(-1, 3)

    def __len__(self) -> int:
        return sum(len(v) for v in self._picks.values())

    def _function(self, cdp_value: float, times: np.ndarray, column: int) -> np.ndarray:
        p = self.picks_at(cdp_value)
        if p.size == 0:
            raise ValueError(f"no picks at CMP {cdp_value:g}")
        return np.interp(times, p[:, 0], p[:, column])

    def field(self, cdps, times, column: int = 1) -> np.ndarray:
        """(len(cdps), len(times)) field of v_rms (column 1) or eta (column 2)."""
        cdps = np.atleast_1d(np.asarray(cdps, dtype=np.float64))
        times = np.asarray(times, dtype=np.float64)
        picked = self.cdps
        if picked.size == 0:
            raise ValueError("no velocity picks")
        funcs = np.stack([self._function(c, times, column) for c in picked])          # (npicked, nt)
        out = np.empty((cdps.size, times.size))
        for j in range(times.size):
            out[:, j] = np.interp(cdps, picked, funcs[:, j])
        return out

    def function(self, cdp: float, times) -> np.ndarray:
        return self.field([cdp], times)[0]

    def eta_field(self, cdps, times) -> np.ndarray:
        return self.field(cdps, times, column=2)

    # ------------------------------------------------------------------ files
    def save(self, path: str) -> None:
        """Text file: one 'cdp  t_ms  v_m/s  eta' line per pick."""
        with open(path, "w", encoding="utf-8") as f:
            f.write("# SeisDesk velocity picks: cdp  t(ms)  vrms(m/s)  eta\n")
            for cdp in self.cdps:
                for t, v, eta in self.picks_at(cdp):
                    f.write(f"{cdp:g}\t{t * 1000.0:.3f}\t{v:.2f}\t{eta:.4f}\n")

    @classmethod
    def load(cls, path: str) -> "VelocityPicks":
        out = cls()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                cdp, t_ms, v = float(parts[0]), float(parts[1]), float(parts[2])
                eta = float(parts[3]) if len(parts) > 3 else 0.0
                out.add(cdp, t_ms / 1000.0, v, eta)
        return out

    @classmethod
    def constant(cls, v: float, cdp: float = 1.0) -> "VelocityPicks":
        out = cls()
        out.add(cdp, 0.0, v)
        return out

    @classmethod
    def from_function(cls, times_s, velocities, cdp: float = 1.0) -> "VelocityPicks":
        out = cls()
        for t, v in zip(times_s, velocities):
            out.add(cdp, float(t), float(v))
        return out

    def to_json(self) -> str:
        return json.dumps({str(k): v for k, v in self._picks.items()})

    @classmethod
    def from_json(cls, text: str) -> "VelocityPicks":
        out = cls()
        for k, rows in json.loads(text).items():
            for t, v, eta in rows:
                out.add(float(k), t, v, eta)
        return out


# ---------------------------------------------------------------------------------------------- conversions
def dix(times: np.ndarray, vrms: np.ndarray) -> np.ndarray:
    """Interval velocity on the same time samples from RMS velocity (two-way time), Dix (1955).

    Negative radicands (RMS velocity decreasing too fast) are clipped to the previous interval velocity."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(vrms, dtype=np.float64)
    vint = np.empty_like(v)
    vint[0] = v[0]
    num = np.diff(t * v * v)
    den = np.diff(t)
    for i in range(1, v.size):
        val = num[i - 1] / den[i - 1] if den[i - 1] > 0 else -1.0
        vint[i] = np.sqrt(val) if val > 0 else vint[i - 1]
    return vint


def vint_to_vrms(times: np.ndarray, vint: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(vint, dtype=np.float64)
    dt = np.diff(t, prepend=0.0)
    cum = np.cumsum(v * v * dt)
    out = np.sqrt(np.divide(cum, t, out=np.full_like(cum, v[0] ** 2), where=t > 0))
    out[t <= 0] = v[0]
    return out


def time_to_depth_axis(times: np.ndarray, vint: np.ndarray) -> np.ndarray:
    """Depth of each two-way time sample for an interval-velocity function v(t)."""
    t = np.asarray(times, dtype=np.float64)
    dt = np.diff(t, prepend=t[0])
    return np.cumsum(0.5 * vint * dt) + 0.5 * vint[0] * t[0]


class DepthModel:
    """Interval velocity v(z, x) on a regular grid: values (nz, nx), z0/dz in metres, x0/dx in metres."""

    def __init__(self, values: np.ndarray, dz: float, dx: float, z0: float = 0.0, x0: float = 0.0):
        self.values = np.asarray(values, dtype=np.float64)
        if self.values.ndim != 2:
            raise ValueError("values must be (nz, nx)")
        self.dz, self.dx, self.z0, self.x0 = float(dz), float(dx), float(z0), float(x0)

    @property
    def shape(self):
        return self.values.shape

    @property
    def z(self) -> np.ndarray:
        return self.z0 + np.arange(self.values.shape[0]) * self.dz

    @property
    def x(self) -> np.ndarray:
        return self.x0 + np.arange(self.values.shape[1]) * self.dx

    def at(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates
        iz = (np.asarray(z) - self.z0) / self.dz
        ix = (np.asarray(x) - self.x0) / self.dx
        return map_coordinates(self.values, [iz, ix], order=1, mode="nearest")

    @classmethod
    def constant(cls, v: float, nz: int, nx: int, dz: float, dx: float, z0: float = 0.0, x0: float = 0.0):
        return cls(np.full((nz, nx), float(v)), dz, dx, z0, x0)

    @classmethod
    def gradient(cls, v0: float, k: float, nz: int, nx: int, dz: float, dx: float, z0: float = 0.0, x0: float = 0.0):
        z = z0 + np.arange(nz) * dz
        return cls(np.repeat((v0 + k * z)[:, None], nx, axis=1), dz, dx, z0, x0)

    @classmethod
    def from_picks(cls, picks: VelocityPicks, cdp_x: dict | None, cdps, nz: int, dz: float, nx: int, dx: float,
                   x0: float = 0.0, tmax: float = 8.0, smooth_m: float = 0.0) -> "DepthModel":
        """Dix-convert the RMS picks at each CMP, map interval velocity from time to depth and sample on the grid.

        ``cdp_x`` maps CMP number to x (metres); without it x = x0 + (cdp - cdps[0]) * dx is assumed.
        """
        times = np.linspace(0.0, tmax, 2001)
        cdps = np.asarray(cdps, dtype=np.float64)
        vrms = picks.field(cdps, times)
        zgrid = np.arange(nz) * dz
        columns = []
        for i in range(cdps.size):
            vint = dix(times, vrms[i])
            depth = time_to_depth_axis(times, vint)
            columns.append(np.interp(zgrid, depth, vint))
        cols = np.stack(columns, axis=1)                        # (nz, ncdp)
        if cdp_x:
            xs = np.array([cdp_x.get(c, np.nan) for c in cdps])
        else:
            xs = x0 + (cdps - cdps[0]) * dx
        keep = np.isfinite(xs)
        order = np.argsort(xs[keep], kind="stable")          # np.interp needs increasing x
        xs, cols = xs[keep][order], cols[:, keep][:, order]
        if xs.size == 0:
            raise ValueError("no CMP positions for the depth model")
        xgrid = x0 + np.arange(nx) * dx
        values = np.empty((nz, nx))
        for k in range(nz):
            values[k] = np.interp(xgrid, xs, cols[k])
        if smooth_m > 0:
            from scipy.ndimage import gaussian_filter
            values = gaussian_filter(values, (smooth_m / dz, smooth_m / dx), mode="nearest")
        return cls(values, dz, dx, 0.0, x0)

    def save(self, path: str) -> None:
        np.savez(path, values=self.values, dz=self.dz, dx=self.dx, z0=self.z0, x0=self.x0)

    @classmethod
    def load(cls, path: str) -> "DepthModel":
        z = np.load(path)
        return cls(z["values"], float(z["dz"]), float(z["dx"]), float(z["z0"]), float(z["x0"]))
