"""3D time migration of small volumes: phase shift (post-stack, v(t)), Stolt (post-stack, constant v) and Kirchhoff
(pre- or post-stack, straight-ray double-square-root traveltimes in v_rms(t)).

Traces sit on the inline/crossline grid of the 3D binning (headers iline and xline; bin sizes from the binning or
given as parameters). The post-stack methods hold the whole cube and its padded 3D spectrum in memory, and the
Kirchhoff summation visits every output bin within the aperture for every input trace, so these steps are meant for
small volumes: tens of inlines and crosslines, as in site surveys and test cubes.
"""

from __future__ import annotations

import numpy as np

from ..core.velocity import dix
from .base import Param, Process, pad_to_time_zero, register, resolve_velocity


def _pad(n: int) -> int:
    """FFT length for one spatial axis: the power of two at or above 1.5 n (room for the migration to spread)."""
    return int(2 ** np.ceil(np.log2(max(2, int(np.ceil(1.5 * n))))))


def _grid_axis(values: np.ndarray):
    """Grid index of every trace along one bin-number axis, the numbering step and the number of grid lines."""
    u = np.unique(values)
    if u.size == 1:
        return np.zeros(values.size, np.int64), 1.0, 1
    step = float(np.min(np.diff(u)))
    idx = np.rint((values - u[0]) / step).astype(np.int64)
    return idx, step, int(idx.max()) + 1


class Cube:
    """The traces of a 3D dataset on its inline/crossline grid (empty bins are zero traces, duplicates averaged)."""

    def __init__(self, data):
        if not (data.has_header("iline") and data.has_header("xline")):
            raise ValueError("inline and crossline numbers (iline, xline) are not set: run 3D binning first")
        ii, self.il_step, self.nil = _grid_axis(data.header("iline"))
        jj, self.xl_step, self.nxl = _grid_axis(data.header("xline"))
        flat = ii * self.nxl + jj
        self.bins, self.first, counts = np.unique(flat, return_index=True, return_counts=True)
        cube = np.zeros((self.nil * self.nxl, data.nt))
        np.add.at(cube, flat, data.traces.astype(np.float64))
        cube[self.bins] /= counts[:, None]
        self.values = cube.reshape(self.nil, self.nxl, data.nt)
        self.duplicates = bool(counts.max() > 1)

    def bin_sizes(self, data, along_inline: float, along_crossline: float) -> tuple[float, float]:
        """Grid spacing (m) between inline numbers and between crossline numbers. geometry_3d's 'inline_bin' is the
        bin size along an inline, i.e. the distance between neighbouring crosslines, and 'xline_bin' the distance
        between neighbouring inlines."""
        geom = data.meta.get("geometry", {})
        d_xl = along_inline if along_inline > 0 else float(geom.get("inline_bin") or 0.0)
        d_il = along_crossline if along_crossline > 0 else float(geom.get("xline_bin") or 0.0)
        if d_xl <= 0 or d_il <= 0:
            raise ValueError("bin sizes unknown: run 3D binning, or give the bin sizes along the inlines and crosslines")
        return d_il * self.il_step, d_xl * self.xl_step

    def to_dataset(self, data, values: np.ndarray, name: str):
        """One output trace per occupied bin, carrying the headers of the bin's first input trace."""
        flat = values.reshape(self.nil * self.nxl, -1)[self.bins]
        out = data.subset(self.first).copy(flat.astype(np.float32), name=name)
        out.set_header("offset", 0.0)
        return out


# ---------------------------------------------------------------------------------------------- post-stack, f-k
def phase_shift_3d(cube: np.ndarray, dt: float, d_il: float, d_xl: float, vint: np.ndarray,
                   fmax: float | None = None, progress=None) -> np.ndarray:
    """Gazdag phase-shift time migration of a zero-offset cube (nil, nxl, nt) with interval velocity vint(tau):
    per time step exp(i w dtau sqrt(1 - u^2 (k_il^2 + k_xl^2) / w^2)), u = v / 2 (exploding reflector)."""
    nil, nxl, nt = cube.shape
    na, nb = _pad(nil), _pad(nxl)
    ntp = int(2 ** np.ceil(np.log2(2 * nt)))
    w = 2 * np.pi * np.fft.rfftfreq(ntp, dt)
    nw = w.size if fmax is None else int(np.searchsorted(w, 2 * np.pi * fmax, side="right"))
    spec = np.fft.rfft(cube.astype(np.float64), ntp, axis=2)[:, :, :nw]
    D = np.fft.fft2(spec, s=(na, nb), axes=(0, 1)).astype(np.complex64)                   # (na, nb, nw)
    k2 = ((2 * np.pi * np.fft.fftfreq(na, d_il))[:, None] ** 2
          + (2 * np.pi * np.fft.fftfreq(nb, d_xl))[None, :] ** 2)[:, :, None]
    w = w[None, None, :nw]
    w_safe2 = np.where(w == 0, 1e-30, w) ** 2
    weights = np.ones(nw, dtype=np.complex64)
    weights[0] = 0.5
    image_k = np.zeros((na, nb, nt), dtype=np.complex64)
    for it in range(nt):
        image_k[:, :, it] = D @ weights
        u = 0.5 * float(vint[it])
        arg = 1.0 - u * u * k2 / w_safe2
        prop = arg > 0
        shift = np.exp(1j * w * dt * np.sqrt(np.where(prop, arg, 0.0))).astype(np.complex64)
        D = np.where(prop, D * shift, 0)
        if progress is not None and it % 20 == 0:
            progress(it / nt)
    return 2.0 * np.real(np.fft.ifft2(image_k, axes=(0, 1)))[:nil, :nxl] / ntp


def stolt_3d(cube: np.ndarray, dt: float, d_il: float, d_xl: float, v: float) -> np.ndarray:
    """Stolt f-k time migration of a zero-offset cube (nil, nxl, nt) with constant velocity v (exploding reflector
    v/2): w -> sqrt(w^2 + u^2 (k_il^2 + k_xl^2)) with the Jacobian w / w_in, as in the 2D Stolt migration."""
    nil, nxl, nt = cube.shape
    na, nb = _pad(nil), _pad(nxl)
    ntp = int(2 ** np.ceil(np.log2(2 * nt)))
    D = np.fft.fft2(np.fft.rfft(cube.astype(np.float64), ntp, axis=2), s=(na, nb), axes=(0, 1))   # (na, nb, nw)
    ka = 2 * np.pi * np.fft.fftfreq(na, d_il)
    kb = 2 * np.pi * np.fft.fftfreq(nb, d_xl)
    w = 2 * np.pi * np.fft.rfftfreq(ntp, dt)
    nw = w.size
    dw = w[1] - w[0]
    u = 0.5 * v
    M = np.zeros_like(D)
    for ia in range(na):
        w_in = np.sqrt(w[None, :] ** 2 + u * u * (ka[ia] ** 2 + kb[:, None] ** 2))          # (nb, nw)
        pos = w_in / dw
        i0 = np.floor(pos).astype(np.int64)
        frac = pos - i0
        ok = i0 + 1 < nw
        i0 = np.minimum(i0, nw - 2)
        row = D[ia]
        val = (1 - frac) * np.take_along_axis(row, i0, axis=1) + frac * np.take_along_axis(row, i0 + 1, axis=1)
        jac = np.divide(np.broadcast_to(w, w_in.shape), w_in, out=np.ones_like(w_in), where=w_in > 0)
        M[ia] = np.where(ok, val, 0.0) * jac
    return np.fft.irfft(np.fft.ifft2(M, axes=(0, 1)), ntp, axis=2)[:nil, :nxl, :nt]


# ---------------------------------------------------------------------------------------------- Kirchhoff
def kirchhoff_time_3d(traces: np.ndarray, t0: float, dt: float, sx: np.ndarray, sy: np.ndarray, gx: np.ndarray,
                      gy: np.ndarray, xo: np.ndarray, yo: np.ndarray, tau: np.ndarray, vrms: np.ndarray,
                      aperture: float, max_angle_deg: float = 60.0, progress=None) -> np.ndarray:
    """Prestack 3D Kirchhoff time migration to the bins (xo, yo): image(tau, bin) is the sum over the traces of the
    time derivative of the data at t = sqrt(tau^2/4 + |b - s|^2 / v^2) + sqrt(tau^2/4 + |b - g|^2 / v^2), weighted
    by the obliquity tau / t. The anti-causal derivative (-i w in NumPy's transform pair) is the 3D counterpart of
    the 2D half derivative: summing over a surface of later data times integrates once, anti-causally.
    vrms: (ntau, nbins)."""
    ntr, nt = traces.shape
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    w = 2 * np.pi * np.fft.rfftfreq(nfft, dt)
    filt = np.fft.irfft(np.fft.rfft(traces.astype(np.float64), nfft, axis=1) * (-1j * w), nfft, axis=1)[:, :nt]
    image = np.zeros((tau.size, xo.size))
    half_tau2 = (0.5 * tau[:, None]) ** 2
    tan_max = np.tan(np.deg2rad(max_angle_deg))
    for i in range(ntr):
        mx, my = 0.5 * (sx[i] + gx[i]), 0.5 * (sy[i] + gy[i])
        dist = np.hypot(xo - mx, yo - my)
        cols = np.flatnonzero(dist <= aperture)
        if cols.size == 0:
            continue
        v2 = vrms[:, cols] ** 2
        ts = np.sqrt(half_tau2 + ((xo[cols] - sx[i]) ** 2 + (yo[cols] - sy[i]) ** 2)[None, :] / v2)
        tg = np.sqrt(half_tau2 + ((xo[cols] - gx[i]) ** 2 + (yo[cols] - gy[i]) ** 2)[None, :] / v2)
        t = ts + tg
        pos = (t - t0) / dt
        ok = (pos >= 0) & (pos < nt - 1)
        ok &= dist[cols][None, :] <= np.maximum(0.5 * tau[:, None] * vrms[:, cols] * tan_max, 1e-6)
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, nt - 2)
        frac = pos - i0
        val = (1 - frac) * filt[i, i0] + frac * filt[i, i0 + 1]
        wgt = np.where(t > 0, tau[:, None] / np.maximum(t, 1e-9), 0.0)
        image[:, cols] += np.where(ok, val * wgt, 0.0)
        if progress is not None and i % 200 == 0:
            progress(i / ntr)
    return image


# ---------------------------------------------------------------------------------------------- steps
_BIN_PARAMS = (Param("bin_inline", "float", 0.0, "Bin size along an inline (0 = from the 3D binning)", unit="m",
                     minimum=0.0),
               Param("bin_crossline", "float", 0.0, "Bin size along a crossline (0 = from the 3D binning)", unit="m",
                     minimum=0.0))


@register
class PhaseShift3D(Process):
    name = "phase_shift_3d"
    label = "3D phase-shift migration (post-stack, v(t))"
    category = "Time migration"
    description = ("Gazdag phase-shift migration of a stacked 3D volume on its inline/crossline grid (3D binning): RMS "
                   "velocities at one CMP bin are Dix-converted to interval velocity v(tau); empty bins count as zero "
                   "traces. The cube and its padded 3D spectrum are held in memory: small volumes only (tens of "
                   "inlines and crosslines).")
    params = (Param("velocity", "velocity", "project", "Velocity", "project picks, a constant, 't:v,...' or a file"),
              Param("cdp", "float", -1.0, "CMP bin for the velocity function (-1 = middle)")) + _BIN_PARAMS + (
              Param("fmax", "float", 90.0, "Highest frequency", unit="Hz", minimum=1.0),)

    def run(self, data, context):
        data = pad_to_time_zero(data)
        cube = Cube(data)
        if cube.duplicates:
            context.log("    several traces share a bin: averaged (the input should be a stack)")
        d_il, d_xl = cube.bin_sizes(data, self["bin_inline"], self["bin_crossline"])
        picks = resolve_velocity(self["velocity"], context)
        c = self["cdp"] if self["cdp"] >= 0 else float(np.median(picks.cdps))
        vint = dix(data.times, picks.function(c, data.times))
        context.log(f"    cube {cube.nil} x {cube.nxl} bins of {d_il:g} x {d_xl:g} m, {data.nt} samples")
        img = phase_shift_3d(cube.values, data.dt, d_il, d_xl, vint, self["fmax"], context.report)
        out = cube.to_dataset(data, img, data.name + " 3D phase shift")
        out.meta["migrated"] = "phase_shift_3d"
        return out


@register
class Stolt3D(Process):
    name = "stolt_3d"
    label = "3D Stolt migration (post-stack, constant v)"
    category = "Time migration"
    description = ("f-k migration of a stacked 3D volume with one constant velocity, on its inline/crossline grid: "
                   "fast, for checks and simple geology. Small volumes only (the padded 3D spectrum is held in "
                   "memory).")
    params = (Param("velocity", "float", 2000.0, "Velocity", unit="m/s", minimum=10.0),) + _BIN_PARAMS

    def run(self, data, context):
        data = pad_to_time_zero(data)
        cube = Cube(data)
        if cube.duplicates:
            context.log("    several traces share a bin: averaged (the input should be a stack)")
        d_il, d_xl = cube.bin_sizes(data, self["bin_inline"], self["bin_crossline"])
        img = stolt_3d(cube.values, data.dt, d_il, d_xl, self["velocity"])
        out = cube.to_dataset(data, img, data.name + " 3D Stolt")
        out.meta["migrated"] = "stolt_3d"
        return out


@register
class KirchhoffPSTM3D(Process):
    name = "kirchhoff_pstm_3d"
    label = "3D Kirchhoff time migration (pre- or post-stack)"
    category = "Time migration"
    description = ("Kirchhoff summation in 3D with straight-ray double-square-root traveltimes in v_rms(t, CMP bin). "
                   "Input: prestack traces with source and receiver x and y, or a stack (treated as zero offset at the "
                   "bin centres cdpx, cdpy). Output: one migrated trace per occupied bin (for prestack input, the "
                   "migrated stack). Every input trace visits every bin within the aperture: small volumes only.")
    params = (Param("velocity", "velocity", "project", "Velocity", "project picks, a constant, 't:v,...' or a file"),
              Param("aperture", "float", 1000.0, "Aperture (radius)", unit="m", minimum=10.0),
              Param("max_angle", "float", 60.0, "Maximum angle", unit="deg", minimum=1.0, maximum=89.0))

    def run(self, data, context):
        if not (data.has_header("iline") and data.has_header("xline")):
            raise ValueError("inline and crossline numbers (iline, xline) are not set: run 3D binning first")
        pairs = np.stack([data.header("iline"), data.header("xline")], axis=1)
        _, first = np.unique(pairs, axis=0, return_index=True)
        cx, cy = data.header("cdpx"), data.header("cdpy")
        if np.ptp(cx) == 0 and np.ptp(cy) == 0:
            raise ValueError("bin centres (cdpx, cdpy) are not set: run 3D binning first")
        xo, yo = cx[first], cy[first]
        picks = resolve_velocity(self["velocity"], context)
        tau = data.times
        vrms = picks.field(data.header("cdp")[first], tau).T                            # (ntau, nbins)
        sx, sy, gx, gy = (data.header(k) for k in ("sx", "sy", "gx", "gy"))
        if not (np.any(sx) or np.any(gx) or np.any(sy) or np.any(gy)):
            sx, gx, sy, gy = cx, cx, cy, cy
        context.log(f"    {data.ntr} traces into {first.size} bins")
        image = kirchhoff_time_3d(data.traces, data.t0, data.dt, sx, sy, gx, gy, xo, yo, tau, vrms,
                                  self["aperture"], self["max_angle"], context.report)
        out = data.subset(first).copy(image.T.astype(np.float32), name=data.name + " 3D PSTM")
        for key, val in (("sx", xo), ("gx", xo), ("sy", yo), ("gy", yo), ("offset", 0.0)):
            out.set_header(key, val)
        out.meta["migrated"] = "kirchhoff_pstm_3d"
        return out
