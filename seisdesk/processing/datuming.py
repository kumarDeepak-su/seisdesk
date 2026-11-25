"""Wave-equation datuming by Kirchhoff extrapolation (Berryhill 1979, 1984).

Receivers (in common-shot gathers) and then sources (in common-receiver gathers) are moved from the recording
surface, the topography, to a flat datum with a constant near-surface velocity, without the vertical-ray assumption of
static corrections. The zero-offset variant datums a stacked section with half the velocity (exploding reflector).
"""

from __future__ import annotations

import numpy as np

from .base import Param, Process, register, trace_positions


def spacing_weights(x: np.ndarray) -> np.ndarray:
    """Integration weight (m) of every position along the line: half the distance between its neighbours (the full
    spacing at the ends), shared by traces at the same position; zero when all positions coincide."""
    x = np.asarray(x, dtype=np.float64)
    ux, inv, counts = np.unique(np.round(x, 6), return_inverse=True, return_counts=True)
    if ux.size < 2:
        return np.zeros(x.size)
    w = np.empty(ux.size)
    w[1:-1] = 0.5 * (ux[2:] - ux[:-2])
    w[0] = ux[1] - ux[0]
    w[-1] = ux[-1] - ux[-2]
    return (w / counts)[np.asarray(inv).reshape(-1)]


def kirchhoff_datum(traces: np.ndarray, dt: float, x: np.ndarray, z: np.ndarray, datum: float, velocity: float,
                    aperture: float = 300.0, max_angle: float = 60.0, fmax: float = 80.0, block: int = 256,
                    progress=None) -> np.ndarray:
    """Extrapolate traces recorded at positions x (m) and elevations z (m, positive up) to the same positions on a
    flat datum, with a constant velocity.

    One operator matrix per frequency, the far-field form of the 2D Kirchhoff integral (Berryhill 1979):

        P(x_o, datum) = sum_i dx_i cos(theta) sqrt(w / (2 pi v r)) exp(-+ i pi/4) exp(+- i w r / v) P(x_i, z_i)

    with r the distance from (x_i, z_i) to (x_o, datum) and cos(theta) = |z_i - datum| / r. The upper signs apply
    where the surface lies above the datum (downward continuation: the data move earlier), the lower ones where it
    lies below (upward continuation: later). At every frequency the dip is limited so that the operator is not aliased
    (sin(theta) < v / (2 f dx)); where the Fresnel zone sqrt(v |dz| / f) is narrower than two trace spacings the
    operator cannot be sampled, and the output trace takes the vertical-ray shift |dz| / v at that frequency instead.
    Frequencies above fmax are removed (cosine roll-off over the top 10 %)."""
    tr = np.asarray(traces, dtype=np.float64)
    n, nt = tr.shape
    if n == 0:
        return tr.copy()
    x = np.asarray(x, dtype=np.float64)
    dz = np.asarray(z, dtype=np.float64) - float(datum)              # > 0: the surface is above the datum
    adz = np.abs(dz)
    sgn = np.where(dz >= 0.0, 1.0, -1.0)                             # +1: time advance, -1: delay
    dxw = spacing_weights(x)
    max_shift = float(np.sqrt(aperture ** 2 + float(adz.max()) ** 2) / velocity)
    pad = int(np.ceil(max_shift / dt)) + 1
    nfft = int(2 ** np.ceil(np.log2(2 * (nt + pad))))
    freqs = np.fft.rfftfreq(nfft, dt)
    spec = np.fft.rfft(tr, nfft, axis=1)
    out = np.zeros_like(spec)
    band = np.flatnonzero((freqs > 0) & (freqs <= fmax))            # contiguous: 1 .. kmax
    if band.size == 0:
        return np.zeros((n, nt))
    roll = np.clip((fmax - freqs) / max(0.1 * fmax, 1e-9), 0.0, 1.0)
    roll = 0.5 * (1.0 - np.cos(np.pi * roll))
    sin_max = np.sin(np.deg2rad(max_angle))
    df = freqs[1] - freqs[0]
    order = np.argsort(x, kind="stable")
    xs = x[order]
    nblocks = int(np.ceil(n / block))
    for nb, b0 in enumerate(range(0, n, block)):
        o_idx = order[b0:b0 + block]                                   # output traces of this block
        i0 = int(np.searchsorted(xs, xs[b0] - aperture, side="left"))
        i1 = int(np.searchsorted(xs, xs[min(b0 + block, n) - 1] + aperture, side="right"))
        i_idx = order[i0:i1]                                           # input traces within the aperture
        h = x[o_idx][:, None] - x[i_idx][None, :]
        r = np.maximum(np.sqrt(h * h + adz[i_idx][None, :] ** 2), 1e-9)
        cos = adz[i_idx][None, :] / r
        sin = np.abs(h) / r
        ap = np.clip((aperture - np.abs(h)) / max(0.2 * aperture, 1e-9), 0.0, 1.0)
        ap = 0.5 * (1.0 - np.cos(np.pi * ap)) * ((sin <= sin_max) & (adz[i_idx][None, :] > 1e-6))
        amp = ap * dxw[i_idx][None, :] * cos / np.sqrt(2.0 * np.pi * velocity * r)
        s_cols = sgn[i_idx][None, :]
        quarter = np.exp(-1j * s_cols * np.pi / 4.0)
        phase = np.exp(1j * s_cols * (2 * np.pi * freqs[band[0]]) * r / velocity)
        step = np.exp(1j * s_cols * (2 * np.pi * df) * r / velocity)
        two_dx = 2.0 * dxw[i_idx][None, :]
        for count, k in enumerate(band):
            if count:
                phase *= step
            f = freqs[k]
            w = 2 * np.pi * f
            unaliased = sin * (two_dx * f) <= velocity                 # sin(theta) < v / (2 f dx)
            col = (amp * unaliased * np.sqrt(w) * quarter * phase) @ spec[i_idx, k]
            static = (np.sqrt(velocity * adz[o_idx] / f) < 2.0 * dxw[o_idx]) | (dxw[o_idx] <= 0.0)
            if np.any(static):
                so = o_idx[static]
                col[static] = spec[so, k] * np.exp(1j * sgn[so] * w * adz[so] / velocity)
            out[o_idx, k] = col * roll[k]
        if progress is not None:
            progress((nb + 1) / nblocks)
    return np.fft.irfft(out, nfft, axis=1)[:, :nt]


def _groups(values: np.ndarray) -> list:
    """Trace indices of every distinct value (rounded to the mm)."""
    keys = np.round(np.asarray(values, dtype=np.float64), 3)
    order = np.argsort(keys, kind="stable")
    bounds = np.flatnonzero(np.diff(keys[order])) + 1
    return [chunk for chunk in np.split(order, bounds) if chunk.size]


_DATUM_PARAMS = (Param("aperture", "float", 300.0, "Aperture (half width)", unit="m", minimum=1.0),
                 Param("max_angle", "float", 60.0, "Maximum angle", unit="deg", minimum=1.0, maximum=89.0),
                 Param("fmax", "float", 80.0, "Highest frequency (output limited to it)", unit="Hz", minimum=1.0))


@register
class DatumKirchhoff(Process):
    name = "datum_kirchhoff"
    label = "Wave-equation datuming (Kirchhoff, prestack)"
    category = "Datuming"
    description = ("Moves receivers and sources from the topography to a flat datum by Kirchhoff extrapolation "
                   "(Berryhill 1979, 1984): receivers in every common-shot gather (from gelev), then sources in "
                   "every common-receiver gather (from selev minus sdepth), with one constant near-surface "
                   "velocity. Unlike elevation statics it follows the actual ray directions, so it stays correct "
                   "for steep rays, rough topography and large elevation changes. A datum below the surface is a "
                   "downward continuation (use the velocity of the layer removed); above the surface an upward "
                   "continuation through a replacement medium. Receivers and sources should be sampled densely "
                   "(a few metres to a few tens of metres); where the spacing is too coarse for the operator the "
                   "step falls back to the vertical-ray shift at those frequencies. Headers gelev/selev are set to "
                   "the datum and sdepth to 0.")
    params = (Param("datum", "float", 0.0, "Datum elevation", unit="m"),
              Param("velocity", "float", 1500.0, "Near-surface velocity", unit="m/s", minimum=1.0),
              Param("side", "choice", "receivers and sources", "Move",
                    choices=("receivers and sources", "receivers only", "sources only"))) + _DATUM_PARAMS

    def run(self, data, context):
        if data.domain != "time":
            raise ValueError("datuming needs time data")
        if not (data.has_header("gx") or data.has_header("sx")):
            raise ValueError("needs source and receiver positions (sx, gx): run a geometry step first")
        if not (data.has_header("gelev") or data.has_header("selev")):
            context.log("    no elevations (selev, gelev) in the headers: the surface is taken at elevation 0")
        side = self["side"]
        do_rec, do_src = side != "sources only", side != "receivers only"
        out = data.traces.astype(np.float64)
        shots = _groups(data.header("fldr") if data.has_header("fldr") else data.header("sx"))
        receivers = _groups(data.header("gx"))
        total = (len(shots) if do_rec else 0) + (len(receivers) if do_src else 0)
        args = (self["datum"], self["velocity"], self["aperture"], self["max_angle"], self["fmax"])
        done = 0
        if do_rec:
            gx, gz = data.header("gx"), data.header("gelev")
            for idx in shots:
                out[idx] = kirchhoff_datum(out[idx], data.dt, gx[idx], gz[idx], *args)
                done += 1
                context.report(done / total)
        if do_src:
            sx, sz = data.header("sx"), data.header("selev") - data.header("sdepth")
            for idx in receivers:
                out[idx] = kirchhoff_datum(out[idx], data.dt, sx[idx], sz[idx], *args)
                done += 1
                context.report(done / total)
        res = data.copy(out.astype(np.float32))
        if do_rec:
            res.set_header("gelev", np.full(data.ntr, float(self["datum"])))
        if do_src:
            res.set_header("selev", np.full(data.ntr, float(self["datum"])))
            res.set_header("sdepth", np.zeros(data.ntr))
        res.meta["datum"] = {"elevation": self["datum"], "velocity": self["velocity"], "method": f"kirchhoff, {side}"}
        context.log(f"    datumed to {self['datum']:g} m with {self['velocity']:g} m/s ({side}): {len(shots)} shot and "
                    f"{len(receivers)} receiver gathers")
        return res


@register
class DatumZeroOffset(Process):
    name = "datum_zero_offset"
    label = "Wave-equation datuming (zero offset, stacked)"
    category = "Datuming"
    description = ("Kirchhoff datuming of a stacked (zero-offset) section from a floating datum, the elevation in a "
                   "header at every CMP, to a flat datum: the exploding-reflector form of the prestack step, with "
                   "half the near-surface velocity because stacked times are two-way. The elevation header is set "
                   "to the datum.")
    params = (Param("datum", "float", 0.0, "Datum elevation", unit="m"),
              Param("velocity", "float", 2000.0, "Near-surface velocity", unit="m/s", minimum=1.0),
              Param("elevation_key", "key", "gelev", "Elevation header (floating datum)")) + _DATUM_PARAMS

    def run(self, data, context):
        if data.domain != "time":
            raise ValueError("datuming needs time data")
        x = trace_positions(data)
        z = data.header(self["elevation_key"])
        out = kirchhoff_datum(data.traces, data.dt, x, z, self["datum"], 0.5 * self["velocity"], self["aperture"],
                              self["max_angle"], self["fmax"], progress=context.report)
        res = data.copy(out.astype(np.float32))
        res.set_header(self["elevation_key"], np.full(data.ntr, float(self["datum"])))
        res.meta["datum"] = {"elevation": self["datum"], "velocity": self["velocity"],
                             "method": "kirchhoff, zero offset"}
        return res
