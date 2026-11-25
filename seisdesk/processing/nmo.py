"""Normal-moveout correction (hyperbolic or VTI non-hyperbolic) with stretch mute, and inverse NMO."""

from __future__ import annotations

import numpy as np

from .base import Param, Process, register, resolve_velocity
from .moveout import moveout_time, sample_at


def nmo_correct(traces: np.ndarray, times: np.ndarray, offsets: np.ndarray, vfield: np.ndarray,
                eta: np.ndarray | float = 0.0, stretch_limit: float = 0.5) -> np.ndarray:
    """out(t0) = in(t(t0, x)) per trace; vfield and eta are (ntr, nt) functions of t0. Stretch > limit is muted."""
    dt = times[1] - times[0]
    t0 = np.broadcast_to(times[None, :], traces.shape)
    x = np.asarray(offsets, dtype=np.float64)[:, None]
    tt = moveout_time(t0, x, vfield, eta)
    vals, valid = sample_at(traces.astype(np.float64), tt, times[0], dt)
    if stretch_limit > 0:
        # output sample spacing dt0 maps to dt = (dt/dt0) dt0 of input: the wavelet is stretched by dt0/dt - 1
        dtt = np.gradient(tt, dt, axis=1)
        valid &= dtt >= 1.0 / (1.0 + stretch_limit)
    return np.where(valid, vals, 0.0).astype(np.float32)


def inverse_nmo(traces: np.ndarray, times: np.ndarray, offsets: np.ndarray, vfield: np.ndarray,
                eta: np.ndarray | float = 0.0) -> np.ndarray:
    """Undo NMO: out(t) = in(t0(t, x)), t0 found by inverting the monotone moveout curve per trace."""
    dt = times[1] - times[0]
    t0 = np.broadcast_to(times[None, :], traces.shape)
    tt = moveout_time(t0, np.asarray(offsets, dtype=np.float64)[:, None], vfield, eta)
    out = np.zeros_like(traces, dtype=np.float64)
    for i in range(traces.shape[0]):
        t0_of_t = np.interp(times, tt[i], times, left=np.nan, right=np.nan)
        pos = (t0_of_t - times[0]) / dt
        ok = np.isfinite(pos)
        out[i, ok] = np.interp(pos[ok], np.arange(times.size), traces[i])
    return out.astype(np.float32)


def velocity_per_trace(data, picks, column: int = 1) -> np.ndarray:
    """(ntr, nt) field of v_rms (or eta) for every trace from its CMP."""
    cdp = data.header("cdp")
    ucdp, inv = np.unique(cdp, return_inverse=True)
    f = picks.field(ucdp, data.times, column=column)
    return f[inv]


@register
class NMO(Process):
    name = "nmo"
    label = "NMO correction"
    category = "Moveout"
    description = ("Normal-moveout correction with RMS velocities (and anisotropy parameter eta, VTI) interpolated "
                   "from the picks; samples stretched more than the limit are muted.")
    params = (Param("velocity", "velocity", "project", "Velocity", "project picks, a constant, 't:v,...' or a file"),
              Param("use_eta", "bool", False, "Use picked eta (VTI non-hyperbolic)"),
              Param("eta", "float", 0.0, "Constant eta (if no picked eta)"),
              Param("stretch_limit", "float", 0.5, "Stretch mute limit (0.5 = 50 %)", minimum=0.0),
              Param("inverse", "bool", False, "Inverse NMO"))

    def run(self, data, context):
        picks = resolve_velocity(self["velocity"], context)
        vf = velocity_per_trace(data, picks)
        eta = velocity_per_trace(data, picks, column=2) if self["use_eta"] else self["eta"]
        off = data.header("offset")
        out = np.empty_like(data.traces)
        block = 2000                                   # bounded temporaries for long lines
        for s in range(0, data.ntr, block):
            sl = slice(s, s + block)
            e = eta[sl] if np.ndim(eta) else eta
            if self["inverse"]:
                out[sl] = inverse_nmo(data.traces[sl], data.times, off[sl], vf[sl], e)
            else:
                out[sl] = nmo_correct(data.traces[sl], data.times, off[sl], vf[sl], e, self["stretch_limit"])
            context.report(min(1.0, (s + block) / data.ntr))
        return data.copy(out)
