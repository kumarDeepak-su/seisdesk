"""Time migration: Stolt (constant velocity), Gazdag phase shift (v(t)), Kirchhoff prestack/poststack (v_rms(t, x)).

Post-stack inputs are stacked sections (one trace per CMP, regular CMP spacing); Kirchhoff takes prestack data with
source and receiver coordinates (or a stack, treated as zero offset).
"""

from __future__ import annotations

import numpy as np

from ..core.dataset import SeismicData
from ..core.velocity import dix
from .base import Param, Process, pad_to_time_zero, register, resolve_velocity
from .moveout import moveout_time


def _section_spacing(data: SeismicData) -> float:
    geom = data.meta.get("geometry", {})
    if geom.get("bin_size"):
        return float(geom["bin_size"])
    x = data.header("cdpx")
    if np.ptp(x) > 0:
        return float(np.median(np.abs(np.diff(x))))
    raise ValueError("trace spacing unknown: run 2D geometry before stacking, or set cdpx")


def stolt(section: np.ndarray, dt: float, dx: float, v: float) -> np.ndarray:
    """Stolt f-k time migration of a zero-offset section (nx, nt) with constant velocity v (exploding reflector v/2)."""
    nx, nt = section.shape
    nxp = int(2 ** np.ceil(np.log2(2 * nx)))
    ntp = int(2 ** np.ceil(np.log2(2 * nt)))
    D = np.fft.fft(np.fft.rfft(section.astype(np.float64), ntp, axis=1), nxp, axis=0)     # (nk, nw)
    kx = 2 * np.pi * np.fft.fftfreq(nxp, dx)
    w = 2 * np.pi * np.fft.rfftfreq(ntp, dt)
    u = 0.5 * v
    M = np.zeros_like(D)
    dw = w[1] - w[0]
    for ik in range(nxp):
        w_in = np.sqrt(w ** 2 + (u * kx[ik]) ** 2)
        pos = w_in / dw
        i0 = np.floor(pos).astype(np.int64)
        frac = pos - i0
        ok = i0 + 1 < w.size
        val = np.zeros(w.size, dtype=np.complex128)
        val[ok] = (1 - frac[ok]) * D[ik, i0[ok]] + frac[ok] * D[ik, i0[ok] + 1]
        jac = np.divide(w, w_in, out=np.ones_like(w), where=w_in > 0)
        M[ik] = val * jac
    return np.fft.irfft(np.fft.ifft(M, axis=0), ntp, axis=1)[:nx, :nt].astype(np.float32)


def phase_shift(section: np.ndarray, dt: float, dx: float, vint: np.ndarray, fmax: float | None = None) -> np.ndarray:
    """Gazdag phase-shift time migration with interval velocity vint(tau) (laterally invariant)."""
    nx, nt = section.shape
    nxp = int(2 ** np.ceil(np.log2(2 * nx)))
    ntp = int(2 ** np.ceil(np.log2(2 * nt)))
    D = np.fft.fft(np.fft.rfft(section.astype(np.float64), ntp, axis=1), nxp, axis=0)
    kx = 2 * np.pi * np.fft.fftfreq(nxp, dx)[:, None]
    w = 2 * np.pi * np.fft.rfftfreq(ntp, dt)
    nw = w.size if fmax is None else int(np.searchsorted(w, 2 * np.pi * fmax, side="right"))
    D = D[:, :nw]
    w = w[None, :nw]
    weights = np.ones(nw)
    weights[0] = 0.5
    image_k = np.zeros((nxp, nt), dtype=np.complex128)
    for it in range(nt):
        image_k[:, it] = D @ weights
        u = 0.5 * vint[it]
        with np.errstate(divide="ignore", invalid="ignore"):
            arg = 1.0 - (u * kx / np.where(w == 0, 1e-30, w)) ** 2
        prop = arg > 0
        D = np.where(prop, D * np.exp(1j * w * dt * np.sqrt(np.where(prop, arg, 0.0))), 0.0)
    return (2.0 * np.real(np.fft.ifft(image_k, axis=0))[:nx] / ntp).astype(np.float32)


def rho_filter(traces: np.ndarray, dt: float, modelling: bool = False) -> np.ndarray:
    """2D Kirchhoff half-derivative filter: sqrt(w) with a 45 degree phase (NumPy's transform pair, d/dt <-> +i w).

    Migration sums every output sample along a curve over data at later times, an anti-causal half-integration that
    the anti-causal half derivative sqrt(-i w) = sqrt(w) exp(-i pi/4) undoes. Modelling (demigration, and the 2D to
    3D correction of line-source synthetics) spreads energy to later times, a causal half-integration undone by
    sqrt(i w) = sqrt(w) exp(+i pi/4): modelling=True."""
    n = traces.shape[1]
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    w = 2 * np.pi * np.fft.rfftfreq(nfft, dt)
    h = np.sqrt(w) * np.exp((1j if modelling else -1j) * np.pi / 4)
    return np.fft.irfft(np.fft.rfft(traces, nfft, axis=1) * h, nfft, axis=1)[:, :n]


def kirchhoff_time(traces: np.ndarray, t0: float, dt: float, sx: np.ndarray, gx: np.ndarray, x_out: np.ndarray,
                   tau: np.ndarray, vrms_out: np.ndarray, aperture: float, max_angle_deg: float = 60.0,
                   offset_bins: np.ndarray | None = None, progress=None, eta_out: np.ndarray | None = None):
    """Prestack Kirchhoff time migration (double-square-root traveltimes, straight rays in v_rms).

    vrms_out: (ntau, nx_out) RMS velocity at every image point. Returns the image (ntau, nx_out) and, if offset_bins
    (edges) are given, common-image gathers (nbins, ntau, nx_out). With eta_out (ntau, nx_out), the anellipticity of
    VTI media, each leg uses the non-hyperbolic time of Alkhalifah & Tsvankin (1995) with t0 = tau / 2:
    t_leg^2 = tau^2/4 + d^2/v^2 - 2 eta d^4 / (v^2 [tau^2 v^2 / 4 + (1 + 2 eta) d^2])."""
    ntr, nt = traces.shape
    filt = rho_filter(traces.astype(np.float64), dt)
    image = np.zeros((tau.size, x_out.size))
    cig = None if offset_bins is None else np.zeros((offset_bins.size - 1, tau.size, x_out.size))
    half_tau = 0.5 * tau[:, None]
    half_tau2 = half_tau ** 2
    tan_max = np.tan(np.deg2rad(max_angle_deg))
    for i in range(ntr):
        mid = 0.5 * (sx[i] + gx[i])
        cols = np.flatnonzero(np.abs(x_out - mid) <= aperture)
        if cols.size == 0:
            continue
        if eta_out is None:
            v2 = vrms_out[:, cols] ** 2
            ts = np.sqrt(half_tau2 + (x_out[cols][None, :] - sx[i]) ** 2 / v2)
            tg = np.sqrt(half_tau2 + (x_out[cols][None, :] - gx[i]) ** 2 / v2)
        else:
            v, e = vrms_out[:, cols], eta_out[:, cols]
            ts = moveout_time(half_tau, x_out[cols][None, :] - sx[i], v, e)
            tg = moveout_time(half_tau, x_out[cols][None, :] - gx[i], v, e)
        t = ts + tg
        pos = (t - t0) / dt
        ok = (pos >= 0) & (pos < nt - 1)
        dist = np.abs(x_out[cols] - mid)[None, :]
        ok &= dist <= np.maximum(0.5 * tau[:, None] * vrms_out[:, cols] * tan_max, 1e-6)
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, nt - 2)
        frac = pos - i0
        val = (1 - frac) * filt[i, i0] + frac * filt[i, i0 + 1]
        wgt = np.where(t > 0, tau[:, None] / np.maximum(t, 1e-9), 0.0)          # obliquity
        contrib = np.where(ok, val * wgt, 0.0)
        image[:, cols] += contrib
        if cig is not None:
            b = int(np.searchsorted(offset_bins, abs(gx[i] - sx[i]), side="right") - 1)
            if 0 <= b < cig.shape[0]:
                cig[b][:, cols] += contrib
        if progress is not None and i % 200 == 0:
            progress(i / ntr)
    return image, cig


def eta_for(picks, cdps: np.ndarray, times: np.ndarray, use_eta: bool, eta: float) -> np.ndarray | None:
    """(ntimes, ncdp) eta for the Kirchhoff steps: the picked eta (falling back to the constant when nothing was
    picked) or the constant; None when it is zero everywhere (isotropic, the faster code path)."""
    field = None
    if use_eta:
        field = picks.eta_field(cdps, times).T
        if not np.any(field):
            field = None
    if field is None and eta != 0:
        field = np.full((np.asarray(times).size, np.asarray(cdps).size), float(eta))
    return field


@register
class StoltMigration(Process):
    name = "stolt"
    label = "Stolt migration (post-stack, constant v)"
    category = "Time migration"
    description = "f-k migration of a stacked section with one constant velocity; fast, for checks and simple geology."
    params = (Param("velocity", "float", 2000.0, "Velocity", unit="m/s", minimum=10.0),)

    def run(self, data, context):
        data = pad_to_time_zero(data)
        out = data.copy(stolt(data.traces, data.dt, _section_spacing(data), self["velocity"]))
        out.meta["migrated"] = "stolt"
        return out


@register
class PhaseShiftMigration(Process):
    name = "phase_shift"
    label = "Phase-shift migration (post-stack, v(t))"
    category = "Time migration"
    description = ("Gazdag phase-shift migration of a stacked section; RMS velocities at one CMP (the middle picked CMP "
                   "by default) are Dix-converted to interval velocity v(tau).")
    params = (Param("velocity", "velocity", "project", "Velocity", "project picks, a constant, 't:v,...' or a file"),
              Param("cdp", "float", -1.0, "CMP for the velocity function (-1 = middle)"),
              Param("fmax", "float", 90.0, "Highest frequency", unit="Hz", minimum=1.0))

    def run(self, data, context):
        data = pad_to_time_zero(data)
        picks = resolve_velocity(self["velocity"], context)
        c = self["cdp"] if self["cdp"] >= 0 else float(np.median(picks.cdps))
        vrms = picks.function(c, data.times)
        vint = dix(data.times, vrms)
        out = data.copy(phase_shift(data.traces, data.dt, _section_spacing(data), vint, self["fmax"]))
        out.meta["migrated"] = "phase_shift"
        return out


@register
class KirchhoffPSTM(Process):
    name = "kirchhoff_pstm"
    label = "Kirchhoff time migration (pre- or post-stack)"
    category = "Time migration"
    description = ("Kirchhoff summation with double-square-root traveltimes in v_rms(t, CMP). Prestack input gives a "
                   "migrated stack and optionally common-image gathers (offset classes) for residual moveout checks. "
                   "With eta (VTI media: picked in the eta scan of the velocity analysis, or a constant) each leg "
                   "uses the non-hyperbolic traveltime of Alkhalifah & Tsvankin (1995).")
    params = (Param("velocity", "velocity", "project", "Velocity", "project picks, a constant, 't:v,...' or a file"),
              Param("aperture", "float", 2000.0, "Aperture (half width)", unit="m", minimum=10.0),
              Param("max_angle", "float", 60.0, "Maximum angle", unit="deg", minimum=1.0, maximum=89.0),
              Param("output", "choice", "image", "Output", choices=("image", "image gathers")),
              Param("offset_class", "float", 200.0, "Offset class width (gathers)", unit="m", minimum=1.0),
              Param("use_eta", "bool", False, "Use picked eta (VTI non-hyperbolic)"),
              Param("eta", "float", 0.0, "Constant eta (if no picked eta)", minimum=-0.2, maximum=0.6))

    def run(self, data, context):
        picks = resolve_velocity(self["velocity"], context)
        cdp = data.header("cdp")
        ucdp, first = np.unique(cdp, return_index=True)
        cx = data.header("cdpx")
        if np.ptp(cx) > 0:
            x_out = cx[first]
        else:
            x_out = 0.5 * (data.header("sx") + data.header("gx"))[first]
        tau = data.times
        vrms = picks.field(ucdp, tau).T                                       # (ntau, nx_out)
        sx, gx = data.header("sx"), data.header("gx")
        if not (np.any(sx) or np.any(gx)):
            sx = gx = x_out[np.searchsorted(ucdp, cdp)]
        edges = None
        if self["output"] == "image gathers":
            omax = float(np.max(np.abs(gx - sx))) + self["offset_class"]
            edges = np.arange(0.0, omax + self["offset_class"], self["offset_class"])
        eta = eta_for(picks, ucdp, tau, self["use_eta"], self["eta"])
        image, cig = kirchhoff_time(data.traces, data.t0, data.dt, sx, gx, x_out, tau, vrms, self["aperture"],
                                    self["max_angle"], edges, progress=context.report, eta_out=eta)
        heads = {"cdp": ucdp, "cdpx": x_out, "sx": x_out, "gx": x_out, "offset": np.zeros(ucdp.size)}
        if cig is None:
            out = SeismicData(image.T.astype(np.float32), data.dt, heads, data.t0, data.name + " PSTM", "time",
                              list(data.history), dict(data.meta))
        else:
            nb = cig.shape[0]
            traces = np.transpose(cig, (2, 0, 1)).reshape(-1, tau.size)          # (nx_out * nb, ntau), CMP-major
            centres = 0.5 * (edges[:-1] + edges[1:])
            heads = {"cdp": np.repeat(ucdp, nb), "cdpx": np.repeat(x_out, nb), "offset": np.tile(centres, ucdp.size)}
            out = SeismicData(traces.astype(np.float32), data.dt, heads, data.t0, data.name + " PSTM gathers", "time",
                              list(data.history), dict(data.meta))
        out.meta["migrated"] = "kirchhoff_pstm"
        return out
