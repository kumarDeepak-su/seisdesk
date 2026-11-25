"""Anisotropic (VTI) time processing: phase-shift migration with eta, migration to zero offset (NMO + DMO with eta)
and residual-moveout semblance scans.

Vertically transversely isotropic (VTI) layers are described in time processing by the NMO velocity and the
anellipticity eta (Alkhalifah & Tsvankin 1995). With eta = 0 every step reduces to its isotropic counterpart.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..core.dataset import SeismicData
from ..core.velocity import dix
from .base import Param, Process, cdp_values, pad_to_time_zero, register, resolve_velocity
from .migration_time import _section_spacing, eta_for, kirchhoff_time, rho_filter
from .moveout import gate_by_energy, moveout_time, sample_at
from .rmo import rmo_scan as residual_semblance


# ---------------------------------------------------------------------------------------------- VTI phase shift
def interval_eta(times: np.ndarray, vrms: np.ndarray, eta_eff: np.ndarray, vint: np.ndarray) -> np.ndarray:
    """Interval eta from the effective (picked) eta by inverting the Dix-type relation of Alkhalifah (1997):

    eta_eff(t) = [ integral_0^t v_int^4 (1 + 8 eta_int) dt' / (t v_rms^4) - 1 ] / 8.

    Unusable values (the integral decreasing) keep the previous interval value; the result is clipped to
    -0.2..0.5, the range met in practice."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(vrms, dtype=np.float64)
    e = np.asarray(eta_eff, dtype=np.float64)
    vi = np.asarray(vint, dtype=np.float64)
    a = t * v ** 4 * (1.0 + 8.0 * e)
    out = np.empty_like(e)
    out[0] = e[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.diff(a) / np.diff(t)
        val = (rate / np.maximum(vi[1:], 1e-9) ** 4 - 1.0) / 8.0
    for i in range(1, e.size):
        out[i] = val[i - 1] if np.isfinite(val[i - 1]) and rate[i - 1] > 0 else out[i - 1]
    return np.clip(out, -0.2, 0.5)


def phase_shift_vti(section: np.ndarray, dt: float, dx: float, vint: np.ndarray, eta: np.ndarray,
                    fmax: float | None = None) -> np.ndarray:
    """Phase-shift time migration of a zero-offset section (nx, nt) in a layered VTI medium.

    Exploding reflector with half velocities u = v_nmo / 2: per vertical-time step dtau the wavefield is shifted by
    exp(i w dtau sqrt(q)) with Alkhalifah's (1998) acoustic VTI dispersion relation

        q = (1 - (1 + 2 eta) s^2) / (1 - 2 eta s^2),   s = u kx / w.

    Only the propagating branch (numerator > 0, which makes the denominator positive too) is kept; this also drops
    the spurious second branch of the acoustic approximation at large kx. eta = 0 is Gazdag's phase shift exactly.
    vint and eta are interval values per time sample."""
    nx, nt = section.shape
    vint = np.broadcast_to(np.asarray(vint, dtype=np.float64), (nt,))
    eta = np.broadcast_to(np.asarray(eta, dtype=np.float64), (nt,))
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
        e = float(eta[it])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s2 = (u * kx / np.where(w == 0, 1e-30, w)) ** 2
        num = 1.0 - (1.0 + 2.0 * e) * s2
        den = 1.0 - 2.0 * e * s2
        prop = num > 0
        q = np.where(prop, num / np.where(prop, den, 1.0), 0.0)
        D = np.where(prop, D * np.exp(1j * w * dt * np.sqrt(q)), 0.0)
    return (2.0 * np.real(np.fft.ifft(image_k, axis=0))[:nx] / ntp).astype(np.float32)


@register
class PhaseShiftVTI(Process):
    name = "phase_shift_vti"
    label = "Phase-shift migration for VTI media (eta)"
    category = "Time migration"
    description = ("Gazdag phase-shift migration of a stacked section for vertically transversely isotropic layers "
                   "(shales): RMS velocities at one CMP are Dix-converted to interval NMO velocity, and eta - picked "
                   "in the eta scan of the velocity analysis (effective values, converted to interval eta) or a "
                   "constant interval eta - enters Alkhalifah's acoustic VTI dispersion relation. Steep dips and "
                   "diffraction flanks then migrate to the right place where isotropic migration over-migrates them. "
                   "eta = 0 gives the isotropic phase-shift migration.")
    params = (Param("velocity", "velocity", "project", "Velocity", "project picks, a constant, 't:v,...' or a file"),
              Param("cdp", "float", -1.0, "CMP for the velocity function (-1 = middle)"),
              Param("eta_source", "choice", "picked", "Eta", "picked: effective eta from the velocity picks; "
                    "constant: the interval eta below", choices=("picked", "constant")),
              Param("eta", "float", 0.0, "Constant eta (interval)", minimum=-0.2, maximum=0.5),
              Param("fmax", "float", 90.0, "Highest frequency", unit="Hz", minimum=1.0))

    def run(self, data, context):
        data = pad_to_time_zero(data)
        picks = resolve_velocity(self["velocity"], context)
        c = self["cdp"] if self["cdp"] >= 0 else float(np.median(picks.cdps))
        times = data.times
        vrms = picks.function(c, times)
        vint = dix(times, vrms)
        if self["eta_source"] == "picked":
            eta = interval_eta(times, vrms, picks.eta_field([c], times)[0], vint)
        else:
            eta = np.full(times.size, self["eta"])
        context.log(f"    interval eta {eta.min():.3f} to {eta.max():.3f}")
        out = data.copy(phase_shift_vti(data.traces, data.dt, _section_spacing(data), vint, eta, self["fmax"]))
        out.meta["migrated"] = "phase_shift_vti"
        return out


# ---------------------------------------------------------------------------------------------- migration to zero offset
def demigrate_zero_offset(image: np.ndarray, tau: np.ndarray, x: np.ndarray, vrms: np.ndarray,
                          eta: np.ndarray | None, aperture: float, t0: float, dt: float, nt: int,
                          max_angle_deg: float = 60.0) -> np.ndarray:
    """Kirchhoff zero-offset demigration (modelling) of a time image (ntau, nx) into a zero-offset section (nx, nt).

    Every image sample at (tau, x) is spread along its zero-offset diffraction curve t = 2 t_leg(tau / 2, d), d the
    lateral distance and t_leg the leg time of moveout_time (VTI when eta is given), weighted by the obliquity
    tau / t - the adjoint of kirchhoff_time. The 2D half-derivative afterwards restores the wavelet that the
    summation integrates. vrms and eta (ntau, nx) are taken at the image point."""
    ntau, nx = image.shape
    out = np.zeros(nt * nx)
    steps = np.abs(np.diff(np.asarray(x, dtype=np.float64)))
    steps = steps[steps > 0]
    dmin = float(steps.min()) if steps.size else 1.0
    kmax = int(min(max(nx - 1, 0), np.ceil(aperture / dmin)))
    half_tau = 0.5 * tau[:, None]
    tan_max = np.tan(np.deg2rad(max_angle_deg))
    for k in range(-kmax, kmax + 1):
        j = np.arange(max(0, -k), min(nx, nx - k))
        if j.size == 0:
            continue
        jo = j + k
        d = np.abs(x[jo] - x[j])
        keep = d <= aperture
        if not np.any(keep):
            continue
        j, jo, d = j[keep], jo[keep], d[keep][None, :]
        v = vrms[:, j]
        t = 2.0 * moveout_time(half_tau, d, v, 0.0 if eta is None else eta[:, j])
        pos = (t - t0) / dt
        ok = (pos >= 0) & (pos < nt - 1) & (d <= np.maximum(half_tau * v * tan_max, 1e-6))
        if not np.any(ok):
            continue
        i0 = np.floor(np.where(ok, pos, 0.0)).astype(np.int64)
        fr = np.where(ok, pos - i0, 0.0)
        val = image[:, j] * np.where(t > 0, tau[:, None] / np.maximum(t, 1e-9), 0.0)
        flat = (i0 * nx + jo[None, :])[ok]
        out += np.bincount(flat, weights=((1.0 - fr) * val)[ok], minlength=nt * nx)
        out += np.bincount(flat + nx, weights=(fr * val)[ok], minlength=nt * nx)
    return rho_filter(out.reshape(nt, nx).T, dt, modelling=True)


def _cmp_positions(data: SeismicData, first: np.ndarray) -> np.ndarray:
    cx = data.header("cdpx")
    if np.ptp(cx) > 0:
        return cx[first]
    return (0.5 * (data.header("sx") + data.header("gx")))[first]


@register
class MZODMO(Process):
    name = "mzo_dmo"
    label = "Anisotropic DMO (migration to zero offset, eta)"
    category = "Moveout"
    description = ("NMO and DMO in one step, also for VTI media. Input: prestack traces WITHOUT NMO, with source and "
                   "receiver x. The traces are sorted into offset classes; each class is Kirchhoff time-migrated "
                   "with v_rms and eta and then demigrated (modelled back) to zero offset with the same v_rms and eta "
                   "(migration to zero offset). Output: one NMO+DMO-corrected trace per CMP and offset class, ready "
                   "to stack, or the stack itself. Dipping events and diffractions then stack at their zero-offset "
                   "times whatever their dip, and eta also flattens the non-hyperbolic far offsets of anisotropic "
                   "layers. The cost is one prestack time migration plus one zero-offset modelling per class, which "
                   "suits 2D lines of a few thousand traces. Do not apply NMO afterwards: the output is corrected.")
    params = (Param("velocity", "velocity", "project", "Velocity", "project picks, a constant, 't:v,...' or a file"),
              Param("use_eta", "bool", True, "Use picked eta (VTI)"),
              Param("eta", "float", 0.0, "Constant eta (if no picked eta)", minimum=-0.2, maximum=0.5),
              Param("offset_class", "float", 200.0, "Offset class width", unit="m", minimum=1.0),
              Param("aperture", "float", 1000.0, "Aperture (half width)", unit="m", minimum=10.0),
              Param("max_angle", "float", 60.0, "Maximum angle", unit="deg", minimum=1.0, maximum=89.0),
              Param("output", "choice", "gathers", "Output", "gathers: one trace per CMP and offset class; stack: "
                    "their average", choices=("gathers", "stack")))

    def run(self, data, context):
        picks = resolve_velocity(self["velocity"], context)
        cdp = cdp_values(data)
        ucdp, first = np.unique(cdp, return_index=True)
        x_out = _cmp_positions(data, first)
        sx, gx = data.header("sx"), data.header("gx")
        if not (np.any(sx) or np.any(gx)):
            raise ValueError("migration to zero offset needs source and receiver x (sx, gx): run a geometry step")
        tau = data.times
        vrms = picks.field(ucdp, tau).T                                        # (ntau, ncmp)
        eta = eta_for(picks, ucdp, tau, self["use_eta"], self["eta"])
        width = self["offset_class"]
        cls = np.floor(np.abs(gx - sx) / width).astype(np.int64)
        classes = np.unique(cls)
        sections = np.zeros((classes.size, ucdp.size, data.nt))
        for n, c in enumerate(classes):
            idx = np.flatnonzero(cls == c)
            image, _ = kirchhoff_time(data.traces[idx], data.t0, data.dt, sx[idx], gx[idx], x_out, tau, vrms,
                                      self["aperture"], self["max_angle"], eta_out=eta)
            zo = demigrate_zero_offset(image, tau, x_out, vrms, eta, self["aperture"], data.t0, data.dt, data.nt,
                                       self["max_angle"])
            # Kirchhoff sums carry an arbitrary sampling factor: match the RMS of the class's input at its CMPs
            here = np.isin(ucdp, np.unique(cdp[idx]))
            r_in = float(np.sqrt(np.mean(data.traces[idx].astype(np.float64) ** 2)))
            r_out = float(np.sqrt(np.mean(zo[here] ** 2))) if np.any(here) else 0.0
            sections[n] = zo * (r_in / r_out) if r_out > 0 else zo
            context.log(f"    offset class {c * width:g}-{(c + 1) * width:g} m: {idx.size} traces")
            context.report((n + 1) / classes.size)
        if self["output"] == "stack":
            live = np.sum(np.abs(sections) > 0, axis=0)
            traces = np.sum(sections, axis=0) / np.maximum(live, 1)
            heads = {"cdp": ucdp, "cdpx": x_out, "sx": x_out, "gx": x_out, "offset": np.zeros(ucdp.size)}
            name = data.name + " MZO stack"
        else:
            nb = classes.size
            traces = np.transpose(sections, (1, 0, 2)).reshape(-1, data.nt)          # CMP-major, offset classes
            off = np.tile((classes + 0.5) * width, ucdp.size)
            cx = np.repeat(x_out, nb)
            heads = {"cdp": np.repeat(ucdp, nb), "cdpx": cx, "offset": off, "sx": cx - 0.5 * off,
                     "gx": cx + 0.5 * off}
            name = data.name + " MZO"
        out = SeismicData(traces.astype(np.float32), data.dt, heads, data.t0, name, "time", list(data.history),
                          dict(data.meta))
        out.meta["moveout_corrected"] = "mzo_dmo"
        return out


# ---------------------------------------------------------------------------------------------- residual moveout scan
def eta_residual_scan(gather: np.ndarray, offsets: np.ndarray, times: np.ndarray, dt: float, vnmo: np.ndarray,
                      etas: np.ndarray, window: float = 0.03) -> np.ndarray:
    """Semblance of a gather NMO-corrected with hyperbolic moveout (velocity vnmo(t0)) for trial values of eta.

    With the right eta, the event of time t0 lies where hyperbolic NMO put it, at
    t = sqrt(t_vti(t0, x)^2 - x^2 / v^2) (Alkhalifah 1997); sampling the gather along that curve flattens it."""
    x = np.abs(np.asarray(offsets, dtype=np.float64))[:, None]
    t0 = np.asarray(times, dtype=np.float64)[None, :]
    v = np.asarray(vnmo, dtype=np.float64)[None, :]
    nwin = max(1, int(round(window / dt)))
    g = gather.astype(np.float64)
    out = np.zeros((len(etas), t0.size))
    energy = np.zeros_like(out)
    for i, e in enumerate(etas):
        tv = moveout_time(t0, x, v, float(e))
        tt = np.sqrt(np.maximum(tv ** 2 - x ** 2 / v ** 2, 0.0))
        vals, valid = sample_at(g, tt, times[0], dt)
        num = uniform_filter1d(vals.sum(axis=0) ** 2, nwin)
        den = uniform_filter1d(np.maximum(valid.sum(axis=0), 1) * (vals ** 2).sum(axis=0), nwin)
        out[i] = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
        energy[i] = uniform_filter1d((vals ** 2).sum(axis=0), nwin)
    return gate_by_energy(out, energy)


@register
class RMOScan(Process):
    name = "rmo_scan"
    label = "Residual moveout semblance scan (anisotropic)"
    category = "Velocity"
    description = ("Checks flattened gathers - NMO-corrected CMP gathers or image gathers from prestack migration - for "
                   "remaining moveout. At each selected CMP the semblance across the offsets is computed for a range "
                   "of residual moveouts: a curvature r (time shift at the far offset, parabolic or fourth order; in "
                   "metres for depth gathers), or eta (the VTI non-hyperbolic term left after hyperbolic NMO with the "
                   "given velocity). The panels come out as a dataset, one panel per CMP with one trace per trial "
                   "value (header rmo_r) and two blank traces between panels: a bright band at r = 0 means the gather "
                   "is flat; elsewhere it shows the correction to apply (Residual moveout correction) or the eta to "
                   "pick. The best value at each time is kept in the results.")
    params = (Param("cdps", "floats", "", "CMPs (empty = three across the line)"),
              Param("mode", "choice", "parabolic", "Residual term", choices=("parabolic", "fourth order", "eta (VTI)")),
              Param("rmin", "float", -0.05, "Minimum r", unit="s"),
              Param("rmax", "float", 0.05, "Maximum r", unit="s"),
              Param("eta_min", "float", 0.0, "Minimum eta (eta mode)", minimum=-0.2, maximum=0.5),
              Param("eta_max", "float", 0.3, "Maximum eta (eta mode)", minimum=-0.2, maximum=0.5),
              Param("nr", "int", 41, "Number of trial values", minimum=3, maximum=201),
              Param("velocity", "velocity", "project", "NMO velocity (eta mode)",
                    "project picks, a constant, 't:v,...' or a file"),
              Param("window", "float", 0.03, "Semblance window", unit="s", minimum=0.001))

    def run(self, data, context):
        cdp = cdp_values(data)
        ucdp = np.unique(cdp)
        wanted = list(self["cdps"]) or [float(ucdp[int(round(q * (ucdp.size - 1)))]) for q in (0.25, 0.5, 0.75)]
        chosen = []
        for c in wanted:
            near = float(ucdp[int(np.argmin(np.abs(ucdp - c)))])
            if near not in chosen:
                chosen.append(near)
        eta_mode = self["mode"] == "eta (VTI)"
        if eta_mode:
            if data.domain != "time":
                raise ValueError("the eta scan needs time gathers")
            values = np.linspace(self["eta_min"], self["eta_max"], self["nr"])
            picks = resolve_velocity(self["velocity"], context)
        else:
            values = np.linspace(self["rmin"], self["rmax"], self["nr"])
            picks = None
        order = 4 if self["mode"] == "fourth order" else 2
        panels, best = [], {}
        heads = {"cdp": [], "rmo_r": [], "offset": []}
        for n, c in enumerate(chosen):
            idx = data.gather_index("cdp", c)
            if idx.size < 2:
                context.log(f"    CMP {c:g}: fewer than two traces, skipped")
                continue
            g, off = data.traces[idx], data.header("offset")[idx]
            if eta_mode:
                sem = eta_residual_scan(g, off, data.times, data.dt, picks.function(c, data.times), values,
                                        self["window"])
            else:
                sem = residual_semblance(g, off, data.times, data.dt, values, self["window"], order)
            panels.append(sem.astype(np.float32))
            panels.append(np.zeros((2, data.nt), np.float32))
            heads["cdp"] += [c] * (values.size + 2)
            heads["rmo_r"] += list(values) + [0.0, 0.0]
            heads["offset"] += [0.0] * (values.size + 2)
            peak = sem.max(axis=0)
            pick = values[np.argmax(sem, axis=0)]
            best[f"{c:g}"] = [float(p) if s > 0 else None for p, s in zip(pick, peak)]
            context.report((n + 1) / len(chosen))
        if not panels:
            raise ValueError("no CMP with at least two traces among the selected ones")
        context.results["rmo_scan"] = {"mode": self["mode"], "times": data.times.tolist(), "values": values.tolist(),
                                       "best": best}
        out = SeismicData(np.concatenate(panels), data.dt, heads, data.t0, f"{data.name} RMO scan", data.domain,
                          list(data.history))
        out.meta["rmo_scan"] = {"mode": self["mode"], "values": values.tolist()}
        return out
