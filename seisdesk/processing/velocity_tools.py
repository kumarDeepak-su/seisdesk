"""Velocity conversion, tau-p correction and asymptotic log-stretch offset continuation."""
import numpy as np
from scipy.fft import next_fast_len

from ..core.velocity import VelocityPicks, vint_to_vrms
from .base import Param, Process, register, resolve_velocity
from .velan import semblance


@register
class IntervalToStacking(Process):
    name, label, category = "interval_to_stacking", "Interval depth model to RMS velocity picks", "Velocity"
    description = "Integrate the project interval model as constant-velocity depth cells to obtain vertical two-way time and RMS velocity. Writes picks at the nearest survey CMPs."
    params = (Param("step", "float", .1, "Time between exported picks", unit="s", minimum=.001),)
    def run(self, data, context):
        model = context.depth_model
        if model is None or model.z0 != 0 or np.any(model.values <= 0):
            raise ValueError("load a positive interval velocity model starting at the seismic datum z=0")
        cdp, x = data.header("cdp", None), data.header("cdpx", None)
        picks = VelocityPicks(); tables = []
        for c in np.unique(cdp):
            position = float(np.mean(x[cdp == c])); col = int(np.argmin(np.abs(model.x-position)))
            v = model.values[:, col]; time = np.r_[0, np.cumsum(2*model.dz/v[1:])]
            rms = vint_to_vrms(time, v)
            selected = np.unique(np.r_[np.arange(0, time[-1], self["step"]), time[-1]])
            for t, vr in zip(selected, np.interp(selected, time, rms)):
                picks.add(c, t, vr)
            tables.append(dict(cdp=float(c), time=time, vrms=rms))
        context.velocity = picks; context.results[self.name] = tables
        return data.copy()


def ti_nmo_velocity(vp0, epsilon, delta, dip_degrees):
    """Homogeneous VTI dip-NMO from weak-anisotropy phase velocity and its angular curvature."""
    phi = np.deg2rad(dip_degrees)
    v = vp0*(1+delta*np.sin(phi)**2*np.cos(phi)**2+epsilon*np.sin(phi)**4)
    dv = vp0*(delta*.5*np.sin(4*phi)+epsilon*(np.sin(2*phi)-.5*np.sin(4*phi)))
    ddv = vp0*(2*delta*np.cos(4*phi)+2*epsilon*(np.cos(2*phi)-np.cos(4*phi)))
    denominator = 1-np.tan(phi)*dv/v
    radicand = 1+ddv/v
    if np.any(radicand <= 0) or np.any(denominator <= 0):
        raise ValueError("non-convex weak-anisotropy slowness curve; reduce dip or anisotropy")
    return v/np.cos(phi)*np.sqrt(radicand)/denominator


@register
class TIVelocityTable(Process):
    name, label, category = "ti_velocity_table", "TI dip-NMO velocity table (weak anisotropy)", "Velocity"
    description = "Tabulate homogeneous VTI NMO velocity versus reflector dip using weak-anisotropy phase-velocity curvature; inspect the table in the log and results."
    params = (Param("vp0", "float", 2500, "Vertical P velocity", unit="m/s", minimum=100),
              Param("epsilon", "float", .1, "Thomsen epsilon", minimum=-.1, maximum=.2),
              Param("delta", "float", .05, "Thomsen delta", minimum=-.1, maximum=.2),
              Param("dips", "floats", "0,10,20,30,40,50,60", "Reflector dips", unit="deg"))
    def run(self, data, context):
        dips = np.asarray(self["dips"])
        if dips.size == 0 or np.any(np.abs(dips) > 70):
            raise ValueError("give dips between -70 and 70 degrees")
        velocity = ti_nmo_velocity(self["vp0"], self["epsilon"], self["delta"], dips)
        context.results[self.name] = dict(dip=dips, vnmo=velocity)
        for d, v in zip(dips, velocity):
            context.log(f"    dip {d:g} degrees: NMO velocity {v:.2f} m/s")
        return data.copy()


@register
class VelocitySemblanceSlice(Process):
    name, label, category = "velocity_semblance_slice", "Velocity semblance at one time along the line", "Velocity"
    description = "Compute a CMP-by-velocity semblance slice at one two-way time. Stores the full slice and logs the maximum at each CMP; optionally writes those velocity picks."
    params = (Param("time", "float", 1, "Analysis time", unit="s", minimum=0),
              Param("vmin", "float", 1500, "Minimum velocity", minimum=10),
              Param("vmax", "float", 4000, "Maximum velocity", minimum=10),
              Param("nv", "int", 51, "Velocities", minimum=3, maximum=501),
              Param("window", "float", .04, "Time window", unit="s", minimum=.001),
              Param("write_picks", "bool", False, "Write maximum-semblance picks"))
    def run(self, data, context):
        if data.domain != "time" or not data.t0 <= self["time"] <= data.tmax or self["vmax"] <= self["vmin"]:
            raise ValueError("choose a time inside the data and an increasing velocity range")
        velocities = np.linspace(self["vmin"], self["vmax"], self["nv"])
        sample = int(round((self["time"]-data.t0)/data.dt)); rows, cdps = [], []
        groups = list(data.iter_gathers("cdp"))
        picks = context.velocity if context.velocity is not None else VelocityPicks()
        for i, (cdp, idx) in enumerate(groups):
            context.report(i/len(groups))
            if len(idx) < 3:
                continue
            spectrum = semblance(data.traces[idx], data.header("offset")[idx], data.times, data.dt, velocities, self["window"])
            row = spectrum[:, sample]; rows.append(row); cdps.append(cdp)
            if row.max() > .1:
                v = velocities[np.argmax(row)]
                context.log(f"    CMP {cdp:g}: {v:g} m/s, semblance {row.max():.3f}")
                if self["write_picks"]:
                    picks.add(cdp, self["time"], v)
        if not rows:
            raise ValueError("at least one CMP needs three traces for semblance")
        context.results[self.name] = dict(cdp=np.asarray(cdps), velocity=velocities, semblance=np.asarray(rows), time=self["time"])
        if self["write_picks"]:
            context.velocity = picks
        return data.copy()


@register
class TauPNMO(Process):
    name, label, category = "taup_nmo", "NMO in tau-p (layered isotropic medium)", "Moveout"
    description = "Correct tau(p,t0)=integral sqrt(1-p² v_interval²) dt0. Input traces must already be in tau-p with p in seconds/metre; critical and post-critical samples are muted."
    params = (Param("velocity", "velocity", "2000", "Interval velocities v(t0)"),
              Param("p_key", "key", "p", "Horizontal slowness header (s/m)"),)
    def run(self, data, context):
        if data.t0 < 0 or data.domain != "time":
            raise ValueError("tau-p correction requires a nonnegative time axis")
        p = data.header(self["p_key"], None)
        velocity = resolve_velocity(self["velocity"], context)
        out = np.zeros_like(data.traces)
        increments = np.diff(data.times, prepend=0)
        for i, trace in enumerate(data.traces):
            v = velocity.function(data.header("cdp")[i], data.times)
            radicand = 1-(p[i]*v)**2
            valid = np.logical_and.accumulate(radicand > 0)
            tau = np.cumsum(increments*np.sqrt(np.maximum(radicand, 0)))
            out[i] = np.where(valid, np.interp(tau, data.times, trace, left=0, right=0), 0)
        return data.copy(out)


def log_offset_continuation(traces, times, dx, half_in, half_out, max_gain=10):
    """Fomel high-frequency log-stretch operator; valid |2kh/Omega|<=1.

    Reference: https://sepwww.stanford.edu/data/media/public/docs/sep107/paper_html/node84.html
    Uses the conjugate phase for NumPy's exp(-i Omega sigma) convention. The
    unreliable low-frequency/large-dip region is tapered out; mean trace is kept.
    """
    if half_in == half_out:
        return np.array(traces, copy=True)
    if half_out < 0 or half_out > half_in:
        raise ValueError("offset continuation currently supports a smaller nonnegative output offset")
    positive = times > 0
    if positive.sum() < 8:
        raise ValueError("offset continuation needs at least eight positive time samples")
    mean = np.mean(traces, axis=0, keepdims=True)
    residual = traces-mean
    nlog = next_fast_len(2*len(times))
    sigma = np.linspace(np.log(times[positive][0]), np.log(times[-1]), nlog)
    ts = np.exp(sigma)
    stretched = np.stack([np.interp(ts, times, row) for row in residual])
    nfft = next_fast_len(2*nlog)
    spectrum = np.fft.fft(np.fft.rfft(stretched, nfft, axis=1), axis=0)
    omega = 2*np.pi*np.fft.rfftfreq(nfft, sigma[1]-sigma[0])[None]
    k = 2*np.pi*np.fft.fftfreq(len(traces), dx)[:, None]
    safe = np.where(omega > 0, omega, 1)
    e1, e2 = 2*k*half_in/safe, 2*k*half_out/safe
    def terms(e):
        root = np.sqrt(1+e*e)
        log_amp = .5*np.log((1+root)/(2*root)) + .5*(1-root)
        phase = .5*(1-root+np.log((1+root)/2))
        return log_amp, phase
    a1, p1 = terms(e1); a2, p2 = terms(e2)
    taper = np.cos(.5*np.pi*np.clip((np.abs(e1)-.8)/.2, 0, 1))**2
    multiplier = np.exp(np.clip(a2-a1, -np.log(max_gain), np.log(max_gain))-1j*omega*(p2-p1))*taper
    multiplier = np.where(omega >= 2, multiplier, 0); multiplier[0] = 1
    continued = np.fft.irfft(np.fft.ifft(spectrum*multiplier, axis=0), nfft, axis=1)[:, :nlog]
    result = np.stack([np.interp(times, ts, row, left=0, right=0) for row in continued])
    return result+mean


@register
class OffsetContinuation(Process):
    name, label, category = "offset_continuation", "Offset continuation (log-stretch f-k approximation)", "Interpolation"
    description = "Continue an NMO-corrected common-offset 2D section to a smaller offset, using Fomel's high-frequency log-stretch operator. Periodic spatial edges; taper/crop first. Output remains NMO corrected."
    params = (Param("offset", "float", 0, "Output full offset", unit="m", minimum=0),
              Param("max_gain", "float", 10, "Maximum amplitude correction", minimum=1, maximum=100))
    def run(self, data, context):
        offset = data.header("offset", None)
        if np.ptp(np.abs(offset)) > .01 or data.domain != "time" or data.t0 < 0:
            raise ValueError("input must be one common-offset time section with nonnegative times")
        # One panel, ordered by physical CMP position.
        ordered = data.sort("cdpx")
        position = ordered.header("cdpx", None); diff = np.diff(position)
        if len(diff) < 1 or np.any(diff <= 0) or not np.allclose(diff, np.median(diff), rtol=.02):
            raise ValueError("offset continuation requires regularly spaced, distinct CMPs")
        values = log_offset_continuation(ordered.traces, ordered.times, float(np.median(diff)), abs(offset[0])/2,
                                          self["offset"]/2, self["max_gain"])
        out = ordered.copy(values.astype(np.float32)); out.set_header("offset", self["offset"])
        sign = 1 if offset[0] >= 0 else -1
        out.set_header("sx", position-sign*self["offset"]/2); out.set_header("gx", position+sign*self["offset"]/2)
        out.meta["nmo_corrected"] = True
        return out
