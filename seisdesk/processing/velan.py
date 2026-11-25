"""Velocity analysis: semblance spectra, automatic picking, constant-velocity stacks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..core.dataset import SeismicData
from ..core.velocity import VelocityPicks
from .base import Param, Process, cdp_values, register
from .moveout import gate_by_energy, moveout_time, sample_at


def semblance(gather: np.ndarray, offsets: np.ndarray, times: np.ndarray, dt: float, velocities: np.ndarray,
              window: float = 0.02, eta: float = 0.0, stretch_limit: float = 0.5) -> np.ndarray:
    """Semblance S(v, t0) of one gather (ntr, nt) over the velocities; window in seconds."""
    ntr, nt = gather.shape
    t0 = times[None, :]
    x = np.asarray(offsets, dtype=np.float64)[:, None]
    nwin = max(1, int(round(window / dt)))
    out = np.zeros((velocities.size, nt))
    energy = np.zeros((velocities.size, nt))
    g = gather.astype(np.float64)
    for iv, v in enumerate(velocities):
        tt = moveout_time(t0, x, v, eta)
        vals, valid = sample_at(g, tt, times[0], dt)
        if stretch_limit > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                stretch = np.where(t0 > 0, tt / np.maximum(t0, 1e-9) - 1.0, np.inf)
            valid &= stretch <= stretch_limit
            vals = np.where(valid, vals, 0.0)
        n_live = valid.sum(axis=0)
        num = vals.sum(axis=0) ** 2
        den = n_live * (vals ** 2).sum(axis=0)
        num = uniform_filter1d(num, nwin, mode="nearest")
        den = uniform_filter1d(den, nwin, mode="nearest")
        out[iv] = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
        energy[iv] = uniform_filter1d((vals ** 2).sum(axis=0), nwin, mode="nearest")
    return gate_by_energy(out, energy)


def auto_pick(spectrum: np.ndarray, times: np.ndarray, velocities: np.ndarray, max_dv_per_s: float = 2000.0,
              pick_interval: float = 0.2, threshold: float = 0.1, tmin: float = 0.05, block: float = 0.02,
              max_decrease_per_s: float = 1000.0) -> np.ndarray:
    """A smooth v(t) path through a semblance spectrum, decimated to picks (t, v).

    Dynamic programming on blocks of ``block`` seconds (the maximum semblance of each block): the velocity may rise by
    up to ``max_dv_per_s`` but fall only by ``max_decrease_per_s``, so the path cannot drop onto the slow multiples
    that sit just below a primary and climb back."""
    nv, nt = spectrum.shape
    dt = times[1] - times[0]
    dv = velocities[1] - velocities[0] if nv > 1 else 1.0
    nb = max(1, int(round(block / dt)))
    nblk = int(np.ceil(nt / nb))
    s_full = spectrum / (spectrum.max() + 1e-12)
    s = np.pad(s_full, ((0, 0), (0, nblk * nb - nt))).reshape(nv, nblk, nb).max(axis=2)
    tb = nb * dt
    up = max(1, int(np.floor(max_dv_per_s * tb / dv)))
    down = max(0, int(np.floor(max_decrease_per_s * tb / dv)))
    score = np.full((nv, nblk), -np.inf)
    back = np.zeros((nv, nblk), dtype=np.int64)
    score[:, 0] = s[:, 0]
    for j in range(1, nblk):
        prev = score[:, j - 1]
        best = np.full(nv, -np.inf)
        arg = np.zeros(nv, dtype=np.int64)
        for k in range(-down, up + 1):              # k = index change from the previous block to this one
            shifted = np.full(nv, -np.inf)
            if k >= 0:
                shifted[k:] = prev[: nv - k]
            else:
                shifted[:k] = prev[-k:]
            better = shifted > best
            best[better] = shifted[better]
            arg[better] = (np.arange(nv) - k)[better]
        score[:, j] = best + s[:, j]
        back[:, j] = arg
    path_b = np.zeros(nblk, dtype=np.int64)
    path_b[-1] = int(np.argmax(score[:, -1]))
    for j in range(nblk - 1, 0, -1):
        path_b[j - 1] = back[path_b[j], j]
    path = np.repeat(path_b, nb)[:nt]
    picks = []
    for tp in np.arange(max(tmin, times[0]), times[-1], pick_interval):
        j = int(round((tp - times[0]) / dt))
        if s_full[path[j], j] >= threshold:
            picks.append((times[j], velocities[path[j]]))
    if not picks:
        j = int(np.argmax(s_full.max(axis=0)))
        picks.append((times[j], velocities[path[j]]))
    return np.array(picks)


def supergather(data: SeismicData, cdp: float, half_width: int = 0) -> np.ndarray:
    """Trace indices of CMP ``cdp`` and its ``half_width`` neighbours on each side, ordered by |offset|."""
    c = data.header("cdp")
    idx = np.flatnonzero(np.abs(c - cdp) <= half_width + 1e-9)
    return idx[np.argsort(np.abs(data.header("offset")[idx]), kind="stable")]


@register
class AutoVelocity(Process):
    name = "auto_velocity"
    label = "Automatic velocity picking"
    category = "Velocity"
    description = ("Semblance at every n-th CMP (supergathers) and a smooth maximum-semblance path picked into the "
                   "project velocity picks (replacing picks at those CMPs). Refine them in the velocity analysis tool.")
    params = (Param("cdp_step", "int", 50, "Analyse every n-th CMP", minimum=1),
              Param("supergather", "int", 2, "Neighbour CMPs on each side", minimum=0),
              Param("vmin", "float", 1400.0, "Minimum velocity", unit="m/s", minimum=10.0),
              Param("vmax", "float", 4500.0, "Maximum velocity", unit="m/s", minimum=20.0),
              Param("nv", "int", 100, "Number of velocities", minimum=10),
              Param("window", "float", 0.03, "Semblance window", unit="s", minimum=0.001),
              Param("pick_interval", "float", 0.2, "Pick every", unit="s", minimum=0.01),
              Param("max_dv_per_s", "float", 2500.0, "Maximum velocity change per second", unit="m/s/s"))

    def run(self, data, context):
        cdps = np.unique(cdp_values(data))
        chosen = cdps[self["cdp_step"] // 2:: self["cdp_step"]]       # centred in each step, not at the line end
        if chosen.size == 0:
            chosen = cdps[[cdps.size // 2]]
        vel = np.linspace(self["vmin"], self["vmax"], self["nv"])
        picks = context.velocity if context.velocity is not None else VelocityPicks()
        off = np.abs(data.header("offset"))
        full = float(off.max()) if off.size else 0.0
        skipped = []
        for n, c in enumerate(chosen):
            idx = supergather(data, c, self["supergather"])
            if idx.size < 6 or np.ptp(off[idx]) < 0.6 * full:    # too few traces or offsets to resolve velocity
                skipped.append(c)
                continue
            spec = semblance(data.traces[idx], data.header("offset")[idx], data.times, data.dt, vel, self["window"])
            p = auto_pick(spec, data.times, vel, self["max_dv_per_s"], self["pick_interval"])
            picks.clear(c)
            for t, v in p:
                picks.add(c, t, v)
            context.report((n + 1) / len(chosen))
        context.velocity = picks
        context.log(f"    {len(picks)} velocity picks at {picks.cdps.size} CMPs"
                    + (f"; skipped {len(skipped)} CMPs with short offset range" if skipped else ""))
        return data.copy()


@register
class ConstantVelocityStacks(Process):
    name = "cvs_panels"
    label = "Constant-velocity stack panels"
    category = "Velocity"
    description = "NMO + stack of a CMP range at a series of constant velocities, panels side by side (for picking checks)."
    params = (Param("cdp_min", "float", 0.0, "First CMP"), Param("cdp_max", "float", 1e9, "Last CMP"),
              Param("vmin", "float", 1500.0, "Minimum velocity", unit="m/s", minimum=10.0),
              Param("vmax", "float", 4000.0, "Maximum velocity", unit="m/s", minimum=20.0),
              Param("npanels", "int", 8, "Panels", minimum=2, maximum=50))

    def run(self, data, context):
        from .nmo import nmo_correct
        from .stack import cmp_stack
        sel = data.subset((data.header("cdp") >= self["cdp_min"]) & (data.header("cdp") <= self["cdp_max"]))
        panels = []
        for v in np.linspace(self["vmin"], self["vmax"], self["npanels"]):
            vf = np.full((sel.ntr, sel.nt), v)
            nm = sel.copy(nmo_correct(sel.traces, sel.times, sel.header("offset"), vf, stretch_limit=0.5))
            st = cmp_stack(nm)
            st.set_header("d2", np.full(st.ntr, v))
            panels.append(st)
            gap = st.copy(np.zeros((2, st.nt), np.float32), headers={k: v_[:2] for k, v_ in st.headers.items()})
            panels.append(gap)
        traces = np.concatenate([p.traces for p in panels])
        headers = {k: np.concatenate([p.header(k) for p in panels]) for k in panels[0].headers}
        out = SeismicData(traces, data.dt, headers, data.t0, data.name + " CVS", data.domain, list(data.history))
        return out


@register
class ConstantVelocityGathers(Process):
    name = "cvg_panels"
    label = "Constant-velocity gather panels"
    category = "Velocity"
    description = ("One CMP (or supergather) NMO-corrected with a series of constant velocities, panels side by side "
                   "(two blank traces between them; the panel velocity is in header 'cvg_v'). Each event is flat "
                   "in the panel of its velocity: a check of the semblance picks.")
    params = (Param("cdp", "float", -1.0, "CMP (-1 = middle of the line)"),
              Param("supergather", "int", 0, "Neighbour CMPs on each side", minimum=0),
              Param("vmin", "float", 1500.0, "Minimum velocity", unit="m/s", minimum=10.0),
              Param("vmax", "float", 3500.0, "Maximum velocity", unit="m/s", minimum=20.0),
              Param("npanels", "int", 8, "Panels", minimum=2, maximum=40),
              Param("stretch_limit", "float", 0.5, "Stretch mute limit", minimum=0.0))

    def run(self, data, context):
        from .nmo import nmo_correct
        cdps = np.unique(cdp_values(data))
        c = self["cdp"] if self["cdp"] >= 0 else float(cdps[cdps.size // 2])
        idx = supergather(data, c, self["supergather"])
        if idx.size < 2:
            raise ValueError(f"CMP {c:g} has fewer than two traces")
        g, off = data.traces[idx], data.header("offset")[idx]
        panels, heads = [], {"cdp": [], "offset": [], "cvg_v": []}
        for v in np.linspace(self["vmin"], self["vmax"], self["npanels"]):
            panels.append(nmo_correct(g, data.times, off, np.full(g.shape, v), 0.0, self["stretch_limit"]))
            panels.append(np.zeros((2, data.nt), np.float32))
            heads["cdp"] += [c] * (idx.size + 2)
            heads["offset"] += list(off) + [0.0, 0.0]
            heads["cvg_v"] += [v] * idx.size + [0.0, 0.0]
        return SeismicData(np.concatenate(panels), data.dt, heads, data.t0, f"{data.name} CVG CMP {c:g}",
                           data.domain, list(data.history))


@register
class SmoothVelocity(Process):
    name = "smooth_velocity"
    label = "Smooth velocity field"
    category = "Velocity"
    description = ("Samples the velocity picks on the CMP-time grid of the dataset, smooths them (Gaussian along the "
                   "line and in time) and replaces the project picks by the smoothed field taken every n CMPs and "
                   "every dt seconds. Removes pick-to-pick jitter before migration and depth conversion.")
    params = (Param("cdp_step", "int", 20, "Output picks every n CMPs", minimum=1),
              Param("time_step", "float", 0.1, "Output picks every", unit="s", minimum=0.004),
              Param("smooth_cdp", "float", 20.0, "Smoothing along the line", unit="CMPs", minimum=0.0),
              Param("smooth_time", "float", 0.1, "Smoothing in time", unit="s", minimum=0.0))

    def run(self, data, context):
        from scipy.ndimage import gaussian_filter
        vp = context.velocity
        if vp is None or len(vp) == 0:
            raise ValueError("there are no velocity picks to smooth")
        cdps = np.unique(cdp_values(data))
        times = data.times
        sig = (self["smooth_cdp"] / max(float(np.median(np.diff(cdps))) if cdps.size > 1 else 1.0, 1e-9),
               self["smooth_time"] / data.dt)
        v = gaussian_filter(vp.field(cdps, times), sig, mode="nearest")
        eta = gaussian_filter(vp.eta_field(cdps, times), sig, mode="nearest")
        out = VelocityPicks()
        keep = list(range(0, cdps.size, self["cdp_step"]))
        if keep[-1] != cdps.size - 1:
            keep.append(cdps.size - 1)
        t_out = np.arange(max(times[0], 0.0), times[-1] + 1e-9, self["time_step"])
        it = np.clip(np.rint((t_out - times[0]) / data.dt).astype(np.int64), 0, times.size - 1)
        for i in keep:
            for j, tj in zip(it, t_out):
                out.add(float(cdps[i]), float(tj), float(v[i, j]), float(eta[i, j]))
        context.velocity = out
        context.log(f"    velocity field smoothed: {len(out)} picks at {len(keep)} CMPs")
        return data.copy()
