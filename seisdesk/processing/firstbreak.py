"""First-break picking (modified energy ratio) and refraction analysis from multi-phase picks."""

from __future__ import annotations

import numpy as np

from ..core.picks import PickSet, refraction_layers
from .base import Param, Process, register, resolve_picks


def mer_picks(traces: np.ndarray, times: np.ndarray, window: float, tmin: np.ndarray, tmax: np.ndarray,
              first_fraction: float = 0.0) -> np.ndarray:
    """Modified-energy-ratio first breaks: argmax of (E_after / E_before * |a|)^3 between tmin and tmax per trace.

    With first_fraction > 0 the pick is instead the earliest local maximum whose E_after / E_before * |a| (before
    cubing) reaches that fraction of the trace's largest one: the first arrival even when a later arrival (a direct
    wave behind a refraction) is stronger."""
    dt = times[1] - times[0]
    L = max(2, int(round(window / dt)))
    a = traces.astype(np.float64)
    e = np.cumsum(a * a, axis=1)
    e = np.concatenate([np.zeros((a.shape[0], 1)), e], axis=1)
    nt = a.shape[1]
    idx = np.arange(nt)
    before = e[:, idx + 1] - e[:, np.maximum(idx + 1 - L, 0)]
    after = e[:, np.minimum(idx + L, nt)] - e[:, idx]
    eps = 1e-3 * np.maximum(e[:, -1:] * L / nt, 1e-30)      # stabiliser: 0.1 % of a trace's mean window energy
    er = (after + eps) / (before + eps)
    mer = (er * np.abs(a)) ** 3
    t = times[None, :]
    mer[(t < tmin[:, None]) | (t > tmax[:, None])] = -np.inf
    if first_fraction <= 0:
        return times[np.argmax(mer, axis=1)]
    peak = np.max(mer, axis=1, keepdims=True)
    first = np.argmax(mer >= first_fraction ** 3 * peak, axis=1)          # the fraction applies before cubing
    return np.array([times[i0 + int(np.argmax(mer[k, i0:i0 + L + 1]))] for k, i0 in enumerate(first)])


@register
class AutoFirstBreaks(Process):
    name = "auto_first_breaks"
    label = "Automatic first-break picking"
    category = "Picking"
    description = ("Modified-energy-ratio picker inside a velocity guide window (|x|/v_fast .. |x|/v_slow + margin); "
                   "picks go to a pick set (one phase) that the picking tool can edit and split into refraction phases.")
    params = (Param("pickset", "str", "first breaks", "Pick set name"),
              Param("gather_key", "key", "fldr", "Gather key"),
              Param("window", "float", 0.02, "Energy window", unit="s", minimum=0.002),
              Param("v_fast", "float", 6000.0, "Fastest apparent velocity", unit="m/s", minimum=10.0),
              Param("v_slow", "float", 300.0, "Slowest apparent velocity", unit="m/s", minimum=10.0),
              Param("margin", "float", 0.05, "Margin", unit="s", minimum=0.0))

    def run(self, data, context):
        x = np.abs(data.header("offset"))
        tmin = np.maximum(x / self["v_fast"] - self["margin"], data.t0)
        tmax = np.minimum(x / self["v_slow"] + self["margin"], data.tmax)
        fb = mer_picks(data.traces, data.times, self["window"], tmin, tmax)
        ps = context.picks.get(self["pickset"]) or PickSet(self["gather_key"], "offset", self["pickset"])
        phase = ps.new_phase("first break")
        g = data.header(self["gather_key"])
        off = data.header("offset")
        for i in range(data.ntr):
            ps.add(g[i], phase, off[i], fb[i], trace_id=data.trace_ids[i])
        context.picks[self["pickset"]] = ps
        out = data.copy()
        out.set_header("laga", fb * 1000.0)
        context.log(f"    {data.ntr} first breaks picked into '{self['pickset']}' (also in header laga, ms)")
        return out


@register
class RefractionAnalysis(Process):
    name = "refraction_analysis"
    label = "Refraction analysis (intercept time)"
    category = "Picking"
    description = ("Straight-line fits to the picked phases in order (direct wave first, then each refracted branch "
                   "as it takes over away from zero offset): apparent velocities, intercept times, crossover distances "
                   "and horizontal-layer thicknesses. Results go to the log and the project results.")
    params = (Param("pickset", "picks", "first breaks", "Pick set"),
              Param("phases", "keys", "", "Phases in order (empty = picking order)"),
              Param("side", "choice", "abs", "Offsets", choices=("abs", "positive", "negative")),
              Param("gather", "float", -1.0, "Gather (-1 = all gathers pooled)"))

    def run(self, data, context):
        ps = resolve_picks(self["pickset"], context)
        order = self["phases"] or [ph for ph in ps.phase_order
                                  if ps.phase_type(ph) in ("direct", "refraction", "unclassified")]
        if any(ps.phase_type(ph) not in ("direct", "refraction", "unclassified") for ph in order):
            raise ValueError("refraction analysis accepts only direct and refracted arrivals")
        gather = None if self["gather"] < 0 else self["gather"]
        fits = [ps.fit_line(gather, ph, self["side"]) for ph in order]
        layers = refraction_layers(fits)
        context.results["refraction"] = {"fits": fits, "layers": layers}
        for f in fits:
            context.log(f"    {f['phase']:>14s}: v = {f['velocity']:8.1f} m/s, intercept {f['intercept'] * 1000:7.2f} ms, "
                        f"rms {f['rms'] * 1000:.2f} ms, {f['n']} picks")
        for L in layers:
            context.log(f"    layer {L['layer']}: v = {L['velocity']:.0f} m/s, thickness {L['thickness']:.1f} m, "
                        f"base at {L['depth_to_base']:.1f} m, crossover {L['crossover_distance']:.0f} m")
        return data.copy()
