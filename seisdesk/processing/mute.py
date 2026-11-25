"""Mutes: from picked phases (top / bottom mute) or from a linear velocity (direct waves, refractions)."""

from __future__ import annotations

import numpy as np

from .base import Param, Process, register, resolve_picks


def apply_mute(traces: np.ndarray, times: np.ndarray, mute_times: np.ndarray, kind: str = "top",
               taper: float = 0.02) -> np.ndarray:
    """Zero samples above (top) or below (bottom) each trace's mute time with a cosine taper of ``taper`` seconds."""
    t = times[None, :]
    m = mute_times[:, None]
    if taper > 0:
        ramp = np.clip((t - m) / taper, 0.0, 1.0) if kind == "top" else np.clip((m - t) / taper, 0.0, 1.0)
        w = 0.5 - 0.5 * np.cos(np.pi * ramp)
    else:
        w = (t >= m) if kind == "top" else (t <= m)
    return (traces * w).astype(np.float32)


@register
class PickedMute(Process):
    name = "picked_mute"
    label = "Mute (picked phase)"
    category = "Editing"
    description = ("Top or bottom mute following a picked phase, interpolated in |offset| and between picked gathers. "
                   "Pick it in the viewer with the picking tool (one phase for the top mute, another for the bottom).")
    params = (Param("picks", "picks", "mute", "Pick set name or file"),
              Param("phase", "str", "phase 1", "Phase"),
              Param("kind", "choice", "top", "Mute", choices=("top", "bottom")),
              Param("taper", "float", 0.02, "Taper length", unit="s", minimum=0.0))

    def run(self, data, context):
        ps = resolve_picks(self["picks"], context)
        mt = ps.times_for(data, self["phase"])
        return data.copy(apply_mute(data.traces, data.times, mt, self["kind"], self["taper"]))


@register
class VelocityMute(Process):
    name = "velocity_mute"
    label = "Mute (linear velocity)"
    category = "Editing"
    description = "Zero everything above t = t0 + |offset| / v (top) or below it (bottom), e.g. to remove direct waves."
    params = (Param("velocity", "float", 1500.0, "Velocity", unit="m/s", minimum=1.0),
              Param("t0", "float", 0.05, "Intercept", unit="s"),
              Param("kind", "choice", "top", "Mute", choices=("top", "bottom")),
              Param("taper", "float", 0.02, "Taper length", unit="s", minimum=0.0))

    def run(self, data, context):
        mt = self["t0"] + np.abs(data.header("offset")) / self["velocity"]
        return data.copy(apply_mute(data.traces, data.times, mt, self["kind"], self["taper"]))


def zone_weight(times: np.ndarray, top: np.ndarray, bottom: np.ndarray, taper: float) -> np.ndarray:
    """1 inside the zone top..bottom (per trace), 0 outside, with cosine ramps of ``taper`` seconds inside the edges."""
    t = times[None, :]
    lo = np.minimum(top, bottom)[:, None]
    hi = np.maximum(top, bottom)[:, None]
    if taper > 0:
        ramp = np.clip((t - lo) / taper, 0.0, 1.0) * np.clip((hi - t) / taper, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * ramp)
    return ((t >= lo) & (t <= hi)).astype(np.float64)


@register
class ZoneMute(Process):
    name = "zone_mute"
    label = "Surgical mute (between two phases)"
    category = "Editing"
    description = ("Zeroes the samples between two picked phases (for example the edges of a ground-roll cone, an "
                   "air wave or a noise train), or keeps only that zone. Picks are interpolated along |offset| and "
                   "between picked gathers, as for the picked mute.")
    params = (Param("picks", "picks", "mute", "Pick set name or file"),
              Param("upper", "str", "phase 1", "Upper phase"),
              Param("lower", "str", "phase 2", "Lower phase"),
              Param("mode", "choice", "zero inside", "Mode", choices=("zero inside", "keep inside")),
              Param("taper", "float", 0.02, "Taper length", unit="s", minimum=0.0))

    def run(self, data, context):
        ps = resolve_picks(self["picks"], context)
        w = zone_weight(data.times, ps.times_for(data, self["upper"]), ps.times_for(data, self["lower"]),
                        self["taper"])
        weight = 1.0 - w if self["mode"] == "zero inside" else w
        return data.copy((data.traces * weight).astype(np.float32))


@register
class ConeMute(Process):
    name = "cone_mute"
    label = "Ground-roll cone mute"
    category = "Editing"
    description = ("Zeroes the cone between t = t0 + |offset| / v_fast and t = t0 + |offset| / v_slow, where slow "
                   "surface waves (ground roll) travel on land shots. Read the two velocities from the edges of the "
                   "ground roll on a shot gather.")
    params = (Param("v_fast", "float", 900.0, "Fast edge velocity", unit="m/s", minimum=1.0),
              Param("v_slow", "float", 250.0, "Slow edge velocity", unit="m/s", minimum=1.0),
              Param("t0", "float", 0.0, "Intercept", unit="s"),
              Param("taper", "float", 0.03, "Taper length", unit="s", minimum=0.0))

    def run(self, data, context):
        x = np.abs(data.header("offset"))
        w = zone_weight(data.times, self["t0"] + x / self["v_fast"], self["t0"] + x / self["v_slow"], self["taper"])
        return data.copy((data.traces * (1.0 - w)).astype(np.float32))
