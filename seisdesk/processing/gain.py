"""Amplitude corrections: AGC, t-power gain, spherical divergence, trace balancing, scaling and clipping."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .base import Param, Process, register, resolve_velocity


def agc(traces: np.ndarray, dt: float, window: float, eps: float = 1e-12) -> np.ndarray:
    """Automatic gain control: divide by the RMS amplitude in a sliding window of ``window`` seconds."""
    n = max(1, int(round(window / dt)))
    power = uniform_filter1d(traces.astype(np.float64) ** 2, size=n, axis=1, mode="nearest")
    rms = np.sqrt(np.maximum(power, 0.0))                 # running sums can dip below zero by round-off
    ref = np.maximum(rms, eps * max(float(rms.max()), 1e-30))
    return (traces / ref).astype(np.float32)


@register
class AGC(Process):
    name = "agc"
    label = "AGC"
    category = "Gain"
    description = "Automatic gain control with a sliding RMS window."
    params = (Param("window", "float", 0.5, "Window length", "length of the RMS window", minimum=0.001, unit="s"),)

    def run(self, data, context):
        return data.copy(agc(data.traces, data.dt, self["window"]))


@register
class TPow(Process):
    name = "tpow"
    label = "t-power gain"
    category = "Gain"
    description = "Multiply each sample by t^power (t in seconds)."
    params = (Param("power", "float", 2.0, "Power", "exponent n in t^n"),)

    def run(self, data, context):
        t = np.maximum(data.times, data.dt)
        return data.copy((data.traces * (t ** self["power"])[None, :]).astype(np.float32))


@register
class SphericalDivergence(Process):
    name = "spherical_divergence"
    label = "Spherical divergence"
    category = "Gain"
    description = "Geometric-spreading correction g(t) = t v_rms(t)^2 / (t_ref v_ref^2)."
    params = (Param("velocity", "velocity", "project", "Velocity", "project picks, a constant, 't:v,...' or a file"),
              Param("t_ref", "float", 1.0, "Reference time", unit="s", minimum=1e-3))

    def run(self, data, context):
        vp = resolve_velocity(self["velocity"], context)
        t = np.maximum(data.times, data.dt)
        cdp = data.header("cdp")
        out = np.empty_like(data.traces)
        cache = {}
        for i in range(data.ntr):
            key = float(cdp[i])
            if key not in cache:
                v = vp.function(key, t)
                vref = float(np.interp(self["t_ref"], t, v))
                cache[key] = (t * v * v) / (self["t_ref"] * vref * vref)
            out[i] = data.traces[i] * cache[key]
        return data.copy(out)


@register
class TraceBalance(Process):
    name = "trace_balance"
    label = "Trace balance"
    category = "Gain"
    description = "Scale every trace to unit RMS (or unit maximum) inside a time window."
    params = (Param("mode", "choice", "rms", "Normalise by", choices=("rms", "max")),
              Param("tmin", "float", 0.0, "Window start", unit="s"),
              Param("tmax", "float", 0.0, "Window end (0 = trace end)", unit="s"))

    def run(self, data, context):
        t = data.times
        sel = (t >= self["tmin"]) & ((t <= self["tmax"]) if self["tmax"] > 0 else True)
        seg = data.traces[:, sel].astype(np.float64)
        norm = np.sqrt(np.mean(seg ** 2, axis=1)) if self["mode"] == "rms" else np.max(np.abs(seg), axis=1)
        norm[norm == 0] = 1.0
        return data.copy((data.traces / norm[:, None]).astype(np.float32))


@register
class Scale(Process):
    name = "scale"
    label = "Scale / clip"
    category = "Gain"
    description = "Multiply by a constant and optionally clip at a percentile of |amplitude|."
    params = (Param("factor", "float", 1.0, "Scale factor"),
              Param("clip_percentile", "float", 0.0, "Clip percentile (0 = off)", minimum=0.0, maximum=100.0))

    def run(self, data, context):
        out = data.traces * self["factor"]
        if self["clip_percentile"] > 0:
            c = np.percentile(np.abs(out), self["clip_percentile"])
            out = np.clip(out, -c, c)
        return data.copy(out.astype(np.float32))


@register
class ExpGain(Process):
    name = "exp_gain"
    label = "Exponential gain"
    category = "Gain"
    description = "Multiplies each sample by exp(a t), with a given in dB per second, to compensate for absorption."
    params = (Param("db_per_s", "float", 6.0, "Gain rate", unit="dB/s"),)

    def run(self, data, context):
        a = self["db_per_s"] / 20.0 * np.log(10.0)
        return data.copy((data.traces * np.exp(a * np.maximum(data.times, 0.0))[None, :]).astype(np.float32))
