"""Time-depth conversion of sections and depth-model building."""

from __future__ import annotations

import numpy as np

from ..core.dataset import SeismicData
from ..core.velocity import DepthModel, dix, time_to_depth_axis
from .base import Param, Process, model_grid, register, resolve_velocity, trace_positions


@register
class BuildDepthModel(Process):
    name = "build_depth_model"
    label = "Build depth model"
    category = "Velocity"
    description = ("Interval-velocity depth model for depth migration and depth conversion: Dix conversion of the "
                   "velocity picks (smoothed), a constant, or a linear gradient v0 + k z. Stored as the project model.")
    params = (Param("source", "choice", "picks (Dix)", "From", choices=("picks (Dix)", "constant", "gradient")),
              Param("velocity", "velocity", "project", "Picks (for Dix)"),
              Param("v0", "float", 1500.0, "v0 (constant / gradient)", unit="m/s", minimum=10.0),
              Param("k", "float", 0.5, "Gradient k", unit="1/s"),
              Param("dz", "float", 10.0, "Depth step", unit="m", minimum=0.5),
              Param("zmax", "float", 5000.0, "Maximum depth", unit="m", minimum=10.0),
              Param("smooth", "float", 200.0, "Smoothing (Dix model)", unit="m", minimum=0.0))

    def run(self, data, context):
        ucdp, first = np.unique(data.header("cdp"), return_index=True)
        xs = trace_positions(data)[first]
        x0, dx, nx = model_grid(xs)
        nz = int(self["zmax"] / self["dz"]) + 1
        if self["source"] == "constant":
            model = DepthModel.constant(self["v0"], nz, nx, self["dz"], dx, 0.0, x0)
        elif self["source"] == "gradient":
            model = DepthModel.gradient(self["v0"], self["k"], nz, nx, self["dz"], dx, 0.0, x0)
        else:
            picks = resolve_velocity(self["velocity"], context)
            model = DepthModel.from_picks(picks, dict(zip(ucdp.tolist(), xs.tolist())), ucdp, nz, self["dz"], nx, dx,
                                          x0, tmax=max(data.tmax, 1.0), smooth_m=self["smooth"])
        context.depth_model = model
        context.log(f"    depth model {model.shape[0]} x {model.shape[1]}, v {model.values.min():.0f}-{model.values.max():.0f} m/s")
        return data.copy()


@register
class TimeToDepth(Process):
    name = "time_to_depth"
    label = "Time to depth conversion"
    category = "Conversion"
    description = ("Convert a time section to depth, trace by trace: with interval velocity from the picks (Dix at each "
                   "trace's CMP) or with the project depth model.")
    params = (Param("using", "choice", "velocity picks", "Velocity from", choices=("velocity picks", "depth model")),
              Param("velocity", "velocity", "project", "Picks (if from picks)"),
              Param("dz", "float", 5.0, "Depth step", unit="m", minimum=0.1),
              Param("zmax", "float", 4000.0, "Maximum depth", unit="m", minimum=1.0))

    def run(self, data, context):
        if data.domain != "time":
            raise ValueError("input is already in depth")
        z = np.arange(0.0, self["zmax"] + self["dz"] / 2, self["dz"])
        out = np.zeros((data.ntr, z.size), dtype=np.float32)
        t = data.times
        if self["using"] == "depth model" or str(self["velocity"]).strip().lower() == "project model":
            if context.depth_model is None:
                raise ValueError("no project depth model; run 'Build depth model' first")
            model = context.depth_model
            x = trace_positions(data)
            for i in range(data.ntr):
                v = model.at(z, np.full(z.size, x[i]))
                tz = 2.0 * np.concatenate([[0.0], np.cumsum(self["dz"] / v[:-1])])
                out[i] = np.interp(tz, t, data.traces[i], left=0.0, right=0.0)
        else:
            picks = resolve_velocity(self["velocity"], context)
            cache = {}
            cdp = data.header("cdp")
            for i in range(data.ntr):
                c = float(cdp[i])
                if c not in cache:
                    vint = dix(t, picks.function(c, t))
                    cache[c] = time_to_depth_axis(t, vint)
                out[i] = np.interp(z, cache[c], data.traces[i], left=0.0, right=0.0)
        return SeismicData(out, self["dz"], {k: v.copy() for k, v in data.headers.items()}, 0.0, data.name + " (depth)",
                           "depth", list(data.history), dict(data.meta))


@register
class DepthToTime(Process):
    name = "depth_to_time"
    label = "Depth to time conversion"
    category = "Conversion"
    description = "Convert a depth section to two-way time with the project depth model."
    params = (Param("dt", "float", 0.004, "Time step", unit="s", minimum=1e-4),
              Param("tmax", "float", 4.0, "Maximum time", unit="s", minimum=0.01))

    def run(self, data, context):
        if data.domain != "depth":
            raise ValueError("input is not a depth section")
        if context.depth_model is None:
            raise ValueError("no project depth model; run 'Build depth model' first")
        z = data.times
        t = np.arange(0.0, self["tmax"] + self["dt"] / 2, self["dt"])
        x = trace_positions(data)
        out = np.zeros((data.ntr, t.size), dtype=np.float32)
        for i in range(data.ntr):
            v = context.depth_model.at(z, np.full(z.size, x[i]))
            tz = 2.0 * np.concatenate([[0.0], np.cumsum(data.dt / v[:-1])])
            out[i] = np.interp(t, tz, data.traces[i], left=0.0, right=0.0)
        return SeismicData(out, self["dt"], {k: v.copy() for k, v in data.headers.items()}, 0.0, data.name + " (time)",
                           "time", list(data.history), dict(data.meta))
