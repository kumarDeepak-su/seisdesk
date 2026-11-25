"""Topographic first-arrival tomography and coverage-aware statics on small 2D lines."""
import numpy as np
from scipy.integrate import trapezoid

from ..core.tomography import NodeGrid, invert
from ..core.velocity import DepthModel
from .base import Param, Process, register, resolve_picks
from .fwi import model_section
from .migration_depth import surface_depth
from .statics import shift_traces


def first_arrival_records(ps, data, phases):
    selected = phases or [ph for ph in ps.phase_order if ps.phase_type(ph) in ("direct", "refraction", "unclassified")]
    if any(ps.phase_type(ph) not in ("direct", "refraction", "unclassified") for ph in selected):
        raise ValueError("first-arrival tomography cannot use reflection, multiple, mute or horizon phases")
    earliest = {}
    for phase in selected:
        for _, _, _, t, i in ps.matched_records(data, phase):
            if np.isfinite(t) and t > 0:
                earliest[i] = min(t, earliest.get(i, np.inf))
    if len(earliest) < 4:
        raise ValueError("at least four positive first-arrival picks with known trace geometry are required")
    idx = np.array(sorted(earliest), dtype=int)
    return idx, np.array([earliest[i] for i in idx])


def column_statics(model, x, surface_z, datum_z, replacement, bottom=None):
    """Vertical replacement correction to a datum, through the covered model depth.

    Positive shifts delay a trace. bottom is a user-selected replacement base; it
    must lie below both the recording point and datum, within the velocity model.
    """
    bottom = model.z[-1] if bottom is None else float(bottom)
    if bottom > model.z[-1] or bottom <= max(float(np.max(surface_z)), datum_z):
        raise ValueError("static replacement base must lie inside the model, below the datum and recording surface")
    shifts = []
    for xi, top in zip(x, surface_z):
        z = np.linspace(top, bottom, max(3, int(np.ceil((bottom - top) / model.dz)) + 1))
        observed = trapezoid(1 / model.at(z, np.full(z.size, xi)), z)
        shifts.append((bottom - datum_z) / replacement - observed)
    return np.asarray(shifts)


@register
class RefractionTomography(Process):
    name, label, category = "refraction_tomography", "First-arrival traveltime tomography (2D)", "Tomography"
    description = ("Shortest-path rays and damped LSQR slowness updates, with topography, smoothing and velocity "
                   "bounds. Select first-arrival phases only. Reports misfit and ray coverage; unilluminated model "
                   "cells remain unconstrained. Save a project model before using it for migration or statics.")
    params = (
        Param("pickset", "picks", "first breaks", "First-arrival pick set"),
        Param("phases", "keys", "", "Phases (empty = direct/refraction/unclassified)"),
        Param("model", "choice", "gradient", "Starting model", choices=("gradient", "project model", "refraction layers")),
        Param("dx", "float", 10.0, "Grid spacing along line", unit="m", minimum=0.1),
        Param("dz", "float", 5.0, "Grid spacing in depth", unit="m", minimum=0.1),
        Param("zmax", "float", 200.0, "Depth below highest elevation", unit="m", minimum=1.0),
        Param("v0", "float", 500.0, "Surface velocity", unit="m/s", minimum=50.0),
        Param("gradient", "float", 10.0, "Velocity gradient", unit="1/s", minimum=0.0),
        Param("vmin", "float", 100.0, "Minimum velocity", unit="m/s", minimum=10.0),
        Param("vmax", "float", 7000.0, "Maximum velocity", unit="m/s", minimum=100.0),
        Param("iterations", "int", 8, "Iterations", minimum=1, maximum=100),
        Param("smoothing", "float", 1.0, "Second-difference smoothing", minimum=0.0),
        Param("damping", "float", 0.1, "Update damping", minimum=0.001),
        Param("aspect", "float", 2.0, "Horizontal smoothing weight", minimum=0.1),
        Param("radius", "int", 3, "Ray stencil radius", minimum=1, maximum=5),
        Param("output", "choice", "velocity model", "Output", choices=("velocity model", "data with tomostatics")),
        Param("datum", "float", 0.0, "Static datum elevation", unit="m"),
        Param("replacement_velocity", "float", 2000.0, "Replacement velocity", unit="m/s", minimum=100.0),
        Param("static_base", "float", 100.0, "Static replacement base below highest elevation", unit="m", minimum=1.0),
        Param("apply", "bool", False, "Apply tomostatics to traces"),
    )

    def run(self, data, context):
        if self["vmax"] <= self["vmin"]:
            raise ValueError("maximum velocity must exceed minimum velocity")
        for key in ("sx", "gx"):
            if key not in data.headers:
                raise ValueError("tomography requires source and receiver coordinates")
        if np.ptp(np.r_[data.header("sy"), data.header("gy")]) > 1e-3:
            raise ValueError("project the source/receiver geometry onto a 2D line before tomography")
        ps = resolve_picks(self["pickset"], context)
        idx, observed = first_arrival_records(ps, data, self["phases"])
        sx, gx = data.header("sx"), data.header("gx")
        dx, dz = self["dx"], self["dz"]
        x0 = float(np.floor(min(sx.min(), gx.min()) / dx) * dx)
        nx = int(np.ceil((max(sx.max(), gx.max()) - x0) / dx)) + 1
        nz = int(np.ceil(self["zmax"] / dz)) + 1
        if nx * nz > 200_000:
            raise ValueError("tomography grid exceeds 200,000 nodes; crop the line or increase spacing")
        x = x0 + np.arange(nx) * dx
        ref, surface = surface_depth(x, sx, gx, data.header("selev"), data.header("gelev"))
        if surface.max() + data.header("sdepth").max() >= (nz - 1) * dz:
            raise ValueError("the depth grid must extend below all sources and receivers")
        grid = NodeGrid(x0, dx, nx, dz, nz, surface, self["radius"])
        src = np.array([grid.surface_node(sx[i], data.header("sdepth")[i]) for i in idx])
        rec = np.array([grid.surface_node(gx[i]) for i in idx])
        keep = src != rec
        if not np.all(keep):
            context.log(f"    omitted {int((~keep).sum())} picks whose source/receiver round to the same grid node")
        src, rec, idx, observed = src[keep], rec[keep], idx[keep], observed[keep]
        if idx.size < 4:
            raise ValueError("too few resolved source/receiver pairs; refine the tomography grid")
        if self["model"] == "project model":
            if context.depth_model is None:
                raise ValueError("no project depth model")
            old = context.depth_model
            start = np.stack([old.at(np.full(nx, z), x) for z in grid.z])
        else:
            local_z = np.maximum(grid.z[:, None] - surface[None, :], 0)
            start = self["v0"] + self["gradient"] * local_z
            if self["model"] == "refraction layers":
                layers = context.results.get("refraction", {}).get("layers", [])
                if not layers:
                    raise ValueError("run refraction layer analysis before selecting its starting model")
                start[:] = layers[-1]["refractor_velocity"]
                for layer in reversed(layers):
                    start[local_z < layer["depth_to_base"]] = layer["velocity"]
        context.report(0)
        result = invert(grid, start, src, rec, observed, self["iterations"], self["smoothing"],
                        aspect=self["aspect"], damping=self["damping"], vmin=self["vmin"], vmax=self["vmax"],
                        report=context.report, log=context.log)
        model = DepthModel(grid.fill_air(result["velocity"]), dz, dx, x0=x0)
        result.update(starting_model=start, x=grid.x, z=grid.z, surface=surface, reference_elevation=ref,
                      observed=observed, trace_ids=data.trace_ids[idx].tolist(),
                      source_nodes=src, receiver_nodes=rec)
        if self["output"] == "data with tomostatics":
            sz = ref - data.header("selev") + data.header("sdepth")
            gz = ref - data.header("gelev")
            ss = column_statics(model, sx, sz, ref - self["datum"], self["replacement_velocity"], self["static_base"])
            gs = column_statics(model, gx, gz, ref - self["datum"], self["replacement_velocity"], self["static_base"])
            out = data.copy()
            out.set_header("sstat", ss * 1000); out.set_header("gstat", gs * 1000)
            if self["apply"]:
                out.traces = shift_traces(out.traces, data.dt, ss + gs)
                out.set_header("tstat", data.header("tstat") + (ss + gs) * 1000)
            result.update(source_statics_ms=ss * 1000, receiver_statics_ms=gs * 1000)
        else:
            out = model_section(model, data.name + " tomography velocity", data.history)
            out.meta["depth_reference_elevation"] = ref
        context.report(1)
        context.depth_model = model
        context.results[self.name] = result
        return out
