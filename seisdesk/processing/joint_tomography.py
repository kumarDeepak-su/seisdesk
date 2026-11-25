"""Joint inversion of multiple named arrival branches, without earliest-pick collapse."""
import json
from pathlib import Path

import numpy as np

from ..core.joint_tomography import invert_joint
from ..core.interface_inversion import invert_interfaces
from ..core.tomography import NodeGrid
from ..core.velocity import DepthModel
from .base import Param, Process, register, resolve_picks
from .fwi import model_section
from .migration_depth import surface_depth


@register
class JointTomography(Process):
    name, label, category = "joint_tomography", "Joint refraction / reflection tomography (2D)", "Tomography"
    description = ("Invert all selected named branches together. Configure each phase as first, reflection or head "
                   "with an interface depth or x,z points and timing uncertainty. Reflected legs bend "
                   "through the velocity model; head waves follow the faster side of the assigned refractor. "
                   "Optionally update bounded interface depths and velocity together. Shared interface IDs tie "
                   "branches to the same boundary. Depth resolution is limited by the ray grid; velocity-depth "
                   "trade-offs still apply. Invalid or subcritical branches raise an error.")
    params = (
        Param("pickset", "picks", "first breaks", "Multi-event pick set"),
        Param("branches", "text", '{"first break": {"kind": "first", "uncertainty_ms": 2}}',
              "Arrival paths and interfaces"),
        Param("dx", "float", 10, "Horizontal grid spacing", unit="m", minimum=.1),
        Param("dz", "float", 5, "Depth grid spacing", unit="m", minimum=.1),
        Param("zmax", "float", 200, "Depth below highest elevation", unit="m", minimum=1),
        Param("model", "choice", "project model", "Starting model", choices=("project model", "gradient")),
        Param("v0", "float", 1500, "Starting surface velocity", unit="m/s", minimum=50),
        Param("gradient", "float", 5, "Starting velocity gradient", unit="1/s", minimum=0),
        Param("vmin", "float", 100, "Minimum velocity", unit="m/s", minimum=10),
        Param("vmax", "float", 8000, "Maximum velocity", unit="m/s", minimum=100),
        Param("iterations", "int", 8, "Iterations", minimum=1, maximum=100),
        Param("smoothing", "float", 1, "Smoothing", minimum=0),
        Param("damping", "float", .1, "Damping", minimum=.001),
        Param("radius", "int", 5, "Ray stencil radius", minimum=1, maximum=5,
              help="Larger stencils improve angular resolution, especially when dx and dz differ. Check stability against grid refinement."),
        Param("invert_interfaces", "bool", False, "Update interface depths"),
        Param("update_velocity", "bool", True, "Update velocity"),
        Param("interface_max_shift", "float", 50, "Maximum depth change from starting interface", unit="m", minimum=.1),
        Param("interface_max_step", "float", 10, "Largest interface search step (at least dz)", unit="m", minimum=.1),
        Param("warp_velocity", "bool", True, "Move velocity layers with interfaces",
              help="Preserves layer contrasts when boundaries move. Disable for an independently known fixed velocity grid."),
    )

    def run(self, data, context):
        if self["vmax"] <= self["vmin"]:
            raise ValueError("maximum velocity must exceed minimum velocity")
        text = self["branches"].strip()
        if not text.startswith("{"):
            path = Path(text)
            text = (path if path.is_absolute() else Path(context.work_dir)/path).read_text(encoding="utf-8")
        branches = json.loads(text)
        if not isinstance(branches, dict) or not branches:
            raise ValueError("give at least one named phase definition")
        ps = resolve_picks(self["pickset"], context)
        records = []
        compatible = {"first": ("direct", "refraction", "unclassified"),
                      "head": ("refraction", "unclassified"), "reflection": ("reflection", "unclassified")}
        for phase, definition in branches.items():
            if phase not in ps.phase_order or not isinstance(definition, dict):
                raise ValueError(f"unknown phase or invalid definition: {phase!r}")
            if ps.phase_type(phase) not in compatible.get(definition.get("kind"), ()):
                raise ValueError(f"phase {phase!r}: pick type and path definition disagree")
            found = list(ps.matched_records(data, phase))
            if not found:
                raise ValueError(f"phase {phase!r} has no picks matched to this dataset")
            records.extend((i, phase, time) for _, _, _, time, i in found)
        if len(records) < 4:
            raise ValueError("joint tomography requires at least four matched picks")
        idx = np.array([r[0] for r in records], int)
        phases = [r[1] for r in records]
        observed = np.array([r[2] for r in records])
        if not np.isfinite(observed).all() or np.any(observed <= 0):
            raise ValueError("all picked times must be finite and positive")
        sx, gx = data.header("sx", None), data.header("gx", None)
        if np.ptp(np.r_[data.header("sy"), data.header("gy")]) > 1e-3:
            raise ValueError("project the geometry onto a 2D line before joint tomography")
        dx, dz = self["dx"], self["dz"]
        x0 = np.floor(min(sx.min(), gx.min())/dx)*dx
        nx = int(np.ceil((max(sx.max(), gx.max())-x0)/dx))+1
        nz = int(np.ceil(self["zmax"]/dz))+1
        if nx*nz > 200_000:
            raise ValueError("joint tomography grid exceeds 200,000 nodes; coarsen or crop the line")
        x = x0 + np.arange(nx)*dx
        reference, surface = surface_depth(x, sx, gx, data.header("selev"), data.header("gelev"))
        grid = NodeGrid(x0, dx, nx, dz, nz, surface, self["radius"])
        if np.max(surface) + np.max(data.header("sdepth")) >= grid.z[-1]:
            raise ValueError("model must extend beneath sources and receivers")
        source = np.array([grid.surface_node(sx[i], data.header("sdepth")[i]) for i in idx])
        receiver = np.array([grid.surface_node(gx[i]) for i in idx])
        if self["model"] == "project model":
            model = context.depth_model
            if model is None:
                raise ValueError("load a depth model, or select a gradient starting model")
            if x[0] < model.x[0] or x[-1] > model.x[-1] or grid.z[-1] > model.z[-1] or model.z0 != 0:
                raise ValueError("the starting model must cover the tomography grid in the same depth datum")
            start = np.stack([model.at(np.full(nx, z), x) for z in grid.z])
        else:
            start = self["v0"] + self["gradient"]*np.maximum(grid.z[:, None]-surface, 0)
        options = dict(iterations=self["iterations"], smoothing=self["smoothing"], damping=self["damping"],
                       vmin=self["vmin"], vmax=self["vmax"], report=context.report, log=context.log)
        if self["invert_interfaces"]:
            result = invert_interfaces(grid, start, source, receiver, observed, phases, branches,
                                       update_velocity=self["update_velocity"], max_shift=self["interface_max_shift"],
                                       max_step=self["interface_max_step"], warp_velocity=self["warp_velocity"], **options)
        else:
            if not self["update_velocity"]:
                raise ValueError("select velocity updates, interface updates, or both")
            result = invert_joint(grid, start, source, receiver, observed, phases, branches, **options)
        result.update(starting_model=start, x=x, z=grid.z, surface=surface, reference_elevation=reference,
                      trace_ids=data.trace_ids[idx], source_nodes=source, receiver_nodes=receiver)
        model = DepthModel(grid.fill_air(result["velocity"]), dz, dx, x0=x0)
        out = model_section(model, data.name + " joint tomography", data.history)
        out.meta["depth_reference_elevation"] = reference
        context.depth_model = model
        context.results[self.name] = result
        return out
