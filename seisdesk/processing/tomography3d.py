"""Trace-associated multi-arrival inversion of a full 3D velocity volume."""
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from ..core.tomography3d import NodeGrid3D, invert3d
from ..core.interfaces3d import invert_interfaces3d, save_interfaces3d
from ..core.volume import VelocityVolume
from .base import Param, Process, register, resolve_picks


def export_arrival_qc(result, path):
    """Export every retained arrival, including its trace identity and uncertainty."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trace_id", "phase", "observed_ms", "predicted_ms", "uncertainty_ms",
                         "residual_ms", "standardized_residual", "flag_over_3sigma",
                         "source_snap_m", "receiver_snap_m"])
        for row, (identity, phase) in enumerate(zip(result["trace_ids"], result["phases"])):
            writer.writerow([identity, phase, result["observed"][row]*1000, result["t_pred"][row]*1000,
                             result["uncertainty"][row]*1000, result["residual"][row]*1000,
                             result["standardized_residual"][row], int(result["outlier"][row]),
                             result["source_snap_m"][row], result["receiver_snap_m"][row]])


@register
class JointTomography3D(Process):
    name, label, category = "joint_tomography_3d", "Joint arrival tomography (3D)", "Tomography"
    description = ("Invert multiple named first arrivals, reflections and head waves in a 3D velocity volume. "
                   "Planes or ny,nx depth arrays can stay fixed or undergo bounded shared-interface updates. Coordinates, depths and velocities use "
                   "metres and m/s. Station depths use the model reference elevation. Ray directions and interfaces "
                   "are discretised on the model grid; refine the grid/stencil to assess accuracy. CPU execution. "
                   "Optional timing-noise ensembles estimate conditional velocity/interface uncertainty, not geological uniqueness.")
    params = (
        Param("pickset", "picks", "first breaks", "Multi-event pick set"),
        Param("model", "infile", "", "Starting 3D velocity model (.npz)"),
        Param("branches", "text", '{"first break": {"kind": "first", "uncertainty_ms": 2}}',
              "Arrival paths and interface controls"),
        Param("invert_interfaces", "bool", False, "Update selected interface depths",
              help="Configure shared IDs, fixed boundaries, control grids and depth bounds in Configure arrivals."),
        Param("update_velocity", "bool", True, "Also update velocity during interface inversion"),
        Param("interface_max_shift", "float", 50, "Maximum interface control shift", unit="m", minimum=.001,
              help="Bound around the original boundary, retained during noise refits; at least one depth-grid cell."),
        Param("interface_step", "float", 20, "Largest interface search step", unit="m", minimum=.001,
              help="The search halves this step down to one depth-grid cell."),
        Param("interface_warp", "bool", True, "Remap velocity layers with moving boundaries",
              help="Linear vertical remapping between ordered boundaries; disable to keep an independently known velocity grid."),
        Param("surface", "infile", "", "Ground depth array (.npy), optional",
              help="Shape ny,nx, positive-down depth in the model datum. Empty means a flat surface at model z0."),
        Param("uncertainties", "infile", "", "Per-pick uncertainty CSV, optional",
              help="Columns trace_id, phase, uncertainty_ms. Overrides branch uncertainty; every CSV row must match a selected pick."),
        Param("vmin", "float", 100, "Minimum velocity", unit="m/s", minimum=10),
        Param("vmax", "float", 8000, "Maximum velocity", unit="m/s", minimum=100),
        Param("iterations", "int", 8, "Iterations per fit", minimum=1, maximum=100),
        Param("smoothing", "float", 1, "Smoothing", minimum=0),
        Param("damping", "float", .1, "Damping", minimum=.001),
        Param("radius", "int", 2, "Ray stencil radius", minimum=1, maximum=5),
        Param("memory_gb", "float", 2, "Graph working memory budget", unit="GiB", minimum=.01),
        Param("bootstrap", "int", 0, "Timing-noise uncertainty fits (0 or 3–100)", minimum=0, maximum=100),
        Param("seed", "int", 0, "Uncertainty random seed", minimum=0),
    )

    def run(self, data, context):
        def path(value):
            candidate = Path(value)
            return candidate if candidate.is_absolute() else Path(context.work_dir)/candidate

        if data.domain != "time":
            raise ValueError("arrival tomography requires a time-domain survey with identified picks")
        if not self["model"]:
            raise ValueError("choose a starting 3D velocity model")
        volume = VelocityVolume.load(path(self["model"]))
        raw = self["branches"].strip()
        branches = json.loads(raw if raw.startswith("{") else path(raw).read_text(encoding="utf-8"))
        if not isinstance(branches, dict) or not branches:
            raise ValueError("select at least one named arrival")
        picks = resolve_picks(self["pickset"], context)
        compatible = {"first": ("direct", "refraction", "unclassified"),
                      "head": ("refraction", "unclassified"), "reflection": ("reflection", "unclassified")}
        records = []
        for phase, definition in branches.items():
            if phase not in picks.phase_order or not isinstance(definition, dict):
                raise ValueError(f"unknown phase or invalid definition: {phase!r}")
            if picks.phase_type(phase) not in compatible.get(definition.get("kind"), ()):
                raise ValueError(f"phase {phase!r}: pick type and path definition disagree")
            if "surface_file" in definition:
                definition["surface"] = np.load(path(definition["surface_file"]), allow_pickle=False)
            found = list(picks.matched_records(data, phase))
            if not found:
                raise ValueError(f"phase {phase!r} has no picks matched to this survey")
            records.extend((index, phase, time) for _, _, _, time, index in found)
        idx = np.array([r[0] for r in records], int)
        phases = [r[1] for r in records]
        observed = np.array([r[2] for r in records])
        sigma = np.array([float(branches[p].get("uncertainty_ms", 2))*.001 for p in phases])
        keys = [(str(data.trace_ids[i]), p) for i, p in zip(idx, phases)]
        if len(set(keys)) != len(keys):
            raise ValueError("each selected phase must have at most one pick per trace")
        if self["uncertainties"]:
            lookup, seen = {key: i for i, key in enumerate(keys)}, set()
            with path(self["uncertainties"]).open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if not {"trace_id", "phase", "uncertainty_ms"}.issubset(reader.fieldnames or []):
                    raise ValueError("uncertainty CSV needs trace_id, phase and uncertainty_ms columns")
                for row in reader:
                    key = row["trace_id"], row["phase"]
                    if key not in lookup or key in seen:
                        raise ValueError(f"unknown or repeated uncertainty pick: {key}")
                    sigma[lookup[key]] = float(row["uncertainty_ms"])*.001
                    seen.add(key)
        surface = np.load(path(self["surface"]), allow_pickle=False) if self["surface"] else None
        context.report(0)
        grid = NodeGrid3D(volume.values.shape, volume.spacing, volume.origin, surface,
                          self["radius"], self["memory_gb"])
        unique, inverse = np.unique(idx, return_inverse=True)
        source_xyz, receiver_xyz = volume.coordinates(data.subset(unique))
        source, source_snap = grid.nodes_for(source_xyz[inverse])
        receiver, receiver_snap = grid.nodes_for(receiver_xyz[inverse])
        context.log(f"3D tomography: {len(idx)} arrivals; {grid.size} nodes; "
                    f"graph working estimate {grid.estimated_graph_bytes/1024**3:.3f} GiB; "
                    f"largest station snap {max(source_snap.max(), receiver_snap.max()):.3g} m")
        options = {}
        solver = invert3d
        if self["invert_interfaces"]:
            solver = invert_interfaces3d
            options = dict(update_velocity=self["update_velocity"], max_shift=self["interface_max_shift"],
                           max_step=self["interface_step"], warp_velocity=self["interface_warp"])
        result = solver(grid, volume.values, source, receiver, observed, phases, branches,
                          uncertainty=sigma, iterations=self["iterations"], smoothing=self["smoothing"],
                          damping=self["damping"], vmin=self["vmin"], vmax=self["vmax"],
                          bootstrap=self["bootstrap"], seed=self["seed"], report=context.report, log=context.log, **options)
        result.update(trace_ids=data.trace_ids[idx], source_nodes=source, receiver_nodes=receiver,
                      source_coordinates=source_xyz[inverse], receiver_coordinates=receiver_xyz[inverse],
                      source_snap_m=source_snap, receiver_snap_m=receiver_snap,
                      reference_elevation=volume.reference_elevation)
        final = VelocityVolume(result["velocity"], volume.spacing, volume.origin, volume.reference_elevation)
        out = final.dataset(name=data.name+" 3D tomography")
        out.history = list(data.history)
        destination = Path(context.work_dir)
        destination.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="tomography3d-", dir=destination))
        final.save(destination/"velocity.npz")
        export_arrival_qc(result, destination/"arrival-qc.csv")
        result.update(final_model=str(destination/"velocity.npz"), arrival_qc=str(destination/"arrival-qc.csv"))
        if "sampled_interfaces" in result:
            save_interfaces3d(result, destination/"interfaces.npz")
            result["interface_bundle"] = str(destination/"interfaces.npz")
        context.results[self.name] = result
        context.log(f"3D model and per-arrival QC saved to {destination}")
        return out
