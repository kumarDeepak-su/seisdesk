"""Bounded 3D boundary updates with re-traced multi-arrival travel times.

Uniform or bilinear control-grid shifts deform each starting interface. Every
candidate is evaluated on the ray grid, with shared boundaries, ordering and
hard bounds enforced. This coordinate search is grid resolved, not a continuous
shape derivative. Optional velocity steps and noise refits use the same picks.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .tomography3d import arrivals3d, invert3d


def _groups(grid, branches, max_shift):
    groups = {}
    for phase, definition in branches.items():
        if definition.get("kind") == "first":
            continue
        base = grid.interface_surface(definition).copy()
        name = str(definition.get("interface", phase)).strip()
        if not name:
            raise ValueError("shared interface identifiers must not be empty")
        shape = np.asarray(definition.get("control_shape", [1, 1]))
        if (shape.shape != (2,) or not np.issubdtype(shape.dtype, np.number) or
                not np.isfinite(shape).all() or np.any(shape != np.floor(shape)) or
                np.any(shape < 1) or np.any(shape > 5) or np.any(shape > grid.shape[1:]) or
                (np.any(shape == 1) and not np.all(shape == 1))):
            raise ValueError("interface control shape must be 1x1 or 2–5 per axis, within the horizontal grid")
        shape = tuple(map(int, shape))
        bounds = np.asarray(definition.get("depth_bounds", [grid.axes[0][0]+grid.spacing[0], grid.axes[0][-2]]), float)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or
                np.any(base < bounds[0]) or np.any(base > bounds[1])):
            raise ValueError(f"interface {name!r}: depth bounds must increase and contain the starting surface")
        active = bool(definition.get("invert_interface", True))
        if name in groups:
            previous = groups[name]
            if (not np.array_equal(previous["base"], base) or previous["shape"] != shape or
                    previous["active"] != active or not np.array_equal(previous["bounds"], bounds)):
                raise ValueError(f"shared interface {name!r} needs identical geometry, controls, bounds and update selection")
            previous["phases"].append(phase)
        else:
            groups[name] = dict(base=base, shape=shape, bounds=bounds, active=active, phases=[phase])
    groups = dict(sorted(groups.items(), key=lambda item: float(item[1]["base"].mean())))
    if not any(g["active"] for g in groups.values()):
        raise ValueError("select at least one reflected/head-wave interface to update")
    count = sum(int(np.prod(g["shape"])) for g in groups.values() if g["active"])
    if count > 64:
        raise ValueError("use at most 64 active interface depth controls")
    for group in groups.values():
        group["max_shift"] = max_shift
    return groups


def _deform(group, controls):
    if controls.shape == (1, 1):
        return group["base"]+float(controls[0, 0])
    ny, nx = group["base"].shape
    iy, ix = np.meshgrid(np.linspace(0, controls.shape[0]-1, ny),
                          np.linspace(0, controls.shape[1]-1, nx), indexing="ij")
    return group["base"]+map_coordinates(controls, [iy, ix], order=1, mode="nearest", prefilter=False)


def _materialize(grid, branches, groups, controls):
    result = deepcopy(branches)
    nominal, sampled = {}, {}
    for name, group in groups.items():
        delta = controls[name]
        if np.any(np.abs(delta) > group["max_shift"]+1e-8):
            raise ValueError("interface control shift exceeds its original bounds")
        depth = _deform(group, delta)
        if np.any(depth < group["bounds"][0]-1e-8) or np.any(depth > group["bounds"][1]+1e-8):
            raise ValueError("trial interface exceeds absolute depth bounds")
        for phase in group["phases"]:
            # A fitted surface is self-contained, rather than dependent on an old external file.
            result[phase].pop("surface_file", None)
            result[phase]["surface"] = depth
        nominal[name] = depth
        sampled[name] = grid.interface(dict(surface=depth))[1]
    anchors = np.asarray([grid.surface, *sampled.values(), np.full(grid.shape[1:], grid.axes[0][-1])])
    raw_anchors = np.asarray([grid.surface, *nominal.values(), np.full(grid.shape[1:], grid.axes[0][-1])])
    if np.any(np.diff(anchors, axis=0) < grid.spacing[0]-1e-7) or np.any(np.diff(raw_anchors, axis=0) < grid.spacing[0]-1e-7):
        raise ValueError("interfaces must remain ordered, separated by a depth cell and below ground; use shared IDs for one boundary")
    return result, nominal, sampled, anchors


def _warp_velocity(grid, velocity, previous, candidate):
    """Linear vertical remapping between ordered sampled boundary anchors."""
    result = np.empty_like(velocity)
    for y, x in np.ndindex(grid.shape[1:]):
        mapped = np.interp(grid.axes[0], candidate[:, y, x], previous[:, y, x])
        result[:, y, x] = np.interp(mapped, grid.axes[0], velocity[:, y, x])
    return result


def invert_interfaces3d(grid, start, source, receiver, observed, phases, branches, *, uncertainty=None,
                        iterations=5, update_velocity=True, max_shift=50., max_step=20., warp_velocity=True,
                        smoothing=1., damping=.1, vmin=100., vmax=8000., bootstrap=0, seed=0, report=None, log=None):
    """Alternate bounded interface search and velocity inversion, then conditional noise refits.

Controls describe shifts of the ORIGINAL surface, also during noise refits.
Hard bounds are never recentered at the optimum. Every successful candidate must
reduce the uncertainty-weighted data misfit. No observation is dropped. Timing
refits keep phase assignments, topology, grid and control parameterisation fixed.
    """
    observed, phases, start = np.asarray(observed, float), np.asarray(phases, str), np.asarray(start, float)
    if (observed.ndim != 1 or len(observed) < 4 or phases.shape != observed.shape or
            not np.isfinite(observed).all() or np.any(observed <= 0)):
        raise ValueError("3D interface inversion requires at least four finite positive picked times")
    if (not np.isfinite([vmin, vmax, smoothing, damping, max_shift, max_step]).all() or
            not 0 < vmin < vmax or smoothing < 0 or damping <= 0 or
            max_shift < grid.spacing[0] or max_step < grid.spacing[0] or
            int(iterations) != iterations or iterations < 1):
        raise ValueError("invalid inversion bounds; interface shift/step must be at least one depth cell")
    if start.shape != grid.shape or not np.isfinite(start).all() or np.any((start < vmin) | (start > vmax)):
        raise ValueError("starting velocity must match the grid and lie inside its bounds")
    if int(bootstrap) != bootstrap or (bootstrap != 0 and not 3 <= bootstrap <= 100):
        raise ValueError("uncertainty replicates must be zero or an integer between 3 and 100")
    bootstrap = int(bootstrap)
    if any(str(p) not in branches for p in phases):
        raise ValueError("every selected phase needs an arrival definition")
    if set(branches) != set(phases):
        raise ValueError("each configured branch must have selected arrivals")
    branches = deepcopy(branches)
    groups = _groups(grid, branches, float(max_shift))
    sigma = (np.asarray([branches[p].get("uncertainty_ms", 2)*.001 for p in phases], float)
             if uncertainty is None else np.asarray(uncertainty, float))
    if sigma.shape != observed.shape or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("give a finite positive timing uncertainty in seconds for every arrival")
    extra = (len(groups)*np.prod(grid.shape[1:])*96 + bootstrap*(grid.size+len(groups)*np.prod(grid.shape[1:]))*40)
    if grid.estimated_graph_bytes+extra > grid.memory_gb*1024**3:
        raise ValueError("interface uncertainty ensemble and graph exceed the memory budget")
    zero = {name: np.zeros(g["shape"]) for name, g in groups.items()}
    initial = _materialize(grid, branches, groups, zero)
    cells = max(1, int(min(max_step, max_shift)//grid.spacing[0]))
    steps = []
    while True:
        steps.append(cells*grid.spacing[0])
        if cells == 1:
            break
        cells = max(1, cells//2)
    moves = []
    for name, group in groups.items():
        if group["active"]:
            for step in steps:
                moves.append((name, step, None))
                if np.prod(group["shape"]) > 1:
                    moves.extend((name, step, c) for c in np.ndindex(group["shape"]))

    def fit(initial_velocity, initial_controls, obs, part):
        def progress(value):
            if report:
                report((part+value)/(bootstrap+1))
        progress(0.)
        velocity = initial_velocity.copy()
        controls = {name: value.copy() for name, value in initial_controls.items()}
        defs, nominal, sampled, anchors = _materialize(grid, branches, groups, controls)
        prediction = arrivals3d(grid, velocity, source, receiver, phases, defs, rays=False,
                                report=lambda _: progress(.01))[0]
        score = lambda times: float(np.mean(((obs-times)/sigma)**2))
        objective = [score(prediction)]
        rms = [float(np.sqrt(np.mean((obs-prediction)**2)))]
        history = [dict(update="initial", iteration=0, weighted_misfit=objective[0],
                        controls={n: c.tolist() for n, c in controls.items()})]
        rejected = 0
        stop = "iteration limit"

        def record(kind, iteration):
            objective.append(score(prediction)); rms.append(float(np.sqrt(np.mean((obs-prediction)**2))))
            history.append(dict(update=kind, iteration=iteration+1, weighted_misfit=objective[-1],
                                controls={n: c.tolist() for n, c in controls.items()}))
            if log:
                log(f"3D joint fit {part+1}, {kind}: RMS {rms[-1]*1000:.4g} ms")

        for iteration in range(int(iterations)):
            changed = False
            for move, (name, step, control) in enumerate(moves):
                fraction = .02+.9*(iteration+move/(len(moves)+1))/iterations
                progress(fraction)
                best = None
                for sign in (-1, 1):
                    trial_controls = {n: c.copy() for n, c in controls.items()}
                    if control is None:
                        trial_controls[name] += sign*step
                    else:
                        trial_controls[name][control] += sign*step
                    try:
                        trial = _materialize(grid, branches, groups, trial_controls)
                        if np.array_equal(trial[3], anchors):
                            continue
                        trial_velocity = _warp_velocity(grid, velocity, anchors, trial[3]) if warp_velocity else velocity
                        times = arrivals3d(grid, trial_velocity, source, receiver, phases, trial[0], rays=False,
                                           report=lambda _: progress(fraction))[0]
                    except ValueError:
                        rejected += 1
                        continue
                    value = score(times)
                    if value < objective[-1]-max(1e-12, objective[-1]*1e-10) and (best is None or value < best[0]):
                        best = value, trial, trial_velocity, times, trial_controls
                if best is not None:
                    _, (defs, nominal, sampled, anchors), velocity, prediction, controls = best
                    record(f"interface {name}, {'all controls' if control is None else control}", iteration)
                    changed = True
            if update_velocity:
                fraction = .02+.9*(iteration+len(moves)/(len(moves)+1))/iterations
                updated = invert3d(grid, velocity, source, receiver, obs, phases, defs, uncertainty=sigma,
                                   iterations=1, smoothing=smoothing, damping=damping, vmin=vmin, vmax=vmax,
                                   report=lambda _: progress(fraction), _noise_realization=part > 0)
                if score(updated["t_pred"]) < objective[-1]-max(1e-12, objective[-1]*1e-10):
                    velocity, prediction = updated["velocity"], updated["t_pred"]
                    record("velocity", iteration); changed = True
            if not changed:
                stop = "no improving grid-resolved interface/velocity update"
                break
        contacts = {}
        prediction, _, hits = arrivals3d(grid, velocity, source, receiver, phases, defs, interface_hits=contacts,
                                         report=lambda _: progress(.97))
        contact = {name: sum((contacts[p] for p in g["phases"]), np.zeros(grid.shape[1:], int))
                   for name, g in groups.items()}
        progress(1.)
        return dict(velocity=velocity, t_pred=prediction, hits=hits, branches=defs, interface_surfaces=nominal,
                    sampled_interfaces=sampled, interface_hits=contact, interface_controls=controls,
                    rms=rms, weighted_misfit=objective, interface_history=history, stop_reason=stop,
                    invalid_interface_trials=rejected)

    result = fit(start, zero, observed, 0)
    residual = result["t_pred"]-observed
    result.update(starting_model=start.copy(), observed=observed.copy(), phases=phases.tolist(),
                  uncertainty=sigma.copy(), residual=residual, standardized_residual=residual/sigma,
                  outlier=np.abs(residual/sigma)>3, used=np.ones(len(observed), bool), axes=grid.axes,
                  origin=grid.origin.copy(), spacing=grid.spacing.copy(), surface=grid.surface.copy(),
                  graph_bytes_estimate=grid.estimated_graph_bytes, starting_branches=branches,
                  starting_interface_surfaces=initial[1], starting_sampled_interfaces=initial[2],
                  interface_grid_step_m=float(grid.spacing[0]), interface_velocity_remap=bool(warp_velocity),
                  interface_update_velocity=bool(update_velocity),
                  interface_parameters={n: dict(phases=g["phases"], active=g["active"], shape=list(g["shape"]),
                    max_shift_m=float(max_shift), depth_bounds=g["bounds"].tolist(),
                    control_at_shift_bound=np.isclose(np.abs(result["interface_controls"][n]), max_shift))
                    for n, g in groups.items()},
                  phase_rms={str(p): float(np.sqrt(np.mean(residual[phases==p]**2))) for p in dict.fromkeys(phases)})
    if bootstrap:
        rng = np.random.default_rng(seed)
        velocities = np.empty((bootstrap, *grid.shape))
        depths = {n: np.empty((bootstrap, *grid.shape[1:])) for n in groups}
        controls = {n: np.empty((bootstrap, *g["shape"])) for n, g in groups.items()}
        fit_rms = []
        for index in range(bootstrap):
            synthetic = observed+rng.normal(size=len(observed))*sigma
            # No clipping of perturbed observations: the Gaussian noise model is retained.
            sample = fit(result["velocity"], result["interface_controls"], synthetic, index+1)
            velocities[index] = sample["velocity"]
            for name in groups:
                depths[name][index] = sample["sampled_interfaces"][name]
                controls[name][index] = sample["interface_controls"][name]
            fit_rms.append(sample["rms"][-1])
        for name, calculate in (("std", lambda a: a.std(axis=0, ddof=1)),
                                ("p05", lambda a: np.quantile(a, .05, axis=0)),
                                ("p95", lambda a: np.quantile(a, .95, axis=0))):
            result["velocity_"+name] = np.where(result["hits"]>0, calculate(velocities), np.nan)
            result["interface_"+name] = {n: np.where(result["interface_hits"][n]>0, calculate(a), np.nan)
                                        for n, a in depths.items()}
        result["interface_control_ensemble"] = controls
        result["uncertainty_ensemble"] = dict(replicates=bootstrap, seed=int(seed), fit_rms_s=fit_rms,
            assumption="Independent Gaussian timing errors; fixed phase assignments, topology, control grid, bounds and velocity regularisation; interfaces and selected velocity updates refitted. Zero spread may reflect ray-grid quantisation")
    return result


def save_interfaces3d(result, path):
    """Portable NPZ containing nominal/ray-grid surfaces, coverage and conditional uncertainty."""
    arrays, metadata = {}, {}
    for index, (name, parameters) in enumerate(result["interface_parameters"].items()):
        prefix = f"interface_{index}"
        metadata[prefix] = {**parameters, "name": name,
                            "control_at_shift_bound": np.asarray(parameters["control_at_shift_bound"]).tolist()}
        for key in ("interface_surfaces", "sampled_interfaces", "starting_interface_surfaces",
                    "starting_sampled_interfaces", "interface_hits", "interface_controls",
                    "interface_std", "interface_p05", "interface_p95", "interface_control_ensemble"):
            if key in result:
                arrays[f"{prefix}_{key}"] = np.asarray(result[key][name])
    metadata["units"] = "metres; positive-down depth in the velocity model datum"
    metadata["spacing_zyx"] = np.asarray(result["spacing"]).tolist()
    metadata["origin_zyx"] = np.asarray(result["origin"]).tolist()
    metadata["reference_elevation"] = float(result.get("reference_elevation", 0))
    metadata["uncertainty"] = result.get("uncertainty_ensemble")
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **arrays)
