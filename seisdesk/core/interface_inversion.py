"""Bounded, grid-resolved interface updates for joint traveltime inversion.

Alternates velocity updates with a multiscale coordinate search over interface
depths. Shared interface IDs move together. The search re-traces every candidate
and accepts only a lower weighted data misfit. This is not a continuous shape
derivative method; depth resolution is limited by the ray grid and acquisition.
"""
from copy import deepcopy

import numpy as np

from .joint_tomography import interface_nodes, invert_joint, joint_arrivals


def _controls(definition):
    if definition.get("depth") is not None:
        return np.array([float(definition["depth"])])
    return np.asarray(definition["points"], float)[:, 1].copy()


def _set_controls(definition, values):
    if definition.get("depth") is not None:
        definition["depth"] = float(values[0])
    else:
        points = np.asarray(definition["points"], float).copy()
        points[:, 1] = values
        definition["points"] = points.tolist()


def _groups(grid, branches, max_shift):
    groups = {}
    for phase, definition in branches.items():
        if definition.get("kind") == "first":
            continue
        interface_nodes(grid, definition)  # Validate before searching.
        name = str(definition.get("interface", phase))
        controls = _controls(definition)
        if len(controls) > 20:
            raise ValueError("interface inversion supports at most 20 depth controls per interface")
        bounds = np.asarray(definition.get("depth_bounds", [0, grid.z[-2]]), float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"interface {name!r}: depth_bounds must be finite [minimum, maximum]")
        low, high = np.maximum(controls-max_shift, bounds[0]), np.minimum(controls+max_shift, bounds[1])
        if np.any(controls < low) or np.any(controls > high):
            raise ValueError(f"interface {name!r}: starting depths lie outside the bounds")
        key = ("depth",) if definition.get("depth") is not None else tuple(np.asarray(definition["points"])[:, 0])
        if name not in groups:
            groups[name] = dict(phases=[phase], initial=controls, key=key, low=low, high=high,
                                active=bool(definition.get("invert_depth", True)))
        else:
            group = groups[name]
            if group["key"] != key or not np.array_equal(group["initial"], controls):
                raise ValueError(f"shared interface {name!r} must have identical geometry in every phase")
            group["low"] = np.maximum(group["low"], low)
            group["high"] = np.minimum(group["high"], high)
            group["active"] &= bool(definition.get("invert_depth", True))
            group["phases"].append(phase)
    if not any(group["active"] for group in groups.values()):
        raise ValueError("select at least one reflected/head-wave interface to update")
    return sorted(groups.items(), key=lambda item: float(np.mean(item[1]["initial"])))


def _surfaces(grid, branches, groups):
    depths = [grid.surface]
    for _, group in groups:
        depths.append(interface_nodes(grid, branches[group["phases"][0]])[1])
    depths.append(np.full(grid.nx, grid.z[-1]))
    surfaces = np.asarray(depths)
    if np.any(np.diff(surfaces, axis=0) < grid.dz-1e-7):
        raise ValueError("inverted interfaces must not cross or touch; phases on the same boundary "
                         "must use a shared interface ID")
    return surfaces


def _warp_velocity(grid, velocity, previous, candidate):
    """Preserve layer contrasts by remapping between ordered interface anchors."""
    result = np.empty_like(velocity)
    for column in range(grid.nx):
        mapped = np.interp(grid.z, candidate[:, column], previous[:, column])
        result[:, column] = np.interp(mapped, grid.z, velocity[:, column])
    return result


def invert_interfaces(grid, start, source, receiver, observed, phases, branches, *, iterations=8,
                      update_velocity=True, max_shift=50., max_step=10., warp_velocity=True,
                      smoothing=1., damping=.1, vmin=100., vmax=8000., report=None, log=None):
    """Alternate velocity and constrained interface-depth updates.

Each depth or x,z control is bounded around its initial value by max_shift and
optional per-interface depth_bounds. invert_depth=false fixes a boundary. A
shared interface ID ties reflected/head-wave branches to the same boundary.
warp_velocity moves velocity contrasts with boundaries, using other boundaries
as fixed anchors; disable it when holding an independently known velocity grid.
"""
    if not np.isfinite([max_shift, max_step]).all() or max_shift <= 0 or max_step < grid.dz:
        raise ValueError("interface shift must be positive and search step at least one depth-grid cell")
    if iterations < 1 or int(iterations) != iterations:
        raise ValueError("interface inversion requires a positive integer iteration count")
    if not np.isfinite([vmin, vmax]).all() or not 0 < vmin < vmax:
        raise ValueError("velocity bounds must be positive, finite and increasing")
    start = np.asarray(start, float)
    if start.shape != (grid.nz, grid.nx) or not np.isfinite(start).all() or np.any(start <= 0):
        raise ValueError("starting velocity must be a positive finite array matching the grid")
    branches = deepcopy(branches)
    initial_branches = deepcopy(branches)
    groups = _groups(grid, branches, max_shift)
    surfaces = _surfaces(grid, branches, groups)
    starting_surfaces = surfaces.copy()
    observed = np.asarray(observed, float)
    if observed.ndim != 1 or not len(observed) or len(observed) != len(source) or len(observed) != len(phases):
        raise ValueError("one observed time is required for every source, receiver and phase")
    sigma = np.array([branches[p].get("uncertainty_ms", 2)*.001 for p in phases])
    if not np.isfinite(observed).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("observations and positive uncertainties must be finite")
    velocity = np.clip(np.asarray(start, float), vmin, vmax)
    score = lambda times: float(np.mean(((observed-times)/sigma)**2))
    predicted, _, hits = joint_arrivals(grid, velocity, source, receiver, phases, branches)
    objective = [score(predicted)]
    rms = [float(np.sqrt(np.mean((observed-predicted)**2)))]
    history = [dict(update="initial", iteration=0, branches=deepcopy(branches), weighted_misfit=objective[0])]
    stop_reason = "iteration limit"

    def record(kind, iteration):
        objective.append(score(predicted))
        rms.append(float(np.sqrt(np.mean((observed-predicted)**2))))
        history.append(dict(update=kind, iteration=iteration+1, branches=deepcopy(branches),
                            weighted_misfit=objective[-1]))
        if log:
            log(f"Joint {kind}: RMS {rms[-1]*1000:.3f} ms; weighted misfit {objective[-1]:.6g}")

    cells = max(1, int(np.floor(max_step/grid.dz)))
    steps = []
    while cells >= 1:
        steps.append(cells*grid.dz)
        if cells == 1:
            break
        cells = max(1, cells//2)
    for iteration in range(iterations):
        changed = False
        if report:
            report(iteration/max(iterations, 1))
        for name, group in groups:
            if not group["active"]:
                continue
            phase = group["phases"][0]
            for step in steps:
                for control in range(len(group["initial"])):
                    best = None
                    for sign in (-1, 1):
                        if report:
                            report(iteration/max(iterations, 1))
                        values = _controls(branches[phase])
                        values[control] += sign*step
                        if np.any(values < group["low"]) or np.any(values > group["high"]):
                            continue
                        trial_branches = deepcopy(branches)
                        for tied in group["phases"]:
                            _set_controls(trial_branches[tied], values)
                        try:
                            trial_surfaces = _surfaces(grid, trial_branches, groups)
                            if np.array_equal(trial_surfaces, surfaces):
                                continue
                            trial_velocity = (_warp_velocity(grid, velocity, surfaces, trial_surfaces)
                                              if warp_velocity else velocity)
                            times = joint_arrivals(grid, trial_velocity, source, receiver, phases,
                                                   trial_branches, rays=False,
                                                   report=(lambda _p: report(iteration/iterations)) if report else None)[0]
                        except ValueError:
                            continue  # Invalid candidate, not discarded observations.
                        value = score(times)
                        if value < objective[-1]-max(1e-12, objective[-1]*1e-10) and (best is None or value < best[0]):
                            best = value, trial_branches, trial_surfaces, trial_velocity, times
                    if best is not None:
                        _, branches, surfaces, velocity, predicted = best
                        record(f"interface {name}, control {control+1}", iteration)
                        changed = True
        if update_velocity:
            fit = invert_joint(grid, velocity, source, receiver, observed, phases, branches, iterations=1,
                               smoothing=smoothing, damping=damping, vmin=vmin, vmax=vmax,
                               report=(lambda _p: report(iteration/iterations)) if report else None)
            if score(fit["t_pred"]) < objective[-1]-max(1e-12, objective[-1]*1e-10):
                velocity, predicted = fit["velocity"], fit["t_pred"]
                record("velocity", iteration)
                changed = True
        if not changed:
            stop_reason = "no improving grid-resolved velocity/interface update"
            break
    predicted, _, hits = joint_arrivals(grid, velocity, source, receiver, phases, branches)
    if report:
        report(1)
    return dict(velocity=velocity, t_pred=predicted, observed=observed, hits=hits, rms=rms,
                weighted_misfit=objective, phases=list(phases), uncertainty=sigma, branches=branches,
                starting_branches=initial_branches, interface_history=history, stop_reason=stop_reason,
                interface_grid_step_m=grid.dz, ray_stencil_radius=grid.radius,
                sampled_interfaces={name: surfaces[index+1].copy() for index, (name, _) in enumerate(groups)},
                starting_sampled_interfaces={name: starting_surfaces[index+1].copy() for index, (name, _) in enumerate(groups)},
                interface_velocity_remap=bool(warp_velocity),
                phase_rms={p: float(np.sqrt(np.mean((observed[np.asarray(phases)==p]-
                                                   predicted[np.asarray(phases)==p])**2)))
                           for p in dict.fromkeys(phases)}, used=np.ones(len(observed), bool))
