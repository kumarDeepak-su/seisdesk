"""Joint 2D first, reflected and head-wave traveltimes on specified interfaces.

Reflection rays minimise source-to-interface plus interface-to-receiver time,
with both bent legs restricted above that interface. Head waves also traverse
the fast side of the interface. Interfaces are fixed and sampled on the node
grid. This does not interpret a pick's geological branch automatically or invert
interface depths. Every row retains its phase and trace, including later arrivals.
"""
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import dijkstra

from .tomography import _path, damped_update, first_arrivals, roughness_operators


def interface_nodes(grid, definition):
    value = definition.get("depth")
    if np.isscalar(value) and value is not None:
        depth = np.full(grid.nx, float(value))
    else:
        points = np.asarray(definition.get("points", []), float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("an interface needs depth (m) or points [[x,z], ...]")
        if not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) <= 0):
            raise ValueError("interface points must be finite with strictly increasing x")
        if points[0, 0] > grid.x[0] or points[-1, 0] < grid.x[-1]:
            raise ValueError("interface points must cover the entire tomography grid")
        depth = np.interp(grid.x, points[:, 0], points[:, 1])
    if not np.isfinite(depth).all() or np.any(depth <= grid.surface + grid.dz) or np.any(depth > grid.z[-2]):
        raise ValueError("interfaces must be at least one cell below ground and above the last model row")
    rows = np.rint(depth / grid.dz).astype(int)
    return rows * grid.nx + np.arange(grid.nx), rows * grid.dz


def _restricted_graph(grid, velocity, depth):
    a, b, length = grid.edge_a, grid.edge_b, grid.edge_len
    za, xa = np.divmod(a, grid.nx)
    zb, xb = np.divmod(b, grid.nx)
    keep = np.ones(len(a), bool)
    for fraction in np.linspace(0, 1, 2 * grid.radius + 1):
        x = xa * (1 - fraction) + xb * fraction
        z = (za * (1 - fraction) + zb * fraction) * grid.dz
        keep &= z <= np.interp(x, np.arange(grid.nx), depth) + 1e-7
    slow = 1 / np.asarray(velocity).ravel()
    weight = length[keep] * .5 * (slow[a[keep]] + slow[b[keep]])
    return csr_matrix((weight, (a[keep], b[keep])), shape=(grid.size, grid.size))


def joint_arrivals(grid, velocity, source, receiver, phases, branches, rays=True, report=None):
    source, receiver = np.asarray(source, int), np.asarray(receiver, int)
    phases = np.asarray(phases, str)
    n = len(source)
    if receiver.shape != source.shape or phases.shape != source.shape:
        raise ValueError("one source, receiver and phase per arrival is required")
    times = np.full(n, np.nan)
    row_list, col_list, length_list = [], [], []
    hits = np.zeros(grid.size)

    def add_path(row, path):
        if path is None or len(path) < 2:
            return
        z, x = np.divmod(path, grid.nx)
        length = np.hypot(np.diff(z) * grid.dz, np.diff(x) * grid.dx)
        row_list.append(np.full(2 * len(length), row))
        col_list.append(np.r_[path[:-1], path[1:]])
        length_list.append(.5 * np.r_[length, length])

    completed = 0
    for phase in dict.fromkeys(phases):
        if phase not in branches:
            raise ValueError(f"missing path definition for phase {phase!r}")
        definition = branches[phase]
        kind = definition.get("kind", "")
        records = np.flatnonzero(phases == phase)
        if kind == "first":
            if np.any(source[records] == receiver[records]):
                raise ValueError(f"phase {phase!r}: source and receiver coincide on the grid; "
                                 "refine the grid or exclude zero-offset first-arrival picks")
            t, matrix, _ = first_arrivals(grid, velocity, source[records], receiver[records], rays=rays)
            times[records] = t
            if rays:
                coo = matrix.tocoo()
                row_list.append(records[coo.row]); col_list.append(coo.col); length_list.append(coo.data)
            completed += len(records)
            continue
        if kind not in ("reflection", "head"):
            raise ValueError(f"phase {phase!r}: kind must be first, reflection or head")
        boundary, depth = interface_nodes(grid, definition)
        graph = _restricted_graph(grid, velocity, depth)
        below = boundary + grid.nx
        if kind == "head":
            if grid.nx > 4000:
                raise ValueError("head-wave interface has over 4000 nodes; coarsen or crop the 2D line")
            # Velocity directly below the interface must exceed that directly above.
            above = boundary - grid.nx
            if np.any(np.asarray(velocity).ravel()[below] <= np.asarray(velocity).ravel()[above]):
                raise ValueError(f"phase {phase!r}: head waves require a faster layer below the interface")
            lengths = np.hypot(grid.dx, np.diff(depth))
            slow = 1 / np.asarray(velocity).ravel()[below]
            arc = np.r_[0, np.cumsum(lengths * .5 * (slow[:-1] + slow[1:]))]
            along = np.abs(arc[:, None] - arc[None, :])
            # Two endpoint distance arrays plus this matrix bound per-branch memory.

        @lru_cache(maxsize=8)
        def field(node):
            return dijkstra(graph, directed=True, indices=int(node), return_predecessors=True)

        for row in records:
            if report:
                report(completed / max(n, 1))
            ds, ps = field(int(source[row]))
            dr, pr = field(int(receiver[row]))
            if kind == "reflection":
                costs = ds[boundary] + dr[boundary]
                entry = exit_node = int(np.argmin(costs))
                travel = costs[entry]
            else:
                costs = ds[boundary, None] + dr[boundary][None, :] + along
                np.fill_diagonal(costs, np.inf)
                entry, exit_node = np.unravel_index(np.argmin(costs), costs.shape)
                travel = costs[entry, exit_node]
                if abs(entry - exit_node) < 2:
                    raise ValueError(f"phase {phase!r}, pick {row}: no resolved head-wave leg; "
                                     "check critical distance, branch assignment or grid spacing")
            if not np.isfinite(travel):
                raise ValueError(f"phase {phase!r}, pick {row}: interface is unreachable from this geometry")
            times[row] = travel
            if rays:
                add_path(row, _path(ps, int(source[row]), int(boundary[entry])))
                add_path(row, _path(pr, int(receiver[row]), int(boundary[exit_node])))
                if kind == "head":
                    lo, hi = sorted((entry, exit_node))
                    # Head-leg geometry follows the interface; its slowness is sampled below it.
                    nodes = below[lo:hi+1]
                    length = lengths[lo:hi]
                    row_list.append(np.full(2 * len(length), row))
                    col_list.append(np.r_[nodes[:-1], nodes[1:]])
                    length_list.append(.5 * np.r_[length, length])
            completed += 1
    matrix = None
    if rays:
        matrix = coo_matrix((np.concatenate(length_list), (np.concatenate(row_list), np.concatenate(col_list))),
                            shape=(n, grid.size)).tocsr() if row_list else csr_matrix((n, grid.size))
        hits = np.asarray((matrix > 0).sum(axis=0)).ravel()
    if not np.isfinite(times).all():
        raise ValueError("some arrivals are unreachable; no picks have been silently discarded")
    return times, matrix, hits.reshape(grid.nz, grid.nx)


def invert_joint(grid, start, source, receiver, observed, phases, branches, *, iterations=8,
                 smoothing=1, damping=.1, vmin=100, vmax=8000, report=None, log=None):
    observed = np.asarray(observed, float)
    sigma = np.array([branches[phase].get("uncertainty_ms", 2) * .001 for phase in phases])
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("phase uncertainties must be positive and finite")
    active_mask = grid.ground()
    active = np.flatnonzero(active_mask.ravel())
    d2, d1 = roughness_operators(active_mask)
    v = np.clip(np.asarray(start, float), vmin, vmax)
    predicted, matrix, hits = joint_arrivals(grid, v, source, receiver, phases, branches, report=report)
    rms = [float(np.sqrt(np.mean((observed - predicted)**2)))]
    objective = [float(np.mean(((observed-predicted)/sigma)**2))]
    for iteration in range(iterations):
        if report:
            report(iteration / iterations)
        slow = 1 / v.ravel()
        jacobian = diags(1 / sigma) @ matrix[:, active] @ diags(slow[active])
        delta = damped_update(jacobian, (observed-predicted)/sigma, d2, d1,
                              smoothing=smoothing, damping=damping, max_change=.2)
        accepted = False
        for step in (1, .5, .25, .125, .0625):
            trial = v.ravel().copy()
            trial[active] = np.clip(1/(slow[active]*(1+step*delta)), vmin, vmax)
            trial = trial.reshape(v.shape)
            try:
                tp, _, _ = joint_arrivals(grid, trial, source, receiver, phases, branches, rays=False)
            except ValueError:
                continue  # A trial may destroy a head-wave velocity contrast; it is not accepted.
            obj = float(np.mean(((observed-tp)/sigma)**2))
            if obj < objective[-1]:
                v = trial
                predicted, matrix, hits = joint_arrivals(grid, v, source, receiver, phases, branches)
                objective.append(obj)
                rms.append(float(np.sqrt(np.mean((observed-predicted)**2))))
                accepted = True
                if log:
                    log(f"Joint tomography {iteration+1}: RMS {rms[-1]*1000:.3f} ms; weighted misfit {obj:.5g}")
                break
        if not accepted:
            break
    if report:
        report(1)
    phase_rms = {p: float(np.sqrt(np.mean((observed[np.asarray(phases)==p]-predicted[np.asarray(phases)==p])**2)))
                 for p in dict.fromkeys(phases)}
    return dict(velocity=v, t_pred=predicted, observed=observed, hits=hits, rms=rms,
                weighted_misfit=objective, phases=list(phases), phase_rms=phase_rms,
                uncertainty=sigma, branches=branches, used=np.ones(len(observed), bool))
