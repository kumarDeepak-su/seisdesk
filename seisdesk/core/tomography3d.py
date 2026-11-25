"""Three-dimensional shortest-path arrival tomography.

Edges integrate nodal slowness by the trapezoidal rule (Moser, 1991). Reflected
legs remain above an assigned interface; head waves additionally travel along
its faster side. The path-length derivative is exact for this discrete graph,
away from path switches. Accuracy must be checked against grid/stencil refinement.
Coordinates and velocities are metres and m/s; arrival times are seconds.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import gcd, prod

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import dijkstra

from .tomography import _path, damped_update


class NodeGrid3D:
    """A z,y,x node grid with a ground-depth surface at every y,x location."""

    def __init__(self, shape, spacing, origin=(0., 0., 0.), surface=None, radius=2, memory_gb=2.):
        if len(shape) != 3 or any(int(n) != n or n < 2 for n in shape):
            raise ValueError("3D tomography needs at least two nodes per axis")
        self.shape = tuple(map(int, shape))
        self.spacing, self.origin = np.asarray(spacing, float), np.asarray(origin, float)
        if (self.spacing.shape != (3,) or self.origin.shape != (3,) or
                not np.isfinite(np.r_[self.spacing, self.origin]).all() or np.any(self.spacing <= 0)):
            raise ValueError("give finite z,y,x origins and positive spacings")
        if int(radius) != radius or not 1 <= radius <= 5:
            raise ValueError("ray stencil radius must be an integer from 1 to 5")
        if not np.isfinite(memory_gb) or memory_gb <= 0:
            raise ValueError("graph memory budget must be positive")
        self.radius, self.memory_gb = int(radius), float(memory_gb)
        self.size = prod(self.shape)
        offsets = [d for d in product(range(-self.radius, self.radius+1), repeat=3)
                   if d != (0, 0, 0) and gcd(gcd(abs(d[0]), abs(d[1])), abs(d[2])) == 1
                   and next(v for v in d if v) > 0 and all(abs(v) < n for v, n in zip(d, self.shape))]
        edge_bound = 2*sum(prod(n-abs(d) for n, d in zip(self.shape, shift)) for shift in offsets)
        self.estimated_graph_bytes = int(edge_bound*96 + self.size*256)
        if self.estimated_graph_bytes > self.memory_gb*1024**3:
            raise ValueError(f"3D ray graph needs approximately {self.estimated_graph_bytes/1024**3:.2f} GiB; "
                             "coarsen/crop the grid, reduce the stencil or increase its memory budget")
        self.axes = tuple(o + np.arange(n)*d for o, d, n in zip(self.origin, self.spacing, self.shape))
        self.surface = (np.full(self.shape[1:], self.origin[0]) if surface is None else np.asarray(surface, float))
        if (self.surface.shape != self.shape[1:] or not np.isfinite(self.surface).all() or
                np.any(self.surface < self.axes[0][0]) or np.any(self.surface > self.axes[0][-1])):
            raise ValueError("ground surface must have shape (ny,nx) and lie inside the depth grid")
        node = np.arange(self.size).reshape(self.shape)
        aa, bb, lengths = [], [], []
        for shift in offsets:
            left = tuple(slice(max(0, -d), n-max(0, d)) for n, d in zip(self.shape, shift))
            right = tuple(slice(max(0, d), n-max(0, -d)) for n, d in zip(self.shape, shift))
            a, b = node[left].ravel(), node[right].ravel()
            keep = self._inside_surface(a, b, self.surface, above=False)
            aa.append(a[keep]); bb.append(b[keep])
            lengths.append(np.full(int(keep.sum()), np.linalg.norm(np.asarray(shift)*self.spacing)))
        if not aa or not sum(len(a) for a in aa):
            raise ValueError("the ground mask has no connected ray edges")
        a, b, length = np.concatenate(aa), np.concatenate(bb), np.concatenate(lengths)
        self.edge_a, self.edge_b, self.edge_len = np.r_[a, b], np.r_[b, a], np.r_[length, length]

    def coordinates(self, nodes):
        return np.column_stack(np.unravel_index(np.asarray(nodes, int).reshape(-1), self.shape))*self.spacing + self.origin

    def _surface_at(self, surface, yz):
        # yz is a pair of floating y/x grid indices, not physical coordinates.
        return map_coordinates(surface, yz, order=1, mode="nearest", prefilter=False)

    def _inside_surface(self, a, b, surface, *, above):
        ac = np.array(np.unravel_index(a, self.shape), float)
        bc = np.array(np.unravel_index(b, self.shape), float)
        keep = np.ones(len(a), bool)
        for fraction in np.linspace(0, 1, 2*self.radius+1):
            point = ac*(1-fraction) + bc*fraction
            depth = self.origin[0] + self.spacing[0]*point[0]
            boundary = self._surface_at(surface, point[1:])
            keep &= depth <= boundary+1e-7*self.spacing[0] if above else depth >= boundary-1e-7*self.spacing[0]
        return keep

    def ground(self):
        return self.axes[0][:, None, None] >= self.surface[None] - 1e-7*self.spacing[0]

    def nodes_for(self, coordinates):
        coordinates = np.asarray(coordinates, float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
            raise ValueError("source and receiver coordinates need finite z,y,x rows")
        relative = (coordinates-self.origin)/self.spacing
        if np.any(relative < -1e-7) or np.any(relative > np.asarray(self.shape)-1+1e-7):
            raise ValueError("arrival geometry lies outside the 3D model")
        if np.any(coordinates[:, 0] < self._surface_at(self.surface, relative[:, 1:].T)-1e-6):
            raise ValueError("arrival geometry lies above the ground surface")
        indices = np.rint(relative).astype(int)
        first_ground = np.ceil((self.surface[indices[:, 1], indices[:, 2]]-self.origin[0])/self.spacing[0]-1e-7).astype(int)
        indices[:, 0] = np.maximum(indices[:, 0], first_ground)
        nodes = np.ravel_multi_index(indices.T, self.shape)
        return nodes, np.linalg.norm(self.coordinates(nodes)-coordinates, axis=1)

    def graph(self, velocity, boundary=None):
        v = np.asarray(velocity, float)
        if v.shape != self.shape or not np.isfinite(v).all() or np.any(v <= 0):
            raise ValueError("give a finite positive velocity at every 3D node")
        a, b, length = self.edge_a, self.edge_b, self.edge_len
        if boundary is not None:
            keep = self._inside_surface(a, b, boundary, above=True)
            a, b, length = a[keep], b[keep], length[keep]
        slow = 1/v.ravel()
        return csr_matrix((.5*length*(slow[a]+slow[b]), (a, b)), shape=(self.size, self.size))

    def interface_surface(self, definition):
        """Validate and return the nominal, unrounded depth surface."""
        if "surface" in definition:
            depth = np.asarray(definition["surface"], float)
        else:
            depth0 = float(definition.get("depth", np.nan))
            yy, xx = np.meshgrid(self.axes[1], self.axes[2], indexing="ij")
            depth = (depth0 + float(definition.get("slope_x", 0))*(xx-self.origin[2]) +
                     float(definition.get("slope_y", 0))*(yy-self.origin[1]))
        if (depth.shape != self.shape[1:] or not np.isfinite(depth).all() or
                np.any(depth < self.surface+self.spacing[0]) or np.any(depth > self.axes[0][-2])):
            raise ValueError("interface must cover (ny,nx), lie a cell below ground and leave one node beneath it")
        return depth

    def interface(self, definition):
        depth = self.interface_surface(definition)
        rows = np.rint((depth-self.origin[0])/self.spacing[0]).astype(int)
        yy, xx = np.indices(self.shape[1:])
        nodes = np.ravel_multi_index((rows, yy, xx), self.shape).ravel()
        return nodes, self.origin[0]+rows*self.spacing[0]


def roughness3d(active, spacing):
    """First/second differences with physical grid-aspect scaling, on active nodes."""
    index = np.full(active.shape, -1, dtype=int)
    index[active] = np.arange(active.sum())
    matrices = []
    for order in (2, 1):
        rows, columns, values = [], [], []
        start = 0
        for axis in range(3):
            if active.shape[axis] <= order:
                continue
            terms = []
            for offset in range(order+1):
                slices = [slice(None)]*3
                slices[axis] = slice(offset, active.shape[axis]-order+offset)
                terms.append(index[tuple(slices)].ravel())
            keep = np.logical_and.reduce([t >= 0 for t in terms])
            rr = np.arange(int(keep.sum())) + start
            weight = (min(spacing)/spacing[axis])**order
            for term, coefficient in zip(terms, (1., -2., 1.) if order == 2 else (-1., 1.)):
                rows.append(rr); columns.append(term[keep]); values.append(np.full(len(rr), coefficient*weight))
            start += len(rr)
        matrices.append(coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                                   shape=(start, int(active.sum()))).tocsr() if rows else csr_matrix((0, int(active.sum()))))
    return tuple(matrices)


def _head_graph(grid, velocity, boundary):
    ny, nx = grid.shape[1:]
    below = boundary + ny*nx
    v = np.asarray(velocity).ravel()
    if np.any(v[below] <= v[boundary-ny*nx]):
        raise ValueError("head waves require a faster layer beneath the entire assigned interface")
    nodes = np.arange(ny*nx).reshape(ny, nx)
    aa, bb, lengths = [], [], []
    for dy, dx in product(range(-grid.radius, grid.radius+1), repeat=2):
        if (dy, dx) == (0, 0) or gcd(abs(dy), abs(dx)) != 1 or abs(dy) >= ny or abs(dx) >= nx:
            continue
        a = nodes[max(0, -dy):ny-max(0, dy), max(0, -dx):nx-max(0, dx)].ravel()
        b = nodes[max(0, dy):ny-max(0, -dy), max(0, dx):nx-max(0, -dx)].ravel()
        aa.append(a); bb.append(b)
        lengths.append(np.linalg.norm(grid.coordinates(boundary[a])-grid.coordinates(boundary[b]), axis=1))
    a, b, length = np.concatenate(aa), np.concatenate(bb), np.concatenate(lengths)
    weights = .5*length*(1/v[below[a]]+1/v[below[b]])
    return a, b, weights, below


def arrivals3d(grid, velocity, source, receiver, phases, branches, *, rays=True, report=None, interface_hits=None):
    """All selected first/reflected/head arrivals, plus their discrete slowness derivative."""
    source, receiver, phases = np.asarray(source), np.asarray(receiver), np.asarray(phases, str)
    if (source.ndim != 1 or receiver.shape != source.shape or phases.shape != source.shape or not len(source) or
            not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(receiver.dtype, np.integer)):
        raise ValueError("give one integer source node, receiver node and phase per arrival")
    if min(source.min(), receiver.min()) < 0 or max(source.max(), receiver.max()) >= grid.size:
        raise ValueError("arrival node index is outside the model")
    if not np.all(grid.ground().ravel()[np.r_[source, receiver]]):
        raise ValueError("arrival nodes must be at or below ground")
    times = np.full(len(source), np.nan)
    rows, columns, lengths = [], [], []

    def add_path(row, path):
        if path is None:
            raise ValueError("an assigned arrival path is unreachable")
        if len(path) < 2:
            return
        segment = np.linalg.norm(np.diff(grid.coordinates(path), axis=0), axis=1)
        rows.append(np.full(2*len(segment), row, dtype=int))
        columns.append(np.r_[path[:-1], path[1:]])
        lengths.append(.5*np.r_[segment, segment])

    completed = 0
    for phase in dict.fromkeys(phases):
        definition = branches.get(str(phase), {})
        kind = definition.get("kind")
        if kind not in ("first", "reflection", "head"):
            raise ValueError(f"phase {phase!r} needs a first, reflection or head path")
        boundary, surface = (None, None) if kind == "first" else grid.interface(definition)
        if rays and interface_hits is not None and kind != "first":
            interface_hits[str(phase)] = np.zeros(grid.shape[1:], dtype=int)
        graph = grid.graph(velocity, surface)

        @lru_cache(maxsize=8)
        def field(node):
            return dijkstra(graph, directed=True, indices=int(node), return_predecessors=True)

        if kind == "head":
            ha, hb, hw, below = _head_graph(grid, velocity, boundary)
            nb = len(boundary)

            @lru_cache(maxsize=4)
            def head_field(node):
                distance = field(node)[0][boundary]
                good = np.flatnonzero(np.isfinite(distance))
                seeded = csr_matrix((np.r_[hw, distance[good]],
                                     (np.r_[ha, np.full(len(good), nb)], np.r_[hb, good])), shape=(nb+1, nb+1))
                return dijkstra(seeded, directed=True, indices=nb, return_predecessors=True)

        for row in np.flatnonzero(phases == phase):
            if report:
                report(completed/len(source))
            ds, ps = field(int(source[row]))
            if kind == "first":
                if source[row] == receiver[row]:
                    raise ValueError("source and receiver coincide for a first-arrival pick; refine or exclude that pick")
                times[row] = ds[receiver[row]]
                if rays:
                    add_path(row, _path(ps, int(source[row]), int(receiver[row])))
            else:
                dr, pr = field(int(receiver[row]))
                if kind == "reflection":
                    costs = ds[boundary]+dr[boundary]
                    entry = exit_node = int(np.argmin(costs))
                else:
                    hd, hp = head_field(int(source[row]))
                    costs = hd[:nb]+dr[boundary]
                    exit_node = int(np.argmin(costs))
                    head_path = _path(hp, nb, exit_node)
                    if head_path is None or len(head_path) < 3:
                        raise ValueError(f"phase {phase!r} is subcritical or has no nonzero head-wave segment")
                    head_path = head_path[1:]
                    entry = int(head_path[0])
                times[row] = costs[exit_node]
                if rays:
                    if interface_hits is not None:
                        contact = [entry] if kind == "reflection" else np.unique(head_path)
                        np.add.at(interface_hits[str(phase)].ravel(), contact, 1)
                    add_path(row, _path(ps, int(source[row]), int(boundary[entry])))
                    add_path(row, _path(pr, int(receiver[row]), int(boundary[exit_node])))
                    if kind == "head":
                        add_path(row, below[head_path])
            completed += 1
    if not np.isfinite(times).all():
        raise ValueError("some 3D arrivals are unreachable; no selected picks were discarded")
    matrix = None
    hits = np.zeros(grid.shape)
    if rays:
        matrix = (coo_matrix((np.concatenate(lengths), (np.concatenate(rows), np.concatenate(columns))),
                             shape=(len(source), grid.size)).tocsr() if rows else csr_matrix((len(source), grid.size)))
        hits = np.asarray((matrix > 0).sum(axis=0)).reshape(grid.shape)
    if report:
        report(1.)
    return times, matrix, hits


def invert3d(grid, start, source, receiver, observed, phases, branches, *, uncertainty=None,
             iterations=8, smoothing=1., damping=.1, vmin=100., vmax=8000.,
             bootstrap=0, seed=0, report=None, log=None, _noise_realization=False):
    """Weighted, bounded slowness inversion and optional timing-noise perturbation ensemble.

    Ensemble intervals are conditional on the starting solution, assigned paths,
    fixed interfaces, grid and regularisation. They are not geological uniqueness
    or total uncertainty estimates. Cells without final ray coverage are masked.
    """
    observed, phases = np.asarray(observed, float), np.asarray(phases, str)
    start = np.asarray(start, float)
    if (observed.ndim != 1 or len(observed) < 4 or phases.shape != observed.shape or
            not np.isfinite(observed).all() or (not _noise_realization and np.any(observed <= 0))):
        raise ValueError("3D tomography requires at least four finite positive picked times")
    if (not np.isfinite([vmin, vmax, smoothing, damping]).all() or vmin <= 0 or vmax <= vmin or
            smoothing < 0 or damping <= 0 or int(iterations) != iterations or iterations < 1):
        raise ValueError("invalid velocity bounds, regularisation or iteration count")
    if start.shape != grid.shape or not np.isfinite(start).all() or np.any((start < vmin) | (start > vmax)):
        raise ValueError("the starting 3D velocity must lie inside the configured bounds")
    if int(bootstrap) != bootstrap or (bootstrap != 0 and not 3 <= bootstrap <= 100):
        raise ValueError("uncertainty replicates must be zero or an integer between 3 and 100")
    bootstrap = int(bootstrap)
    if bootstrap*grid.size*32 > grid.memory_gb*1024**3:
        raise ValueError("uncertainty ensemble exceeds the model memory budget; reduce replicates or grid size")
    sigma = (np.array([branches.get(str(p), {}).get("uncertainty_ms", 2)*.001 for p in phases])
             if uncertainty is None else np.asarray(uncertainty, float))
    if sigma.shape != observed.shape or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("give one finite positive timing uncertainty (seconds) per pick")
    active_mask = grid.ground()
    active = np.flatnonzero(active_mask.ravel())
    d2, d1 = roughness3d(active_mask, grid.spacing)

    def fit(initial, obs, part):
        def progress(fraction):
            if report:
                report((part+fraction)/(bootstrap+1))
        velocity = initial.copy()
        prediction, matrix, hits = arrivals3d(grid, velocity, source, receiver, phases, branches,
                                              report=lambda f: progress(.05*f))
        objective = [float(np.mean(((obs-prediction)/sigma)**2))]
        rms = [float(np.sqrt(np.mean((obs-prediction)**2)))]
        for iteration in range(int(iterations)):
            current_progress = .05+.9*iteration/iterations
            progress(current_progress)
            slow = 1/velocity.ravel()
            jacobian = diags(1/sigma) @ matrix[:, active] @ diags(slow[active])
            change = damped_update(jacobian, (obs-prediction)/sigma, d2, d1,
                                   smoothing=smoothing, damping=damping, max_change=.2)
            accepted = False
            for step in (1., .5, .25, .125, .0625, .03125):
                trial = velocity.ravel().copy()
                trial[active] = np.clip(1/(slow[active]*(1+step*change)), vmin, vmax)
                trial = trial.reshape(grid.shape)
                try:
                    tp, _, _ = arrivals3d(grid, trial, source, receiver, phases, branches, rays=False,
                                          report=lambda f: progress(current_progress))
                except ValueError:
                    continue
                value = float(np.mean(((obs-tp)/sigma)**2))
                if value < objective[-1]:
                    velocity = trial
                    prediction, matrix, hits = arrivals3d(grid, velocity, source, receiver, phases, branches,
                                                          report=lambda f: progress(current_progress))
                    objective.append(value); rms.append(float(np.sqrt(np.mean((obs-prediction)**2))))
                    accepted = True
                    if log:
                        log(f"3D tomography fit {part+1}, iteration {iteration+1}: RMS {rms[-1]*1000:.4g} ms")
                    break
            if not accepted:
                break
        progress(1.)
        return velocity, prediction, hits, objective, rms

    velocity, prediction, hits, objective, rms = fit(start, observed, 0)
    residual = prediction-observed
    result = dict(velocity=velocity, starting_model=start.copy(), t_pred=prediction, observed=observed.copy(),
                  residual=residual, standardized_residual=residual/sigma, outlier=np.abs(residual/sigma)>3,
                  uncertainty=sigma, phases=phases.tolist(), branches=branches, hits=hits, rms=rms,
                  weighted_misfit=objective, used=np.ones(len(observed), bool), axes=grid.axes,
                  spacing=grid.spacing.copy(), origin=grid.origin.copy(), surface=grid.surface.copy(),
                  graph_bytes_estimate=grid.estimated_graph_bytes,
                  phase_rms={str(p): float(np.sqrt(np.mean(residual[phases==p]**2))) for p in dict.fromkeys(phases)})
    if bootstrap:
        rng = np.random.default_rng(seed)
        samples = []
        for i in range(bootstrap):
            sample = fit(velocity, observed+rng.normal(size=len(observed))*sigma, i+1)[0]
            samples.append(sample)
        samples = np.stack(samples)
        for name, field in (("velocity_std", samples.std(axis=0, ddof=1)),
                            ("velocity_p05", np.quantile(samples, .05, axis=0)),
                            ("velocity_p95", np.quantile(samples, .95, axis=0))):
            result[name] = np.where(hits > 0, field, np.nan)
        result["uncertainty_ensemble"] = dict(replicates=bootstrap, seed=int(seed),
            assumption="Independent Gaussian timing errors; fixed interfaces, grid, regularisation and starting solution")
    return result
