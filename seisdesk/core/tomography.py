"""Traveltime tomography on a 2D node grid.

First arrivals and rays come from shortest paths through a graph of grid nodes (Moser 1991): every node is linked to
its neighbours within a stencil radius (radius 3 gives 32 directions), an edge costs its length times the mean slowness
of its two nodes, and Dijkstra's algorithm returns the traveltime field of a source together with the predecessor of
every node, so the ray to a receiver is the chain of predecessors back to the source. An edge's cost is linear in the
nodal slownesses, so the ray-length (Frechet) matrix is exact for these traveltimes: half of every edge's length goes
to each of its two nodes.

Depth z is measured down from a reference elevation (the highest ground), so sources and receivers can sit on
topography. Nodes above the ground are air: they get a very low velocity, no ray crosses them, and they are never
updated. Updates solve for the relative slowness change by damped, smoothed least squares (LSQR, Paige & Saunders
1982) with second- and first-difference roughness penalties weighted separately along and across the line, as in
regularised refraction tomography (Zhang & Toksoz 1998).
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, identity, vstack
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsqr

AIR_FACTOR = 1e-3          # air velocity as a fraction of the lowest ground velocity


def stencil(radius: int) -> list[tuple[int, int]]:
    """Node offsets (row, column) with coprime components inside the radius: the graph's edge directions."""
    out = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if (di, dj) != (0, 0) and gcd(abs(di), abs(dj)) == 1:
                out.append((di, dj))
    return out


class NodeGrid:
    """Regular node grid: columns at x0 + j dx (j < nx), rows at z = i dz (i < nz) below the reference elevation, and
    the depth of the ground below the reference at every column (zero everywhere for flat ground)."""

    def __init__(self, x0: float, dx: float, nx: int, dz: float, nz: int, surface=None, radius: int = 3):
        self.x0, self.dx, self.nx, self.dz, self.nz = float(x0), float(dx), int(nx), float(dz), int(nz)
        if not np.isfinite([self.dx, self.dz, self.x0]).all() or min(self.dx, self.dz) <= 0:
            raise ValueError("grid spacing must be finite and positive")
        if self.nx < 2 or self.nz < 2:
            raise ValueError("the tomography grid needs at least two rows and two columns")
        self.surface = np.zeros(self.nx) if surface is None else np.asarray(surface, dtype=np.float64).reshape(-1)
        if self.surface.size != self.nx:
            raise ValueError("give one ground depth per grid column")
        self.radius = max(1, int(radius))
        self._edges()

    @property
    def size(self) -> int:
        return self.nz * self.nx

    @property
    def x(self) -> np.ndarray:
        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def z(self) -> np.ndarray:
        return np.arange(self.nz) * self.dz

    def ground(self) -> np.ndarray:
        """(nz, nx) True for nodes at or below the ground."""
        return self.z[:, None] >= self.surface[None, :] - 1e-6 * self.dz

    def _edges(self) -> None:
        node = np.arange(self.size).reshape(self.nz, self.nx)
        a_list, b_list, l_list = [], [], []
        for di, dj in stencil(self.radius):
            if abs(di) >= self.nz or abs(dj) >= self.nx:
                continue
            zs = slice(max(0, -di), self.nz - max(0, di))
            xs = slice(max(0, -dj), self.nx - max(0, dj))
            zt = slice(max(0, di), self.nz - max(0, -di))
            xt = slice(max(0, dj), self.nx - max(0, -dj))
            a = node[zs, xs].ravel()
            if a.size == 0:
                continue
            a_list.append(a)
            b_list.append(node[zt, xt].ravel())
            l_list.append(np.full(a.size, np.hypot(di * self.dz, dj * self.dx)))
        self.edge_a = np.concatenate(a_list)
        self.edge_b = np.concatenate(b_list)
        self.edge_len = np.concatenate(l_list)
        # Long stencil edges must not cut through an air gap between two ground nodes.
        za, xa = np.divmod(self.edge_a, self.nx)
        zb, xb = np.divmod(self.edge_b, self.nx)
        valid = np.ones(self.edge_a.size, dtype=bool)
        for f in np.linspace(0, 1, 2 * self.radius + 1):
            x = (1 - f) * xa + f * xb
            z = ((1 - f) * za + f * zb) * self.dz
            valid &= z >= np.interp(x, np.arange(self.nx), self.surface) - 1e-6 * self.dz
        self.edge_a, self.edge_b, self.edge_len = (a[valid] for a in
                                                 (self.edge_a, self.edge_b, self.edge_len))

    def graph(self, velocity) -> csr_matrix:
        """Edge costs (traveltimes) for a node velocity field: length x mean slowness of the two nodes."""
        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if velocity.size != self.size or not np.isfinite(velocity).all() or np.any(velocity <= 0):
            raise ValueError("give a finite positive velocity at every grid node")
        slow = 1.0 / velocity
        w = 0.5 * self.edge_len * (slow[self.edge_a] + slow[self.edge_b])
        return csr_matrix((w, (self.edge_a, self.edge_b)), shape=(self.size, self.size))

    def column(self, x: float) -> int:
        return int(np.clip(np.rint((float(x) - self.x0) / self.dx), 0, self.nx - 1))

    def surface_node(self, x: float, depth: float = 0.0) -> int:
        """The node at or just below the ground (plus depth, e.g. a buried source) under position x."""
        col = self.column(x)
        row = int(np.clip(np.ceil((self.surface[col] + max(float(depth), 0.0)) / self.dz - 1e-6), 0, self.nz - 1))
        return row * self.nx + col

    def node_at(self, x: float, z: float) -> int:
        """The nearest node to (x, z), z below the reference elevation."""
        row = int(np.clip(np.rint(float(z) / self.dz), 0, self.nz - 1))
        return row * self.nx + self.column(x)

    def with_air(self, velocity) -> np.ndarray:
        """The velocity with the air nodes set very low, so that no path crosses the air."""
        v = np.array(velocity, dtype=np.float64, copy=True).reshape(self.nz, self.nx)
        g = self.ground()
        floor = float(v[g].min()) if g.any() else 1.0
        v[~g] = AIR_FACTOR * floor
        return v

    def fill_air(self, velocity) -> np.ndarray:
        """The velocity with every air node given the value of the first ground node below it (for a depth model
        that later steps sample anywhere)."""
        v = np.array(velocity, dtype=np.float64, copy=True).reshape(self.nz, self.nx)
        g = self.ground()
        for j in range(self.nx):
            rows = np.flatnonzero(g[:, j])
            if rows.size and rows[0] > 0:
                v[:rows[0], j] = v[rows[0], j]
        return v


def _path(pred_row: np.ndarray, src: int, rec: int) -> np.ndarray | None:
    """Nodes from src to rec along the predecessor chain of a shortest-path search from src (None if unreached)."""
    nodes = [rec]
    node = rec
    limit = pred_row.size
    while node != src:
        node = int(pred_row[node])
        if node < 0 or len(nodes) > limit:
            return None
        nodes.append(node)
    return np.array(nodes[::-1], dtype=np.int64)


def first_arrivals(grid: NodeGrid, velocity, src_nodes, rec_nodes, rays: bool = True, chunk: int = 32):
    """Shortest-path traveltimes of the node pairs (src_nodes[k], rec_nodes[k]).

    With rays, also the ray-length matrix (npairs x grid nodes; ray k has half of each of its edges on both end
    nodes) and the number of rays through every node, (nz, nx). Unreached pairs get NaN and an empty matrix row.
    Returns (times, matrix or None, hits)."""
    g = grid.graph(grid.with_air(velocity))
    src = np.asarray(src_nodes, dtype=np.int64).reshape(-1)
    rec = np.asarray(rec_nodes, dtype=np.int64).reshape(-1)
    n = src.size
    times = np.full(n, np.nan)
    hits = np.zeros(grid.size)
    rows, cols, vals = [], [], []
    if n == 0:
        return times, (csr_matrix((0, grid.size)) if rays else None), hits.reshape(grid.nz, grid.nx)
    usrc, inv = np.unique(src, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(usrc.size + 1))
    for start in range(0, usrc.size, chunk):
        batch = usrc[start:start + chunk]
        if rays:
            dist, pred = dijkstra(g, directed=True, indices=batch, return_predecessors=True)
        else:
            dist, pred = dijkstra(g, directed=True, indices=batch), None
        dist = np.atleast_2d(dist)
        for k in range(batch.size):
            u = start + k
            pairs = order[bounds[u]:bounds[u + 1]]
            t = dist[k, rec[pairs]]
            good = np.isfinite(t)
            times[pairs[good]] = t[good]
            if not rays:
                continue
            p_row = np.atleast_2d(pred)[k]
            s = int(batch[k])
            for p in pairs[good]:
                path = _path(p_row, s, int(rec[p]))
                if path is None or path.size < 2:
                    continue
                r_i, c_i = np.divmod(path, grid.nx)
                seg = np.hypot(np.diff(r_i) * grid.dz, np.diff(c_i) * grid.dx)
                rows.append(np.full(2 * seg.size, p, dtype=np.int64))
                cols.append(np.concatenate([path[:-1], path[1:]]))
                vals.append(0.5 * np.concatenate([seg, seg]))
                hits[np.unique(path)] += 1.0
    matrix = None
    if rays:
        if rows:
            matrix = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(n, grid.size)).tocsr()
        else:
            matrix = csr_matrix((n, grid.size))
    return times, matrix, hits.reshape(grid.nz, grid.nx)


def roughness_operators(active: np.ndarray, aspect: float = 2.0):
    """Second- and first-difference operators between neighbouring active nodes; columns are the active nodes in
    row-major order. Differences along the line are weighted by aspect, those in depth by 1."""
    nz, nx = active.shape
    n = int(active.sum())
    idx = np.full((nz, nx), -1, dtype=np.int64)
    idx[active] = np.arange(n)
    wh, wv = float(aspect), 1.0

    def build(terms):
        rows, cols, vals = [], [], []
        r = 0
        for members, weights in terms:
            if members[0].size == 0:
                continue
            ok = np.ones(members[0].shape, dtype=bool)
            for m in members:
                ok &= m >= 0
            k = int(ok.sum())
            if k == 0:
                continue
            rid = r + np.arange(k)
            for m, w in zip(members, weights):
                rows.append(rid)
                cols.append(m[ok])
                vals.append(np.full(k, w))
            r += k
        if r == 0:
            return csr_matrix((0, n))
        return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, n)).tocsr()

    d2 = build([([idx[:, :-2], idx[:, 1:-1], idx[:, 2:]], (wh, -2.0 * wh, wh)),
                ([idx[:-2, :], idx[1:-1, :], idx[2:, :]], (wv, -2.0 * wv, wv))])
    d1 = build([([idx[:, :-1], idx[:, 1:]], (-wh, wh)),
                ([idx[:-1, :], idx[1:, :]], (-wv, wv))])
    return d2, d1


def damped_update(matrix, residual, d2=None, d1=None, smoothing: float = 1.0, flatness: float = 0.0,
                  damping: float = 0.1, max_change: float = 0.3) -> np.ndarray:
    """The relative slowness change minimising |A d - r|^2 + (a |D2 d|)^2 + (b |D1 d|)^2 + (c |d|)^2.

    A is the ray-length matrix times the slowness of every column (relative-slowness form); a, b and c are smoothing,
    flatness and damping times the rms norm of the covered columns of A, so the weights do not depend on units or on
    the number of rays. The change is clipped to +-max_change."""
    a_mat = csr_matrix(matrix)
    n = a_mat.shape[1]
    colsq = np.asarray(a_mat.multiply(a_mat).sum(axis=0)).ravel()
    covered = colsq > 0
    scale = float(np.sqrt(colsq[covered].mean())) if covered.any() else 1.0
    blocks, rhs = [a_mat], [np.asarray(residual, dtype=np.float64).reshape(-1)]
    for op, w in ((d2, smoothing), (d1, flatness)):
        if op is not None and w > 0 and op.shape[0] > 0:
            blocks.append(csr_matrix(op) * (w * scale))
            rhs.append(np.zeros(op.shape[0]))
    if damping > 0:
        blocks.append(identity(n, format="csr") * (damping * scale))
        rhs.append(np.zeros(n))
    system = vstack(blocks).tocsr()
    delta = lsqr(system, np.concatenate(rhs), atol=1e-10, btol=1e-10, iter_lim=max(300, 4 * n))[0]
    return np.clip(delta, -max_change, max_change)


def invert(grid: NodeGrid, v_start, src_nodes, rec_nodes, t_obs, iterations: int = 8, smoothing: float = 1.0,
           flatness: float = 0.0, aspect: float = 2.0, damping: float = 0.1, vmin: float = 100.0,
           vmax: float = 8000.0, max_change: float = 0.3, report=None, log=None) -> dict:
    """First-arrival traveltime tomography: Gauss-Newton iterations of ray tracing, a damped smoothed update of the
    ground nodes' slowness and a step-halving line search (an update must lower the rms misfit).

    Returns the velocity (nz, nx), the predicted times, ray hits, the rms misfit history (s, the starting model
    first) and the mask of picks with a ray."""
    ground = grid.ground()
    active = np.flatnonzero(ground.ravel())
    if active.size == 0:
        raise ValueError("the tomography grid has no node below the ground")
    d2, d1 = roughness_operators(ground, aspect)
    v = np.clip(np.array(v_start, dtype=np.float64).reshape(grid.nz, grid.nx), vmin, vmax)
    t_obs = np.asarray(t_obs, dtype=np.float64).reshape(-1)

    def misfit(tp):
        ok = np.isfinite(tp) & np.isfinite(t_obs)
        if not ok.any():
            return np.inf, ok
        return float(np.sqrt(np.mean((t_obs[ok] - tp[ok]) ** 2))), ok

    t_pred, mat, hits = first_arrivals(grid, v, src_nodes, rec_nodes)
    rms, ok = misfit(t_pred)
    history = [rms]
    for it in range(max(int(iterations), 0)):
        if report is not None:
            report(it / max(int(iterations), 1))
        if not np.isfinite(rms) or rms == 0:
            break
        s = 1.0 / v.ravel()
        a_mat = mat[np.flatnonzero(ok)][:, active] @ diags(s[active])
        delta = damped_update(a_mat, (t_obs - t_pred)[ok], d2, d1, smoothing, flatness, damping, max_change)
        step, accepted = 1.0, None
        for _ in range(3):
            v_new = v.ravel().copy()
            v_new[active] = np.clip(1.0 / (s[active] * (1.0 + step * delta)), vmin, vmax)
            v_new = v_new.reshape(grid.nz, grid.nx)
            tp, mat_new, hits_new = first_arrivals(grid, v_new, src_nodes, rec_nodes)
            rms_new, ok_new = misfit(tp)
            if rms_new < rms:
                accepted = (v_new, tp, mat_new, hits_new, rms_new, ok_new)
                break
            step *= 0.5
        if accepted is None:
            if log is not None:
                log(f"    iteration {it + 1}: no further reduction of the misfit, stopped")
            break
        v, t_pred, mat, hits, rms, ok = accepted
        history.append(rms)
        if log is not None:
            log(f"    iteration {it + 1}: rms misfit {rms * 1000.0:.2f} ms")
        if report is not None:
            report((it + 1) / max(int(iterations), 1))
    return {"velocity": v, "t_pred": t_pred, "hits": hits, "rms": history, "used": ok}
