"""First-arrival traveltimes on a velocity grid by shortest paths (Dijkstra) over a 16-direction stencil.

Pure SciPy (no compiled extension beyond scipy.sparse.csgraph), so it ships everywhere SciPy does. Errors are about
1 % for smooth models with the 16-direction stencil; positions are snapped to grid nodes.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def _stencil(radius: int = 2):
    out = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if (di, dj) != (0, 0) and gcd(abs(di), abs(dj)) == 1:
                out.append((di, dj))
    return out


def build_graph(velocity: np.ndarray, dz: float, dx: float, radius: int = 2) -> csr_matrix:
    """Sparse graph whose edge weights are traveltimes between nodes (slowness averaged along the edge)."""
    nz, nx = velocity.shape
    slow = 1.0 / np.asarray(velocity, dtype=np.float64)
    node = np.arange(nz * nx).reshape(nz, nx)
    rows, cols, vals = [], [], []
    for di, dj in _stencil(radius):
        zs = slice(max(0, -di), nz - max(0, di))
        xs = slice(max(0, -dj), nx - max(0, dj))
        zt = slice(max(0, di), nz - max(0, -di))
        xt = slice(max(0, dj), nx - max(0, -dj))
        a = node[zs, xs].ravel()
        b = node[zt, xt].ravel()
        length = np.hypot(di * dz, dj * dx)
        # average slowness along the segment, sampled at the two ends and the midpoint
        s_mid = 0.5 * (slow[zs, xs] + slow[zt, xt])
        w = length * (0.25 * slow[zs, xs] + 0.5 * s_mid + 0.25 * slow[zt, xt]).ravel()
        rows.append(a)
        cols.append(b)
        vals.append(w)
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nz * nx, nz * nx))


def traveltime_tables(velocity: np.ndarray, dz: float, dx: float, x0: float, positions_x: np.ndarray,
                      depth_m: float = 0.0, radius: int = 2, chunk: int = 64) -> dict:
    """Traveltime table (nz, nx) from each surface position (snapped to the grid), keyed by node column."""
    nz, nx = velocity.shape
    g = build_graph(velocity, dz, dx, radius)
    iz = int(round(depth_m / dz))
    cols = np.unique(np.clip(np.rint((np.asarray(positions_x) - x0) / dx).astype(np.int64), 0, nx - 1))
    tables = {}
    for start in range(0, cols.size, chunk):
        c = cols[start:start + chunk]
        dist = dijkstra(g, directed=True, indices=iz * nx + c)
        for k, col in enumerate(c):
            tables[int(col)] = dist[k].reshape(nz, nx).astype(np.float32)
    return tables


def traveltime_tables_xz(velocity: np.ndarray, dz: float, dx: float, x0: float, positions_x: np.ndarray,
                         positions_z: np.ndarray, radius: int = 2, chunk: int = 64) -> dict:
    """Traveltime table (nz, nx) from each (x, z) position (snapped to the grid), keyed by (row, column).

    For sources and receivers below the top of the grid, e.g. on topography when the grid starts at the highest
    elevation; z is depth below the top of the grid."""
    nz, nx = velocity.shape
    g = build_graph(velocity, dz, dx, radius)
    cols = np.clip(np.rint((np.asarray(positions_x, dtype=np.float64) - x0) / dx).astype(np.int64), 0, nx - 1)
    rows = np.clip(np.rint(np.asarray(positions_z, dtype=np.float64) / dz).astype(np.int64), 0, nz - 1)
    nodes = np.unique(rows * nx + cols)
    tables = {}
    for start in range(0, nodes.size, chunk):
        nd = nodes[start:start + chunk]
        dist = dijkstra(g, directed=True, indices=nd)
        for k, node in enumerate(nd):
            tables[(int(node // nx), int(node % nx))] = dist[k].reshape(nz, nx).astype(np.float32)
    return tables
