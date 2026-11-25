"""Earth models for seismic modelling: flat layers from a short text, layers bounded by polylines from a model file,
or the project depth model, sampled onto regular grids of P velocity, S velocity and density.

Flat layers: ``"1500, 2000, 2600 / 300, 700"`` gives three layers with P velocities 1500, 2000 and 2600 m/s whose tops
lie at 0, 300 and 700 m depth. S velocities and densities may follow after more slashes:
``"1500, 2000, 2600 / 300, 700 / 0, 1150, 1500 / 1000, 2100, 2300"`` (vp / interface depths / vs / rho). Missing S
velocities are vp / 1.73 (write 0 for water), missing densities follow Gardner et al. (1974), rho = 310 vp^0.25
(kg/m3 with vp in m/s).

Model file: one layer per line from the top down, ``vp vs rho : x1 z1, x2 z2, ...``. The polyline is the top of the
layer (depth z positive down, metres); a single number after the colon is a flat top at that depth. vs and rho may be
written as ``-`` for the defaults. The first layer's top is the surface (topography); above it is air. Lines starting
with ``#`` are comments.

Depths are measured from a reference level (z = 0). A point belongs to the deepest layer whose top lies above it, so
an interface that runs below the next one pinches its layer out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

EPS = 1e-6                 # metres: points on an interface belong to the layer below it


def gardner_density(vp) -> np.ndarray:
    """Density (kg/m3) from P velocity (m/s), Gardner, Gardner & Gregory (1974)."""
    return 310.0 * np.maximum(np.asarray(vp, dtype=np.float64), 1.0) ** 0.25


@dataclass
class Layer:
    vp: float
    vs: float
    rho: float
    top_x: np.ndarray                  # polyline of the layer top, x increasing
    top_z: np.ndarray

    def top_at(self, x) -> np.ndarray:
        """Depth of the layer top at x (constant beyond the ends of the polyline)."""
        return np.interp(np.asarray(x, dtype=np.float64), self.top_x, self.top_z)

    @property
    def flat(self) -> bool:
        return bool(np.ptp(self.top_z) == 0)


def _polyline(x, z) -> tuple[np.ndarray, np.ndarray]:
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    if x.size != z.size or x.size == 0:
        raise ValueError("a polyline needs as many x as z values")
    order = np.argsort(x, kind="stable")
    return x[order], z[order]


class LayeredModel:
    """Homogeneous layers below polyline tops; the first layer's top is the surface."""
    kind = "layers"

    def __init__(self, layers: list[Layer]):
        if not layers:
            raise ValueError("a model needs at least one layer")
        for layer in layers:
            if not (layer.vp > 0 and layer.rho > 0 and layer.vs >= 0):
                raise ValueError("every layer needs vp > 0, vs >= 0 and rho > 0")
            if layer.vs >= layer.vp / np.sqrt(2.0) - 1e-9 and layer.vs > 0:
                raise ValueError(f"vs = {layer.vs:g} m/s is too high for vp = {layer.vp:g} m/s (vs < vp / sqrt 2)")
        self.layers = list(layers)

    def surface_at(self, x) -> np.ndarray:
        return self.layers[0].top_at(x)

    def with_surface(self, x, z) -> "LayeredModel":
        """The same layers with the surface (top of the first layer) replaced by the polyline (x, z)."""
        px, pz = _polyline(x, z)
        first = self.layers[0]
        return LayeredModel([Layer(first.vp, first.vs, first.rho, px, pz)] + self.layers[1:])

    @property
    def is_flat(self) -> bool:
        tops = [layer.top_z[0] for layer in self.layers]
        return all(layer.flat for layer in self.layers) and all(np.diff(tops) >= 0)

    def deepest(self) -> float:
        """Greatest depth of any layer top (the surface for a one-layer model)."""
        return float(max(np.max(layer.top_z) for layer in self.layers))

    def layer_index(self, z, x) -> np.ndarray:
        """Index of the layer at the points (z, x) (broadcast together); -1 above the surface (air)."""
        z, x = np.broadcast_arrays(np.asarray(z, dtype=np.float64), np.asarray(x, dtype=np.float64))
        idx = np.full(z.shape, -1, dtype=np.int64)
        for k, layer in enumerate(self.layers):
            idx[z >= layer.top_at(x) - EPS] = k
        return idx

    def grid_index(self, x, z) -> np.ndarray:
        """Layer index on the grid (nz, nx) of depths z and positions x."""
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        idx = np.full((z.size, x.size), -1, dtype=np.int64)
        for k, layer in enumerate(self.layers):
            idx[z[:, None] >= layer.top_at(x)[None, :] - EPS] = k
        return idx

    def properties(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.array([la.vp for la in self.layers]), np.array([la.vs for la in self.layers]),
                np.array([la.rho for la in self.layers]))

    def sample(self, x, z):
        """vp, vs, rho on the grid (nz, nx) and the air mask; air cells carry the first layer's properties."""
        idx = self.grid_index(x, z)
        air = idx < 0
        vp, vs, rho = self.properties()
        k = np.where(air, 0, idx)
        return vp[k], vs[k], rho[k], air


class GridModel:
    """The project depth model (interval velocity on a grid) as P velocity; vs = vp / 1.73, Gardner density."""
    kind = "grid"

    def __init__(self, depth_model, surface_x=None, surface_z=None):
        self.depth_model = depth_model
        self.surface = None if surface_x is None else _polyline(surface_x, surface_z)

    def surface_at(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.surface is None:
            return np.zeros(x.shape)
        return np.interp(x, self.surface[0], self.surface[1])

    def with_surface(self, x, z) -> "GridModel":
        return GridModel(self.depth_model, x, z)

    is_flat = False

    def deepest(self) -> float:
        return float(self.depth_model.z[-1])

    def sample(self, x, z):
        zz, xx = np.meshgrid(np.asarray(z, dtype=np.float64), np.asarray(x, dtype=np.float64), indexing="ij")
        vp = np.maximum(self.depth_model.at(zz, xx), 1.0)
        air = zz < self.surface_at(x)[None, :] - EPS
        return vp, vp / 1.73, gardner_density(vp), air


# ---------------------------------------------------------------------------------------------- reading models
def _numbers(text: str) -> list[float]:
    return [float(s) for s in text.replace(";", ",").replace(",", " ").split()]


def parse_layers(text: str) -> LayeredModel:
    """Flat layers from 'vp1, vp2, ... / z1, z2, ... [/ vs1, ... [/ rho1, ...]]' (depths of the layer tops below
    the first, which starts at 0)."""
    parts = [p.strip() for p in str(text).split("/")]
    if not parts or not parts[0]:
        raise ValueError("give the P velocities, e.g. '1500, 2000 / 300'")
    vp = _numbers(parts[0])
    n = len(vp)
    depths = _numbers(parts[1]) if len(parts) > 1 else []
    if len(depths) != n - 1:
        raise ValueError(f"{n} velocities need {n - 1} interface depth(s), got {len(depths)}")
    if depths and (min(depths) <= 0 or np.any(np.diff(depths) <= 0)):
        raise ValueError("interface depths must be positive and increasing")
    vs = _numbers(parts[2]) if len(parts) > 2 and parts[2] else [v / 1.73 for v in vp]
    rho = _numbers(parts[3]) if len(parts) > 3 and parts[3] else list(gardner_density(vp))
    if len(vs) != n or len(rho) != n:
        raise ValueError(f"give {n} S velocities and {n} densities (or leave them out)")
    tops = [0.0] + depths
    return LayeredModel([Layer(float(a), float(b), float(c), np.array([0.0]), np.array([float(t)]))
                         for a, b, c, t in zip(vp, vs, rho, tops)])


def load_model_file(path: str) -> LayeredModel:
    """Layers from a model file: 'vp vs rho : x1 z1, x2 z2, ...' per line, top layer first."""
    layers = []
    with open(path, encoding="utf-8") as fh:
        for number, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"{os.path.basename(path)} line {number}: expected 'vp vs rho : x z, x z, ...'")
            left, right = line.split(":", 1)
            props = left.split()
            if not 1 <= len(props) <= 3:
                raise ValueError(f"{os.path.basename(path)} line {number}: give vp [vs [rho]] before the colon")
            vp = float(props[0])
            vs = vp / 1.73 if len(props) < 2 or props[1] == "-" else float(props[1])
            rho = float(gardner_density(vp)) if len(props) < 3 or props[2] == "-" else float(props[2])
            points = [p.split() for p in right.replace(";", ",").split(",") if p.strip()]
            if len(points) == 1 and len(points[0]) == 1:
                px, pz = np.array([0.0]), np.array([float(points[0][0])])
            else:
                if any(len(p) != 2 for p in points):
                    raise ValueError(f"{os.path.basename(path)} line {number}: polyline points are 'x z' pairs")
                px, pz = _polyline([float(p[0]) for p in points], [float(p[1]) for p in points])
            layers.append(Layer(vp, vs, rho, px, pz))
    return LayeredModel(layers)


def model_from_text(text: str, work_dir: str = ".", depth_model=None):
    """A model from a step parameter: 'project model', the path of a model file, or flat layers in a string."""
    t = str(text).strip()
    if t.lower() in ("project model", "project", "depth model"):
        if depth_model is None:
            raise ValueError("no depth model in the project: build or load one, or give layers")
        return GridModel(depth_model)
    path = t if os.path.isabs(t) else os.path.join(work_dir, t)
    if t and os.path.isfile(path):
        return load_model_file(path)
    return parse_layers(t)


def elevation_surface(sx, selev, gx, gelev):
    """Surface polyline from station elevations: x, depth below the highest station, and that reference elevation."""
    x = np.concatenate([np.asarray(sx, dtype=np.float64), np.asarray(gx, dtype=np.float64)])
    e = np.concatenate([np.asarray(selev, dtype=np.float64), np.asarray(gelev, dtype=np.float64)])
    ux, inv = np.unique(np.round(x, 3), return_inverse=True)
    inv = inv.reshape(-1)
    ue = np.bincount(inv, weights=e) / np.bincount(inv)
    ref = float(ue.max())
    return ux, ref - ue, ref


# ---------------------------------------------------------------------------------------------- reflectivity
def reflection_coefficient(vp1, vs1, rho1, vp2, vs2, rho2, p=None):
    """P-P reflection coefficient for a wave in medium 1 meeting medium 2: normal incidence when p is None,
    otherwise the linearised Aki & Richards (1980) approximation at ray parameter p (s/m)."""
    vp1, vs1, rho1, vp2, vs2, rho2 = (np.asarray(v, dtype=np.float64) for v in (vp1, vs1, rho1, vp2, vs2, rho2))
    z1, z2 = rho1 * vp1, rho2 * vp2
    if p is None:
        return (z2 - z1) / np.maximum(z2 + z1, 1e-12)
    p = np.asarray(p, dtype=np.float64)
    s1 = np.clip(p * vp1, 0.0, 0.9999)
    s2 = np.clip(p * vp2, 0.0, 0.9999)
    theta = 0.5 * (np.arcsin(s1) + np.arcsin(s2))
    a, b, r = 0.5 * (vp1 + vp2), 0.5 * (vs1 + vs2), 0.5 * (rho1 + rho2)
    p2 = p * p
    return (0.5 * (1.0 - 4.0 * p2 * b * b) * (rho2 - rho1) / r + (vp2 - vp1) / (2.0 * a * np.cos(theta) ** 2)
            - 4.0 * p2 * b * (vs2 - vs1))


def interface_normals(model: LayeredModel, k: int, x) -> tuple[np.ndarray, np.ndarray]:
    """Unit normal (nx, nz) of the top of layer k at positions x (pointing down)."""
    x = np.asarray(x, dtype=np.float64)
    zk = model.layers[k].top_at(x)
    slope = np.gradient(zk, x) if x.size > 1 else np.zeros(x.size)
    norm = np.sqrt(1.0 + slope ** 2)
    return -slope / norm, 1.0 / norm


def reflectivity_grid(model, x, z):
    """Normal-incidence reflectivity on the depth grid (nz, nx) and the unit normals (nx, nz) of the reflectors.

    Layered models: one sample per column at every interface, from the layers actually above and below it (pinched
    out interfaces give nothing). Grid models: the impedance contrast between consecutive depth samples."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    nz, nx = z.size, x.size
    refl = np.zeros((nz, nx))
    nrm_x = np.zeros((nz, nx))
    nrm_z = np.ones((nz, nx))
    if isinstance(model, LayeredModel):
        dz = float(z[1] - z[0]) if nz > 1 else 1.0
        vp, vs, rho = model.properties()
        cols = np.arange(nx)
        for k in range(1, len(model.layers)):
            zk = model.layers[k].top_at(x)
            above = model.layer_index(zk - 1e-3, x)
            below = model.layer_index(zk + 1e-3, x)
            iz = np.rint((zk - z[0]) / dz).astype(np.int64)
            ok = (below == k) & (above >= 0) & (above != k) & (iz >= 0) & (iz < nz)
            if not np.any(ok):
                continue
            a = above[ok]
            r = reflection_coefficient(vp[a], vs[a], rho[a], vp[k], vs[k], rho[k])
            nxk, nzk = interface_normals(model, k, x)
            refl[iz[ok], cols[ok]] += r
            nrm_x[iz[ok], cols[ok]] = nxk[ok]
            nrm_z[iz[ok], cols[ok]] = nzk[ok]
    else:
        vp, _, rho, air = model.sample(x, z)
        imp = vp * rho
        refl[:-1] = (imp[1:] - imp[:-1]) / np.maximum(imp[1:] + imp[:-1], 1e-12)
        refl[:-1][air[:-1] | air[1:]] = 0.0
        refl[np.abs(refl) < 1e-4] = 0.0
    return refl, nrm_x, nrm_z
