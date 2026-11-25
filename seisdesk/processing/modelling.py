"""Seismic modelling with the geometry of the input dataset: acoustic and elastic finite differences, Kirchhoff
modelling and ray-traced reflection synthetics.

Every step keeps the headers of the traces (sources sx, receivers gx, optional elevations selev/gelev and source
depth sdepth) and replaces the samples with modelled data, so the result can be displayed, processed and compared
with field data. Models come from ``seisdesk.core.models``: flat layers in a short text, a model file of layers
bounded by polylines, or the project depth model. Amplitudes are relative: each result is scaled to a maximum of one.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ..core.dataset import SeismicData
from ..core.models import (LayeredModel, elevation_surface, interface_normals, model_from_text,
                           reflection_coefficient, reflectivity_grid)
from ..core.synthetic import events_to_traces, ricker
from .base import Param, Process, register
from .migration_time import rho_filter
from .traveltime import build_graph

C1, C2 = 9.0 / 8.0, -1.0 / 24.0                  # fourth-order staggered-grid coefficients
CFL = 0.9 / (np.sqrt(2.0) * (C1 - C2))           # 90 % of the 2D stability limit: dt <= CFL h / vmax


# ---------------------------------------------------------------------------------------------- staggered operators
def _dxp(f, h):
    """d/dx at the half nodes j + 1/2 from the nodes j - 1 .. j + 2 (x is axis 1)."""
    out = np.zeros_like(f)
    out[:, 1:-2] = (C1 * (f[:, 2:-1] - f[:, 1:-2]) + C2 * (f[:, 3:] - f[:, :-3])) / h
    return out


def _dxm(f, h):
    """d/dx at the nodes j from half-node values (index j holds the value at j + 1/2)."""
    out = np.zeros_like(f)
    out[:, 2:-1] = (C1 * (f[:, 2:-1] - f[:, 1:-2]) + C2 * (f[:, 3:] - f[:, :-3])) / h
    return out


def _dzp(f, h):
    """d/dz at the half nodes i + 1/2 (z is axis 0)."""
    out = np.zeros_like(f)
    out[1:-2] = (C1 * (f[2:-1] - f[1:-2]) + C2 * (f[3:] - f[:-3])) / h
    return out


def _dzm(f, h):
    """d/dz at the nodes i from half-node values (index i holds the value at i + 1/2)."""
    out = np.zeros_like(f)
    out[2:-1] = (C1 * (f[2:-1] - f[1:-2]) + C2 * (f[3:] - f[:-3])) / h
    return out


def _pml_profile(n, nabs, h, vmax, low=True, high=True, half=False):
    """PML damping (1/s) along one axis: quadratic ramp over nabs cells for a theoretical reflection of 1e-4
    (Collino & Tsogka 2001)."""
    d = np.zeros(n)
    if nabs <= 0:
        return d
    pos = np.arange(n) + (0.5 if half else 0.0)
    d0 = 3.0 * vmax * np.log(1.0e4) / (2.0 * nabs * h)
    if low:
        d += d0 * np.clip((nabs - pos) / nabs, 0.0, None) ** 2
    if high:
        d += d0 * np.clip((pos - (n - 1 - nabs)) / nabs, 0.0, None) ** 2
    return d


def _pml_coefficients(d, dt, axis):
    """Update factors of a damped field: f = a f + c (right-hand side), as a row (axis 1) or column (axis 0)."""
    a = (1.0 - 0.5 * d * dt) / (1.0 + 0.5 * d * dt)
    c = dt / (1.0 + 0.5 * d * dt)
    shape = (1, -1) if axis == 1 else (-1, 1)
    return a.reshape(shape).astype(np.float32), c.reshape(shape).astype(np.float32)


def _sponge(nz, nx, nabs, top=True, alpha=0.3):
    """Damping factors of Cerjan et al. (1985), applied to every field at every time step."""
    def ramp(n, low):
        g = np.ones(n)
        m = min(nabs, n // 2)
        if m > 0:
            r = np.exp(-(alpha * (m - np.arange(m)) / m) ** 2)
            if low:
                g[:m] = r
            g[n - m:] = r[::-1]
        return g
    return (ramp(nz, top)[:, None] * ramp(nx, True)[None, :]).astype(np.float32)


def _time_stepping(dt_out, nt_out, h, vmax, f0):
    """Time step (a whole fraction of the output interval below the stability limit), source delay in steps, number
    of steps and the source function: the time integral of a Ricker wavelet, t exp(-(pi f0 t)^2), whose injection
    as a rate radiates the Ricker itself."""
    nsub = max(1, int(np.ceil(dt_out / (CFL * h / vmax) - 1e-9)))
    dt = dt_out / nsub
    n_delay = int(round(1.5 / f0 / dt))
    nsteps = n_delay + (nt_out - 1) * nsub + 1
    t = (np.arange(nsteps) - n_delay) * dt
    return dt, nsub, n_delay, nsteps, t * np.exp(-(np.pi * f0 * t) ** 2)


# ---------------------------------------------------------------------------------------------- one shot
def acoustic_shot(vp, rho, h, dt_out, nt_out, src, rec_iz, rec_ix, f0, nabs, pml=True, top_absorbing=True,
                  air=None, progress=None) -> np.ndarray:
    """One shot of 2D variable-density acoustic modelling: velocity-pressure staggered grid, second order in time,
    fourth order in space (Levander 1988), split-field PML (Collino & Tsogka 2001) or Cerjan sponge borders.

    vp, rho: (nz, nx) on a square grid of spacing h; src: (iz, ix); receivers at (rec_iz, rec_ix). With ``air`` (the
    cells above a free surface) the pressure is held at zero there. Returns the pressure (nrec, nt_out) on the output
    time axis, t = 0 at the peak of the source wavelet."""
    nz, nx = vp.shape
    vmax = float(np.max(vp))
    dt, nsub, n_delay, nsteps, q = _time_stepping(dt_out, nt_out, h, vmax, f0)
    kappa = (rho * vp ** 2).astype(np.float32)
    b = 1.0 / rho
    bx = b.copy()
    bx[:, :-1] = 0.5 * (b[:, :-1] + b[:, 1:])
    bz = b.copy()
    bz[:-1] = 0.5 * (b[:-1] + b[1:])
    bx, bz = bx.astype(np.float32), bz.astype(np.float32)
    if pml:
        ax, cx = _pml_coefficients(_pml_profile(nx, nabs, h, vmax), dt, 1)
        axh, cxh = _pml_coefficients(_pml_profile(nx, nabs, h, vmax, half=True), dt, 1)
        az, cz = _pml_coefficients(_pml_profile(nz, nabs, h, vmax, low=top_absorbing), dt, 0)
        azh, czh = _pml_coefficients(_pml_profile(nz, nabs, h, vmax, low=top_absorbing, half=True), dt, 0)
        sponge = None
    else:
        ax = axh = az = azh = 1.0
        cx = cxh = cz = czh = dt
        sponge = _sponge(nz, nx, nabs, top_absorbing)
    vx = np.zeros((nz, nx), np.float32)
    vz = np.zeros_like(vx)
    px = np.zeros_like(vx)                       # pressure split into its x and z parts (exact sum inside)
    pz = np.zeros_like(vx)
    out = np.zeros((rec_iz.size, nt_out), np.float32)
    si, sj = int(src[0]), int(src[1])
    for n in range(nsteps):
        p = px + pz
        vx = axh * vx - cxh * bx * _dxp(p, h)
        vz = azh * vz - czh * bz * _dzp(p, h)
        px = ax * px - cx * kappa * _dxm(vx, h)
        pz = az * pz - cz * kappa * _dzm(vz, h)
        px[si, sj] += 0.5 * dt * q[n]
        pz[si, sj] += 0.5 * dt * q[n]
        if air is not None:
            px[air] = 0.0
            pz[air] = 0.0
        if sponge is not None:
            vx *= sponge
            vz *= sponge
            px *= sponge
            pz *= sponge
        m = n - n_delay
        if m >= 0 and m % nsub == 0:
            out[:, m // nsub] = px[rec_iz, rec_ix] + pz[rec_iz, rec_ix]
        if progress is not None and n % 200 == 0:
            progress(n / nsteps)
    return out


def elastic_shot(vp, vs, rho, h, dt_out, nt_out, src, rec_iz, rec_ix, f0, nabs, component="vertical",
                 top_absorbing=True, air=None, progress=None, epsilon=0.0, delta=0.0) -> np.ndarray:
    """One shot of 2D isotropic elastic (P-SV) modelling: velocity-stress staggered grid (Virieux 1986), fourth order
    in space (Levander 1988), Cerjan sponge borders, explosive source.

    Cells in ``air`` are vacuum (no stiffness, no motion), which makes the surface below them traction-free; the shear
    modulus between nodes is the harmonic mean of the four corners, zero next to fluid or air (Moczo et al. 2002).
    component: vertical or horizontal particle velocity, or pressure -(sxx + szz) / 2."""
    nz, nx = vp.shape
    solid = np.ones(vp.shape, bool) if air is None else ~air
    vmax = float(np.max(vp[solid])) * (np.sqrt(2*(1+2*abs(epsilon)+2*abs(delta))) if epsilon or delta else 1)
    dt, nsub, n_delay, nsteps, q = _time_stepping(dt_out, nt_out, h, vmax, f0)
    mu = rho * vs ** 2
    lam = rho * vp ** 2 - 2.0 * mu
    b = 1.0 / rho
    if air is not None:
        mu = np.where(air, 0.0, mu)
        lam = np.where(air, 0.0, lam)
        b = np.where(air, 0.0, b)
    bx = b.copy()
    bx[:, :-1] = 0.5 * (b[:, :-1] + b[:, 1:])
    bz = b.copy()
    bz[:-1] = 0.5 * (b[:-1] + b[1:])
    mp = np.pad(mu, ((0, 1), (0, 1)), mode="edge")
    corners = np.stack([mp[:-1, :-1], mp[1:, :-1], mp[:-1, 1:], mp[1:, 1:]])
    inv = np.divide(1.0, corners, out=np.full(corners.shape, np.inf), where=corners > 0).sum(axis=0)
    muxz = np.where(np.all(corners > 0, axis=0), 4.0 / inv, 0.0)
    lam2mu = (lam + 2.0 * mu).astype(np.float32)
    c11 = lam2mu*(1+2*epsilon)
    radicand = (lam2mu-mu)**2+2*delta*lam2mu*(lam2mu-mu)
    if np.any(radicand[solid] < 0):
        raise ValueError("Thomsen parameters give an imaginary C13 stiffness")
    c13 = np.sqrt(np.maximum(radicand, 0))-mu
    if np.any((c11*lam2mu-c13*c13)[solid] < -1e-5*(lam2mu[solid]**2)):
        raise ValueError("unstable VTI stiffness tensor")
    bx, bz, lam, muxz = (a.astype(np.float32) for a in (bx, bz, lam, muxz))
    sponge = _sponge(nz, nx, nabs, top_absorbing)
    vx = np.zeros((nz, nx), np.float32)
    vz = np.zeros_like(vx)
    sxx = np.zeros_like(vx)
    szz = np.zeros_like(vx)
    sxz = np.zeros_like(vx)
    out = np.zeros((rec_iz.size, nt_out), np.float32)
    si, sj = int(src[0]), int(src[1])
    for n in range(nsteps):
        vx += dt * bx * (_dxp(sxx, h) + _dzm(sxz, h))
        vz += dt * bz * (_dxm(sxz, h) + _dzp(szz, h))
        exx = _dxm(vx, h)
        ezz = _dzm(vz, h)
        sxx += dt * (c11 * exx + c13 * ezz)
        szz += dt * (c13 * exx + lam2mu * ezz)
        sxz += dt * muxz * (_dzp(vx, h) + _dxp(vz, h))
        sxx[si, sj] += dt * q[n]
        szz[si, sj] += dt * q[n]
        for f in (vx, vz, sxx, szz, sxz):
            f *= sponge
        m = n - n_delay
        if m >= 0 and m % nsub == 0:
            if component == "horizontal":
                out[:, m // nsub] = vx[rec_iz, rec_ix]
            elif component == "pressure":
                out[:, m // nsub] = -0.5 * (sxx[rec_iz, rec_ix] + szz[rec_iz, rec_ix])
            else:
                out[:, m // nsub] = vz[rec_iz, rec_ix]
        if progress is not None and n % 200 == 0:
            progress(n / nsteps)
    return out


# ---------------------------------------------------------------------------------------------- shared helpers
def _model(text, context):
    return model_from_text(text, context.work_dir, context.depth_model)


def _stations(data, model, topography, receiver_depth):
    """The model (with the surface from the elevation headers if asked) and the source and receiver positions
    (x, depth): sources sdepth below the surface, receivers receiver_depth below it."""
    sx, gx = data.header("sx"), data.header("gx")
    if np.ptp(np.concatenate([sx, gx])) <= 0:
        raise ValueError("source and receiver positions (sx, gx) are not set: run a geometry step first")
    if topography == "from elevation headers":
        selev, gelev = data.header("selev"), data.header("gelev")
        if not (np.any(selev) or np.any(gelev)):
            raise ValueError("the dataset has no elevations (selev, gelev); use the model surface")
        x, z, _ = elevation_surface(sx, selev, gx, gelev)
        model = model.with_surface(x, z)
    sz = model.surface_at(sx) + np.maximum(data.header("sdepth"), 0.0)
    gz = model.surface_at(gx) + receiver_depth
    return model, sx, sz, gx, gz


def _time_axis(data, tmax, dt_ms):
    dt = dt_ms / 1000.0 if dt_ms > 0 else (data.dt if data.domain == "time" else 0.002)
    length = tmax if tmax > 0 else (data.tmax if data.domain == "time" else 2.0)
    return int(round(length / dt)) + 1, dt


def _depth_limit(model, zmax):
    return zmax if zmax > 0 else max(1.2 * model.deepest() + 100.0, 300.0)


def _shots(sx, sz):
    """Distinct source positions and the shot number of every trace."""
    uniq, inv = np.unique(np.round(np.stack([sx, sz], axis=1), 3), axis=0, return_inverse=True)
    return uniq, inv.reshape(-1)


def _auto_spacing(model, xlo, xhi, zbot, f0, use_vs, ppw=6.0):
    """Grid spacing giving ``ppw`` points per shortest wavelength (frequencies up to 2.5 f0)."""
    xs = np.linspace(xlo, xhi, 60)
    top = float(np.min(model.surface_at(xs)))
    zs = np.linspace(top, max(zbot, top + 1.0), 120)
    vp, vs, _, air = model.sample(xs, zs)
    v = vp[~air] if np.any(~air) else vp.ravel()
    if use_vs:
        s = vs[~air]
        s = s[s > 0]
        if s.size:
            v = s
    h = float(np.min(v)) / (2.5 * f0) / ppw
    return max(0.5, float(f"{h:.2g}"))


def _nodes(x, z, xg, zg, h, air=None, below=0):
    """Grid nodes (iz, ix) of positions (x, z); with an air mask, at least ``below`` cells under the first solid
    cell of the column."""
    ix = np.clip(np.rint((np.asarray(x, dtype=np.float64) - xg[0]) / h).astype(np.int64), 0, xg.size - 1)
    iz = np.clip(np.rint((np.asarray(z, dtype=np.float64) - zg[0]) / h).astype(np.int64), 0, zg.size - 1)
    if air is not None:
        solid = ~air
        first = np.where(solid.any(axis=0), np.argmax(solid, axis=0), 0)
        iz = np.minimum(np.maximum(iz, first[ix] + below), zg.size - 1)
    return iz, ix


def _node_tables(velocity, h, nodes, keep, context=None, chunk=32):
    """First-arrival traveltimes (shortest paths, see traveltime.py) from every distinct grid node in ``nodes``,
    kept at the node indices ``keep`` only: {node: times}."""
    graph = build_graph(velocity, h, h)
    uniq = np.unique(np.asarray(nodes, dtype=np.int64))
    tables = {}
    for start in range(0, uniq.size, chunk):
        part = uniq[start:start + chunk]
        dist = dijkstra(graph, directed=True, indices=part)
        for k, node in enumerate(part):
            tables[int(node)] = dist[k][keep]
        if context is not None:
            context.report(0.5 * min(1.0, (start + part.size) / uniq.size))
    return tables


def _with_wavelet(spikes, dt, f0):
    """Spike traces convolved with a zero-phase Ricker wavelet of peak frequency f0."""
    w, c = ricker(f0, dt)
    n = spikes.shape[1]
    nfft = int(2 ** np.ceil(np.log2(n + w.size)))
    wf = np.fft.rfft(np.roll(np.pad(w, (0, nfft - w.size)), -c))
    return np.fft.irfft(np.fft.rfft(spikes, nfft, axis=1) * wf, nfft, axis=1)[:, :n]


def _finish(data, traces, dt, label, name):
    """The modelled dataset: input headers, t0 = 0, traces scaled to a maximum of one."""
    traces = np.nan_to_num(np.asarray(traces, dtype=np.float64))
    peak = float(np.max(np.abs(traces))) if traces.size else 0.0
    if peak > 0:
        traces = traces / peak
    heads = {k: v.copy() for k, v in data.headers.items()}
    out = SeismicData(traces.astype(np.float32), dt, heads, 0.0, f"{data.name} {label}", "time", list(data.history),
                      dict(data.meta))
    out.meta["modelled"] = name
    return out


def _model_params(f0=25.0):
    return (Param("model", "str", "1500, 2000, 2600 / 300, 700", "Model",
                  "flat layers 'vp1, vp2, ... / z1, z2, ...' (optionally '/ vs1, ... / rho1, ...'), the name of a "
                  "model file, or 'project model'"),
            Param("topography", "choice", "model surface", "Surface",
                  "the model's surface, or the topography from the selev/gelev headers (depths then count from the "
                  "highest station)", choices=("model surface", "from elevation headers")),
            Param("receiver_depth", "float", 0.0, "Receiver depth below the surface", unit="m", minimum=0.0),
            Param("f0", "float", f0, "Ricker peak frequency", unit="Hz", minimum=1.0),
            Param("tmax", "float", 0.0, "Record length (0 = as input)", unit="s", minimum=0.0),
            Param("dt_ms", "float", 0.0, "Sample interval (0 = as input)", unit="ms", minimum=0.0),
            Param("zmax", "float", 0.0, "Model depth (0 = below the deepest interface)", unit="m", minimum=0.0),
            Param("dx", "float", 0.0, "Grid spacing (0 = automatic)", unit="m", minimum=0.0))


# ---------------------------------------------------------------------------------------------- finite differences
def _fd_grid(model, sx, sz, gx, gz, h, zmax, nabs, free_surface):
    """Regular grid around the stations: 20-cell margins, absorbing borders of nabs cells, and above the surface
    either three air rows (free surface) or an absorbing border filled with the first layer."""
    xlo = float(min(sx.min(), gx.min()))
    xhi = float(max(sx.max(), gx.max()))
    margin = (20 + nabs) * h
    xg = xlo - margin + np.arange(int(np.ceil((xhi - xlo + 2 * margin) / h)) + 1) * h
    top = float(np.min(model.surface_at(xg)))
    z0 = top - (3 if free_surface else nabs) * h
    zbot = max(zmax, float(max(sz.max(), gz.max())) + 10 * h) + nabs * h
    zg = z0 + np.arange(int(np.ceil((zbot - z0) / h)) + 1) * h
    if xg.size * zg.size > 6_000_000:
        raise ValueError(f"the grid would have {zg.size} x {xg.size} cells: use a larger grid spacing or a smaller "
                         "model depth")
    vp, vs, rho, air = model.sample(xg, zg)
    return xg, zg, vp, vs, rho, air


def _run_fd(step, data, context, elastic):
    model = _model(step["model"], context)
    model, sx, sz, gx, gz = _stations(data, model, step["topography"], step["receiver_depth"])
    nt, dt = _time_axis(data, step["tmax"], step["dt_ms"])
    f0 = step["f0"]
    if 2.5 * f0 > 0.4 / dt:
        context.log(f"    warning: the wavelet reaches {2.5 * f0:.0f} Hz, close to the Nyquist frequency of the "
                    "output sample interval")
    zmax = _depth_limit(model, step["zmax"])
    xlo, xhi = float(min(sx.min(), gx.min())), float(max(sx.max(), gx.max()))
    h = step["dx"] if step["dx"] > 0 else _auto_spacing(model, xlo, xhi, zmax, f0, elastic)
    free = step["free_surface"]
    nabs = step["absorbing"]
    xg, zg, vp, vs, rho, air = _fd_grid(model, sx, sz, gx, gz, h, zmax, nabs, free)
    solid = ~air
    shear = solid & (vs > 0)
    vmin = float(np.min(vs[shear])) if elastic and np.any(shear) else float(np.min(vp[solid]))
    ppw = vmin / (2.5 * f0) / h
    context.log(f"    grid {zg.size} x {xg.size} cells of {h:g} m, {ppw:.1f} points per shortest wavelength")
    if ppw < 5.0:
        context.log("    warning: fewer than 5 grid points per shortest wavelength; expect numerical dispersion "
                    "(use a smaller grid spacing or a lower frequency)")
    if free:
        context.log("    free surface: sources sit at least two cells below it" +
                    ("" if elastic else ", receivers too"))
    shots, inv = _shots(sx, sz)
    out = np.zeros((data.ntr, nt))
    air_mask = air if free else None
    for k in range(len(shots)):
        idx = np.flatnonzero(inv == k)
        s_iz, s_ix = _nodes(sx[idx[:1]], sz[idx[:1]], xg, zg, h, air_mask, 2)
        r_iz, r_ix = _nodes(gx[idx], gz[idx], xg, zg, h, air_mask, 0 if elastic else 2)

        def report(fraction, k=k):
            context.report((k + fraction) / len(shots))

        if elastic:
            rec = elastic_shot(vp, vs, rho, h, dt, nt, (s_iz[0], s_ix[0]), r_iz, r_ix, f0, nabs, step["component"],
                               not free, air_mask, report, epsilon=step.values.get('epsilon', 0), delta=step.values.get('delta', 0))
        else:
            rec = acoustic_shot(vp, rho, h, dt, nt, (s_iz[0], s_ix[0]), r_iz, r_ix, f0, nabs,
                                step["boundary"] == "PML", not free, air_mask, report)
        out[idx] = rec
    if step["correct_2d"]:
        out = rho_filter(out, dt, modelling=True)
    context.results[step.name] = {"dx": h, "nx": int(xg.size), "nz": int(zg.size), "x0": float(xg[0]),
                                  "z0": float(zg[0]), "shots": int(len(shots)), "points_per_wavelength": ppw}
    return out, dt


@register
class AcousticFD(Process):
    name = "fd_acoustic"
    label = "Acoustic finite-difference modelling (2D)"
    category = "Modelling"
    description = ("Full-waveform acoustic modelling of every shot of the dataset, at its source and receiver "
                   "positions: pressure and particle velocity on a staggered grid, second order in time and fourth "
                   "order in space, absorbing PML or sponge borders and an optional free surface. Direct, refracted, "
                   "reflected and diffracted waves and multiples are all included. Unless given, the grid spacing "
                   "gives six points per shortest wavelength; the time step is set for stability. The 2D correction "
                   "turns the line-source waveform of a 2D model into the point-source (Ricker) waveform.")
    params = _model_params() + (
        Param("free_surface", "bool", False, "Free surface (stations at least two cells below it)"),
        Param("boundary", "choice", "PML", "Absorbing borders", choices=("PML", "sponge")),
        Param("absorbing", "int", 30, "Absorbing border width", unit="cells", minimum=5, maximum=200),
        Param("correct_2d", "bool", True, "2D to 3D waveform correction"))

    def run(self, data, context):
        out, dt = _run_fd(self, data, context, elastic=False)
        return _finish(data, out, dt, "FD acoustic", self.name)


@register
class ElasticFD(Process):
    name = "fd_elastic"
    label = "Elastic finite-difference modelling (2D)"
    category = "Modelling"
    description = ("Elastic (P-SV) finite-difference modelling of every shot: velocity-stress staggered grid "
                   "(Virieux 1986), fourth order in space, explosive source, sponge borders and a free surface "
                   "(vacuum above the model surface), so ground roll and converted waves appear. Records the vertical "
                   "or horizontal particle velocity, or pressure. S velocities come from the model (vp / 1.73 when "
                   "not given; 0 makes a fluid layer). The slowest S velocity sets the grid spacing, so elastic runs "
                   "take longer than acoustic ones.")
    params = _model_params(20.0) + (
        Param("component", "choice", "vertical", "Recorded component", choices=("vertical", "horizontal", "pressure")),
        Param("free_surface", "bool", True, "Free surface (sources at least two cells below it)"),
        Param("absorbing", "int", 40, "Absorbing border width", unit="cells", minimum=5, maximum=200),
        Param("correct_2d", "bool", True, "2D to 3D waveform correction"))

    def run(self, data, context):
        out, dt = _run_fd(self, data, context, elastic=True)
        return _finish(data, out, dt, f"FD elastic {self['component']}", self.name)


# ---------------------------------------------------------------------------------------------- Kirchhoff modelling
@register
class KirchhoffModelling(Process):
    name = "kirchhoff_modelling"
    label = "Kirchhoff modelling from a depth model"
    category = "Modelling"
    description = ("Primary reflections by Kirchhoff summation over the reflectors of the model: every reflector point "
                   "of a depth grid adds its normal-incidence reflection coefficient at the traveltime source - point "
                   "- receiver (shortest paths through the model), weighted for geometric spreading and obliquity; "
                   "the sum is convolved with a Ricker wavelet. Dipping and curved reflectors get their diffraction "
                   "tails; there are no direct waves and no multiples.")
    params = _model_params() + (
        Param("aperture", "float", 2000.0, "Aperture (half width around the midpoint)", unit="m", minimum=10.0),)

    def run(self, data, context):
        model = _model(self["model"], context)
        model, sx, sz, gx, gz = _stations(data, model, self["topography"], self["receiver_depth"])
        nt, dt = _time_axis(data, self["tmax"], self["dt_ms"])
        f0 = self["f0"]
        zmax = _depth_limit(model, self["zmax"])
        xlo, xhi = float(min(sx.min(), gx.min())), float(max(sx.max(), gx.max()))
        h = self["dx"] if self["dx"] > 0 else _auto_spacing(model, xlo, xhi, zmax, f0, False, ppw=4.0)
        pad = min(0.5 * self["aperture"], 1000.0) + 2 * h
        xg = xlo - pad + np.arange(int(np.ceil((xhi - xlo + 2 * pad) / h)) + 1) * h
        top = float(np.min(model.surface_at(xg)))
        zg = top + np.arange(int(np.ceil((max(zmax, top + h) - top) / h)) + 1) * h
        if xg.size * zg.size > 4_000_000:
            raise ValueError(f"the grid would have {zg.size} x {xg.size} cells: use a larger grid spacing")
        vp, _, _, air = model.sample(xg, zg)
        refl, nrm_x, nrm_z = reflectivity_grid(model, xg, zg)
        refl[air] = 0.0
        cells = np.flatnonzero(refl.ravel())
        if cells.size == 0:
            raise ValueError("the model has no reflecting interface inside the grid")
        nxg = xg.size
        rc = refl.ravel()[cells]
        xc, zc = xg[cells % nxg], zg[cells // nxg]
        cn_x, cn_z = nrm_x.ravel()[cells], nrm_z.ravel()[cells]
        col_w = h / np.maximum(np.abs(cn_z), 0.2)               # reflector length represented by one column
        s_iz, s_ix = _nodes(sx, sz, xg, zg, h)
        r_iz, r_ix = _nodes(gx, gz, xg, zg, h)
        s_node, r_node = s_iz * nxg + s_ix, r_iz * nxg + r_ix
        context.log(f"    traveltime grid {zg.size} x {xg.size} cells of {h:g} m, {cells.size} reflector points")
        tables = _node_tables(vp, h, np.concatenate([s_node, r_node]), cells, context)
        spikes = np.zeros((data.ntr, nt))
        for i in range(data.ntr):
            t = tables[int(s_node[i])] + tables[int(r_node[i])]
            ds = np.maximum(np.hypot(xc - sx[i], zc - sz[i]), h)
            dr = np.maximum(np.hypot(xc - gx[i], zc - gz[i]), h)
            length = ds + dr
            v_eff = length / np.maximum(t, 1e-9)
            cos_s = np.abs(cn_x * (sx[i] - xc) + cn_z * (sz[i] - zc)) / ds
            cos_r = np.abs(cn_x * (gx[i] - xc) + cn_z * (gz[i] - zc)) / dr
            # stationary phase of the sum along a reflector gives coefficient / path length after the half
            # derivative (rho filter) below
            w = rc * 0.5 * (cos_s + cos_r) * np.sqrt(2.0 / (np.pi * v_eff * length)) / length * col_w
            pos = t / dt
            ok = (np.isfinite(pos) & (pos >= 0) & (pos < nt - 1)
                  & (np.abs(xc - 0.5 * (sx[i] + gx[i])) <= self["aperture"]))
            if np.any(ok):
                i0 = np.floor(pos[ok]).astype(np.int64)
                fr = pos[ok] - i0
                spikes[i] += (np.bincount(i0, w[ok] * (1.0 - fr), minlength=nt)[:nt]
                              + np.bincount(i0 + 1, w[ok] * fr, minlength=nt)[:nt])
            if i % 100 == 0:
                context.report(0.5 + 0.5 * i / data.ntr)
        traces = _with_wavelet(rho_filter(spikes, dt, modelling=True), dt, f0)
        context.results[self.name] = {"dx": h, "reflector_points": int(cells.size)}
        return _finish(data, traces, dt, "Kirchhoff modelling", self.name)


# ---------------------------------------------------------------------------------------------- ray tracing
def _ray_parameter(thick, v, offset, vmax, iterations=64):
    """Ray parameter (s/m) per row giving the horizontal distance ``offset``: bisection on p vmax in [0, 1) of
    X(p) = sum_j thick_j p v_j / sqrt(1 - p^2 v_j^2), which increases with p."""
    lo = np.zeros(offset.size)
    hi = np.full(offset.size, 1.0 - 1e-12)
    for _ in range(iterations):
        u = 0.5 * (lo + hi)
        sv = np.where(thick > 0, (u / vmax)[:, None] * v[None, :], 0.0)
        x = (thick * sv / np.sqrt(1.0 - sv ** 2)).sum(axis=1)
        far = x > offset
        hi = np.where(far, u, hi)
        lo = np.where(far, lo, u)
    return 0.5 * (lo + hi) / vmax


def flat_reflections(model: LayeredModel, sx, sz, gx, gz, angle: bool = False):
    """Two-way times and amplitudes (ntr, ninterfaces) of the primary reflections in flat layers by exact two-point
    ray tracing (Snell's law; the ray parameter is found by bisection). Amplitude: reflection coefficient divided by
    the spherical divergence v_rms^2 t / v_source (Newman 1973). NaN where a station lies below the interface."""
    vp, vs, rho = model.properties()
    tops = np.array([la.top_z[0] for la in model.layers])
    n = tops.size
    sx, sz, gx, gz = (np.asarray(a, dtype=np.float64) for a in (sx, sz, gx, gz))
    off = np.abs(gx - sx)
    ntr = off.size
    times = np.full((ntr, max(n - 1, 0)), np.nan)
    amps = np.zeros((ntr, max(n - 1, 0)))
    src_layer = np.clip(np.searchsorted(tops, sz + 1e-6, side="right") - 1, 0, n - 1)
    for k in range(1, n):
        zk = tops[k]
        lo, hi = tops[:k], tops[1:k + 1]
        down = np.clip(np.minimum(hi[None, :], zk) - np.maximum(lo[None, :], sz[:, None]), 0.0, None)
        up = np.clip(np.minimum(hi[None, :], zk) - np.maximum(lo[None, :], gz[:, None]), 0.0, None)
        thick = down + up
        v = vp[:k]
        vmax = np.max(np.where(thick > 0, v[None, :], 0.0), axis=1)
        ok = (sz < zk) & (gz < zk) & (vmax > 0)
        if not np.any(ok):
            continue
        p = _ray_parameter(thick, v, off, np.where(vmax > 0, vmax, 1.0))
        sv = np.where(thick > 0, p[:, None] * v[None, :], 0.0)
        tj = thick / (v[None, :] * np.sqrt(1.0 - sv ** 2))
        t = tj.sum(axis=1)
        vrms2 = (tj * v[None, :] ** 2).sum(axis=1) / np.maximum(t, 1e-12)
        spread = vrms2 * t / vp[src_layer]
        r = np.broadcast_to(reflection_coefficient(vp[k - 1], vs[k - 1], rho[k - 1], vp[k], vs[k], rho[k],
                                                   p if angle else None), (ntr,))
        times[:, k - 1] = np.where(ok, t, np.nan)
        amps[:, k - 1] = np.where(ok, r / np.maximum(spread, 1e-9), 0.0)
    return times, amps


def grid_reflections(model: LayeredModel, xg, zg, h, sx, sz, gx, gz, angle: bool = False, context=None):
    """Primary reflections in a layered model of any shape: for each interface, shortest-path traveltimes through
    the layers above it (the cells below are blocked), and the reflection time is the minimum over the interface of
    T_source + T_receiver (Fermat's principle). Amplitude: reflection coefficient / path length."""
    vp = model.sample(xg, zg)[0]
    vpl, vsl, rhol = model.properties()
    nz, nx = vp.shape
    s_iz, s_ix = _nodes(sx, sz, xg, zg, h)
    r_iz, r_ix = _nodes(gx, gz, xg, zg, h)
    s_node, r_node = s_iz * nx + s_ix, r_iz * nx + r_ix
    nl = len(model.layers)
    ntr = sx.size
    times = np.full((ntr, max(nl - 1, 0)), np.nan)
    amps = np.zeros((ntr, max(nl - 1, 0)))
    for k in range(1, nl):
        layer = model.layers[k]
        zk = layer.top_at(xg)
        above = model.layer_index(zk - 1e-3, xg)
        below = model.layer_index(zk + 1e-3, xg)
        rows = np.floor((zk - zg[0]) / h + 1e-9).astype(np.int64)
        valid = (below == k) & (above >= 0) & (above != k) & (rows >= 0) & (rows < nz)
        if not np.any(valid):
            continue
        cols = np.flatnonzero(valid)
        rw = rows[valid]
        delta = zk[valid] - zg[rw]                         # node to interface, 0 <= delta < h
        up_layer = above[valid]
        v_up = vpl[up_layer]
        blocked = vp.copy()
        blocked[zg[:, None] > zk[None, :] + 1e-6] = 1e-3 * float(np.min(vp))
        blocked[rw, cols] = v_up                           # a node on the interface belongs to the layer above
        tables = _node_tables(blocked, h, np.concatenate([s_node, r_node]), rw * nx + cols)
        nrm_x, nrm_z = interface_normals(model, k, xg)
        nrm_x, nrm_z = nrm_x[valid], nrm_z[valid]
        xc, zc = xg[cols], zk[valid]
        reach = (sz < layer.top_at(sx)) & (gz < layer.top_at(gx))
        for i in np.flatnonzero(reach):
            dsx, dsz = sx[i] - xc, sz[i] - zc
            drx, drz = gx[i] - xc, gz[i] - zc
            ds = np.maximum(np.hypot(dsx, dsz), 1e-6)
            dr = np.maximum(np.hypot(drx, drz), 1e-6)
            # the nodes lie up to one cell above the interface: add the rest of each leg (dT/dz = cos / v)
            t = tables[int(s_node[i])] + tables[int(r_node[i])] + delta * (np.abs(dsz) / ds + np.abs(drz) / dr) / v_up
            j = int(np.argmin(t))
            if not np.isfinite(t[j]):
                continue
            cos_i = abs(nrm_x[j] * dsx[j] + nrm_z[j] * dsz[j]) / ds[j]
            p = np.sqrt(max(0.0, 1.0 - cos_i ** 2)) / v_up[j]
            a = int(up_layer[j])
            r = float(reflection_coefficient(vpl[a], vsl[a], rhol[a], vpl[k], vsl[k], rhol[k], p if angle else None))
            times[i, k - 1] = t[j]
            amps[i, k - 1] = r / (ds[j] + dr[j])
        if context is not None:
            context.report(k / nl)
    return times, amps


def first_arrivals(vp, xg, zg, h, sx, sz, gx, gz):
    """First-arrival times (direct and head waves) from shortest-path traveltimes, amplitude 1 / distance."""
    nx = vp.shape[1]
    s_iz, s_ix = _nodes(sx, sz, xg, zg, h)
    r_iz, r_ix = _nodes(gx, gz, xg, zg, h)
    s_node, r_node = s_iz * nx + s_ix, r_iz * nx + r_ix
    graph = build_graph(vp, h, h)
    uniq = np.unique(s_node)
    t = np.full(sx.size, np.nan)
    for start in range(0, uniq.size, 32):
        part = uniq[start:start + 32]
        dist = dijkstra(graph, directed=True, indices=part)
        for k, node in enumerate(part):
            sel = np.flatnonzero(s_node == node)
            t[sel] = dist[k][r_node[sel]]
    return t, 1.0 / np.maximum(np.hypot(gx - sx, gz - sz), 2.0 * h)


@register
class RaySynthetics(Process):
    name = "ray_synthetics"
    label = "Ray-traced reflection synthetics (layered model)"
    category = "Modelling"
    description = ("Primary reflections by ray tracing in a layered model: in flat layers by exact two-point ray "
                   "tracing (Snell's law), otherwise by the minimum over each interface of the traveltimes from "
                   "source and receiver through the layers above it. Amplitude: reflection coefficient (normal "
                   "incidence or Aki-Richards with angle) over spherical divergence; Ricker wavelet. Optionally adds "
                   "the first arrivals (direct and head waves). Fast; the event times are kept for comparison with "
                   "picks.")
    params = _model_params() + (
        Param("method", "choice", "auto", "Ray tracing", "auto: exact for flat layers, otherwise minimum time",
              choices=("auto", "exact (flat layers)", "minimum time (grid)")),
        Param("coefficients", "choice", "normal incidence", "Reflection coefficients",
              choices=("normal incidence", "angle (Aki-Richards)")),
        Param("first_arrivals", "bool", False, "Add first arrivals (direct and head waves)"))

    def run(self, data, context):
        model = _model(self["model"], context)
        if not isinstance(model, LayeredModel):
            raise ValueError("ray synthetics need a layered model (flat layers or a model file), not the project "
                             "depth model; use Kirchhoff or finite-difference modelling for that")
        if len(model.layers) < 2 and not self["first_arrivals"]:
            raise ValueError("the model has no interface to reflect from")
        model, sx, sz, gx, gz = _stations(data, model, self["topography"], self["receiver_depth"])
        nt, dt = _time_axis(data, self["tmax"], self["dt_ms"])
        method = self["method"]
        exact = method == "exact (flat layers)" or (method == "auto" and model.is_flat)
        if exact and not model.is_flat:
            raise ValueError("the exact method needs flat layers and a flat surface; use 'minimum time (grid)'")
        angle = self["coefficients"].startswith("angle")
        times = np.zeros((data.ntr, 0))
        amps = np.zeros((data.ntr, 0))
        if exact and len(model.layers) > 1:
            times, amps = flat_reflections(model, sx, sz, gx, gz, angle)
        if not exact or self["first_arrivals"]:
            zmax = _depth_limit(model, self["zmax"])
            xlo, xhi = float(min(sx.min(), gx.min())), float(max(sx.max(), gx.max()))
            top = float(np.min(model.surface_at(np.array([xlo, xhi]))))
            pad = max(200.0, 0.5 * (zmax - top))
            span = max(xhi - xlo + 2 * pad, zmax - top)
            h = self["dx"] if self["dx"] > 0 else float(np.clip(span / 400.0, 2.0, 10.0))
            xg = xlo - pad + np.arange(int(np.ceil((xhi - xlo + 2 * pad) / h)) + 1) * h
            top = float(np.min(model.surface_at(xg)))
            zg = top + np.arange(int(np.ceil((zmax - top) / h)) + 3) * h
            context.log(f"    traveltime grid {zg.size} x {xg.size} cells of {h:g} m")
            if not exact:
                times, amps = grid_reflections(model, xg, zg, h, sx, sz, gx, gz, angle, context)
            if self["first_arrivals"]:
                tf, af = first_arrivals(model.sample(xg, zg)[0], xg, zg, h, sx, sz, gx, gz)
                times = np.concatenate([times, tf[:, None]], axis=1)
                amps = np.concatenate([amps, af[:, None]], axis=1)
        good = np.isfinite(times)
        w, c = ricker(self["f0"], dt)
        traces = events_to_traces(np.where(good, times, -1.0), np.where(good, amps, 0.0), nt, dt, w, c)
        context.results[self.name] = {"times": times, "amplitudes": amps,
                                      "method": "exact" if exact else "minimum time"}
        return _finish(data, traces, dt, "ray synthetics", self.name)
