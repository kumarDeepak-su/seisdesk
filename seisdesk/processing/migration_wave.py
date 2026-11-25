"""Wave-equation depth migrations: finite difference (45 and 65 degrees), Fourier finite difference, phase shift plus
interpolation (PSPI), prestack shot-profile migration and Gaussian-beam migration.

The post-stack methods continue a stacked (zero-offset) section down in the exploding-reflector model (half the
interval velocity) and image at t = 0, like the split-step migration in ``migration_depth``. With NumPy's transform
pair (forward exp(-i w t)) continuing down by dz multiplies by exp(+i kz dz). The prestack migration continues the
recorded wavefield and the complex conjugate of the source wavefield with that same operator and the full velocity,
and images with their zero-lag cross-correlation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter

from ..core.dataset import SeismicData
from .base import Param, Process, pad_to_time_zero, register, taper_window
from .migration_depth import _model_for
from .migration_time import _section_spacing

EDGE = 24                     # absorbing traces added on each side of the wavefield
FD_COEFFS = {"45": (0.5, 0.25), "65": (0.478242060, 0.376369527)}     # Claerbout (1985); Lee & Suh (1985)
GAMMA = 1.0 / 12.0            # d2/dx2 ~ D2 / (dx^2 (1 + D2 / 12)): fourth-order accurate second difference


# ---------------------------------------------------------------------------------------------- shared pieces
def _layout(nx: int) -> tuple[int, int]:
    """Padded width (a power of two for the FFTs) and the index of the first live trace."""
    return int(2 ** np.ceil(np.log2(nx + 2 * EDGE))), EDGE


def _damping(nxp: int, i0: int, nx: int) -> np.ndarray:
    """Damping applied at every depth step (Cerjan et al. 1985), 1 on the live traces. The padding is periodic for the
    Fourier steps, so the distance to the live traces is measured both ways round."""
    j = np.arange(nxp)
    last = i0 + nx - 1
    d = np.where(j < i0, np.minimum(i0 - j, j + nxp - last),
                 np.where(j > last, np.minimum(j - last, nxp - j + i0), 0))
    return np.exp(-(0.016 * d) ** 2)


def _pad_velocity(u: np.ndarray, nxp: int, i0: int) -> np.ndarray:
    return u[np.clip(np.arange(nxp) - i0, 0, u.size - 1)]


def _phase_shift_k(pk: np.ndarray, w: np.ndarray, kx: np.ndarray, u_ref: float, dz: float) -> np.ndarray:
    """Phase shift exp(i kz dz) of a wavefield in the kx domain for one velocity; evanescent energy is dropped."""
    kz2 = (w[None, :] / u_ref) ** 2 - kx ** 2
    prop = kz2 > 0
    return np.where(prop, pk * np.exp(1j * np.sqrt(np.where(prop, kz2, 0.0)) * dz), 0.0)


def _thin_lens(p: np.ndarray, w: np.ndarray, u: np.ndarray, dz: float) -> np.ndarray:
    return p * np.exp(1j * w[None, :] * dz / u[:, None])


def _fd_step(p: np.ndarray, w: np.ndarray, u: np.ndarray, dz: float, dx: float, a, b) -> np.ndarray:
    """Crank-Nicolson step of the diffraction term of the one-way equation kz = w/u - (w/u) a X^2 / (1 - b X^2),
    X = u kx / w (Claerbout 1985; Lee & Suh 1985), with the fourth-order second difference in x. One tridiagonal
    system per frequency; all of them are solved together as one block-diagonal banded system (LAPACK, with
    pivoting). a and b: scalars or one value per trace (Fourier finite difference)."""
    nxp, nw = p.shape
    out = np.zeros_like(p)
    live = w > 0
    if not live.any():
        return out
    ww = w[live][None, :]
    a = np.broadcast_to(np.asarray(a, dtype=np.float64), (nxp,))[:, None]
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), (nxp,))[:, None]
    uu = u[:, None]
    alpha = b * (uu / ww) ** 2 / dx ** 2
    beta = a * uu * dz / (2.0 * ww) / dx ** 2
    c_rhs = GAMMA + alpha + 1j * beta
    c_lhs = GAMMA + alpha - 1j * beta
    q = p[:, live]
    lap = -2.0 * q
    lap[1:] += q[:-1]
    lap[:-1] += q[1:]
    rhs = q + c_rhs * lap
    m = q.shape[1]
    lower = c_lhs.T.copy()                       # (frequency, x): row j couples P[j-1], P[j], P[j+1]
    upper = c_lhs.T.copy()
    diag = 1.0 - 2.0 * c_lhs.T
    lower[:, 0] = 0.0                            # zero edges: the frequencies do not couple
    upper[:, -1] = 0.0
    ab = np.empty((3, m * nxp), dtype=np.complex128)
    ab[0, 0] = 0.0
    ab[0, 1:] = upper.ravel()[:-1]
    ab[1] = diag.ravel()
    ab[2, :-1] = lower.ravel()[1:]
    ab[2, -1] = 0.0
    sol = solve_banded((1, 1), ab, rhs.T.ravel(), overwrite_ab=True, overwrite_b=True, check_finite=False)
    out[:, live] = sol.reshape(m, nxp).T
    return out


def _ssf_step(p, w, kx, u, dz, u_ref=None):
    """Split-step Fourier step (Stoffa et al. 1990): phase shift with a reference velocity, then the time-shift
    correction for the true velocity of each trace."""
    u_ref = float(np.mean(u)) if u_ref is None else float(u_ref)
    p = np.fft.ifft(_phase_shift_k(np.fft.fft(p, axis=0), w, kx, u_ref, dz), axis=0)
    return p * np.exp(1j * w[None, :] * dz * (1.0 / u[:, None] - 1.0 / u_ref))


def _ffd_step(p, w, kx, u, dz, dx):
    """Fourier finite-difference step (Ristow & Ruhl 1994): split-step with the lowest velocity c of the step, then
    a finite-difference correction for c/u < 1 with a = (1 - c/u)/2 and b = (1 + c/u + (c/u)^2)/4."""
    c = float(u.min())
    p = _ssf_step(p, w, kx, u, dz, u_ref=c)
    ratio = c / u
    if float(np.max(1.0 - ratio)) > 1e-6:
        p = _fd_step(p, w, u, dz, dx, 0.5 * (1.0 - ratio), 0.25 * (1.0 + ratio + ratio ** 2))
    return p


def _pspi_step(p, w, kx, u, dz, max_refs=8):
    """Phase shift plus interpolation (Gazdag & Sguazzero 1984): the vertical time shift with the true velocity,
    then the diffraction term exp(i (kz - w/u_r) dz) for reference velocities u_r about 10 % apart, interpolated
    linearly in velocity at each trace."""
    p = _thin_lens(p, w, u, dz)
    pk = np.fft.fft(p, axis=0)
    umin, umax = float(u.min()), float(u.max())
    if umax <= 1.001 * umin or max_refs < 2:
        refs = np.array([0.5 * (umin + umax)])
    else:
        n = int(min(max_refs, np.ceil(np.log(umax / umin) / np.log(1.1)) + 1))
        refs = np.geomspace(umin, umax, max(n, 2))
    fields = []
    for ur in refs:
        kz2 = (w[None, :] / ur) ** 2 - kx ** 2
        prop = kz2 > 0
        phase = np.sqrt(np.where(prop, kz2, 0.0)) - w[None, :] / ur
        fields.append(np.fft.ifft(np.where(prop, pk * np.exp(1j * phase * dz), 0.0), axis=0))
    if refs.size == 1:
        return fields[0]
    k = np.clip(np.searchsorted(refs, u, side="right") - 1, 0, refs.size - 2)
    wgt = np.clip((u - refs[k]) / (refs[k + 1] - refs[k]), 0.0, 1.0)
    f = np.stack(fields)
    cols = np.arange(u.size)
    return (1.0 - wgt)[:, None] * f[k, cols] + wgt[:, None] * f[k + 1, cols]


def _step(method: str, p, w, kx, u, dz, dx, opts: dict):
    if method == "split-step":
        return _ssf_step(p, w, kx, u, dz)
    if method == "PSPI":
        return _pspi_step(p, w, kx, u, dz, int(opts.get("max_refs", 8)))
    if method == "FFD":
        return _ffd_step(p, w, kx, u, dz, dx)
    a, b = FD_COEFFS[str(opts.get("angle", "65"))]
    return _fd_step(_thin_lens(p, w, u, dz), w, u, dz, dx, a, b)


# ---------------------------------------------------------------------------------------------- post-stack
def poststack_migration(section: np.ndarray, dt: float, dx: float, vel: np.ndarray, dz: float,
                        fmax: float | None = None, method: str = "FD", progress=None, **opts) -> np.ndarray:
    """Exploding-reflector depth migration of a zero-offset section (nx, nt) through the interval velocity vel
    (nz, nx) on the section's traces, with the extrapolator 'FD' (opts angle='45' or '65'), 'FFD', 'PSPI' (opts
    max_refs) or 'split-step'. Returns the depth image (nz, nx)."""
    nx, nt = section.shape
    nz = vel.shape[0]
    nxp, i0 = _layout(nx)
    ntp = int(2 ** np.ceil(np.log2(2 * nt)))
    w = 2 * np.pi * np.fft.rfftfreq(ntp, dt)
    nw = w.size if fmax is None else max(2, int(np.searchsorted(w, 2 * np.pi * fmax, side="right")))
    w = w[:nw]
    p = np.zeros((nxp, nw), dtype=np.complex128)
    p[i0:i0 + nx] = np.fft.rfft(section.astype(np.float64), ntp, axis=1)[:, :nw]
    p[:, 0] = 0.0                                # zero frequency: no wave, and the FD coefficients divide by w
    kx = 2 * np.pi * np.fft.fftfreq(nxp, dx)[:, None]
    damp = _damping(nxp, i0, nx)[:, None]
    vel = np.maximum(np.asarray(vel, dtype=np.float64), 1.0)
    image = np.zeros((nz, nx))
    for iz in range(nz):
        image[iz] = 2.0 * np.real(p[i0:i0 + nx].sum(axis=1)) / ntp
        if iz == nz - 1:
            break
        u = _pad_velocity(0.5 * vel[iz], nxp, i0)
        p = _step(method, p, w, kx, u, dz, dx, opts) * damp
        if progress is not None and iz % 10 == 0:
            progress(iz / nz)
    return image


# ---------------------------------------------------------------------------------------------- prestack
def _receiver_spacing(gx: np.ndarray) -> float:
    ux = np.unique(np.round(np.asarray(gx, dtype=np.float64), 3))
    d = np.diff(ux)
    d = d[d > 1e-6]
    return float(np.median(d)) if d.size else 25.0


def shot_profile_migration(traces: np.ndarray, dt: float, sx: np.ndarray, gx: np.ndarray, shot_ids: np.ndarray,
                           x0: float, dx: float, nx: int, velocity, dz: float, nz: int, fmin: float, fmax: float,
                           method: str = "split-step", aperture: float = 1000.0, source: str = "impulse",
                           f0: float = 25.0, normalise: bool = True, progress=None, **opts) -> np.ndarray:
    """Shot-profile wave-equation depth migration (Claerbout 1971; Biondi 2006) on the image grid x0 + i dx.

    velocity(z, x): interval velocity at arrays of points. For each shot the receiver wavefield and the conjugated
    source wavefield (impulse or zero-phase Ricker at the source position) are continued down with the chosen
    extrapolator ('split-step', 'PSPI' or 'FFD'); the image is the zero-lag cross-correlation summed over
    frequencies, optionally divided by the source illumination, and summed over shots. Returns (nz, nx)."""
    ntr, nt = traces.shape
    ntp = int(2 ** np.ceil(np.log2(max(8, int(1.5 * nt)))))
    f = np.fft.rfftfreq(ntp, dt)
    band = np.flatnonzero((f > 0) & (f >= fmin) & (f <= fmax))
    if band.size == 0:
        raise ValueError("no frequencies between the lowest and highest frequency")
    w = 2 * np.pi * f[band]
    taper = taper_window(band.size, 0.1)
    spec = np.fft.rfft(traces.astype(np.float64), ntp, axis=1)[:, band] * taper[None, :]
    if source == "Ricker":
        wr = 2 * np.pi * f0
        src = (w / wr) ** 2 * np.exp(1.0 - (w / wr) ** 2)          # zero-phase Ricker spectrum, 1 at f0
    else:
        src = np.ones(w.size)
    src = src * taper
    nap = int(round(aperture / dx))
    ncomp = nx + 2 * nap                         # the image grid plus the aperture on both sides
    xc0 = x0 - nap * dx
    image = np.zeros((nz, ncomp))
    z = np.arange(nz) * dz
    shots = np.unique(shot_ids)
    for n, s in enumerate(shots):
        idx = np.flatnonzero(shot_ids == s)
        xs = float(np.median(sx[idx]))
        xr = gx[idx]
        lo = max(int(np.floor((min(xs, xr.min()) - aperture - xc0) / dx)), 0)
        hi = min(int(np.ceil((max(xs, xr.max()) + aperture - xc0) / dx)) + 1, ncomp)
        nxl = hi - lo
        if nxl < 2:
            continue
        nxp, i0 = _layout(nxl)
        scol = int(np.rint((xs - xc0) / dx)) - lo + i0
        if not i0 <= scol < i0 + nxl:
            continue
        kx = 2 * np.pi * np.fft.fftfreq(nxp, dx)[:, None]
        damp = _damping(nxp, i0, nxl)[:, None]
        xl = xc0 + (lo + np.arange(nxl)) * dx
        vloc = np.maximum(np.asarray(velocity(np.repeat(z, nxl), np.tile(xl, nz)), dtype=np.float64), 1.0)
        vloc = vloc.reshape(nz, nxl)
        rec = np.zeros((nxp, w.size), dtype=np.complex128)
        cols = np.rint((xr - xc0) / dx).astype(np.int64) - lo + i0
        ok = (cols >= i0) & (cols < i0 + nxl)
        np.add.at(rec, cols[ok], spec[idx[ok]])
        srcf = np.zeros((nxp, w.size), dtype=np.complex128)
        srcf[scol] = src                         # a real spectrum is its own conjugate
        corr = np.zeros((nz, nxl))
        illum = np.zeros((nz, nxl))
        for iz in range(nz):
            s_live, r_live = srcf[i0:i0 + nxl], rec[i0:i0 + nxl]
            corr[iz] = np.real(np.sum(s_live * r_live, axis=1))
            illum[iz] = np.sum(np.abs(s_live) ** 2, axis=1)
            if iz == nz - 1:
                break
            u = _pad_velocity(vloc[iz], nxp, i0)
            rec = _step(method, rec, w, kx, u, dz, dx, opts) * damp
            srcf = _step(method, srcf, w, kx, u, dz, dx, opts) * damp
            if progress is not None and iz % 20 == 0:
                progress((n + iz / nz) / shots.size)
        if normalise:
            corr = corr / (illum + 1e-3 * float(illum.max()) + 1e-30)
        image[:, lo:hi] += corr
    return image[:, nap:nap + nx]


# ---------------------------------------------------------------------------------------------- Gaussian beams
def _bilinear(fields: np.ndarray, iz: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of stacked grids (nf, nz, nx) at fractional indices; (nf, n)."""
    nzf, nxf = fields.shape[1:]
    iz = np.clip(iz, 0.0, nzf - 1.000001)
    ix = np.clip(ix, 0.0, nxf - 1.000001)
    i0 = np.floor(iz).astype(np.int64)
    j0 = np.floor(ix).astype(np.int64)
    fz = iz - i0
    fx = ix - j0
    top = (1 - fx) * fields[:, i0, j0] + fx * fields[:, i0, j0 + 1]
    bottom = (1 - fx) * fields[:, i0 + 1, j0] + fx * fields[:, i0 + 1, j0 + 1]
    return (1 - fz) * top + fz * bottom


def _ray_rhs(y: np.ndarray, fields: np.ndarray, dx: float, dz: float) -> np.ndarray:
    """Kinematic and dynamic ray equations with traveltime as the parameter (Cerveny 2001): state x, z, px, pz, then
    two real solutions (q1, p1) and (q2, p2) of dQ/dt = u^2 P, dP/dt = -(u_nn / u) Q."""
    x, z, px, pz, q1, p1, q2, p2 = y
    u, ux, uz, uxx, uxz, uzz = _bilinear(fields, z / dz, x / dx)
    u2 = u * u
    tx, tz = u * px, u * pz                      # unit tangent; the normal is (tz, -tx)
    unn = uxx * tz * tz - 2.0 * uxz * tz * tx + uzz * tx * tx
    return np.stack([u2 * px, u2 * pz, -ux / u, -uz / u, u2 * p1, -unn / u * q1, u2 * p2, -unn / u * q2])


def _trace_beams(x0, s0, u0, fields, dx, dz, nx, nz, dtau, nsteps, tmax):
    """Rays leaving the surface at x0 (relative to the first trace) with sin(theta) = s0, traced with fourth-order
    Runge-Kutta steps of dtau (less than half a depth step). Returns, for every depth row a ray crosses on its way
    down, a table (nz, nrays) of x, traveltime, direction, velocity and the dynamic solutions, and a validity mask."""
    nr = x0.size
    y = np.zeros((8, nr))
    cos0 = np.sqrt(1.0 - s0 ** 2)
    y[0], y[2], y[3], y[4], y[7] = x0, s0 / u0, cos0 / u0, 1.0, 1.0
    tau = np.zeros(nr)
    tab = {k: np.zeros((nz, nr)) for k in ("x", "tau", "sx", "cz", "u", "q1", "p1", "q2", "p2")}
    tab["x"][0], tab["sx"][0], tab["cz"][0], tab["u"][0], tab["q1"][0], tab["p2"][0] = x0, s0, cos0, u0, 1.0, 1.0
    valid = np.zeros((nz, nr), dtype=bool)
    valid[0] = True
    active = np.ones(nr, dtype=bool)
    xmax, zmax, h = (nx - 1) * dx, (nz - 1) * dz, dtau
    for _ in range(nsteps):
        ids = np.flatnonzero(active)
        if ids.size == 0:
            break
        ya = y[:, ids]
        k1 = _ray_rhs(ya, fields, dx, dz)
        k2 = _ray_rhs(ya + 0.5 * h * k1, fields, dx, dz)
        k3 = _ray_rhs(ya + 0.5 * h * k2, fields, dx, dz)
        k4 = _ray_rhs(ya + h * k3, fields, dx, dz)
        yn = ya + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        un = _bilinear(fields[:1], yn[1] / dz, yn[0] / dx)[0]
        norm = un * np.hypot(yn[2], yn[3])       # keep |p| = 1/u
        yn[2] /= norm
        yn[3] /= norm
        tn = tau[ids] + h
        z_old, z_new = ya[1], yn[1]
        row = np.floor(z_new / dz + 1e-9).astype(np.int64)
        cross = (row > np.floor(z_old / dz + 1e-9)) & (row < nz) & (z_new > z_old)
        if cross.any():
            c = ids[cross]
            r = (row[cross] * dz - z_old[cross]) / (z_new[cross] - z_old[cross])
            yi = ya[:, cross] + r * (yn[:, cross] - ya[:, cross])
            ui = _bilinear(fields[:1], row[cross].astype(np.float64), yi[0] / dx)[0]
            sx, cz = ui * yi[2], ui * yi[3]
            s = np.hypot(sx, cz)
            rows = row[cross]
            tab["x"][rows, c] = yi[0]
            tab["tau"][rows, c] = tau[c] + r * h
            tab["sx"][rows, c] = sx / s
            tab["cz"][rows, c] = cz / s
            tab["u"][rows, c] = ui
            for k, i in (("q1", 4), ("p1", 5), ("q2", 6), ("p2", 7)):
                tab[k][rows, c] = yi[i]
            valid[rows, c] = True
        y[:, ids] = yn
        tau[ids] = tn
        alive = (z_new <= zmax) & (yn[0] >= -dx) & (yn[0] <= xmax + dx) & (yn[3] > 0) & (tn <= tmax)
        active[ids[~alive]] = False
    return tab, valid


def _trace_beams_vti(x0, px0, fields, dx, dz, nx, nz, eta):
    """Depth-domain Hamiltonian rays and paraxial variations for acoustic VTI, delta=0.

    Vertical slowness q = sqrt((1/u²-(1+2eta)p²)/(1-2eta*u²*p²)).
    x'=-q_p, p'=q_x and tau'=q-p*q_p. Q/P are variations in the horizontal
    plane at fixed depth, so P/Q is the horizontal travel-time curvature.
    """
    count = len(x0)
    state = np.zeros((7, count)); state[0] = x0; state[1] = px0; state[3] = 1; state[6] = 1
    table = {k: np.zeros((nz, count)) for k in ('x', 'tau', 'sx', 'cz', 'u', 'q1', 'p1', 'q2', 'p2', 'px')}
    valid = np.zeros((nz, count), bool); active = np.ones(count, bool)

    def slowness(u, p):
        numerator = 1/u**2-(1+2*eta)*p*p; denominator = 1-2*eta*u*u*p*p
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where((numerator > 0) & (denominator > 0), np.sqrt(numerator/denominator), np.nan)

    def rhs(y, depth):
        pos = np.nan_to_num(y[0], nan=0, posinf=(nx-1)*dx, neginf=0)
        u, ux, _, uxx, _, _ = _bilinear(fields, np.full(count, depth/dz), pos/dx)
        p = y[1]; hu, hp = 1e-4*u, 1e-4/u
        q = slowness(u, p)
        up, um = slowness(u+hu, p), slowness(u-hu, p)
        pp, pm = slowness(u, p+hp), slowness(u, p-hp)
        qu = (up-um)/(2*hu); qp = (pp-pm)/(2*hp)
        quu = (up-2*q+um)/hu**2; qpp = (pp-2*q+pm)/hp**2
        qup = (slowness(u+hu, p+hp)-slowness(u+hu, p-hp)
               -slowness(u-hu, p+hp)+slowness(u-hu, p-hp))/(4*hu*hp)
        qx, qxx, qxp = qu*ux, quu*ux*ux+qu*uxx, qup*ux
        return np.stack([-qp, qx, q-p*qp, -qxp*y[3]-qpp*y[4], qxx*y[3]+qxp*y[4],
                          -qxp*y[5]-qpp*y[6], qxx*y[5]+qxp*y[6]])

    for iz in range(nz):
        u = _bilinear(fields[:1], np.full(count, iz, float), np.nan_to_num(state[0])/dx)[0]
        q = slowness(u, state[1])
        active &= np.all(np.isfinite(state), axis=0) & np.isfinite(q) & (state[0] >= -dx) & (state[0] <= nx*dx)
        valid[iz] = active
        for key, index in (('x', 0), ('px', 1), ('tau', 2), ('q1', 3), ('p1', 4), ('q2', 5), ('p2', 6)):
            table[key][iz] = np.where(active, state[index], 0)
        table['u'][iz] = u; table['sx'][iz] = state[1]*u; table['cz'][iz] = 1
        if iz == nz-1 or not np.any(active):
            break
        # Quarter-depth RK4 steps keep rapidly changing beam curvature resolved.
        h = dz/4
        for substep in range(4):
            z = iz*dz+substep*h
            k1 = rhs(state, z); k2 = rhs(state+.5*h*k1, z+.5*h)
            k3 = rhs(state+.5*h*k2, z+.5*h); k4 = rhs(state+h*k3, z+h)
            state += h/6*(k1+2*k2+2*k3+k4)
            # Invalid rays remain inactive; safe values prevent invalid interpolation indices.
            active &= np.all(np.isfinite(state), axis=0)
            state[:, ~active] = 0
    return table, valid


def _spread_beams(image, tab, valid, traces, u0, eps, w_ref, dx, dt):
    """Add every beam to the image: at each depth row the time of a point at horizontal distance h from the ray is
    tau + h sin/u + Re(M) n^2 / 2 (n = h cos, M = P/Q), weighted by the Gaussian exp(-w_ref Im(M) n^2 / 2) and the
    beam amplitude sqrt(|Q0| u / (u0 |Q|))."""
    nz, nx = image.shape
    nt = traces.shape[1]
    tlast = (nt - 1) * dt
    for r in range(traces.shape[0]):
        rows = np.flatnonzero(valid[:, r])
        if rows.size == 0:
            continue
        xr, tau, sx, cz, ur = (tab[k][rows, r] for k in ("x", "tau", "sx", "cz", "u"))
        q = eps * tab["q1"][rows, r] + tab["q2"][rows, r]
        p = eps * tab["p1"][rows, r] + tab["p2"][rows, r]
        m = p / q
        mi = np.maximum(m.imag, 1e-30)
        amp = np.sqrt(abs(eps) * ur / (u0[r] * np.abs(q)))
        hmax = 3.0 / np.sqrt(w_ref * mi) / np.maximum(cz, 0.1)
        half = int(min(nx, 500, np.ceil(float(hmax.max()) / dx)))
        cols = np.rint(xr / dx).astype(np.int64)[:, None] + np.arange(-half, half + 1)[None, :]
        hh = cols * dx - xr[:, None]
        nn = hh * cz[:, None]
        slope = tab['px'][rows, r] if 'px' in tab else sx/ur
        t = tau[:, None] + hh * slope[:, None] + 0.5 * m.real[:, None] * nn ** 2
        keep = (cols >= 0) & (cols < nx) & (np.abs(hh) <= hmax[:, None]) & (t >= 0) & (t < tlast)
        if not keep.any():
            continue
        g = amp[:, None] * np.exp(-0.5 * w_ref * mi[:, None] * nn ** 2)
        pos = t[keep] / dt
        i0 = np.minimum(np.floor(pos).astype(np.int64), nt - 2)
        fr = pos - i0
        tr = traces[r]
        val = (1 - fr) * tr[i0] + fr * tr[i0 + 1]
        rr = np.broadcast_to(rows[:, None], cols.shape)[keep]
        image[rr, cols[keep]] += g[keep] * val


def gaussian_beam_migration(section: np.ndarray, dt: float, dx: float, vel: np.ndarray, dz: float,
                            fmin: float = 5.0, fmax: float = 50.0, beam_width: float = 0.0, max_angle: float = 60.0,
                            smooth_m: float = 300.0, progress=None, log=None, eta: float = 0.0) -> np.ndarray:
    """Zero-offset Gaussian-beam depth migration (Hill 1990) in the exploding-reflector model (half velocity).

    The section is decomposed into local plane waves: slant stacks under Gaussian windows (a partition of unity) at
    beam centres one beam width apart. Each local plane wave is sent down along a ray traced in the smoothed model
    and spread as a Gaussian beam. Simplifications: one beam width for all frequencies (set at the band's centre
    frequency), real traveltimes with the paraxial correction, beam amplitudes without the KMAH phase.
    Returns the depth image (nz, nx)."""
    nx, nt = section.shape
    nz = vel.shape[0]
    if nx < 2 or nz < 2 or nt < 2:
        raise ValueError("Gaussian-beam migration needs at least two traces, samples and depth steps")
    x = np.arange(nx) * dx
    u = 0.5 * np.maximum(np.asarray(vel, dtype=np.float64), 1.0)
    if smooth_m > 0:
        u = gaussian_filter(u, (smooth_m / dz, smooth_m / dx), mode="nearest")
    uz, ux = np.gradient(u, dz, dx)
    uzz, uzx = np.gradient(uz, dz, dx)
    uxz, uxx = np.gradient(ux, dz, dx)
    fields = np.stack([u, ux, uz, uxx, 0.5 * (uzx + uxz), uzz])
    ntp = int(2 ** np.ceil(np.log2(2 * nt)))
    f = np.fft.rfftfreq(ntp, dt)
    band = np.flatnonzero((f > 0) & (f >= fmin) & (f <= fmax))
    if band.size == 0:
        raise ValueError("no frequencies between the lowest and highest frequency")
    wb = 2.0 * np.pi * f[band]
    f_hi = float(f[band[-1]])
    f_ref = 0.5 * (float(f[band[0]]) + f_hi)
    w_ref = 2.0 * np.pi * f_ref
    w0 = beam_width if beam_width > 0 else max(3.0 * dx, float(u[0].mean()) / f_ref)
    spec = np.fft.rfft(section.astype(np.float64), ntp, axis=1)[:, band] * taper_window(band.size, 0.1)[None, :]
    nc = int(np.floor(x[-1] / w0)) + 1
    centres = np.linspace(0.0, x[-1], nc) if nc > 1 else np.array([0.5 * x[-1]])
    nc = centres.size
    win = np.exp(-0.5 * ((x[None, :] - centres[:, None]) / w0) ** 2)
    win /= np.maximum(win.sum(axis=0, keepdims=True), 1e-12)
    sin_max = np.sin(np.deg2rad(max_angle))
    dp = 0.7 / (2.0 * np.pi * f_hi * w0)        # slowness step: below the slant-stack resolution at f_hi
    npl = int(min(200, np.ceil(sin_max / float(np.interp(centres, x, u[0]).min()) / dp)))
    ps = np.arange(-npl, npl + 1) * dp
    eps = -0.5j * w_ref * w0 ** 2               # Q(0) with P(0) = 1: M(0) = 2i / (w_ref w0^2), flat wavefront
    tmax = (nt - 1) * dt
    dtau = 0.5 * min(dz, dx) / float(u.max())
    nsteps = int(np.ceil(tmax / dtau))
    if log is not None:
        log(f"    {nc} beam centres, {ps.size} slownesses, beam width {w0:.0f} m")
    image = np.zeros((nz, nx))
    group = max(1, 1500 // ps.size)
    for g0 in range(0, nc, group):
        cs = np.arange(g0, min(nc, g0 + group))
        beams = np.zeros((cs.size, ps.size, nt))
        for k, ic in enumerate(cs):
            sel = np.flatnonzero(win[ic] > 1e-4)
            h = x[sel] - centres[ic]
            local = win[ic, sel][:, None] * spec[sel]
            stack = np.empty((ps.size, wb.size), dtype=np.complex128)
            for j0 in range(0, wb.size, 64):                     # frequency blocks keep the phase array small
                phase = np.exp(1j * wb[None, None, j0:j0 + 64] * ps[:, None, None] * h[None, :, None])
                stack[:, j0:j0 + 64] = np.einsum("psw,sw->pw", phase, local[:, j0:j0 + 64])
            stack *= (wb * dp * dx / (2.0 * np.pi))[None, :]
            full = np.zeros((ps.size, ntp // 2 + 1), dtype=np.complex128)
            full[:, band] = stack
            beams[k] = np.fft.irfft(full, ntp, axis=1)[:, :nt]
        cc = np.repeat(cs, ps.size)
        u0 = np.interp(centres[cc], x, u[0])
        s0 = -np.tile(ps, cs.size) * u0         # the ray continued down heads towards the event
        ok = np.abs(s0) < sin_max
        if ok.any():
            if eta == 0:
                tab, valid = _trace_beams(centres[cc][ok], s0[ok], u0[ok], fields, dx, dz, nx, nz, dtau, nsteps, tmax)
            else:
                tab, valid = _trace_beams_vti(centres[cc][ok], s0[ok]/u0[ok], fields, dx, dz, nx, nz, eta)
            _spread_beams(image, tab, valid, beams.reshape(-1, nt)[ok], u0[ok], eps, w_ref, dx, dt)
        if progress is not None:
            progress(float(cs[-1] + 1) / nc)
    return image


# ---------------------------------------------------------------------------------------------- steps
_VELOCITY = Param("velocity", "velocity", "project model", "Velocity model", "'project model', picks or a file")


class _PostStackWave(Process):
    """A post-stack depth migration of a stacked section in v(x, z); subclasses supply migrate()."""
    category = "Depth migration"

    def migrate(self, section, dt, dx, vel, dz, context):              # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, data, context):
        data = pad_to_time_zero(data)
        if data.has_header("cdp"):
            data = data.sort("cdp")
        nz = int(self["zmax"] / self["dz"]) + 1
        model, x = _model_for(data, context, nz, self["dz"], self["velocity"], self["smooth"])
        if x.size != data.ntr:
            raise ValueError("post-stack migration needs a stacked section with one trace per CMP")
        vel = np.stack([model.at(np.full(x.size, z), x) for z in np.arange(nz) * self["dz"]])
        img = self.migrate(data.traces, data.dt, _section_spacing(data), np.maximum(vel, 1.0), self["dz"], context)
        out = SeismicData(img.T.astype(np.float32), self["dz"], {k: v.copy() for k, v in data.headers.items()}, 0.0,
                          data.name + " depth", "depth", list(data.history), dict(data.meta))
        out.meta["migrated"] = self.name
        return out


@register
class FDMigration(_PostStackWave):
    name = "fd_migration"
    label = "Finite-difference depth migration (post-stack, 45/65 degrees)"
    description = ("Depth migration of a stacked section by finite differences in frequency and space: Claerbout's "
                   "45-degree equation, or the 65-degree equation of Lee and Suh, solved implicitly at every depth "
                   "step. Follows strong lateral velocity changes; dips steeper than the chosen angle are "
                   "under-migrated. Velocity: the project depth model, or a Dix model from the picks. Output is a "
                   "depth section.")
    params = (_VELOCITY,
              Param("angle", "choice", "65", "Dip accuracy", choices=("45", "65"), unit="deg"),
              Param("dz", "float", 10.0, "Depth step", unit="m", minimum=0.5),
              Param("zmax", "float", 4000.0, "Maximum depth", unit="m", minimum=10.0),
              Param("smooth", "float", 200.0, "Smoothing of a Dix model", unit="m", minimum=0.0),
              Param("fmax", "float", 60.0, "Highest frequency", unit="Hz", minimum=1.0))

    def migrate(self, section, dt, dx, vel, dz, context):
        return poststack_migration(section, dt, dx, vel, dz, self["fmax"], "FD", context.report, angle=self["angle"])


@register
class FFDMigration(_PostStackWave):
    name = "ffd_migration"
    label = "Fourier finite-difference depth migration (post-stack)"
    description = ("Depth migration of a stacked section by the Fourier finite-difference method of Ristow and Ruhl: "
                   "at every depth step a phase shift with the lowest velocity of the step, a time-shift correction "
                   "for the true velocity and a finite-difference correction for the rest. Accurate for steep dips "
                   "and strong lateral velocity changes. Output is a depth section.")
    params = (_VELOCITY,
              Param("dz", "float", 10.0, "Depth step", unit="m", minimum=0.5),
              Param("zmax", "float", 4000.0, "Maximum depth", unit="m", minimum=10.0),
              Param("smooth", "float", 200.0, "Smoothing of a Dix model", unit="m", minimum=0.0),
              Param("fmax", "float", 60.0, "Highest frequency", unit="Hz", minimum=1.0))

    def migrate(self, section, dt, dx, vel, dz, context):
        return poststack_migration(section, dt, dx, vel, dz, self["fmax"], "FFD", context.report)


@register
class PSPIMigration(_PostStackWave):
    name = "pspi_migration"
    label = "Phase-shift-plus-interpolation depth migration (post-stack)"
    description = ("Depth migration of a stacked section by phase shift plus interpolation (Gazdag and Sguazzero): "
                   "at every depth step the wavefield is continued with a few reference velocities, about 10 % "
                   "apart, and interpolated with the true velocity at each trace. Exact for any dip where the "
                   "velocity is laterally constant; cost grows with the number of reference velocities.")
    params = (_VELOCITY,
              Param("dz", "float", 10.0, "Depth step", unit="m", minimum=0.5),
              Param("zmax", "float", 4000.0, "Maximum depth", unit="m", minimum=10.0),
              Param("smooth", "float", 200.0, "Smoothing of a Dix model", unit="m", minimum=0.0),
              Param("fmax", "float", 60.0, "Highest frequency", unit="Hz", minimum=1.0),
              Param("references", "int", 8, "Maximum reference velocities per step", minimum=1, maximum=30))

    def migrate(self, section, dt, dx, vel, dz, context):
        return poststack_migration(section, dt, dx, vel, dz, self["fmax"], "PSPI", context.report,
                                   max_refs=self["references"])


@register
class GaussianBeamMigration(_PostStackWave):
    name = "gaussian_beam"
    label = "Gaussian-beam migration (post-stack)"
    description = ("Depth migration of a stacked section by Gaussian beams (Hill 1990): the section is split into "
                   "local plane waves (slant stacks under overlapping Gaussian windows); each is sent down along a "
                   "ray traced in the smoothed velocity model and spread as a Gaussian beam. Handles caustics and "
                   "several arrivals, which single-arrival Kirchhoff migration does not. Simplified: one beam width "
                   "for all frequencies and real traveltimes. Optional acoustic VTI uses delta=0 and constant eta, "
                   "with Hamiltonian rays and horizontal paraxial curvature; amplitudes remain approximate. "
                   "Beam width 0 = automatic (about one wavelength).")
    params = (_VELOCITY,
              Param("dz", "float", 10.0, "Depth step", unit="m", minimum=0.5),
              Param("zmax", "float", 4000.0, "Maximum depth", unit="m", minimum=10.0),
              Param("smooth", "float", 300.0, "Model smoothing for the rays", unit="m", minimum=0.0),
              Param("fmin", "float", 5.0, "Lowest frequency", unit="Hz", minimum=0.0),
              Param("fmax", "float", 50.0, "Highest frequency", unit="Hz", minimum=1.0),
              Param("beam_width", "float", 0.0, "Initial beam width (0 = automatic)", unit="m", minimum=0.0),
              Param("max_angle", "float", 60.0, "Maximum ray angle", unit="deg", minimum=5.0, maximum=85.0),
              Param("eta", "float", 0.0, "Acoustic VTI eta (delta=0)", minimum=-.1, maximum=.3))

    def migrate(self, section, dt, dx, vel, dz, context):
        return gaussian_beam_migration(section, dt, dx, vel, dz, self["fmin"], self["fmax"], self["beam_width"],
                                       self["max_angle"], self["smooth"], context.report, context.log, eta=self["eta"])


@register
class WavePSDM(Process):
    name = "wave_psdm"
    label = "Prestack wave-equation depth migration (shot profile)"
    category = "Depth migration"
    description = ("Prestack depth migration shot by shot: the source wavefield (from the source position) and the "
                   "recorded wavefield are continued down through v(x, z) with a split-step, PSPI or Fourier "
                   "finite-difference extrapolator and correlated at zero lag; the images of all shots are summed. "
                   "Needs shot gathers with source and receiver x. The recording surface is taken as flat: on land "
                   "apply statics or datuming first. Output is a depth image on a regular x grid.")
    params = (_VELOCITY,
              Param("extrapolator", "choice", "split-step", "Extrapolator", choices=("split-step", "PSPI", "FFD", "finite difference")),
              Param("angle", "choice", "65", "Finite-difference dip accuracy", choices=("45", "65"), unit="deg"),
              Param("dz", "float", 10.0, "Depth step", unit="m", minimum=0.5),
              Param("zmax", "float", 3000.0, "Maximum depth", unit="m", minimum=10.0),
              Param("dx", "float", 0.0, "Image trace spacing (0 = receiver spacing)", unit="m", minimum=0.0),
              Param("aperture", "float", 1000.0, "Aperture beyond the spread", unit="m", minimum=0.0),
              Param("fmin", "float", 3.0, "Lowest frequency", unit="Hz", minimum=0.0),
              Param("fmax", "float", 40.0, "Highest frequency", unit="Hz", minimum=1.0),
              Param("source", "choice", "impulse", "Source wavelet", choices=("impulse", "Ricker")),
              Param("source_f0", "float", 25.0, "Ricker peak frequency", unit="Hz", minimum=1.0),
              Param("normalise", "bool", True, "Divide by the source illumination"),
              Param("smooth", "float", 200.0, "Smoothing of a Dix model", unit="m", minimum=0.0))

    def run(self, data, context):
        data = pad_to_time_zero(data)
        sx, gx = data.header("sx"), data.header("gx")
        if np.ptp(np.concatenate([sx, gx])) <= 0:
            raise ValueError("prestack migration needs source and receiver x (sx, gx): run a geometry step")
        shot_ids = data.header("fldr") if data.has_header("fldr") else sx
        nz = int(self["zmax"] / self["dz"]) + 1
        model, _ = _model_for(data, context, nz, self["dz"], self["velocity"], self["smooth"])
        dx = self["dx"] if self["dx"] > 0 else _receiver_spacing(gx)
        x0 = float(min(sx.min(), gx.min()))
        nx = int(round((float(max(sx.max(), gx.max())) - x0) / dx)) + 1
        if nx > 20000:
            raise ValueError(f"{nx} image traces: increase the image trace spacing")
        context.log(f"    {np.unique(shot_ids).size} shots, image {nz} x {nx} ({dx:g} m)")

        def velocity(z, x):
            return model.at(z, x)

        img = shot_profile_migration(data.traces, data.dt, sx, gx, shot_ids, x0, dx, nx, velocity, self["dz"], nz,
                                     self["fmin"], self["fmax"], self["extrapolator"], self["aperture"],
                                     self["source"], self["source_f0"], self["normalise"], context.report, angle=self["angle"])
        xg = x0 + np.arange(nx) * dx
        heads = {"cdp": np.arange(1, nx + 1, dtype=np.float64), "cdpx": xg, "sx": xg, "gx": xg,
                 "offset": np.zeros(nx)}
        meta = dict(data.meta)
        meta["geometry"] = dict(meta.get("geometry", {}), bin_size=dx)
        meta["migrated"] = "wave_psdm"
        return SeismicData(img.T.astype(np.float32), self["dz"], heads, 0.0, data.name + " WEM", "depth",
                           list(data.history), meta)
