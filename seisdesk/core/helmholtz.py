"""Small-grid, constant-density 2D acoustic frequency-domain modelling.

The five-point flux stencil uses complex-coordinate PML on every edge. With the
NumPy Fourier convention exp(-i omega t), outgoing waves and the PML have negative
phase. LU factors are reused for sources and conjugate-transpose adjoint solves.
The discrete velocity gradient includes the dependence of replicated PML velocities
on physical boundary cells. This is a CPU research implementation, not a 3D solver.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu


class Helmholtz:
    def __init__(self, velocity, dx, dz, frequency, nabs=16, x0=0.0, z0=0.0, alpha=0.0):
        self.velocity = np.asarray(velocity, dtype=float)
        if self.velocity.ndim != 2 or min(self.velocity.shape) < 2:
            raise ValueError("velocity must be a 2D grid with at least two cells in each direction")
        if not np.isfinite(self.velocity).all() or np.any(self.velocity <= 0):
            raise ValueError("velocity must be finite and positive")
        if min(dx, dz, frequency) <= 0 or alpha < 0 or nabs < 4:
            raise ValueError("positive spacing/frequency, nonnegative damping and at least four PML cells required")
        self.dx, self.dz, self.x0, self.z0 = float(dx), float(dz), float(x0), float(z0)
        self.nabs, self.frequency, self.alpha = int(nabs), float(frequency), float(alpha)
        self.omega = 2 * np.pi * frequency - 1j * alpha
        self.nz, self.nx = self.velocity.shape
        self.vpad = np.pad(self.velocity, self.nabs, mode="edge")
        self.shape = self.vpad.shape
        nz, nx = self.shape
        if nz * nx > 160_000:
            raise ValueError("frequency-domain grid exceeds 160,000 padded cells; use a smaller 2D model")
        self.matrix = self._matrix()
        self.lu = splu(self.matrix)

    def _stretch(self, n, h, half=0.0):
        q = np.arange(n, dtype=float) + half
        distance = np.maximum.reduce([self.nabs - q, q - (n - self.nabs - 1), np.zeros(n)])
        # Fixed reference speed makes the PML independent of the inversion parameters.
        sigma = 3 * 4000.0 * np.log(1e5) / (2 * self.nabs * h) * (distance / self.nabs) ** 2
        return 1 - 1j * sigma / (2 * np.pi * self.frequency)

    def _matrix(self):
        nz, nx = self.shape
        sx, sz = self._stretch(nx, self.dx), self._stretch(nz, self.dz)
        xp = 1 / (sx * self._stretch(nx, self.dx, 0.5) * self.dx ** 2)
        xm = 1 / (sx * self._stretch(nx, self.dx, -0.5) * self.dx ** 2)
        zp = 1 / (sz * self._stretch(nz, self.dz, 0.5) * self.dz ** 2)
        zm = 1 / (sz * self._stretch(nz, self.dz, -0.5) * self.dz ** 2)
        diagonal = xp[None, :] + xm[None, :] + zp[:, None] + zm[:, None] - (self.omega / self.vpad) ** 2
        right = np.broadcast_to(-xp, (nz, nx)).copy(); right[:, -1] = 0
        left = np.broadcast_to(-xm, (nz, nx)).copy(); left[:, 0] = 0
        down = np.broadcast_to(-zp[:, None], (nz, nx)).copy()
        up = np.broadcast_to(-zm[:, None], (nz, nx)).copy()
        return diags([diagonal.ravel(), right.ravel()[:-1], left.ravel()[1:],
                      down.ravel()[:-nx], up.ravel()[nx:]], [0, 1, -1, nx, -nx], format="csc")

    def sampling(self, positions):
        """Bilinear extraction matrix; its transpose injects at exactly the same points."""
        p = np.asarray(positions, dtype=float).reshape(-1, 2)
        x, z = (p[:, 0] - self.x0) / self.dx, (p[:, 1] - self.z0) / self.dz
        if not np.isfinite(p).all() or np.any((x < 0) | (x > self.nx - 1) | (z < 0) | (z > self.nz - 1)):
            raise ValueError("source or receiver lies outside the physical model grid")
        ix, iz = np.minimum(x.astype(int), self.nx - 2), np.minimum(z.astype(int), self.nz - 2)
        fx, fz = x - ix, z - iz
        base = (iz + self.nabs) * self.shape[1] + ix + self.nabs
        columns = np.stack([base, base + 1, base + self.shape[1], base + self.shape[1] + 1], axis=1)
        weights = np.stack([(1 - fx) * (1 - fz), fx * (1 - fz), (1 - fx) * fz, fx * fz], axis=1)
        return coo_matrix((weights.ravel(), (np.repeat(np.arange(p.shape[0]), 4), columns.ravel())),
                          shape=(p.shape[0], int(np.prod(self.shape)))).tocsr()

    def source(self, position):
        return self.sampling([position]).T.toarray().ravel().astype(complex) / (self.dx * self.dz)

    def solve(self, rhs, adjoint=False):
        return self.lu.solve(np.asarray(rhs, dtype=complex), trans="H" if adjoint else "N")

    def crop(self, field):
        n = self.nabs
        return np.asarray(field).reshape(self.shape)[n:n + self.nz, n:n + self.nx]

    def fold_gradient(self, gradient):
        """Adjoint of edge padding: accumulate PML sensitivities into physical boundary cells."""
        z = np.clip(np.arange(self.shape[0]) - self.nabs, 0, self.nz - 1)
        x = np.clip(np.arange(self.shape[1]) - self.nabs, 0, self.nx - 1)
        out = np.zeros(self.velocity.shape)
        np.add.at(out, (z[:, None], x[None, :]), np.asarray(gradient).reshape(self.shape))
        return out


def objective_gradient(velocity, dx, dz, frequencies, shots, observations, *, x0=0.0, z0=0.0,
                       nabs=16, alpha=0.0, estimate_source=True, source_spectra=None, report=None):
    """Least-squares objective and exact discrete adjoint gradient with respect to velocity.

    shots is a list of (source[x,z], receivers[nr,2]); observations[f][shot] is complex
    receiver data. Optional variable projection estimates one source scalar per shot
    and frequency. Its derivative vanishes at the least-squares optimum.
    """
    velocity = np.asarray(velocity, dtype=float)
    grad = np.zeros_like(velocity)
    illumination = np.zeros_like(velocity)
    loss, predictions, signatures = 0.0, [], []
    energy = sum(float(np.vdot(d, d).real) for ff in observations for d in ff)
    norm = max(energy, 1e-30)
    for fi, frequency in enumerate(frequencies):
        if report is not None:
            report(fi / max(len(frequencies), 1))
        op = Helmholtz(velocity, dx, dz, frequency, nabs, x0, z0, alpha)
        predicted, wavelets = [], []
        for si, (source, receivers) in enumerate(shots):
            if report is not None:
                report((fi + si / max(len(shots), 1)) / max(len(frequencies), 1))
            r = op.sampling(receivers)
            u0 = op.solve(op.source(source))
            p0 = r @ u0
            obs = np.asarray(observations[fi][si], dtype=complex)
            if obs.shape != p0.shape or not np.isfinite(obs).all():
                raise ValueError("observed data must be finite with one value per receiver")
            q = source_spectra[fi][si] if source_spectra is not None else 1.0 + 0j
            if estimate_source:
                q = np.vdot(p0, obs) / max(float(np.vdot(p0, p0).real), 1e-30)
            u, pred = q * u0, q * p0
            residual = pred - obs
            loss += 0.5 * float(np.vdot(residual, residual).real) / norm
            adj = op.solve(r.conj().T @ (residual / norm), adjoint=True)
            derivative = 2 * op.omega ** 2 / op.vpad.ravel() ** 3
            grad += op.fold_gradient(-np.real(np.conj(adj) * derivative * u))
            illumination += op.crop(np.abs(u) ** 2)
            predicted.append(pred); wavelets.append(q)
        predictions.append(predicted); signatures.append(wavelets)
    if report is not None:
        report(1.0)
    return loss, grad, {"predictions": predictions, "source_spectra": signatures, "illumination": illumination}
