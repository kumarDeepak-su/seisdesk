"""Parabolic Radon transform (least squares or high-resolution sparse, frequency domain) and Radon demultiple."""

from __future__ import annotations

import numpy as np

from .base import Param, Process, register


def parabolic_radon(gather: np.ndarray, offsets: np.ndarray, dt: float, q: np.ndarray, damping: float = 0.01,
                    fmin: float = 0.0, fmax: float | None = None, xref: float | None = None,
                    sparse_iterations: int = 0, sparsity: float = 0.05):
    """Least-squares Radon model m(tau, q) of a gather, parabola t = tau + q (x / xref)^2.

    With sparse_iterations > 0 every frequency is refined by iteratively reweighted least squares with a Cauchy
    model norm (Sacchi & Ulrych 1995): the damping of a coefficient falls as |m|^2 grows past sparsity x the largest
    |m|^2, which focuses the events in q (high resolution) and separates close curvatures.

    Returns the model in the frequency domain (nq, nf), the operator builder and the FFT length."""
    ntr, nt = gather.shape
    xref = float(np.max(np.abs(offsets))) if xref is None else float(xref)
    xn2 = (np.asarray(offsets, dtype=np.float64) / max(xref, 1e-9)) ** 2
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    freqs = np.fft.rfftfreq(nfft, dt)
    D = np.fft.rfft(gather.astype(np.float64), nfft, axis=1)              # (ntr, nf)
    fmax = freqs[-1] if fmax is None else fmax
    M = np.zeros((q.size, freqs.size), dtype=np.complex128)
    band = np.flatnonzero((freqs >= fmin) & (freqs <= fmax))

    def operator(j):
        return np.exp(-2j * np.pi * freqs[j] * xn2[:, None] * q[None, :])    # (ntr, nq)

    for j in band:
        L = operator(j)
        LhL = L.conj().T @ L
        LhD = L.conj().T @ D[:, j]
        mu = damping * np.real(np.trace(LhL)) / q.size
        m = np.linalg.solve(LhL + mu * np.eye(q.size), LhD)
        for _ in range(int(sparse_iterations)):
            power = np.abs(m) ** 2
            s2 = sparsity * float(power.max()) + 1e-30
            m = np.linalg.solve(LhL + np.diag(mu / (1.0 + power / s2)), LhD)
        M[:, j] = m
    return M, operator, nfft, band


def radon_forward(M: np.ndarray, operator, nfft: int, band: np.ndarray, ntr: int, nt: int) -> np.ndarray:
    nf = M.shape[1]
    D = np.zeros((ntr, nf), dtype=np.complex128)
    for j in band:
        D[:, j] = operator(j) @ M[:, j]
    return np.fft.irfft(D, nfft, axis=1)[:, :nt]


@register
class RadonDemultiple(Process):
    name = "radon_demultiple"
    label = "Radon demultiple (parabolic)"
    category = "Multiples"
    description = ("On NMO-corrected CMP gathers (primary velocities): model events with residual moveout above q_cut "
                   "(multiples, under-corrected) by a least-squares parabolic Radon transform and subtract them. "
                   "'High resolution' refines the transform by sparse (Cauchy-norm) reweighting, which separates "
                   "multiples whose curvature is close to that of the primaries.")
    params = (Param("qmin", "float", -0.1, "Minimum moveout at far offset", unit="s"),
              Param("qmax", "float", 0.8, "Maximum moveout at far offset", unit="s"),
              Param("nq", "int", 121, "Number of q values", minimum=8),
              Param("q_cut", "float", 0.05, "Multiples above (moveout at far offset)", unit="s"),
              Param("damping", "float", 0.02, "Damping (fraction of mean diagonal)", minimum=0.0),
              Param("fmin", "float", 3.0, "Lowest frequency", unit="Hz", minimum=0.0),
              Param("fmax", "float", 80.0, "Highest frequency", unit="Hz", minimum=1.0),
              Param("output", "choice", "primaries", "Output", choices=("primaries", "multiples")),
              Param("high_resolution", "bool", False, "High resolution (sparse, Sacchi & Ulrych 1995)"),
              Param("sparse_iterations", "int", 3, "Sparse iterations (high resolution)", minimum=1, maximum=20))

    def run(self, data, context):
        out = data.traces.copy()
        sparse = self["sparse_iterations"] if self["high_resolution"] else 0
        q = np.linspace(self["qmin"], self["qmax"], self["nq"])
        xref = float(np.max(np.abs(data.header("offset")))) or 1.0     # q means the same curvature in every CMP
        gathers = list(data.iter_gathers("cdp"))
        for n, (_, idx) in enumerate(gathers):
            if idx.size < 4:
                continue
            g = data.traces[idx]
            off = data.header("offset")[idx]
            M, op, nfft, band = parabolic_radon(g, off, data.dt, q, self["damping"], self["fmin"], self["fmax"], xref,
                                                sparse)
            M[q < self["q_cut"], :] = 0.0
            mult = radon_forward(M, op, nfft, band, idx.size, data.nt)
            out[idx] = (g - mult) if self["output"] == "primaries" else mult
            context.report((n + 1) / len(gathers))
        return data.copy(out.astype(np.float32))
