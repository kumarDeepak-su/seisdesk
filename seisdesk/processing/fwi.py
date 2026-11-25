"""Bounded, multiscale 2D acoustic frequency-domain full-waveform inversion.

Pratt (1999), Geophysics 64, 888-901, doi:10.1190/1.1444597. The exact discrete
adjoint gradient is checked by directional finite differences. Five-point spatial
differences require at least ten grid points per shortest wavelength. Constant
density, absorbing surface, no attenuation except optional Laplace damping.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..core.dataset import SeismicData
from ..core.helmholtz import objective_gradient
from ..core.velocity import DepthModel
from .base import Param, Process, register


GRID_PARAMS = (
    Param("model", "choice", "project model", "Starting velocity", choices=("project model", "gradient")),
    Param("dx", "float", 10.0, "Grid spacing x", unit="m", minimum=0.5),
    Param("dz", "float", 10.0, "Grid spacing z", unit="m", minimum=0.5),
    Param("zmax", "float", 500.0, "Maximum depth (gradient model)", unit="m", minimum=10.0),
    Param("v0", "float", 1800.0, "Surface velocity (gradient model)", unit="m/s", minimum=100.0),
    Param("gradient", "float", 0.5, "Velocity gradient", unit="1/s", minimum=0.0),
    Param("nabs", "int", 16, "PML cells per edge", minimum=8, maximum=60),
    Param("max_offset", "float", 0.0, "Maximum offset (0 = all)", unit="m", minimum=0.0),
    Param("alpha", "float", 0.0, "Laplace damping", unit="1/s", minimum=0.0),
    Param("f0", "float", 20.0, "Ricker peak frequency (fixed source)", unit="Hz", minimum=0.1),
    Param("source_delay", "float", 0.0, "Ricker wavelet delay (fixed source)", unit="s", minimum=0.0),
    Param("estimate_source", "bool", True, "Estimate source spectrum per shot and frequency"),
)


def frequency_groups(text):
    groups = [[float(v.strip()) for v in group.split(",") if v.strip()] for group in str(text).split(";")]
    if not groups or any(not g or not np.isfinite(g).all() or min(g) <= 0 for g in groups):
        raise ValueError("give positive frequencies: 3,4,5;5,6,7 (semicolon separates stages)")
    if any(max(a) > max(b) for a, b in zip(groups, groups[1:])):
        raise ValueError("frequency stages must progress from low to high frequencies")
    return groups


def prepare_wave_problem(data, context, params):
    if data.domain != "time" or data.nt < 4:
        raise ValueError("wave-equation imaging needs time-domain shot gathers")
    for key in ("sx", "gx", "fldr"):
        if key not in data.headers:
            raise ValueError(f"wave-equation imaging requires the {key} geometry header")
    if np.ptp(np.r_[data.header("sy"), data.header("gy")]) > 1e-3:
        raise ValueError("this solver is 2D; project the survey geometry onto the line first")
    if np.ptp(data.header("selev")) > 1e-3 or np.ptp(data.header("gelev")) > 1e-3:
        raise ValueError("this acoustic solver assumes flat recording levels; datum the land data first")
    ref = float(max(data.header("selev").max(), data.header("gelev").max()))
    sx, gx = data.header("sx"), data.header("gx")
    sz, gz = ref - data.header("selev") + data.header("sdepth"), ref - data.header("gelev")
    shots, indices = [], []
    for _, idx in data.iter_gathers("fldr", None):
        if np.ptp(sx[idx]) > 1e-4 or np.ptp(sz[idx]) > 1e-4:
            raise ValueError("each shot number must have one source position")
        if params["max_offset"] > 0:
            idx = idx[np.abs(gx[idx] - sx[idx]) <= params["max_offset"]]
        if idx.size:
            shots.append((np.array([sx[idx[0]], sz[idx[0]]]), np.column_stack([gx[idx], gz[idx]])))
            indices.append(idx)
    if not shots:
        raise ValueError("no traces remain in the requested offset range")
    if params["model"] == "project model":
        if context.depth_model is None:
            raise ValueError("load a project depth model or select a gradient starting model")
        old = context.depth_model
        model = DepthModel(old.values.copy(), old.dz, old.dx, old.z0, old.x0)
    else:
        dx, dz = params["dx"], params["dz"]
        x0 = float(np.floor(min(sx.min(), gx.min()) / dx) * dx - 2 * dx)
        nx = int(np.ceil((max(sx.max(), gx.max()) - x0) / dx)) + 3
        nz = int(np.ceil(params["zmax"] / dz)) + 1
        model = DepthModel.gradient(params["v0"], params["gradient"], nz, nx, dz, dx, x0=x0)
    return model, shots, indices


def observed_spectra(data, frequencies, indices, alpha=0.0):
    f = np.asarray(frequencies, dtype=float)
    if np.max(f) >= 0.5 / data.dt:
        raise ValueError("requested frequencies must be below the data Nyquist frequency")
    kernel = np.exp(-(alpha + 2j * np.pi * f[:, None]) * data.times[None, :]) * data.dt
    spectra = data.traces.astype(float) @ kernel.T
    return [[spectra[idx, fi] for idx in indices] for fi in range(len(f))]


def model_section(model, name, history=()):
    n = model.shape[1]
    return SeismicData(model.values.T.astype(np.float32), model.dz,
                       {"cdp": np.arange(1, n + 1), "cdpx": model.x, "sx": model.x, "gx": model.x,
                        "offset": np.zeros(n)}, model.z0, name, "depth", list(history),
                       {"model_kind": "interval velocity", "sample_units": "m/s"})


def smooth_penalty(q):
    """Squared first differences and their exact gradient; q is dimensionless velocity."""
    grad = np.zeros_like(q)
    count = (q.shape[0] - 1) * q.shape[1] + q.shape[0] * (q.shape[1] - 1)
    dz, dx = np.diff(q, axis=0), np.diff(q, axis=1)
    grad[:-1] -= dz; grad[1:] += dz
    grad[:, :-1] -= dx; grad[:, 1:] += dx
    return 0.5 * (np.sum(dz ** 2) + np.sum(dx ** 2)) / count, grad / count


def invert_frequency(velocity, dx, dz, frequencies, shots, observations, *, iterations=8,
                     vmin=500.0, vmax=6000.0, smoothing=0.01, precondition=False, report=None, **kwargs):
    """One frequency stage. A fixed change of variables provides optional illumination scaling.

    Gradients are never smoothed or rescaled behind L-BFGS-B's back: the objective,
    bounds and gradient all use the same parameterisation.
    """
    start = np.asarray(velocity, dtype=float).copy()
    if not (0 < vmin < vmax) or np.any((start < vmin) | (start > vmax)):
        raise ValueError("starting velocity must lie within positive increasing inversion bounds")
    scale = 2000.0
    q0, metric = start / scale, np.ones_like(start)
    history, evaluations = [], [0]

    def wave_progress(fraction):
        # Also checks the worker's cancellation flag between frequencies and shots.
        if report is not None:
            report(min((evaluations[0] + fraction) / max(10 * iterations, 1), 0.95))

    if precondition:
        _, _, info = objective_gradient(start, dx, dz, frequencies, shots, observations,
                                        report=wave_progress, **kwargs)
        illum = info["illumination"]
        metric = np.clip(np.sqrt((np.median(illum) + 1e-30) / (illum + 0.01 * illum.max() + 1e-30)), 0.1, 3.0)
        evaluations[0] += 1

    def evaluate(z):
        q = q0 + metric * z.reshape(start.shape)
        value, gradient, _ = objective_gradient(scale * q, dx, dz, frequencies, shots, observations,
                                                report=wave_progress, **kwargs)
        penalty, pg = smooth_penalty(q)
        evaluations[0] += 1
        if report is not None:
            report(min(evaluations[0] / max(10 * iterations, 1), 0.95))
        return value + smoothing * penalty, ((scale * gradient + smoothing * pg) * metric).ravel()

    initial = np.zeros(start.size)
    history.append(float(evaluate(initial)[0]))

    def callback(z):
        history.append(float(evaluate(z)[0]))

    lower = ((vmin / scale - q0) / metric).ravel()
    upper = ((vmax / scale - q0) / metric).ravel()
    opt = minimize(evaluate, initial, jac=True, method="L-BFGS-B", bounds=list(zip(lower, upper)),
                   callback=callback, options={"maxiter": iterations, "maxls": 20, "ftol": 1e-10, "gtol": 1e-8})
    final = scale * (q0 + metric * opt.x.reshape(start.shape))
    value, _, details = objective_gradient(final, dx, dz, frequencies, shots, observations,
                                           report=wave_progress, **kwargs)
    details.update(history=history, data_misfit=value, converged=bool(opt.success), message=str(opt.message),
                   iterations=int(opt.nit), evaluations=evaluations[0])
    if report is not None:
        report(1.0)
    return final, details


@register
class FullWaveformInversion(Process):
    name, label, category = "fwi", "Full-waveform inversion (2D acoustic, frequency domain)", "Inversion"
    description = ("Bounded multiscale acoustic inversion of shot gathers, with exact discrete adjoint gradients, "
                   "optional source estimation and Laplace damping. Requires a close starting model, flat recording "
                   "levels, calibrated amplitudes and at least 10 grid points per shortest wavelength. Small 2D CPU grids.")
    params = GRID_PARAMS + (
        Param("frequencies", "str", "3,4,5;5,6,7", "Frequency groups (Hz; semicolon separates stages)"),
        Param("iterations", "int", 8, "Iterations per group", minimum=1, maximum=100),
        Param("vmin", "float", 1000.0, "Minimum velocity", unit="m/s", minimum=50.0),
        Param("vmax", "float", 6000.0, "Maximum velocity", unit="m/s", minimum=100.0),
        Param("smoothing", "float", 0.01, "Model roughness penalty", minimum=0.0),
        Param("precondition", "bool", True, "Fixed illumination scaling of inversion variables"),
    )

    def run(self, data, context):
        model, shots, indices = prepare_wave_problem(data, context, self)
        start = model.values.copy()
        groups = frequency_groups(self["frequencies"])
        if self["vmin"] / (max(map(max, groups)) * max(model.dx, model.dz)) < 10:
            raise ValueError("inversion lower velocity bound gives fewer than 10 grid points per wavelength; "
                             "refine dx/dz, reduce frequencies, or raise vmin")
        stages = []
        for stage, frequencies in enumerate(groups):
            observations = observed_spectra(data, frequencies, indices, self["alpha"])
            spectra = [[(f / self["f0"]) ** 2 * np.exp(-(f / self["f0"]) ** 2) *
                        np.exp(-2j * np.pi * f * self["source_delay"])] * len(shots) for f in frequencies]
            context.log(f"    FWI stage {stage + 1}: {frequencies} Hz")
            values, detail = invert_frequency(
                model.values, model.dx, model.dz, frequencies, shots, observations,
                iterations=self["iterations"], vmin=self["vmin"], vmax=self["vmax"], smoothing=self["smoothing"],
                precondition=self["precondition"], nabs=self["nabs"], x0=model.x0, z0=model.z0,
                alpha=self["alpha"], estimate_source=self["estimate_source"], source_spectra=spectra,
                report=lambda f: context.report((stage + f) / len(groups)))
            model = DepthModel(values, model.dz, model.dx, model.z0, model.x0)
            detail["frequencies"] = frequencies
            stages.append(detail)
            context.log(f"    {detail['message']}; normalised data misfit {detail['data_misfit']:.6g}")
        context.report(1)
        context.depth_model = model
        context.results["fwi"] = {"starting_model": start, "final_model": model.values.copy(), "stages": stages,
                                  "x": model.x, "z": model.z, "physics": "2D constant-density acoustic"}
        return model_section(model, data.name + " FWI velocity", data.history)
