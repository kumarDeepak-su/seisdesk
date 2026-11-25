"""Two-way acoustic migration by frequency-domain source/receiver correlation."""
from dataclasses import replace

import numpy as np
from scipy.ndimage import laplace

from ..core.helmholtz import Helmholtz
from ..core.velocity import DepthModel
from .base import Param, Process, register
from .fwi import GRID_PARAMS, model_section, observed_spectra, prepare_wave_problem


@register
class ReverseTimeMigration(Process):
    name, label, category = "rtm", "Reverse-time migration (2D acoustic, frequency domain)", "Depth migration"
    description = ("Two-way acoustic source and adjoint receiver wavefields with PML boundaries; zero-lag "
                   "correlation, optional illumination normalisation and Laplacian filtering. Constant density, "
                   "flat recording levels, small 2D CPU grids. This is relative-amplitude imaging.")
    params = tuple(replace(p, default=False) if p.name == "estimate_source" else p for p in GRID_PARAMS) + (
        Param("fmin", "float", 3.0, "Lowest frequency", unit="Hz", minimum=0.1),
        Param("fmax", "float", 20.0, "Highest frequency", unit="Hz", minimum=0.1),
        Param("frequency_step", "int", 1, "Use every Nth Fourier bin", minimum=1, maximum=100),
        Param("normalise", "bool", True, "Normalise by source illumination"),
        Param("laplacian", "bool", True, "Laplacian filter (suppress low wavenumbers)"),
    )

    def run(self, data, context):
        model, shots, indices = prepare_wave_problem(data, context, self)
        frequencies = np.fft.rfftfreq(data.nt, data.dt)
        frequencies = frequencies[(frequencies >= self["fmin"]) & (frequencies <= self["fmax"])][::self["frequency_step"]]
        if not frequencies.size:
            raise ValueError("no Fourier bins in the selected frequency range")
        if model.values.min() / (frequencies.max() * max(model.dx, model.dz)) < 10:
            raise ValueError("RTM needs at least 10 grid points per shortest wavelength; refine the model or reduce fmax")
        observations = observed_spectra(data, frequencies, indices, self["alpha"])
        image, illumination = np.zeros(model.shape), np.zeros(model.shape)
        for fi, f in enumerate(frequencies):
            op = Helmholtz(model.values, model.dx, model.dz, f, self["nabs"], model.x0, model.z0, self["alpha"])
            for si, (source, receivers) in enumerate(shots):
                context.report((fi + si / len(shots)) / len(frequencies))
                u = op.solve(op.source(source))
                sample = op.sampling(receivers)
                q = (f / self["f0"]) ** 2 * np.exp(-(f / self["f0"]) ** 2) * np.exp(
                    -2j * np.pi * f * self["source_delay"])
                if self["estimate_source"]:
                    pred = sample @ u
                    q = np.vdot(pred, observations[fi][si]) / max(float(np.vdot(pred, pred).real), 1e-30)
                u *= q
                receiver = op.solve(sample.conj().T @ observations[fi][si], adjoint=True)
                image += op.crop(np.real(np.conj(u) * receiver))
                illumination += op.crop(np.abs(u) ** 2)
        if self["normalise"]:
            image /= illumination + 0.01 * max(float(illumination.max()), 1e-30)
        if self["laplacian"]:
            # Dimensionless discrete high-pass; unequal dx/dz accounted for separately.
            image = -(laplace(image, mode="nearest") if model.dx == model.dz else
                      np.gradient(np.gradient(image, model.dz, axis=0), model.dz, axis=0) +
                      np.gradient(np.gradient(image, model.dx, axis=1), model.dx, axis=1))
        context.report(1)
        context.results["rtm"] = {"illumination": illumination, "frequencies": frequencies,
                                  "velocity": model.values.copy(), "x": model.x, "z": model.z}
        out = model_section(DepthModel(image, model.dz, model.dx, model.z0, model.x0), data.name + " RTM", data.history)
        out.meta = {"migrated": "2D acoustic RTM", "stacked": True, "sample_units": "relative amplitude"}
        return out
