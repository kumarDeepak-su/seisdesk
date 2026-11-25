"""Output and utility steps."""

from __future__ import annotations

import os

import numpy as np

from ..core.segy_io import write_segy, write_su
from .base import Param, Process, register


@register
class ExportSEGY(Process):
    name = "export_segy"
    label = "Write SEG-Y"
    category = "Input/Output"
    description = "Write the current data to an IEEE-float SEG-Y file (the data pass on unchanged)."
    params = (Param("path", "file", "output.sgy", "File"),)

    def run(self, data, context):
        path = self["path"] if os.path.isabs(self["path"]) else os.path.join(context.work_dir, self["path"])
        write_segy(data, path)
        context.log(f"    wrote {path}")
        return data.copy()


@register
class ExportSU(Process):
    name = "export_su"
    label = "Write Seismic Unix file"
    category = "Input/Output"
    description = "Write the current data in Seismic Unix format (little-endian)."
    params = (Param("path", "file", "output.su", "File"),)

    def run(self, data, context):
        path = self["path"] if os.path.isabs(self["path"]) else os.path.join(context.work_dir, self["path"])
        write_su(data, path)
        context.log(f"    wrote {path}")
        return data.copy()


@register
class AddNoise(Process):
    name = "add_noise"
    label = "Add random noise"
    category = "Utility"
    description = "Add Gaussian noise at a signal-to-noise ratio (dB, RMS) — for tests and training."
    params = (Param("snr_db", "float", 20.0, "SNR", unit="dB"), Param("seed", "int", 0, "Random seed"))

    def run(self, data, context):
        rng = np.random.default_rng(self["seed"])
        rms = float(np.sqrt(np.mean(data.traces.astype(np.float64) ** 2)))
        sigma = rms / (10 ** (self["snr_db"] / 20.0))
        return data.copy((data.traces + sigma * rng.standard_normal(data.traces.shape)).astype(np.float32))
