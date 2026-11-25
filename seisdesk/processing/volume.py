"""GUI and flow adapters for fully spatially varying 3D acoustic jobs."""
from pathlib import Path
import tempfile

import numpy as np

from ..backends.volume_jobs import run_volume_job, stage_shots
from ..core.dataset import SeismicData
from ..core.streaming import iter_file_gathers
from ..core.volume import VelocityVolume
from .base import Param, Process, register


VOLUME_PARAMS = (
    Param("model", "infile", "velocity3d.npz", "3D velocity model (.npz or model directory)"),
    Param("gather_key", "key", "fldr", "Shot header"),
    Param("device", "choice", "cpu", "Compute device", choices=("cpu", "cuda", "auto"),
          help="CUDA uses the selected NVIDIA GPU on the execution host. Auto checks CUDA and reports any CPU fallback."),
    Param("device_index", "int", 0, "CUDA GPU index (visible device number)", minimum=0, maximum=63),
    Param("workers", "int", 1, "Local shot workers", minimum=1, maximum=64),
    Param("scheduler", "str", "", "Dask scheduler address (optional)",
          help="Workers must share the job directory. CUDA workers must advertise GPU=1."),
    Param("wavelet", "infile", "", "Source wavelet: time(s), acceleration (optional)"),
    Param("frequency", "float", 5, "Ricker frequency when no wavelet file", unit="Hz", minimum=.1),
    Param("source_delay", "float", .3, "Ricker peak time", unit="s", minimum=0),
    Param("source_amplitude", "float", 1e6, "Ricker acceleration amplitude"),
    Param("padding", "int", 12, "Absorbing cells on every boundary", minimum=2, maximum=100),
    Param("block", "int", 24, "Time steps per checkpoint block", minimum=1, maximum=2000),
    Param("memory_gb", "float", 4, "Memory budget per worker", unit="GiB", minimum=.1),
    Param("allow_coarse_grid", "bool", False, "Allow under-resolved grid (diagnostic runs only)",
          help="Normally require at least 10 grid points per shortest source wavelength."),
    Param("vmin", "float", 100, "Minimum admissible velocity", unit="m/s", minimum=10),
    Param("vmax", "float", 8000, "Maximum admissible velocity", unit="m/s", minimum=100),
    Param("remove_background", "bool", True, "Subtract modelled background before RTM"),
    Param("iterations", "int", 5, "FWI iterations", minimum=1, maximum=100),
    Param("max_update", "float", .03, "Maximum FWI step / median velocity", minimum=.0001, maximum=.2),
    Param("smooth_cells", "float", 1, "FWI gradient smoothing", unit="cells", minimum=0, maximum=20),
    Param("job_directory", "directory", "", "Job directory (empty = new project subfolder)"),
)


def execute_volume_job(process, data, context, mode, input_file=""):
    params = dict(process.values)
    def path(value):
        p = Path(value)
        return p if p.is_absolute() else Path(context.work_dir)/p
    model = VelocityVolume.load(path(process["model"]))
    Path(context.work_dir).mkdir(parents=True, exist_ok=True)
    directory = path(process["job_directory"]) if process["job_directory"] else Path(
        tempfile.mkdtemp(prefix="volume-", dir=context.work_dir))
    directory.mkdir(parents=True, exist_ok=True)
    if params["wavelet"]:
        params["wavelet"] = str(path(params["wavelet"]))
    if input_file:
        gathers = iter_file_gathers(path(input_file), process["gather_key"])
    else:
        labels = data.header(process["gather_key"], None)
        gathers = ((idx, data.subset(idx)) for idx in
                   (np.flatnonzero(labels == label) for label in np.unique(labels)))
    context.log(f"Preparing 3D {mode} job: {directory}")
    shots = stage_shots(gathers, model, directory/"shots", context.report)
    result = run_volume_job(shots, model, directory, params, mode=mode, cancelled=context.cancelled,
                            report=context.report, log=context.log)
    context.results[process.name] = result
    final = VelocityVolume.load(result["final_model"])
    if mode == "rtm":
        output = final.dataset(np.load(result["image"], allow_pickle=False), "3D acoustic RTM")
    elif mode == "fwi":
        output = final.dataset(name="3D acoustic FWI velocity")
    elif data is not None and not input_file:
        output = data.copy()
        for shot, response in zip(shots, result["shot_results"]):
            with np.load(shot["path"], allow_pickle=False) as saved:
                output.traces[saved["indices"]] = np.load(response["files"]["predicted"], allow_pickle=False).T
    else:
        # File jobs retain every shot on disk and open a bounded preview.
        with np.load(shots[0]["path"], allow_pickle=False) as saved:
            headers = {key[2:]: saved[key] for key in saved.files if key.startswith("h_")}
            output = SeismicData(np.load(result["shot_results"][0]["files"]["predicted"], allow_pickle=False).T,
                                 float(saved["dt"]), headers, t0=float(saved["t0"]), name="3D model — first shot preview")
    output.meta["volume_job"] = str(directory/"job.json")
    context.log(f"Completed 3D {mode}. Inputs, predicted shots, diagnostics and models: {directory}")
    return output


@register
class AcousticModelling3D(Process):
    name, label, category = "acoustic_model_3d", "3D acoustic finite-difference modelling", "Modelling"
    description = ("Full v(z,y,x) isotropic constant-density acoustic propagation with trilinear geometry and "
                   "absorbing boundaries. CPU or optional CUDA, disk checkpoints and parallel shots. "
                   "Source samples are acceleration forcing. Grid resolution must resolve the shortest wavelength.")
    params = VOLUME_PARAMS

    def run(self, data, context):
        return execute_volume_job(self, data, context, "model")


@register
class AcousticRTM3D(Process):
    name, label, category = "acoustic_rtm_3d", "3D acoustic reverse-time depth migration", "Depth migration"
    description = ("Prestack RTM through a fully varying v(z,y,x) model. Exact discrete adjoint, "
                   "zero-lag correlation and illumination volumes; relative amplitudes. "
                   "Flat or buried recording coordinates inside the model; no free surface or elastic physics.")
    params = VOLUME_PARAMS

    def run(self, data, context):
        return execute_volume_job(self, data, context, "rtm")


@register
class AcousticFWI3D(Process):
    name, label, category = "acoustic_fwi_3d", "3D acoustic full-waveform inversion", "Inversion"
    description = ("Velocity inversion in a full 3D model using checkpointed discrete-adjoint gradients and a "
                   "bounded line search. Known source waveform and a suitable starting model are required. "
                   "Isotropic constant-density acoustic physics; cycle skipping and acquisition coverage still apply.")
    params = VOLUME_PARAMS

    def run(self, data, context):
        return execute_volume_job(self, data, context, "fwi")


@register
class VolumeFileJob(Process):
    name, label, category = "volume_file_job", "3D acoustic file job (streamed shot input)", "Input/Output"
    description = ("Read a SEG-Y or SU survey one shot at a time and run 3D modelling, RTM or FWI. "
                   "No full survey waveform matrix is loaded into memory. Optional Dask workers require "
                   "the same SeisDesk version and a shared job directory. All output shots remain in the job folder.")
    params = (Param("input_file", "infile", "", "Input SEG-Y or SU survey"),
              Param("mode", "choice", "rtm", "Operation", choices=("model", "rtm", "fwi"))) + VOLUME_PARAMS

    def run(self, data, context):
        if not self["input_file"]:
            raise ValueError("choose an input survey file")
        return execute_volume_job(self, None, context, self["mode"], self["input_file"])
