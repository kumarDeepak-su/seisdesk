"""File-based 3D modelling, RTM and FWI with parallel independent shots.

Native process workers run on one host; an optional Dask scheduler distributes
shots over workers sharing the job directory. Each worker can use CPU or CUDA.
The velocity grid must fit on one worker; this is shot parallelism, not spatial
domain decomposition. All job inputs, histories and output volumes are retained.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from .. import __version__
from ..core.acoustic3d import Acoustic3D
from ..core.volume import VelocityVolume


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_shots(gathers, volume, directory, report=None):
    """Stage either GUI gathers or streamed file gathers without accumulating waveforms."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shots = []
    for count, (indices, data) in enumerate(gathers):
        if report:
            report(0)
        if data.domain != "time" or data.t0 < 0 or data.nt < 3:
            raise ValueError("3D jobs require time records starting at or after zero")
        source, receiver = volume.coordinates(data)
        if not np.allclose(source, source[0], atol=1e-4, rtol=0):
            raise ValueError("each gather must contain one source location; check the shot headers")
        if not np.isfinite(data.traces).all():
            raise ValueError("input contains nonfinite samples")
        path = directory/f"shot-{count:06d}.npz"
        arrays = dict(source=source[:1], receiver=receiver, observed=data.traces.T, dt=data.dt, t0=data.t0,
                      indices=np.asarray(indices), trace_ids=data.trace_ids)
        arrays.update({"h_"+key: values for key, values in data.headers.items()})
        np.savez(path, **arrays)
        shots.append({"path": str(path.resolve()), "sha256": file_digest(path), "ntr": data.ntr, "nt": data.nt})
    if not shots:
        raise ValueError("no shots to process")
    return shots


def _wavelet(times, params):
    if params.get("wavelet"):
        samples = np.loadtxt(params["wavelet"], ndmin=2)
        if samples.shape[1] != 2 or not np.isfinite(samples).all() or np.any(np.diff(samples[:, 0]) <= 0):
            raise ValueError("source wavelet file requires increasing time (seconds), amplitude columns")
        return np.interp(times, samples[:, 0], samples[:, 1], left=0, right=0)
    frequency = float(params.get("frequency", 5))
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("source frequency must be positive and finite")
    phase = np.pi*frequency*(times-float(params.get("source_delay", 1.5/frequency)))
    return float(params.get("source_amplitude", 1e6))*(1-2*phase**2)*np.exp(-phase**2)


def shot_task(task, cancelled=None):
    """Serializable worker entry point. Workers need the same SeisDesk installation."""
    directory = Path(task["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    record = directory/"result.json"
    key = task["key"]
    if record.is_file():
        result = json.loads(record.read_text(encoding="utf-8"))
        if result.get("key") == key and all(Path(p).is_file() for p in result.get("files", {}).values()):
            return result
    params = task["params"]
    volume = VelocityVolume.load(task["model"])
    with np.load(task["shot"]["path"], allow_pickle=False) as archive:
        source, receiver = archive["source"], archive["receiver"]
        observed, sample_dt, t0 = archive["observed"], float(archive["dt"]), float(archive["t0"])
    # Fix the solver sampling from the configured velocity upper bound, so line
    # search candidates always minimise the same discretised objective.
    bound = max(float(params.get("vmax", 8000)), float(volume.values.max()))
    stable = .85/(bound*np.sqrt(np.sum(1/np.asarray(volume.spacing)**2)))
    stride = max(1, int(np.ceil(sample_dt/stable)))
    dt = sample_dt/stride
    start = int(round(t0/dt))
    if abs(start*dt-t0) > dt*1e-5:
        raise ValueError("recording delay must align with the modelling time grid")
    indices = start + np.arange(len(observed))*stride
    nt = int(indices[-1])+1
    padding, block = int(params.get("padding", 12)), int(params.get("block", 24))
    shape = np.array(volume.values.shape)+2*padding
    grid_bytes = int(np.prod(shape))*4
    estimate_ram = grid_bytes*(block+18)
    estimate_disk = grid_bytes*2*int(np.ceil(nt/block))
    if estimate_ram > float(params.get("memory_gb", 4))*1024**3:
        raise ValueError(f"shot needs approximately {estimate_ram/1024**3:.2f} GiB; decrease checkpoint block "
                         "size/grid size, or increase the per-worker memory limit")
    if estimate_disk > shutil.disk_usage(directory).free*.85:
        raise ValueError(f"insufficient checkpoint disk space (estimate {estimate_disk/1024**3:.2f} GiB per shot)")
    if params.get("device", "cpu") == "cuda":
        from ..core.compute import cuda_module
        gpu = cuda_module(params.get("device_index", 0))
        free, total = gpu.cuda.runtime.memGetInfo()
        if estimate_ram > .85*free:
            raise ValueError(f"shot needs approximately {estimate_ram/1024**3:.2f} GiB, but the selected GPU has "
                             f"{free/1024**3:.2f} GiB free; reduce the grid/checkpoint block or choose another device")
    solver = Acoustic3D(volume, dt, padding=padding, device=params.get("device", "cpu"),
                        device_index=params.get("device_index", 0), block=block)
    wavelet = _wavelet(np.arange(nt)*dt, params)
    if not np.isfinite(wavelet).all() or not np.any(wavelet):
        raise ValueError("source wavelet must be finite and nonzero within the modelling interval")
    if params.get("wavelet"):
        amplitude = np.abs(np.fft.rfft(wavelet))
        energetic = np.flatnonzero(amplitude > .01*amplitude.max())
        highest = float(np.fft.rfftfreq(nt, dt)[energetic[-1]]) if energetic.size else 0.
    else:
        highest = 2.5*float(params.get("frequency", 5))
    points_per_wavelength = float(volume.values.min())/(max(highest, 1e-20)*max(volume.spacing))
    if points_per_wavelength < 10 and not params.get("allow_coarse_grid", False):
        raise ValueError(f"only {points_per_wavelength:.2f} grid points per shortest source wavelength; "
                         "use at least 10 for this second-order stencil. Refine the model grid or lower "
                         "the source bandwidth. The coarse-grid override is for numerical diagnostics.")
    def check(_fraction):
        if Path(task["cancel_file"]).exists() or (cancelled is not None and cancelled()):
            raise InterruptedError("3D job cancelled")
    files = {}
    with tempfile.TemporaryDirectory(prefix="checkpoints-", dir=directory) as checkpoints:
        needs_adjoint = task["operation"] in ("gradient", "rtm")
        predicted_full = solver.forward(source, receiver, wavelet, checkpoints=checkpoints if needs_adjoint else None,
                                        report=check)
        predicted = predicted_full[indices]
        residual = predicted-observed
        objective = .5*float(np.sum(residual.astype(float)**2))
        np.save(directory/"predicted.npy", predicted)
        files["predicted"] = str((directory/"predicted.npy").resolve())
        if needs_adjoint:
            residual_full = np.zeros_like(predicted_full)
            residual_full[indices] = residual if task["operation"] == "gradient" else (
                observed-predicted if params.get("remove_background", True) else observed)
            adjoint = solver.adjoint(source, receiver, wavelet, residual_full, checkpoints,
                                     image=task["operation"] == "rtm", report=check)
            for name in (("gradient",) if task["operation"] == "gradient" else ("image", "illumination")):
                np.save(directory/(name+".npy"), adjoint[name])
                files[name] = str((directory/(name+".npy")).resolve())
    result = dict(key=key, objective=objective, files=files, dt=dt, stride=stride, nt=nt,
                  estimate_ram_bytes=estimate_ram, estimate_checkpoint_bytes=estimate_disk,
                  source_band_hz=highest, points_per_wavelength=points_per_wavelength,
                  coarse_grid_override=bool(params.get("allow_coarse_grid", False)))
    _json(record, result)
    return result


def _evaluate(shots, model_path, params, operation, root, cancelled, report, log):
    digest = file_digest(model_path)
    tasks = []
    for index, shot in enumerate(shots):
        key = hashlib.sha256(json.dumps([__version__, shot["sha256"], digest, params, operation],
                                       sort_keys=True).encode()).hexdigest()
        tasks.append(dict(shot=shot, model=str(model_path), params=params, operation=operation, key=key,
                          directory=str(root/"tasks"/key), cancel_file=str(root/"cancel")))
    workers = int(params.get("workers", 1))
    scheduler = params.get("scheduler", "")
    if workers < 1:
        raise ValueError("worker count must be positive")
    if not scheduler and params.get("device") == "cuda" and workers > 1:
        raise ValueError("use one local CUDA worker, or a Dask cluster with one GPU resource per worker")
    results = [None]*len(tasks)
    executor = client = None
    try:
        if scheduler:
            try:
                from distributed import Client
            except ImportError as exc:
                raise RuntimeError("distributed jobs require the optional dask[distributed] package") from exc
            client = Client(scheduler)
            options = {"resources": {"GPU": 1}} if params.get("device") == "cuda" else {}
            futures = {client.submit(shot_task, task, pure=False, **options): i for i, task in enumerate(tasks)}
        elif workers > 1:
            executor = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
            futures = {executor.submit(shot_task, task): i for i, task in enumerate(tasks)}
        else:
            for i, task in enumerate(tasks):
                if cancelled():
                    raise InterruptedError("3D job cancelled")
                results[i] = shot_task(task, cancelled)
                report((i+1)/len(tasks))
            return results
        pending = set(futures)
        number = 0
        while pending:
            if cancelled():
                raise InterruptedError("3D job cancelled")
            for future in list(pending):
                if not future.done():
                    continue
                results[futures[future]] = future.result()
                pending.remove(future)
                number += 1
                log(f"3D {operation}: completed shot {number}/{len(tasks)}")
                report(number/len(tasks))
            if pending:
                time.sleep(.1)
        return results
    except BaseException:
        (root/"cancel").touch()
        if client is not None:
            client.cancel(list(futures), force=True)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if client is not None:
            client.close()


def run_volume_job(shots, volume, root, params, *, mode="rtm", cancelled=lambda: False,
                   report=lambda f: None, log=print):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/"cancel").exists():
        (root/"cancel").unlink()
    if mode not in ("model", "rtm", "fwi"):
        raise ValueError("mode must be model, rtm or fwi")
    params = dict(params)
    if params.get("wavelet"):
        source_wavelet = Path(params["wavelet"]).resolve()
        snapshot = root/"source-wavelet.txt"
        if source_wavelet != snapshot:
            shutil.copyfile(source_wavelet, snapshot)
        params["wavelet"] = str(snapshot)
        params["wavelet_sha256"] = file_digest(snapshot)
    vmin, vmax = float(params.get("vmin", 100)), float(params.get("vmax", 8000))
    if vmin <= 0 or vmax <= vmin or volume.values.min() < vmin or volume.values.max() > vmax:
        raise ValueError("the model must lie within positive vmin/vmax bounds")
    state = dict(seisdesk_version=__version__, mode=mode, params=params, shots=shots,
                 state="running", objective=[], iteration=0)
    _json(root/"job.json", state)
    current = root/"model-000.npz"
    volume.save(current)
    def checked(fraction):
        if cancelled():
            (root/"cancel").touch()
            raise InterruptedError("3D job cancelled")
        report(fraction)
    try:
        if params.get("scheduler"):
            if params.get("device") == "auto":
                raise ValueError("choose cpu or cuda explicitly for distributed workers")
            state["compute"] = dict(selected=params.get("device", "cpu"), location="distributed workers")
        else:
            from ..core.compute import resolve_device
            params["device"], state["compute"] = resolve_device(params.get("device", "cpu"),
                                                               params.get("device_index", 0), log=log)
        _json(root/"job.json", state)
        iterations = int(params.get("iterations", 5)) if mode == "fwi" else 1
        if iterations < 1:
            raise ValueError("iterations must be positive")
        results = []
        for iteration in range(iterations):
            checked(iteration/iterations)
            results = _evaluate(shots, current, params, "gradient" if mode == "fwi" else mode, root,
                                cancelled, lambda f: checked((iteration+.5*f)/iterations), log)
            objective = sum(r["objective"] for r in results)
            if not state["objective"]:
                state["objective"].append(objective)
            if mode != "fwi":
                break
            gradient = np.zeros(volume.values.shape, dtype=float)
            for result in results:
                gradient += np.load(result["files"]["gradient"], mmap_mode="r", allow_pickle=False)
            direction = gaussian_filter(gradient, float(params.get("smooth_cells", 1)))
            peak = float(np.max(np.abs(direction)))
            if peak == 0 or not np.isfinite(peak):
                if not np.isfinite(peak):
                    raise ValueError("FWI gradient became nonfinite")
                state["stop_reason"] = "zero gradient"
                break
            step = float(params.get("max_update", .03))*float(np.median(volume.values))/peak
            accepted = False
            for search in range(7):
                checked((iteration+.5)/iterations)
                candidate = VelocityVolume(np.clip(volume.values-step*direction, vmin, vmax).astype(np.float32),
                                           volume.spacing, volume.origin, volume.reference_elevation)
                path = root/f"trial-{iteration:03d}-{search}.npz"
                candidate.save(path)
                trial = _evaluate(shots, path, params, "model", root, cancelled,
                                  lambda f: checked((iteration+.5+.45*f)/iterations), log)
                new_objective = sum(r["objective"] for r in trial)
                if new_objective < objective:
                    volume, current, results = candidate, path, trial
                    state["objective"].append(new_objective)
                    state["iteration"] = iteration+1
                    log(f"3D FWI {iteration+1}: objective {objective:.6g} -> {new_objective:.6g}")
                    accepted = True
                    break
                step *= .5
            if not accepted:
                state["stop_reason"] = "line search found no improving step"
                break
            _json(root/"job.json", state)
        state["shot_results"] = results
        if mode == "rtm":
            image = np.zeros(volume.values.shape, dtype=np.float64)
            illumination = np.zeros_like(image)
            for result in results:
                image += np.load(result["files"]["image"], mmap_mode="r", allow_pickle=False)
                illumination += np.load(result["files"]["illumination"], mmap_mode="r", allow_pickle=False)
            np.save(root/"image.npy", image.astype(np.float32))
            np.save(root/"illumination.npy", illumination.astype(np.float32))
            state.update(image=str(root/"image.npy"), illumination=str(root/"illumination.npy"))
        volume.save(root/"final-model.npz")
        state.update(state="completed", final_model=str(root/"final-model.npz"))
        checked(1)
        return state
    except BaseException as exc:
        (root/"cancel").touch()
        state.update(state="cancelled" if isinstance(exc, InterruptedError) else "failed", error=str(exc))
        raise
    finally:
        _json(root/"job.json", state)
