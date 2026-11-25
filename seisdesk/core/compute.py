"""Local compute-device discovery and explicit CUDA selection.

Importing this module does not load CUDA or start a calculation. Discovery and
the small kernel check run only on request or when a CUDA/automatic job starts.
"""
from __future__ import annotations

import importlib
import platform


def _version(number):
    return f"{number//1000}.{number%1000//10}"


def probe_cuda(*, test=False, device_index=None):
    result = dict(cpu=True, platform=platform.system(), package_available=False, available=False,
                  devices=[], state="missing_package", message="This build has no CuPy CUDA backend.")
    try:
        cupy = importlib.import_module("cupy")
        result.update(package_available=True, cupy_version=cupy.__version__)
    except Exception as exc:
        result["message"] = ("CUDA backend could not load. Use the CUDA edition of SeisDesk, or install its "
                             "GPU dependencies in a source environment. " + str(exc))
        return result
    try:
        runtime = cupy.cuda.runtime
        result["driver_version"] = _version(runtime.driverGetVersion())
        result["runtime_version"] = _version(runtime.runtimeGetVersion())
        count = runtime.getDeviceCount()
        if count == 0:
            result.update(state="no_device", message="No NVIDIA CUDA device is visible to this process.")
            return result
        for index in range(count):
            props = runtime.getDeviceProperties(index)
            name = props["name"]
            name = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
            with cupy.cuda.Device(index):
                free, total = runtime.memGetInfo()
            result["devices"].append(dict(index=index, name=name, free_bytes=int(free), total_bytes=int(total),
                                           capability=f"{props['major']}.{props['minor']}", tested=False))
        if test:
            selected = list(range(count)) if device_index is None else [int(device_index)]
            if any(index < 0 or index >= count for index in selected):
                raise ValueError(f"GPU index {device_index} is not visible; available indices are 0 to {count-1}")
            for index in selected:
                with cupy.cuda.Device(index):
                    # Compiles and runs an elementwise CUDA kernel and a reduction,
                    # detecting missing NVRTC/toolkit components as well as drivers.
                    values = cupy.arange(1024, dtype=cupy.float32)
                    calculated = float(cupy.sum(values*2+1).get())
                    cupy.cuda.get_current_stream().synchronize()
                    if calculated != 1024**2:
                        raise RuntimeError("CUDA arithmetic check returned an incorrect value")
                    del values
                result["devices"][index]["tested"] = True
        result.update(available=True, state="ready", message="CUDA device check passed." if test else "CUDA devices detected.")
    except Exception as exc:
        result.update(available=False, state="cuda_error", message=str(exc))
    return result


def resolve_device(requested, device_index=0, *, log=lambda s: None):
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")
    if int(device_index) != device_index or int(device_index) < 0:
        raise ValueError("CUDA device index must be a nonnegative integer")
    if requested == "cpu":
        return "cpu", dict(requested="cpu", selected="cpu")
    info = probe_cuda(test=True, device_index=int(device_index))
    if info["available"]:
        selected = next((d for d in info["devices"] if d["index"] == int(device_index)), None)
        if selected is None:
            raise ValueError(f"CUDA device {device_index} was not found")
        log(f"Using local CUDA device {device_index}: {selected['name']}")
        return "cuda", dict(requested=requested, selected="cuda", device=selected,
                             driver_version=info["driver_version"], runtime_version=info["runtime_version"],
                             cupy_version=info["cupy_version"])
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable: "+info["message"])
    log("Automatic selection chose CPU: "+info["message"])
    return "cpu", dict(requested="auto", selected="cpu", reason=info["message"])


def cuda_module(device_index=0):
    """Select a visible GPU in the calling thread. Explicit CUDA never silently uses CPU."""
    try:
        cupy = importlib.import_module("cupy")
        count = cupy.cuda.runtime.getDeviceCount()
        if int(device_index) != device_index or not 0 <= int(device_index) < count:
            raise ValueError(f"CUDA device index {device_index} is not available ({count} visible devices)")
        cupy.cuda.Device(int(device_index)).use()
        return cupy
    except Exception as exc:
        raise RuntimeError("Cannot initialise the selected CUDA GPU: "+str(exc)) from exc
