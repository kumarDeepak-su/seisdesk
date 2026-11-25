"""Validate packaged CUDA DLLs, headers and NVRTC compilation without a GPU.

This is a packaging check. Actual device execution is checked separately by
core.compute.probe_cuda(test=True) and the acoustic CUDA validation suite.
"""
import json
from pathlib import Path
import sys
import tempfile
import traceback


def check():
    import cupy
    from cuda.pathfinder import find_nvidia_header_directory, load_nvidia_dynamic_lib
    from cupy_backends.cuda.libs import nvrtc
    result = dict(cupy_version=cupy.__version__, libraries={})
    root = Path(sys._MEIPASS).resolve() if getattr(sys, "frozen", False) else None
    for name in ("cudart", "nvrtc", "cublas", "cufft", "curand", "cusolver", "cusparse"):
        library = load_nvidia_dynamic_lib(name)
        path = Path(library.abs_path).resolve()
        if root is not None and not path.is_relative_to(root):
            raise RuntimeError(f"{name} loaded outside the packaged application: {path}")
        result["libraries"][name] = str(path)
    headers = Path(find_nvidia_header_directory("cudart")).resolve()
    if root is not None and not headers.is_relative_to(root):
        raise RuntimeError("CUDA runtime headers are outside the packaged application")
    for name in ("cuda_fp16.h", "vector_types.h", "cuda_runtime.h"):
        if not (headers/name).is_file():
            raise RuntimeError("CUDA header is missing: "+name)
    source = '#include <cuda_fp16.h>\nextern "C" __global__ void check(float *p) { p[threadIdx.x] = __half2float(__float2half(2.0f)); }'
    program = nvrtc.createProgram(source, "seisdesk_runtime_check.cu", (), ())
    try:
        includes = [headers]
        cccl = Path(cupy.__file__).parent/"_core/include/cupy/_cccl/libcudacxx/include"
        if cccl.is_dir():
            includes.append(cccl)
        nvrtc.compileProgram(program, ("--std=c++11", "--gpu-architecture=compute_75",
                                       *("-I"+str(path) for path in includes)))
        ptx = nvrtc.getPTX(program)
        if not ptx:
            raise RuntimeError("NVRTC returned no compiled PTX")
        result.update(nvrtc_version=list(nvrtc.getVersion()), headers=str(headers), ptx_bytes=len(ptx))
    except Exception as exc:
        raise RuntimeError(str(exc)+"\n"+nvrtc.getProgramLog(program)) from exc
    finally:
        nvrtc.destroyProgram(program)
    from ..core.compute import probe_cuda
    result["device_detection"] = probe_cuda()
    result["state"] = "passed"
    return result


def main():
    destination = Path(tempfile.gettempdir())/"seisdesk_gpu_runtime_check.json"
    code = 0
    try:
        result = check()
    except Exception:
        code, result = 1, dict(state="failed", error=traceback.format_exc())
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return code
