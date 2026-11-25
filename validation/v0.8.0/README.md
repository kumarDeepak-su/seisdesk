# Workstation CUDA validation

v0.8 adds local CPU/CUDA/automatic selection, visible GPU indices, a real CUDA kernel check, project defaults,
actual-device job records and GPU memory estimates. The CUDA Windows edition includes CuPy and NVIDIA runtime,
compiler and header packages. CUDA currently accelerates full-volume 3D constant-density acoustic modelling,
RTM and FWI; this milestone does not port the other processing methods to CUDA.

## Hardware execution

- Athena Slurm job **3489949**, source `1f978f7`, Python 3.10/CuPy 13.6: **14 passed, zero skips** on an NVIDIA
  A100-SXM4-40GB. `v08-cuda-results.xml` and `v08-device-report.json` retain the unmodified evidence.
- Athena Slurm job **3490030**, source `7758b6a`, Python 3.11.9/CuPy 14.2 with the same NVIDIA component package
  selection as the Windows CUDA build: **14 passed, zero skips**, exit `0:0`, on A100 node `t0020`.
  `v08-cupy14-results.xml`, `v08-cupy14-device-report.json` and `v08-cupy14-runtime-report.json` retain the evidence.
  This environment uses packaged NVIDIA libraries without a system Python or CUDA module. NVRTC reports **12.8**;
  CuPy's statically linked runtime reports **12.9** and the driver reports **13.2**. These are different components,
  not a claim that every CUDA library has the same patch version.
- Checks include real device discovery, kernel execution, automatic GPU selection and persistence, explicit-CUDA
  failure reporting, CPU fallback, source-adjoint and velocity-gradient checks, CPU/GPU agreement, complete
  modelling/RTM/FWI jobs and the existing streamed/local-process job tests. The independent v0.5 propagation
  benchmarks remain separate; these small synthetic jobs do not establish field-scale throughput or accuracy.
- The CuPy 14 environment also loads all seven required CUDA libraries, checks headers and compiles a half-precision
  kernel to PTX with NVRTC. The runtime report records the exact paths used.

## Windows packaging

Windows build 41613398476, source `45b7663`,
**succeeds for both CPU and CUDA editions**. Each passes **320 tests with eight optional-environment skips**,
zero failures and zero errors. Unmodified reports are `windows-cpu-results-45b7663.xml` and
`windows-cuda-results-45b7663.xml`. The skips are three real-GPU checks, two actual-SU checks, two ObsPy comparisons
and one Dask check. The three GPU checks run successfully in the Athena suite; prior SU/ObsPy/Dask measurements
are recorded in the earlier validation milestones.

Both frozen executables pass the seismic workflow, two-process 3D and portable compute-source/bundle self-tests.
The CUDA edition additionally passes bundled DLL/header location checks and real NVRTC compilation to **709 bytes
of PTX**; see `windows-cuda-runtime-45b7663.json`. Missing GPU/driver detection is expected on this Windows runner.
Both installers and manuals were produced successfully. The local batch entry point's help route also passes;
the full installers were built by the workflow. The corrected v0.8 control preview was inspected and is readable.

Installer artifacts from this build:

- CPU installer:
  `SeisDesk-0.8.0-setup.exe`.
- CUDA installer:
  `SeisDesk-0.8.0-CUDA-setup.exe`. This edition also supports CPU execution.

The CUDA archive is approximately 1.18 GB because it includes the NVIDIA component libraries and headers.
These are draft-build artifacts, not a published release. GitHub access to the repository is required.

The Windows runner has no GPU. Its checks validate packaged imports, bundled DLL/header locations and PTX
compilation. Actual CUDA kernel/solver execution runs on Athena Linux compute nodes. Running the frozen Windows
installer on a physical NVIDIA workstation remains an additional deployment check; no Windows GPU has been
available in this session. The device screenshot is explicitly illustrative.

## Reproduction and scope

`tools/athena_v08_validate.sh` runs the older CuPy environment. `tools/athena_v08_cupy14_validate.sh` runs the isolated
Python 3.11/CuPy 14 environment. Both must run through Slurm on a compute node. The Windows workflow builds both
editions, runs tests and frozen executable checks, renders controls, builds the manual and creates installers.
The local build entry point accepts `build_windows.bat cpu` or `build_windows.bat cuda` and uses separate environments.

On a GPU workstation, install the CUDA edition and an appropriate NVIDIA driver, then open
**Processing > Workstation CPU / CUDA devices**, select a GPU, run **Run CUDA check** and save the choice.
New local 3D steps and survey jobs inherit it; existing flows preserve their explicit settings. Explicit CUDA never
silently falls back to CPU. Automatic selection records the fallback reason. Local CUDA uses one selected GPU;
multi-GPU/multi-node scale validation remains outstanding.

The broader processing expansion remains incomplete. See `DEVELOPMENT_STATUS.md` for the remaining physics,
survey reconstruction, input-format, uncertainty and field-validation work.
