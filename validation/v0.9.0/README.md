# 3D arrivals and survey reconstruction validation

v0.9 adds CPU shortest-path tomography through a full velocity volume and CPU frequency-blocked 3D/5D Fourier
POCS reconstruction. It includes phase/interface controls, per-pick uncertainty and residual tables, conditional
timing-noise ensembles, model/coverage/uncertainty slices, reconstruction fold maps, persisted diagnostics and
background single-step execution with progress and cancellation. These operators do not use CUDA; the existing
3D acoustic modelling/RTM/FWI CUDA backend remains available.

## Athena scientific execution

Slurm job **3491028**, source **`3fad78e`**, node `t0005`, Python 3.11.9, exits `0:0` in nine seconds. The ten new
checks pass without skips. `v09-scientific-results.xml`, `athena-v09-3491028.log` and `report.json` preserve the
unmodified output. The numerical implementation is unchanged by the later frozen-build self-test addition.

The tests cover independent homogeneous direct/mirror-reflection times, two-layer analytical head-wave times,
reciprocity, discrete slowness derivatives, topography and memory rejection, bounded inversion convergence,
conditional timing uncertainty, cancellation, per-trace multi-event retention, uncertainty CSV matching, model/QC
export and project round trips. Reconstruction checks withhold analytic events in two/four spatial dimensions,
preserve exact observed float32 samples (including live zeros), retain identities, reject bad geometry/duplicate
bins by default, apply an explicit averaging policy, preserve full signed-offset geometry, and remove temporary
spectrum storage after cancellation.

The small 3D example retains 96 observations across P, R1 and R2. Travel-time RMS decreases from **5.35003 to
0.07958 ms** over five accepted updates. Only **269 of 729 nodes** have final ray coverage. The uncertainty example
uses three timing-noise fits and masks uncovered cells. This is a diagnostic example: three fits do not establish
stable percentiles, total geological uncertainty or unique velocity recovery. Interfaces remain fixed.

The periodic 5D plane-wave example retains 585 traces and predicts 315 withheld traces, with relative L2 error
`1.18685e-7` and zero change to recorded samples. Its exact sparse Fourier representation is deliberately favourable;
it does not establish comparable field accuracy.

## Curved reflections and amplitude variation

Slurm job **3491045**, node `t0005`, exits `0:0` in four seconds using the same implementation. The independently
defined Ricker events have curved/dipping travel times, offset moveout and spatial/offset amplitude variation.
Approximately 30% of traces are randomly withheld. `curved-events-report.json` and `v09-curved-3491045.log` retain
the measurements. `tools/validate_v09_events.py` reproduces the test inside a Slurm CPU allocation.

| Test | Recorded / withheld | Withheld waveform relative L2 | First-event amplitude relative L2 | Recorded-sample change |
|---|---:|---:|---:|---:|
| 3D, 16×16 spatial grid | 171 / 85 | 21.08% | 7.09% | 0 |
| 5D, 9×9×5×5 spatial grid | 1414 / 611 | 21.89% | 13.11% | 0 |

Zero filling has 100% relative L2 error over the same withheld samples. Amplitudes are assessed by projection onto
the independently specified first-event wavelet. These results show material residual waveform/amplitude errors;
the method is not validated for field amplitude interpretation, large gaps, irregular NUFFT sampling or survey-scale
performance. The finite Fourier grid assumes periodic spatial support; global edge effects remain visible.

## Windows validation

Windows build 41662508196, application source
**`d61930a`**, succeeds for **both CPU and CUDA editions**. Each passes **335 tests, eight skips, zero failures and
zero errors**. Unmodified JUnit files are `windows-cpu-results-d61930a.xml` and `windows-cuda-results-d61930a.xml`; the complete job/step
status record is `windows-build-d61930a.json`. The eight skips are three GPU, two SU, two ObsPy and one Dask checks
that require optional environments. Earlier independent environment checks remain documented in previous milestones.

New GUI checks cover phase editing and flow persistence without JSON, masked uncertainty slices, per-arrival
tables, reconstruction maps, responsiveness during work, cancellation and publication of successful results only.
All five new control/result previews were rendered on Windows and inspected for readable labels and units.

Both frozen executables execute 3D graph inversion and disk-backed survey FFT reconstruction in addition to the
existing seismic, two-process 3D modelling and portable-bundle checks. Both manuals and installers were produced.
The CUDA edition loads all seven packaged CUDA DLLs, locates headers and compiles a kernel to **709 bytes of PTX**
with NVRTC 12.8; `cuda-runtime-d61930a/gpu-runtime-check.json` is the unmodified report. Missing driver/device
detection is expected on this CPU-only CI runner. Actual CUDA solver execution on Athena is documented in v0.8;
this milestone does not add new CUDA physics. The user's physical Windows GPU deployment check remains deferred.

Draft-build installers:

- CPU installer,
  `SeisDesk-0.9.0-setup.exe`. A local copy and the manual are in `dist/v0.9.0-d61930a`;
  `local-downloads.json` records their exact SHA-256 hashes.
- CUDA installer,
  `SeisDesk-0.9.0-CUDA-setup.exe`. Its archive is approximately 1.18 GB, including NVIDIA libraries and headers.
  The CUDA edition also supports CPU execution. This archive has not been downloaded locally.

These are private draft-build artifacts, not a published release or a merged PR.

## Reproduction and artifacts

- Submit `tools/athena_v09_validate.sh` inside the documented Athena environment. It runs the scientific tests,
  stores example arrays with `save_results`, writes JSON metrics and generates two scientific PNG figures.
- Run `tools/validate_v09_events.py OUTPUT_DIRECTORY` in a CPU Slurm allocation for the two curved-event figures
  and their report. The tool refuses execution on the Mac or a Linux login node without `SLURM_JOB_ID`.
- Windows CI runs all project tests, renders `tools/v09_screenshots.py`, builds both application editions and runs
  the frozen executable's end-to-end checks, including the new inversion and disk-backed FFT paths.
- All nine local scientific and Windows GUI PNGs are in `dist/result-images/v0.9.0`; `image-manifest.json` records
  original dimensions, source jobs and hashes. The larger synthetic arrays are on Athena under
  `$HOME/seisdesk-validation-20261102/v09-benchmark/synthetic-results.npz`.

The full expansion remains incomplete; see `DEVELOPMENT_STATUS.md` for the remaining physics, format, field
validation, interface-inversion and distributed-execution work.
