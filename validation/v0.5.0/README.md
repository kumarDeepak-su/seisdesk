# SeisDesk v0.5 validation record

These results establish the specific numerical, integration and packaging checks below. They do not establish
production accuracy for every land or marine survey, and are not a full-survey interpretation benchmark.
This is the v0.5 historical record. Subsequent land checks and bounded interface-depth updates are documented in
the [v0.6 validation record](../v0.6.0/README.md).

## Revisions and environments

- Numerical source: `be6f464acf2b770c5ad5a383e499b4da63a09aec`.
- Final Windows packaging, quantitative slice display and frozen-worker check:
  `f3d79714e2514bb8a52ac953ce010b6a1eb5e02c`.
  Successful Windows build and installer.
  The numerical engine is unchanged from the Athena revision; the later commits add package smoke coverage
  and GUI colour scales/cursor readout. The preceding packaging build at `a42ec2d` also passed.
- Athena Slurm job `3486492`, 2 November 2026: completed with exit `0:0` in 33 seconds on an
  NVIDIA A100-SXM4-40GB. Module environment: Python 3.10.4 and CUDA 12.4.0; CuPy CUDA 12 wheel.
- Real SU: upstream `JohnWStockwellJr/SeisUnix`, revision `4db4181f654e474507721412bd333708b62f81a1`.
  Isolated Linux build, native little-endian format, non-X11 targets. SU is not bundled into the Windows installer.

## Automated checks

| Check | Result | Scope |
|---|---|---|
| Windows pytest | 273 passed, 5 skipped | Full application suite, including the native engine and GUI |
| Frozen Windows executable | Passed | Synthetic NMO/stack/migration/I/O workflow and two CPU 3D process workers |
| Athena pytest | 19 passed, no skips | CPU/CUDA adjoints and gradients, checkpoint consistency, modelling/RTM/FWI jobs, streaming, local/Dask worker agreement, joint arrivals, persistence and actual SU |
| Real SU failure handling | Passed | Invalid upstream stage reports failure and preserves the previous output |
| Independent 3D propagation reference | Passed | Homogeneous free-space Green function, arrival times, waveform amplitudes and spherical spreading |
| Marine field sample | Passed configured operator checks | Streamed input, real SU scaling, NMO and stacking comparison on 2,400 traces |

The five Windows skips are two CUDA tests, the optional Dask test and two installed-SU tests. All five ran and
passed on Athena. These are overlapping suites; their counts must not be added and described as distinct tests.
The Dask check used two CPU workers on one node. Multi-node and multiple-GPU scaling have not been measured.

The small adjoint/job fixtures deliberately use a coarse-grid override to isolate algebra and job handling.
Physical propagation is checked separately on a grid with 13.33 points per shortest reference wavelength.
Normal processing rejects fewer than ten points per shortest source wavelength.

## Independent 3D physical comparison

`tools/validate_3d_physics.py` compares a 161 x 161 x 161 homogeneous grid with the analytical retarded
3D Green function. Spacing is 5 m, velocity 2,000 m/s, time step 1 ms and the Ricker frequency 12 Hz.
Receivers span 100 m and 200 m offsets in axial, horizontal-oblique and vertical-oblique directions.

- Correlation: 0.999802 to 0.999966.
- Relative waveform L2 difference: 0.83% to 1.99%.
- All sampled arrival peaks coincide with the analytical peak times to the 1 ms sample interval.
- Peak ratio at 100 m versus 200 m: 2.00179; spherical-spreading prediction: 2.
- Predeclared limits: correlation >= 0.99, relative L2 <= 10%, peak error <= 2.1 ms,
  spreading-ratio relative error <= 5%.

This comparison does not validate heterogeneous field inversion, anisotropy, free-surface response or attenuation.
Raw measurements and thresholds are in [physics3d-report.json](physics3d-report.json).

## Marine field comparison

The input is the first 2,400 complete traces (20 shots, 1,500 samples, 4 ms) from the
[SEG Mobil AVO Viking Graben Line 12 dataset](https://s3.amazonaws.com/open.source.geoscience/open_data/Mobil_Avo_Viking_Graben_Line_12/mobil_avo.html).
Credit: Mobil; Keys and Foster, SEG Open File Publication 4. The source SEG-Y was downloaded directly from
the SEG open-data bucket; a fallback mirror was not used.

Sample SHA-256: `10763374d4860589fa70d8a85504d41269e22d82b4e7c09db7a8c78cb8c3dc9c`.

Streaming reproduces the loaded trace samples exactly. Real SU `sugain scale=2` reproduces doubled samples
within the specified float tolerance. Both NMO implementations receive identical sorted, filtered input and a
constant 2,000 m/s velocity with stretch muting disabled; this is an operator comparison, not a field velocity model.

| Operator | Correlation with SU | Relative L2 difference |
|---|---:|---:|
| NMO | 0.998813 | 7.4584% |
| Live stack | 0.998594 | 7.0534% |

The predeclared correlation threshold was 0.95. The results are not sample-identical: SU uses sinc interpolation
and native NMO uses linear interpolation. The residual difference remains reported; it is not interpreted as
proof of amplitude fidelity. See [marine-report.json](marine-report.json).

## Open validation and method limits

- A land field-data comparison is still pending. The public Alaska archive links returned HTTP 403; the
  obtainable author archive contains processing scripts and tables, without the raw SEG-Y. No land field result
  is claimed. A land survey with geometry and a trusted comparison would enable this check.
- Full-survey land/marine processing, heterogeneous 3D field RTM/FWI recovery, acquisition-specific noise and
  source-signature adequacy remain unvalidated. There is no claim of well-tie or geological accuracy from these tests.
- Real SU was tested on Linux. Actual Windows WSL SU execution still needs a WSL environment check.
- The 3D solver is isotropic, constant-density acoustic with a second-order spatial stencil and sponge boundaries.
  It has no elastic, variable-density, anisotropic, free-surface or attenuation inversion.
- Joint traveltime inversion is 2D with assigned, fixed interfaces. It retains multiple arrivals, but does not
  invert interface depths or general converted/multiple-reflection paths.
- Distribution is across shots, with a shared directory and a model fitting on each worker. There is no spatial
  domain decomposition or GUI SSH/Slurm submission. Athena jobs use the documented headless CLI.
- Legacy operator limits remain documented in the user guide. In particular, 3D SRME and full 3D/5D
  acquisition regularisation are not supplied by the new acoustic engine.

## Reproduction

On a configured Athena allocation, `tools/athena_validate.sh` runs the 19 tests, the independent 3D reference
and the marine comparison. `SEISDESK_VALIDATION_ROOT` can override its working directory. The root requires
the source tree, isolated Python environment, installed SU and downloaded field sample as described by the
companion scripts. Slurm account/partition directives must match the executing user's allocation.

Windows builds use `.github/workflows/windows-build.yml` and pinned `packaging/requirements-windows.txt`.
The raw test results are [Athena XML](athena-results.xml) and [final Windows XML](windows-results-f3d7971.xml).
