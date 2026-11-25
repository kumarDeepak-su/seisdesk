# v0.10 field import and layered dispersion validation

Application numerical source: `20f793a`. Validated Windows build: `a09b283` (packaging/manual changes only).
`source-manifest.json` confirms that all 138 application Python files on the Athena compute snapshot match the local
build. Numerical work ran in Slurm compute allocations; the Mac was used for editing, file transfer, static
inspection and viewing existing images. These additions execute on CPU.

## Scientific and real-data results

- Athena job **3491347**: **80 passed, zero skips**, including the two independent ObsPy SEG-2 comparisons.
  The final job and earlier job 3491318 are COMPLETED, exit 0, elapsed 10 seconds, node t0005.
  Earlier job 3491318 passes the 77-check set and both benchmarks. The additional three checks cover SEG-Y feet conversion, CLI import persistence,
  and saved-flow dispersion-model/result persistence.
- Resampling: mixed rates and fractional starts agree with an analytical bandlimited signal; above-target-Nyquist
  content is attenuated; unchanged integer-grid samples remain exact. Both endian orders, variable trace lengths,
  supported SEG-Y encodings, extended textual headers, corruption, output bounds and cancellation are exercised.
- SEG-D: byte-level fixtures cover all nine rev 2 sample encodings, MP descaling, skew, concatenated records and
  malformed/truncated input. Public SmartSolo data has **9 traces × 450,001 samples**, dt **4 ms**. Every descaled
  float32 sample matches the independent reader exactly (maximum absolute difference **0**). This validates that
  specimen, not every vendor or SEG-D revision. Vendor timestamp/position extensions are not inferred.
- SPS: fixed-column matching, descending receiver sequences, multicomponent groups, point indices, coordinate-unit
  conversion, static-header copying, ambiguous/missing relationships and project persistence are checked.
- Dispersion: Rayleigh half-space velocities match the secular polynomial; Love fundamental/higher modes match
  the independent two-layer equation; group velocities match a numerical derivative of phase dispersion.
  Noiseless Vs and thickness recovery, conditional refits, absent modes, bounds, cancellation and persistence pass.
- Noisy synthetic example: **32 Rayleigh/Love picks**, 2 m/s stated errors. RMS **28.4809 → 2.31023 m/s**;
  fitted thickness **9.701 m** versus 10 m truth, Vs **199.109 / 392.166 m/s** versus 200 / 400 m/s.
  **30/30** Gaussian pick-error refits complete. This ensemble is conditional on the model topology, mode assignments,
  Vp/Vs and density; it does not quantify structural ambiguity or establish field accuracy.

The alternate fast-delta solver failed on the shallow example during development. It is excluded from this integration;
the supported Rayleigh solver is Dunkin. Independent Love/Rayleigh analytical tests remain in the suite.

## Windows status

Final Windows build 41689367571, source
**`a09b283`**, succeeds for both CPU and CUDA editions. Each passes **378 tests, eight skips, zero failures and
zero errors**. Unmodified reports are `windows-cpu-results-a09b283.xml` and `windows-cuda-results-a09b283.xml`;
`windows-build-a09b283.json` records every job and step. Skips cover three GPU, two SU, two ObsPy and one Dask checks
requiring optional environments. The two ObsPy checks pass in the final Athena run; earlier SU/CUDA/Dask evidence
remains in the corresponding milestone records.

Both frozen executables compile and execute the disba/CPS kernel, compare its velocity with an analytical result,
and check common-axis import. They also execute the previous 3D tomography/reconstruction, two-process modelling,
portable-bundle and seismic/SEG-Y checks. The four new GUI previews were rendered on Windows and inspected.
Both PDF manuals and installers are produced successfully.
The new manual chapter (pages 139–141 of the 229-page CPU PDF) was rendered and inspected for readable text,
figures and unclipped layout. `packaged-checks-a09b283.log` retains the unaltered self-test and pytest result lines;
`build-log-manifest.json` records the complete local build logs and hashes.

The CUDA edition loads all seven bundled CUDA DLLs, locates headers and compiles **709 bytes of PTX** with NVRTC
12.8. `cuda-runtime-a09b283.json` is the unmodified report. Windows CI has no GPU/driver; this is packaging and compiler
validation, not hardware execution. The existing CUDA solvers were exercised on Athena A100 in v0.8. The user's
physical Windows GPU deployment check remains deferred. The v0.10 import and surface-wave additions execute on CPU.

The initial source `2b1e21b`, run 41677707757, passed 375 tests per edition but exposed a missing Matplotlib import
in the frozen CPU executable. The final build includes Matplotlib 3.10.7/Agg, retains the dispersion source for Numba,
uses a writable per-user compilation cache and passes the failing path. Earlier JUnit reports remain for traceability.

Draft-build downloads:

- CPU installer,
  `SeisDesk-0.10.0-setup.exe`.
- CUDA installer,
  `SeisDesk-0.10.0-CUDA-setup.exe`; approximately 1.21 GB archive. It also supports CPU execution.
- User manual.

These are private draft-build artifacts, not a published release or merged PR.
Local CPU installer and manual copies are in `dist/v0.10.0-a09b283`; `local-downloads.json` records their exact sizes
and SHA-256 hashes. The CUDA installer remains available through its artifact link.

## Evidence and limits

- `field-report.json`, `surface-report.json`: reference, hashes, fit and uncertainty results.
- `v10-scientific-results.xml`, `athena-v10-3491347.log`: final scientific checks and benchmarks.
- `source-manifest.json`: application-source identity across the Mac checkout and compute snapshot.
- Result images: `dist/result-images/v0.10.0/` (two Athena plots and four inspected Windows previews).
  `image-manifest.json` preserves their original sizes, SHA-256 hashes and source jobs. The scientific plot uses
  30 conditional refits; the Windows preview uses eight for a shorter screenshot run.

Remaining format scope includes SEG-D rev 3, tape wrappers/proprietary descriptors, arbitrary vendor timing/navigation
metadata and nonuniform SEG-Y rev 2. Surface-wave inversion is 1D isotropic solid layering with fixed Vp/Vs/density;
it is not a 3D elastic waveform inversion. The wider application checklist in `DEVELOPMENT_STATUS.md` remains open.

References: [SEG SPS 2.1](https://seg.org/wp-content/uploads/2025/11/seg_sps_rev2.1.pdf),
[SEG-D 2.1](https://seg.org/wp-content/uploads/2025/11/seg_d_rev2.1.pdf),
[disba](https://github.com/keurfonluu/disba),
[independent SEG-D reader and public example](https://github.com/SEGDreader/SEGD_reader/tree/0dc57b6a1eecb266a05bc265031490bc5d9cbe68).
