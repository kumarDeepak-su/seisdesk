# SeisDesk expansion status

The requested scientific expansion is **not complete**. v0.12 adds a light desktop workspace, command search,
drag-and-drop workflows and flow undo/redo. The numerical methods remain the v0.11 baseline, including bounded
shared 3D interface/velocity updates and conditional depth uncertainty. Both v0.12 Windows editions pass their
build and validation checks.
This file distinguishes implemented methods from remaining work; adding a process to the palette is not
evidence that its physics, scale or field behaviour has been validated.

The static inventory in `validation/method-inventory.json` identifies 171 registered process classes. It is a
navigation aid for the audit, not a claim of 171 fully validated production methods. Detailed v0.5 evidence is
in `validation/v0.5.0/README.md`; v0.6 additions and public land checks are in `validation/v0.6.0/README.md`.

## Current work

- v0.12: light appearance by default, optional dark theme, a Home workspace with recent projects, command search,
  dataset filtering, file/process drops, flow reordering, bypass, undo/redo, progress/cancellation feedback and
  remembered layout. Viewer/velocity controls fit a checked 1280 x 800 workspace. Flow-only project saves and
  read-only project viewing preserve the correct unsaved state. Application source **`651eb1d`**, Windows build
  **41780615893**, passes **399 tests with eight optional-environment skips per CPU/CUDA edition**. Both packaged
  executables, CUDA runtime/compiler checks, manuals and installers pass. Focused workspace run **41780609401**
  passes **23 tests without skips**; these overlap the full suite. Nine GUI renders and the changed pages of the
  235-page manual were checked. Numerical modules are unchanged, so the earlier Athena evidence still applies to
  those algorithms. See `validation/v0.12.0/README.md`. The physical Windows GPU check remains deferred to the user.
- v0.11: bounded shared 3D interface updates, uniform/bilinear control grids, optional layer
  remapping and joint velocity updates, conditional timing-noise refits, contact masks, boundary maps/sections and
  reusable exports are implemented. Athena job **3492660** passes **18 tests without skips** plus an independent
  noisy plane-reflection example (RMS **4.115 → 0.947 ms**, 20 refits). A dipping-plane forward comparison improves
  from **3.08%** to **1.84%** maximum time error when spacing decreases from 10 m to 5 m. Only 9/81 nodes per example
  boundary have contacts; zero spread and a saturated control illustrate remaining resolution/ambiguity limits.
  Final application source **`5ba6dd4`**, Windows build **41742119084**, passes **388 tests with eight
  optional-environment skips per CPU/CUDA edition**. Both frozen executables recover a reflector depth and export
  the boundary bundle; previous seismic, 3D, process-worker, reconstruction and dispersion checks also pass.
  CUDA runtime/compiler checks, both installers and manuals pass. Four new GUI previews and manual pages 141–144
  were inspected; the five final result images are in `dist/result-images/v0.11.0/`, with earlier renders retained.
  The new interface search executes on CPU. See `validation/v0.11.0/README.md` for reports and installer links.
- v0.10: explicit anti-aliased common-axis import for mixed-rate/variable-length SEG-Y/SU/SEG-2;
  SEG-D 2.0/2.1 demultiplexed disk records with all nine standard sample decoders; SPS S/R/X geometry matching;
  stricter exact-timing export checks; layered Rayleigh/Love modal phase/group dispersion inversion with bounded
  Vs/thickness, multistart fits and conditional Gaussian pick-error refits. GUI import controls, layer/pick tables,
  cancellation, fit/QC views, project persistence and saved-flow input are implemented. Final Athena job **3491347**
  passes **80 tests without skips**, including independent ObsPy, analytical Rayleigh/Love and layer-recovery checks.
  All samples in a public nine-trace SmartSolo SEG-D file match an independent reader exactly. A noisy 32-pick
  Rayleigh/Love example reduces RMS from 28.481 to 2.310 m/s; 30/30 conditional pick-error refits complete.
  An alternate fast-delta solver failed during development and is excluded; Dunkin is the supported Rayleigh solver.
  Final Windows source `a09b283`, build **41689367571**, passes **378 tests with eight optional-environment skips
  in each CPU/CUDA edition**. Both frozen executables compile and execute the dispersion kernel, check common-axis
  import and pass the previous seismic/3D/process-worker/bundle checks. GUI previews, manuals, installers and the
  CUDA runtime/compiler checks pass. Six new result images are in `dist/result-images/v0.10.0`.
  See `validation/v0.10.0/README.md` for exact reports, installer links and remaining limits. These additions run on CPU.
- v0.9: full-volume 3D first/reflected/head-arrival tomography, fixed plane/surface interfaces, timing-noise
  uncertainty ensembles, per-pick uncertainty overrides and QC exports, phase controls and orthogonal model/coverage/
  uncertainty views are implemented. Athena CPU jobs 3490953 and 3490978 pass the four core checks and the five-check
  core/processing/project-persistence set respectively. Survey-wide 3D/5D binned Fourier POCS with an explicit observed
  mask, duplicate policy, aperture distance limit, memory/disk checks, temporary spectrum storage and fold maps is
  implemented. Athena CPU job 3490997 passes all ten new scientific/processing checks, including held-out 3D/5D
  analytic-event reconstruction and cleanup on cancellation. Final numerical snapshot `3fad78e` passes the ten
  checks and example job 3491028: 96 arrivals, RMS 5.350 to 0.080 ms. Curved-reflection job 3491045 has held-out
  trace L2 errors of 21.08% (3D) and 21.89% (5D), with first-event amplitude errors of 7.09% and 13.11% respectively;
  recorded samples remain exact. These are controlled examples, not field-scale accuracy guarantees.
  Windows build 41662508196, final application source `d61930a`, passes **335 tests with eight optional-environment
  skips in each CPU/CUDA edition**. Frozen checks execute the new graph inversion and disk-backed FFT reconstruction;
  GUI controls/cancellation, previews, manuals, installers and bundled CUDA runtime compilation all pass.
  See `validation/v0.9.0/README.md`. Nine new test images are saved under `dist/result-images/v0.9.0`.
  These additions run on CPU.
- v0.8: workstation device detection, an explicit CPU/CUDA/automatic selector, GPU-index selection,
  saved project defaults, runtime checks and separate Windows CPU/CUDA installers are implemented. Athena A100 jobs
  3489949 (CuPy 13.6) and 3490030 (CuPy 14.2) each pass 14 tests without skips. Windows build 41613398476, source
  `45b7663`, passes 320 tests with eight optional-environment skips in each edition. Frozen executable self-tests,
  bundled CUDA libraries/headers/NVRTC compilation, readable controls, manuals and both installers pass.
  See `validation/v0.8.0/README.md`. The user will test the installed application on their Windows GPU later;
  that deployment check is deferred to the user and does not block the remaining software development.
- v0.7: GUI SSH/Slurm submission, portable input/source/result bundles, durable job records, monitoring,
  cancellation and result import pass Windows tests and real Athena lifecycle checks. Windows source `c73ee6d`
  passes 313 tests and packaged executable checks; real CPU, CUDA, expected-failure and cancellation jobs pass.
  See `validation/v0.7.0/README.md`. This launcher requests one node and at most one GPU; distributed scaling and
  new wave physics remain outstanding. Windows OpenSSH authentication must be configured on the user's workstation.
- Public land data: Sirente RFL 1, 26 SEG-2 shot records, 1.1 MB archive; published Koenigsee refraction picks,
  63 stations and 714 observations, approximately 10 kB. Independent reader/operator and held-out-pick checks
  passed on Athena job 3489391. Held-out refraction RMS fell from 3.61 to 0.69 ms; the 312 reflection traces
  match ObsPy, and NMO/stack correlation with SU exceeds 0.999999. These are software/prediction checks.
  The Sirente archive's coordinate headers disagree with its repository description;
  this must be reported and resolved before treating either geometry as surveyed truth.
- Joint interface depths: bounded updates of flat-depth or x,z control-point interfaces, shared boundaries
  between phases, optional velocity-layer remapping, alternate velocity updates, retained objective history
  and initial/final interface overlays. Numerical, analytical-depth, flow-persistence and graphical-editor checks
  passed. Final Windows build 41580143865 passes 303 tests and the packaged executable's self-test.
- SEG-2 packed 20-bit samples and per-trace DELAY alignment pass Athena byte-level, analytic timing and independent
  ObsPy checks. SKEW is preserved and applied only through an explicit import option. Windows control tests pass.

## Remaining capabilities to implement or strengthen

| Area | Present baseline | Remaining scope |
|---|---|---|
| Formats and geometry | SEG-Y, SU, all five SEG-2 encodings and selected GPR; explicit mixed-rate common-axis resampling; SEG-D 2.0/2.1 disk records; SPS 2.0/2.1 matching; trace identities and 2D/3D geometry tools | SEG-D rev 3/tape wrappers/proprietary descriptors, vendor navigation/timestamps, nonuniform SEG-Y rev 2, broader field round trips |
| Picking | Multiple typed events and trace-associated picks; visual branch/interface configuration; 3D per-pick uncertainty and residual QC | Broader picking uncertainty and branch-QC workflows across methods |
| Refraction/reflection tomography | 2D bounded/shared interface-depth updates; 3D shortest-path first/head/reflected branches, bounded shared control-grid interface updates and conditional timing uncertainty | General/overturning/converted path families, structural and phase ambiguity, independent heterogeneous field benchmarks |
| Wave physics | 2D acoustic/elastic tools; full-volume 3D acoustic CPU/CUDA with adjoint checks | 3D elastic and anisotropic physics, variable density, attenuation, free-surface response, matching adjoints/inversion |
| Multiple attenuation | Existing 2D SRME/Radon and adaptive subtraction | General 3D SRME and acquisition-aware multiple prediction, with independent comparisons |
| Regularisation/interpolation | Gather-wise native methods; survey-wide binned 3D/5D Fourier POCS | Survey-scale field/amplitude validation, guided de-aliasing, irregular-grid and acquisition-consistent reconstruction |
| Velocity building/migration | Semblance, moveout, several time/depth methods and acoustic RTM | Broader 3D image-gather/MVA workflows; explicitly audit each approximation against an independent reference |
| Surface waves/statics | Rayleigh/Love phase/group multimode layered Vs/thickness inversion, multistart and conditional pick-error ensembles; existing refraction/GRM/residual statics | Structural/mode uncertainty, independent surface-wave field examples, wider field statics benchmarks |
| Execution | Local/WSL SU adapter; CPU workers; CUDA; Dask shot distribution; Slurm CLI and GUI launcher | Actual WSL execution test, Windows-to-Athena SSH setup check, multi-node/multi-GPU scaling, spatial decomposition |
| Reproducibility | Saved projects, diagnostics, portable job/source/result bundles; scheduler/application states and logs | Broader resource/performance reporting and end-to-end field processing examples with known provenance |
| User interface | Light workspace, command search, file/process drops, flow undo/redo, gather tools and model/result windows | Remove avoidable JSON/manual-host setup from common workflows; broader usability checks of full processing sequences |
| Field evidence | Controlled numerical tests, selected marine/land operator comparisons and held-out field refraction picks | Confirmed Sirente geometry, independent heterogeneous 3D field benchmarks and survey-scale QA |

## Completion standard

Each addition must include usable controls or a documented compute-host interface, persisted results, explicit
units/assumptions, meaningful numerical checks, and comparison with independent analytical, package or field
references where available. Native and external-engine capabilities must be labelled separately. External
engine adapters do not make untested upstream methods validated SeisDesk features.

User approval covers continuing this expansion. No additional approval is needed for ordinary implementation,
isolated validation or updates to the existing draft PR. Numerical and GUI execution remain off the Mac:
Windows CI handles desktop checks, and Athena compute allocations handle scientific/CUDA validation.
