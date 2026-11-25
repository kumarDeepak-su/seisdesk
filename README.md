# SeisDesk

Interactive seismic processing for 2D lines and 3D acoustic surveys, with a Windows installer.

SeisDesk has a NumPy/SciPy processing engine and a bridge to installed Seismic Unix programs on a local host
or Windows WSL. Literal SU pipelines and explicit stdin/stdout shell scripts can run from the GUI or saved flows.
Installed-program discovery and self documentation are available in the SU workbench. SU itself is installed separately.

- **Input/output**: SEG-Y, Seismic Unix (SU), SEG-2 and supported SEG-D 2.0/2.1 disk records, SPS survey geometry,
  explicit common-axis resampling, plus projects saved as folders.
- **Display**: gathers by FFID, shot, receiver, CMP, offset, inline or crossline, as density or wiggle traces, with
  read-out and spectra.
- **Processing**:
  - Geometry, quality control, trace editing, despiking and mutes; gain, including surface-consistent amplitudes.
  - Filters: band-pass, Butterworth, notch, time-variant, f-k, f-x deconvolution, median, tau-p, eigenimage (SVD),
    Harlan signal/noise separation, deghosting.
  - Deconvolution: spiking, predictive, surface-consistent, signature and homomorphic; Wiener shaping, phase rotation,
    zero- and minimum-phase conversion; vibroseis correlation and Klauder-to-minimum-phase.
  - Statics: elevation, refraction (delay-time inversion), surface-consistent residual; refraction convolution
    section; Kirchhoff wave-equation datuming.
  - Velocity analysis: semblance, automatic picking, eta scan, constant-velocity panels, residual moveout scan.
  - NMO (with eta), DMO, anisotropic DMO (migration to zero offset), residual moveout; live, median, diversity and
    phase-weighted stacks; trace interpolation (f-x Spitz, POCS).
  - Time migration: Stolt, phase shift (isotropic and VTI), Kirchhoff (with eta); 3D phase-shift, Stolt and
    Kirchhoff for small volumes.
  - Depth migration: split-step, finite difference, Fourier finite difference, PSPI, Gaussian beam, Kirchhoff (also
    from topography), prestack shot-profile wave-equation migration; time-depth conversion.
  - Modelling: acoustic and elastic finite differences, Kirchhoff and ray-traced synthetics.
  - Attributes: Hilbert transform, instantaneous attributes, spectral decomposition, time-frequency analysis.
  - Multicomponent rotation (3C, Alford 4C), polarisation analysis and filtering.
- **Tools**: MASW dispersion analysis and layered Rayleigh/Love phase/group inversion with multiple modes,
  bounded Vs/thickness and conditional pick-error ensembles; well logs (LAS) with synthetic seismogram, well tie,
  Backus averaging and Gassmann fluid substitution.
- **Picking**: multiple named direct, refracted, reflected, multiple and horizon events. Trace identities preserve
  separate picks on traces with equal offsets through sorting and project saves. Refraction line fits use eligible
  arrival types; reflection picks remain available for interpretation and picked mutes.
- **Flows**: saved as JSON, run in the GUI or in batch with `seisdesk-cli`.

## Version 0.12 desktop workspace

- Home workspace with import, examples, project counts and recent projects.
- Light appearance by default, an optional dark theme, remembered layout, command search (Ctrl+K), searchable
  datasets and workspace shortcuts.
- Drag seismic files into the application, drag processes into flows and drag steps to change their order.
- Flow undo/redo for structure, parameters, bypass states and notes; saved flow changes and clear run/cancel feedback.
- Expandable viewer and velocity controls, with a checked 1280 x 800 workspace and retained scientific colour scales.

Both Windows editions pass **399 tests with eight optional-environment skips each**, packaged executable checks,
manual generation and installer creation. The focused workspace review passes **23 checks without skips**.
See [v0.12 validation and installers](validation/v0.12.0/README.md).
These changes update the desktop workflow; the scientific scope and limitations below still apply.

## Version 0.11 shared 3D interfaces

- The 3D multi-arrival process can now update shared reflecting/refracting boundaries with uniform or bilinear
  control-grid shifts. Fixed boundaries, absolute depth limits, original shift bounds and layer ordering are enforced.
  Optional layer remapping and velocity updates allow joint fitting; a known velocity model can also remain fixed.
- Boundary controls, depth/change/contact/uncertainty maps, orthogonal sections and surface/diagnostic exports are
  available from the 3D arrival workflow. Conditional timing-noise refits retain the original bounds and mask depth
  summaries where the fitted rays have no boundary contacts. Projects preserve the controls and result arrays.
- Athena passes 18 scientific checks without skips. An independent 18-arrival plane-reflection example reduces RMS
  from 4.115 to 0.947 ms and completes 20 conditional refits. This small fixed-velocity example does not establish
  unique geological recovery; its contact coverage, saturated control and grid-dependent spread are reported.

Both Windows editions pass 388 tests with eight optional-environment skips, frozen executable checks, manual
generation and installer creation. These additions execute on CPU and support ordered single-valued depth surfaces.
Ray/control-grid resolution, assigned phases and velocity-depth ambiguity remain limits; see
[v0.11 evidence and remaining scope](validation/v0.11.0/README.md).

## Version 0.10 field import and surface waves

- Mixed-rate/variable-length SU, SEG-Y rev 0/1 and SEG-2 import onto an explicit common time axis, with anti-alias
  filtering, fractional starts, window choices, resource checks and original timing provenance. Export rejects
  timing that the chosen SU/SEG-Y fields cannot represent exactly.
- SEG-D 2.0/2.1 demultiplexed disk records with nine sample decoders and explicit channel/auxiliary/descaling controls.
  SPS S/R/X matching supports point indices, multicomponent receiver groups, unit conversion and optional static headers.
  Proprietary navigation/timestamp extensions and SEG-D rev 3 remain outside this reader's scope.
- Layered surface-wave inversion fits Vs and optional thickness using Rayleigh/Love phase/group picks and assigned
  modes. Editable model/pick tables, bounds, multistart, cancellation, fit/QC views, exports and project persistence
  are available under **Tools > Layered surface-wave inversion**. Vp/Vs, density and model topology remain fixed.
- Athena scientific validation passes 80 checks without skips, including independent analytical dispersion and
  exact sample agreement with an independent public SEG-D reader. Both Windows editions pass 378 tests with eight
  optional-environment skips, frozen executable checks, manual generation and installer creation.

See [v0.10 evidence and limits](validation/v0.10.0/README.md). These additions execute on CPU.

## Version 0.9 3D arrivals and survey reconstruction

- Full-volume 3D first/reflected/head-arrival velocity tomography with fixed interfaces, per-pick uncertainty,
  conditional timing-noise ensembles, phase controls and orthogonal model/coverage/uncertainty views.
- Survey-wide binned 3D/5D Fourier POCS reconstruction with explicit observed masks, duplicate policies, aperture
  limits, memory/disk checks and fold maps. Recorded bin samples remain exact; generated traces are flagged.

See [v0.9 evidence and measured reconstruction errors](validation/v0.9.0/README.md). These operators execute on CPU.

## Version 0.8 workstation GPU and version 0.7 remote jobs

- Workstation CPU, NVIDIA CUDA or automatic selection, GPU detection and a real kernel check under
  **Processing > Workstation CPU / CUDA devices**. Projects save defaults for new 3D steps and local survey jobs.
- A separate Windows CUDA installer bundles CuPy, CUDA runtime libraries, NVRTC and headers. It needs an NVIDIA
  driver; a separate system CUDA Toolkit is unnecessary. CUDA accelerates the full-volume 3D acoustic modelling,
  RTM and FWI operators. Other processes retain their documented CPU or external-engine implementation.
- GUI SSH/Slurm submission, monitoring, cancellation and result import. Portable input/source/result bundles,
  persistent job records and checksums allow recovery after closing the GUI or losing the SSH connection.
- Explicit CUDA fails if the selected device cannot run. Automatic mode records any CPU fallback. Local jobs
  record the actual device and check their memory estimate against its free VRAM.

See [workstation validation](validation/v0.8.0/README.md) and [remote job evidence](validation/v0.7.0/README.md).
The broader expansion is **not complete**; [development status](DEVELOPMENT_STATUS.md) records the remaining scope.

## Version 0.6 expansion

- Joint interface-depth updates: flat and sloping boundaries, shared reflection/head-wave interfaces, depth bounds,
  fixed boundaries, optional velocity-layer remapping and alternating velocity updates. Accepted updates reduce
  the weighted arrival misfit. This is a grid-resolved coordinate search; it does not resolve velocity-depth ambiguity.
- A visual arrival/interface editor with phase selection, uncertainty controls and draggable boundaries; saved results
  show the starting and final interfaces.
- All five SEG-2 encodings, including packed 20-bit samples; per-trace timing validation, fractional-delay alignment,
  explicit SKEW controls and retained original headers.
- Public Sirente reflection waveforms and Koenigsee refraction picks used for independent reader/SU operator checks
  and held-out arrival prediction on Athena. These checks do not establish a unique geological interpretation.

The broader expansion is **not complete**. [Development status](DEVELOPMENT_STATUS.md) tracks remaining physics,
survey reconstruction, distributed execution and field-validation gaps. The v0.6 Windows build passes 303 tests and
the packaged executable's seismic/two-worker checks; the targeted Athena suite passes 46 tests with no skips.
See [v0.6 evidence and public land checks](validation/v0.6.0/README.md).

## Version 0.5 foundation

- Streamed SU file jobs with cancellation, timeouts, per-stage exit checks, retained logs and atomic output replacement.
- Joint 2D first-arrival, reflected and head-wave tomography, preserving every selected phase/trace pair. Reflected
  legs bend through the model. This version used fixed assigned interfaces; optional depth updates are added in v0.6.
- Full v(z,y,x) acoustic modelling, RTM and FWI with a checked discrete adjoint, disk checkpoints and line search.
  CPU and optional CuPy CUDA execution; local process workers and optional Dask shot distribution.
- SEG-Y/SU input read one shot at a time for 3D file jobs; model building and orthogonal slice inspection in the GUI.
- Project saves now retain diagnostic arrays, phase misfits, coverage and convergence histories without pickle.

Source-based GPU and distributed execution require optional dependencies in the **compute host's Python environment**.
The standard Windows installer includes the CPU engine; the CUDA edition includes the local GPU dependencies.
Actual CUDA execution is validated on Athena A100 compute nodes; Windows CI validates packaging without a GPU.
The model must fit on each worker; parallelism is across shots, not spatial domain decomposition. The 3D solver is
isotropic constant-density acoustic, with a second-order spatial stencil and sponge boundaries. Under-resolved
source wavelengths are rejected by default. These features do not establish production accuracy on every survey.

See the v0.5 chapter of `seisdesk/docs/USER_GUIDE.md` for configuration, units and current limits.
The [validation record](validation/v0.5.0/README.md) lists the measured Windows, Athena CUDA, real-SU,
analytical-propagation and marine-sample results, and the checks still outstanding.

## Version 0.4 foundation

- First-arrival tomography with coverage/misfit plots, GRM diagnostics and bounded residual-depth-moveout updates.
- 2D constant-density acoustic frequency-domain RTM and FWI, with discrete-adjoint gradient checks, source estimation,
  frequency continuation and model/convergence plots.
- Wavelet extraction, manual and automatic time ties, applied time-depth export, deviated wells and AVO synthetics.
- Angle gathers/stacks, AVO attributes/crossplots, hyperbolic Radon, tau-p deconvolution, SRME and adaptive subtraction.
- Time-variant/Gabor deconvolution, structure smoothing, Fourier panel filters, trace statistics and similarity.
- Sensors & Software DT1 GPR import with real nanosecond timing; geometry, datum, velocity and modelling utilities.
- Small-volume 3D Kirchhoff depth migration with v(z), 3D DMO, Fowler panels, PS workflows, VTI elastic/ray modelling,
  Chebyshev acoustic propagation and sliding four-component shear rotation.

These earlier methods retain their model, sampling and geometry limits. The frequency-domain FWI/RTM operators
are 2D and CPU-based; MVA uses a small-contrast vertical or straight-ray update; the layered 3D Kirchhoff operator
uses one velocity column. The separate v0.5 acoustic operators accept a full 3D model. Processing outputs, models,
picks, flows and diagnostic arrays are saved in projects. File-job directories also retain inputs and predictions;
keep these directories when moving a project, because their paths are recorded in job diagnostics.

See `seisdesk/docs/USER_GUIDE.md` (also under Help > User guide in the program).

## Install on Windows

Use `SeisDesk-0.12.0-setup.exe` for CPU processing or `SeisDesk-0.12.0-CUDA-setup.exe` for an NVIDIA GPU workstation.
Download both editions, the manual, source and examples from the
[v0.12.0 release](https://github.com/kumarDeepak-su/seisdesk/releases/tag/v0.12.0).
The [v0.12 validation record](validation/v0.12.0/README.md) records the successful Windows build and verified hashes.
Installation does not need administrator rights unless you choose an installation for all users.
To build an installer yourself on a Windows PC:

1. Install Python 3.11 (64-bit) from python.org.
2. Double-click `build_windows.bat` for CPU, or run `build_windows.bat cuda` in a terminal for CUDA.
   It creates a separate build environment for the selected edition and runs the tests (report in
   `test_report.txt`). It then builds the program with PyInstaller, checks it with `SeisDesk.exe --self-test` and
   packs it with Inno Setup, installing Inno Setup through winget when it is missing.
3. The installer is written to `dist\SeisDesk-0.12.0-setup.exe` or `dist\SeisDesk-0.12.0-CUDA-setup.exe`.
   The CUDA build additionally checks its packaged libraries, headers and kernel compiler without requiring a GPU
   on the build machine. Use **Run CUDA check** in the installed application to validate execution on your GPU.

The same build runs on GitHub Actions (`.github/workflows/windows-build.yml`). Tests and the packaged-program smoke
test must pass before an installer is uploaded. The build also renders the added controls and result windows and
produces the PDF manual. The installer and manual are attached to each successful run as artifacts.

## Run from source

```
python -m pip install -e .[test]
python -m seisdesk            # the GUI
seisdesk-cli processes        # the command line
QT_QPA_PLATFORM=offscreen python -m pytest
```

Requirements include Python 3.10+, NumPy, SciPy, segyio, PySide6, pyqtgraph, disba and Numba. The exact versions used for the installer
are in `packaging/requirements-windows.txt`.

## Layout

| Path | Contents |
|---|---|
| `seisdesk/core` | Dataset, headers, SEG-Y/SU I/O, geometry, velocities, picks, synthetic data |
| `seisdesk/processing` | One module per family of processing steps; every step registers itself with a parameter schema |
| `seisdesk/workflow` | Flows, templates and projects |
| `seisdesk/gui` | Qt front end: viewer, picking, velocity analysis, flow editor, dialogs |
| `tests` | pytest suite on synthetic data, including GUI tests (pytest-qt) |
| `packaging` | PyInstaller spec, Inno Setup script, icon generator, pinned requirements |

## Licence

SeisDesk is licensed under the GNU General Public License, version 3 or later
(`GPL-3.0-or-later`), as declared in `pyproject.toml`. The full licence text is in
[Licence.txt](Licence.txt). Third-party dependencies retain their respective licences.
Seismic Unix is installed separately and retains its upstream licence.
