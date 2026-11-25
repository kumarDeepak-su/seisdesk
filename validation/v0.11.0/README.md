# Bounded joint 3D interface and velocity inversion

Numerical implementation: `a3f3693`. Validated Windows application: `5ba6dd4`. The later application changes correct the
GUI uncertainty colour scale and colour-bar labels when switching fields. Scientific figure changes use shared
scales, contact markers and a wider title margin. The numerical solver is unchanged by these corrections.

The existing 3D multi-arrival process now supports uniform or bilinear control-grid depth shifts, shared boundaries,
fixed boundaries, original-shift and absolute-depth bounds, ordered surfaces and optional velocity remapping/updates.
Every accepted step reduces the weighted arrival misfit. The GUI includes controls, depth/change/contact/uncertainty
maps, orthogonal boundary sections and reusable surface/diagnostic exports. Projects preserve the same arrays.

This is CPU coordinate search with re-traced graph arrivals, not a continuous shape-gradient method. Ray-grid
resolution, the control grid and limited illumination constrain the result. It supports ordered single-valued surfaces,
with one assigned reflection or head-wave boundary per phase. Overturning interfaces, converted/multiple-leg paths,
global structural uncertainty and heterogeneous 3D field validation remain outstanding.

## Scientific checks on Athena

Slurm **3492660**, node `t0005`, completed with exit `0:0` in **3 minutes 17 seconds**. It passes **18 tests without
skips**, then completes the independent benchmark and 20 conditional noise refits. The earlier job 3492648 passes
the same 18-check set before the larger example was added. No numerical tests or GUI were run on the Mac.

The checks cover two reflector depths from independent mirror-source times, a shared reflection/head-wave boundary
against analytical two-layer times, layer remapping, lateral control-grid deformation, joint velocity updates, fixed
boundaries, monotonically decreasing weighted data error, original uncertainty bounds, contact masks, malformed/shared/
crossing boundaries, cancellation, portable exports, retained multi-event trace identities and project persistence.
Existing 3D tomography and 2D interface tests are included in the 18-check total.

The independent noisy example uses **18 zero-offset plane-reflection times**, a known fixed 2000 m/s velocity,
0.4 ms injected noise, 2 ms stated pick errors and a 5 m depth grid. It is a deliberately small synthetic example:

- Arrival RMS: **4.11482 → 0.94747 ms**.
- Dipping interface: **2.09877 m** mean absolute depth error over the entire grid. One of its four controls reaches
  the original shift bound. The fitted surface is not a unique recovery of the analytical plane.
- Deeper flat interface: its sampled depth recovers 55 m exactly in this example.
- Each interface has only **9 of 81 nodes** with final boundary contacts. Surfaces elsewhere are extrapolations.
- **20/20** timing-noise refits finish with the original bounds. The maximum covered-node depth standard deviation
  is **2.56495 m** for the dipping interface and **0 m** for the deeper interface. Zero spread does not establish
  exact depth knowledge; the noise refits cannot resolve shifts smaller than the effective ray/control grid.

The separate dipping-plane forward check uses analytical mirror-source times with nonzero offsets. Maximum relative
timing error decreases from **3.07693%** on a 10 m grid to **1.83963%** on a 5 m grid, using stencil radius 3.
These residual errors demonstrate the need for refinement and model-error assessment. The pick-noise ensemble does
not include ray discretisation, phase mistakes, alternate topologies or total velocity/depth ambiguity.

Unmodified metrics and test output are `report.json`, `v11-scientific-results.xml` and `v11-3492660.log`.
`source-manifest.json` records application source identity. The fitted arrays remain on Athena under
`$HOME/seisdesk-validation-20261102/v11-benchmark/`, in `interfaces.npz` and
`project-results.npz`. `tools/athena_v11_validate.sh` reproduces the scientific run in a compute allocation.
Exact local copies of both array bundles and the metric report are in `dist/result-data/v0.11.0/`.
`result-data-manifest.json` records their hashes and successful NPZ archive integrity checks.

## Windows and image validation

Final Windows build 41742119084, application
source **`5ba6dd4`**, succeeds for both CPU and CUDA editions. Each passes **388 tests, eight skips, zero failures
and zero errors**. `windows-cpu-results-5ba6dd4.xml` and `windows-cuda-results-5ba6dd4.xml` retain the unmodified
reports; `windows-build-5ba6dd4.json` records every job and step. Skips require three GPU, two SU, two ObsPy and one
Dask environment. Earlier Athena evidence for these optional environments remains in the preceding milestones.

Both frozen executables recover an analytical reflector depth and export its portable boundary bundle. They also
pass the previous seismic/SEG-Y, two-process 3D modelling, source/bundle verification, tomography/reconstruction,
disba/CPS compilation and common-axis import checks. `packaged-checks-5ba6dd4.log` retains the raw result lines;
`build-log-manifest.json` records the full local logs and hashes.

The CUDA package loads seven bundled runtime DLLs, finds headers and compiles **709 bytes of PTX** using NVRTC 12.8.
`cuda-runtime-5ba6dd4.json` retains the original report. Windows CI has no GPU/driver: this validates packaging and
compilation. Existing acoustic CUDA execution was tested on Athena A100 in v0.8. The new interface search runs on CPU.

All four new Windows GUI previews were inspected, including the corrected uncertainty colour-bar label after field
switching. Pages **141–144** of the **232-page** final CPU manual were rendered and inspected: text, controls, figures
and captions are readable and unclipped. The matching field-switch behaviour is covered by a GUI regression check.
The earlier `windows-cpu-results-89abd1d.xml` is retained for traceability; its source is superseded by `5ba6dd4`.

Five final images are in `dist/result-images/v0.11.0/`: one Athena scientific figure and four Windows control/result
previews. Figures distinguish sampled depths from analytical truth, show boundary contacts and keep standard-deviation
scales nonnegative. The Windows preview uses three diagnostic refits; the scientific benchmark uses 20.
The final scientific plot was rendered from the saved arrays in Athena job **3492711**, node `t0021`, exit `0:0`,
two seconds. This render changes scales, contact labels and title margins only; it does not rerun or alter numerical
results. The initial render and the intermediate render from job 3492694 are retained under
`dist/result-images/v0.11.0/earlier/`.

`image-manifest.json` records source jobs, original dimensions and SHA-256 hashes for the saved plots and final
Windows previews. The complete local collection, including earlier variants, is rebuilt at
`dist/test-image-gallery/index.html` and `dist/seisdesk-test-images.zip`. Every image is copied without changing pixels.
The collection has **126 distinct saved images across 109 views**, including 17 earlier/alternate renders.
`gallery-summary.json` records the gallery/archive hashes, counts and static integrity checks.
The gallery's files, hashes and ZIP integrity are checked; browser interaction was not revalidated because local
browser access was blocked. This does not affect the inspected Windows application previews.

## Draft-build downloads

- CPU installer,
  `SeisDesk-0.11.0-setup.exe`.
- CUDA installer,
  `SeisDesk-0.11.0-CUDA-setup.exe`, approximately 1.21 GB archive. It also supports CPU execution.
- User manual.

These are private draft-build artifacts; no release or merge has been performed. Final previews, reports, build logs,
the original CPU manual and both installers are saved under `dist/v0.11.0-5ba6dd4/`.
`local-downloads.json` lists verified local installer/manual files; `windows-artifacts-5ba6dd4.json` records all
23 downloadable build artifacts and their archive sizes.

The Windows handoff is copied to `~/Desktop/SeisDesk-0.11.0-Windows/`, with a matching transfer ZIP and SHA-256
file list. It includes both unmodified installers, a five-page installation/quick-start guide, the full manual with
its installation text on PDF page 8 corrected, the source snapshot identified in `PACKAGE_MANIFEST.json`, 17 built-in flow templates, 126 saved
images, diagnostic arrays and 26 public Sirente SEG-2 records with attribution and the known geometry caveat.
All other full-manual pages retain the original extracted text; the corrected page and all quick-start pages were
rendered and inspected. The source archive's 140 application Python hashes match the validated snapshot.
`windows-handoff.json` records the exact locations, archive checksum and file verification results. The downloaded
CUDA artifact matches GitHub's SHA-256 digest and passes its ZIP integrity check. Documentation changes after
`5ba6dd4` do not change the application code.
The selected-image preview uses local browser storage. The gallery scripts are regenerated together with their
HTML pages, and the Windows workflow accepts development branches without requiring a particular name prefix.

The graph-arrival foundation follows the shortest-path approximation described in
[Moser's Utrecht thesis](https://research-portal.uu.nl/en/publications/the-shortest-path-method-for-seismic-ray-tracing-in-complicated-m/).
The bounded interface coordinate search is a SeisDesk implementation. The full expansion remains incomplete;
`DEVELOPMENT_STATUS.md` tracks the remaining work. The user's physical Windows GPU test remains deferred.
