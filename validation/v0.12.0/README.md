# Light desktop workspace and interactive flows

Validated application: **`651eb1d264abb41e163e320fdd3ecdaf92274b83`**. This update changes desktop controls,
navigation and flow editing. The numerical files in `seisdesk/core`, `seisdesk/processing`, `seisdesk/backends`
and `seisdesk/workflow` are unchanged from the preceding `7991be8` source snapshot. No application, GUI or
scientific tests were executed on the Mac. Windows CI performed execution and rendering.

## Implemented interactions

- Light appearance by default, optional dark appearance and remembered dock layout.
- Home actions, live project counts, recent projects, dataset filtering and Ctrl+1/2/3/4 navigation.
- Ctrl+K command search covering menus, datasets and registered processing steps, with keyboard navigation.
- Supported local file/project drops, process-library drops and drag-to-reorder flow steps.
- Step bypass, duplicate/remove actions and undo/redo for structure, parameters, names and notes.
- Run readiness, progress, completion/error/cancellation feedback and editing locks during processing.
- Collapsible viewer/velocity controls and bounded field sizes for a 1280 x 800 workspace.
- Saved flow-only projects, removal of cleared saved flows, reset edit history on project changes and correct
  clean/dirty state when reopening or simply viewing a saved project.

Undo/redo retains up to 100 flow states and does not remove or recompute previous result datasets. Ctrl+Z/Y
operates on the flow when its step list has focus; text fields retain normal text undo. Cancellation takes effect
at a process cancellation check. The toolbar's compute label reports a preference; actual execution devices
remain recorded by the individual jobs.

## Windows validation

Windows build 41780615893 succeeds for both
CPU and CUDA editions. Each passes **399 tests, eight skips, zero failures and zero errors**. The unmodified
reports are `test-results-CPU.xml` and `test-results-CUDA.xml`; `windows-build-651eb1d.json` records every step.
The eight skips require three GPU checks, two real SU checks, two independent ObsPy comparisons and one Dask
environment. Earlier Athena evidence for those environments remains in the previous milestone records.

Focused workspace run 41780609401 passes
**23 tests without skips**: 11 new workspace checks and 12 existing GUI checks. These overlap the full suite.
They cover command search, dataset selection, reordered/bypassed flow persistence, parameter undo/redo,
project save/reopen, running/cancelled editor state, theme persistence, file/process drops and compact sizing,
along with existing picking, velocity and diagnostic controls. `workspace-tests.xml` retains the report.

Both packaged executables pass seismic/SEG-Y, two-worker 3D modelling, portable source/bundle verification,
3D tomography/reconstruction, disba/CPS compilation, common-axis import and shared-interface/export checks.
`packaged-checks-651eb1d.log` retains their result lines, and `build-log-manifest.json` identifies the full log.
The CUDA package loads seven runtime libraries, finds its headers and compiles **709 bytes of PTX** with NVRTC
12.8. `cuda-runtime-651eb1d.json` retains the original report. Windows CI has no GPU/driver; this is a packaging
and compilation check. Actual acoustic CUDA execution was previously validated on Athena A100. The user's
physical Windows GPU installation/execution check remains deferred.

## Visual and documentation checks

Nine real Windows renders show light/dark Home, Viewer and Flow views, command search and compact workspaces.
Standard windows are 1500 x 920; compact views are 1280 x 800. The measured minimum main-window width is 981.
`workspace-layout.json` records dimensions; the renderer rejects missing Latin glyphs and oversized windows.
`gui-images.json` records the full-build image hashes. Eight images match the focused-run renders exactly;
the command-search view was separately inspected after its cursor/selection render differed.

The full manual has **235 pages**, identifies version 0.12.0, describes the actual Windows installer choices,
and includes the new controls and light-theme screenshots. The cover and changed pages **8-11** were rendered
and inspected for legible text, complete figures and captions. `manual-review.json` records the review and hash.
The manual's scientific chapters retain their stated assumptions and limitations.

## Downloads and local package

The [v0.12.0 release](https://github.com/kumarDeepak-su/seisdesk/releases/tag/v0.12.0) provides the CPU/CUDA installers,
manual, source, examples and checksums. The original validated build artifacts are also linked below.
All eight published files match their verified local SHA-256 digests and byte lengths. Publication was completed
while the repository was private. `github-publication.json` records the release, successful GitHub-side CUDA transfer, and removal of
22 superseded installer/manual Actions downloads from v0.8-v0.11. Earlier milestone download links are historical;
their scientific reports and figures remain available, and the current v0.12 build artifacts are retained.

- CPU installer:
  `SeisDesk-0.12.0-setup.exe`.
- CUDA installer:
  `SeisDesk-0.12.0-CUDA-setup.exe`.
- Full manual.

The Desktop handoff is `~/Desktop/SeisDesk-0.12.0-Windows`. It contains both original installers,
the updated manual, instructions, nine new GUI previews, earlier scientific images/results, the 17 example
flows, public Sirente files with attribution/geometry caveats, and a source archive with referenced manual images.
`PACKAGE_MANIFEST.json` and `SHA256SUMS.txt` inside that folder record identity and integrity. Both downloaded
installer archives match GitHub's SHA-256 digest and pass ZIP CRC checks; `installer-download-CPU.json` and
`installer-download-CUDA.json` record the checks and executable hashes. `source-manifest.json` identifies all
142 application Python files. The source-archive and final Desktop-copy checks are in `windows-handoff.json`.

This is an interface update, not completion of every seismic processing method. The limitations in
`DEVELOPMENT_STATUS.md` remain applicable. Publication does not expand the scientific validation claims.
