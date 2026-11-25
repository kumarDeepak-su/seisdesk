# Remote job validation

v0.7 adds a desktop SSH/Slurm launcher for the existing 3D acoustic modelling, RTM and FWI engine. It bundles
inputs and the desktop build's Python source, persists job records before submission, reports scheduler/application
states separately, cancels selected jobs, and downloads hash-verified results with local path mapping. Results retain
original inputs, source, staged shots, predictions, models and diagnostics. A saved job can be recovered after closing
the GUI or losing its SSH connection. The launcher currently requests one node and at most one GPU.

## Evidence

- Windows build 41603445024, source `c73ee6d`:
  **313 passed, 7 skipped, zero failures/errors**. GUI worker callbacks, persisted job recovery, cancellation routing,
  archive integrity, failure handling and portable subprocess execution pass. The frozen executable's workflow,
  two-process 3D modelling and packaged compute-source/bundle self-tests pass. Installer/manual generation passes.
  `windows-results-c73ee6d.xml` is the unmodified test report. The seven optional-dependency skips are unchanged
  from v0.6; this build does not constitute a fresh run of those optional engines.
- Athena CPU job **3489747**, source `75b84cc`: **10 passed, zero skips**, covering portable-bundle execution,
  missing/tampered/unsafe files, actual compute subprocesses, scheduler semantics, failed-job output retention and
  existing streamed/local-worker volume jobs. The later Windows run covers the Windows archive-name and GUI fixes.
- Actual SSH client lifecycle: CPU **3489878** completed `0:0`; deliberately invalid velocity bounds in **3489881**
  failed `1:0` and returned logs/partial results; held validation job **3489886** was cancelled through the client;
  CUDA **3489905** completed `0:0`. Success and failure archives were downloaded, verified and extracted, and
  predicted/model/input paths were checked. CPU/failure/cancel use source `c73ee6d`; the CUDA job uses `79849ee`
  (a stale queue-reason display fix and documentation change). `lifecycle-summary.json` records hashes and outcomes.
- The initial CUDA profile omitted the CUDA module: job **3489888** failed with a missing `libnvrtc.so.12`.
  Loading **Python/3.10.4 and CUDA/12.4.0** corrected the environment; the subsequent CUDA job passed.
- CPU postprocessing job **3489909** compared both predicted shots and rendered `athena-cpu-cuda-predictions.png`.
  All **510 samples** agree exactly for this fixture; see `cpu-cuda-comparison.json`. The fixture is a tiny,
  deliberately coarse, constant-1900-m/s model used to test execution and portability. It is not a field result,
  dispersion benchmark or validation of additional physics. Independent acoustic physics checks remain in v0.5.

The real SSH lifecycle was driven by the standard-library client on the Mac; all numerical computation and plotting
ran on Slurm compute nodes. Windows GUI tests use a mocked connection, and frozen Windows packaging checks the
portable bundle. No Athena credentials were copied to CI. Actual Windows-to-Athena SSH authentication/configuration
still depends on the user's Windows OpenSSH setup.

## Reproduction

Generate the fixture on a compute node with `tools/prepare_remote_fixture.py`. Use a host profile with a working
Python environment, Slurm account and partition. `tools/validate_remote_lifecycle.py` is an I/O-only controller that
runs CPU, expected-failure, held-job cancellation and CUDA cases; it supports rerunning individual cases with
`--cases`. CUDA needs the matching toolkit module in the host profile. Use `tools/render_remote_validation.py`
inside a compute allocation to compare and plot the completed CPU/CUDA jobs.

This milestone does not implement survey-wide 3D/5D interpolation, general 3D SRME, advanced 3D wave physics,
multi-node/GPU scaling or spatial decomposition. Those and the other remaining items are tracked in
`DEVELOPMENT_STATUS.md`; the broader requested expansion remains incomplete.
