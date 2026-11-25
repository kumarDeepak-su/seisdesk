"""Start the SeisDesk window (``--self-test`` runs a short end-to-end check and exits, for installers and CI)."""

from __future__ import annotations

import os
import sys
import tempfile
import traceback


def self_test(win) -> int:
    """Seismic workflow, viewer, velocity analysis and frozen 3D process workers.
    Writes the outcome to %TEMP%/seisdesk_selftest.txt and returns the exit code."""
    from ..core import synthetic
    from ..core.dataset import SeismicData
    from ..core.segy_io import read_segy, write_segy
    from ..core.velocity import VelocityPicks
    from ..core.volume import VelocityVolume
    from ..processing import Context, create
    import numpy as np

    from . import common
    common.HEADLESS = True
    lines = []
    code = 0
    try:
        d = synthetic.marine_line(nshots=6, ngroups=24, nt=376)
        name = win.add_dataset(d, "self test")
        ctx = Context(velocity=VelocityPicks.constant(2000.0))
        st = create("stack")(create("nmo")(d.sort(["cdp", "offset"]), ctx), ctx)
        mig = create("stolt", velocity=2000.0)(st, ctx)
        path = os.path.join(tempfile.gettempdir(), "seisdesk_selftest.sgy")
        write_segy(mig, path)
        back = read_segy(path)
        if back.ntr != mig.ntr or back.nt != mig.nt:
            raise RuntimeError("SEG-Y round trip changed the size")
        win.add_dataset(mig, "self test migrated")
        win.open_velan(name)
        if win.velan.spectrum is None:
            raise RuntimeError("velocity analysis produced no semblance")
        with tempfile.TemporaryDirectory(prefix="seisdesk-frozen-3d-") as folder:
            model_path = os.path.join(folder, "model.npz")
            VelocityVolume(np.full((6, 6, 6), 1500., dtype=np.float32), (10, 10, 10)).save(model_path)
            shots = SeismicData(np.zeros((4, 40), dtype=np.float32), .001,
                                dict(sx=[10, 10, 30, 30], sy=20, selev=-20, sdepth=0,
                                     gx=[20, 40, 20, 40], gy=20, gelev=-20, fldr=[1, 1, 2, 2]))
            job_context = Context(work_dir=folder, log=lines.append)
            modelled = create("acoustic_model_3d", model=model_path, workers=2, vmin=1000, vmax=2000,
                              padding=3, block=7, frequency=45, source_delay=.015, memory_gb=.1,
                              allow_coarse_grid=True)(shots, job_context)
            if not np.isfinite(modelled.traces).all() or not np.any(modelled.traces):
                raise RuntimeError("frozen 3D process workers did not produce finite nonzero shots")
            if len(job_context.results["acoustic_model_3d"]["shot_results"]) != 2:
                raise RuntimeError("frozen 3D process workers did not complete both shots")
            lines.append("Frozen CPU 3D modelling passed with two process workers (small numerical fixture).")
            from pathlib import Path
            from ..backends.job_bundle import build_volume_bundle, extract_bundle
            from ..core.segy_io import write_su
            source = Path(folder)/"remote-input.su"
            write_su(shots, str(source))
            bundle = build_volume_bundle(dict(input_file=str(source), model=model_path, mode="model"),
                                         Path(folder)/"remote-inputs.zip")
            extracted = Path(folder)/"portable"
            extract_bundle(bundle["path"], extracted, expected_sha256=bundle["sha256"], kind="volume-input")
            if not (extracted/"code/seisdesk/backends/remote_agent.py").is_file():
                raise RuntimeError("packaged compute source is missing the remote runner")
            lines.append("Frozen portable job bundle and compute source verification passed.")
            from ..core.tomography3d import NodeGrid3D, arrivals3d, invert3d
            from ..core.reconstruction import reconstruct_survey
            grid = NodeGrid3D((3, 3, 3), (10, 10, 10), radius=1)
            source_nodes, receiver_nodes = np.array([0, 2, 6, 8]), np.array([26, 24, 20, 18])
            phases, branches = ["P"]*4, {"P": {"kind": "first", "uncertainty_ms": .2}}
            observed = arrivals3d(grid, np.full(grid.shape, 2000.), source_nodes, receiver_nodes, phases, branches)[0]
            fitted = invert3d(grid, np.full(grid.shape, 1800.), source_nodes, receiver_nodes, observed,
                              phases, branches, iterations=2)
            if fitted["rms"][-1] >= fitted["rms"][0]:
                raise RuntimeError("frozen 3D arrival inversion did not reduce travel-time error")
            reconstructed = reconstruct_survey(np.ones((15, 16), np.float32), np.arange(15), (4, 4),
                                                iterations=8, work_dir=folder)
            if not np.allclose(reconstructed, 1., atol=1e-5):
                raise RuntimeError("frozen survey FFT reconstruction failed")
            lines.append("Frozen 3D arrival inversion and disk-backed survey FFT reconstruction passed.")
            from ..core.surface_inversion import predict_dispersion
            from ..core.sampling import common_samples
            predicted = predict_dispersion([[0, 519.615242, 300, 1900]],
                                           [dict(frequency_hz=10, velocity_ms=275)])
            if not abs(predicted[0]-275.8205) < .03:
                raise RuntimeError("frozen layered dispersion kernel failed its half-space check")
            aligned, _, _, _ = common_samples([np.arange(4), np.arange(3)], [.002]*2, [0, .002], .002, "union")
            if aligned.shape != (2,4) or not np.array_equal(aligned[1], [0,0,1,2]):
                raise RuntimeError("frozen common-axis import failed")
            lines.append("Frozen disba/CPS dispersion compilation and common-axis sample import passed.")
            from ..core.interfaces3d import invert_interfaces3d, save_interfaces3d
            interface_grid = NodeGrid3D((6, 3, 3), (10, 10, 10), radius=1)
            stations = interface_grid.nodes_for([[0,0,0], [0,0,20], [0,20,0], [0,20,20]])[0]
            interfaces = invert_interfaces3d(interface_grid, np.full(interface_grid.shape, 2000.), stations, stations,
                np.full(4, .02), ["R"]*4, {"R":dict(kind="reflection", depth=10)}, iterations=2,
                update_velocity=False, warp_velocity=False, max_shift=10, max_step=10)
            if not np.allclose(interfaces["sampled_interfaces"]["R"], 20) or interfaces["rms"][-1] > 1e-10:
                raise RuntimeError("frozen 3D interface inversion failed its independent reflection-depth check")
            save_interfaces3d(interfaces, Path(folder)/"interfaces.npz")
            lines.append("Frozen 3D shared-interface inversion and portable boundary export passed.")
        lines.append(f"SeisDesk self-test passed: {d.ntr} traces processed, SEG-Y written to {path}")
    except Exception:                                      # noqa: BLE001
        lines.append("SeisDesk self-test FAILED")
        lines.append(traceback.format_exc())
        code = 1
    win.dirty = False
    with open(os.path.join(tempfile.gettempdir(), "seisdesk_selftest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    import pyqtgraph as pg
    from PySide6 import QtWidgets

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False, background="w", foreground="k")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("SeisDesk")
    app.setOrganizationName("SeisDesk")
    app.setStyle("Fusion")
    from .main_window import MainWindow

    win = MainWindow()
    win.show()
    if "--self-test" in argv:
        return self_test(win)
    for path in argv[1:]:
        if path.startswith("--"):
            continue
        if os.path.isfile(path):
            win.read_file({"path": path, "little": path.lower().endswith(".su"), "dt": 0.0, "name": None})
        elif os.path.isdir(path) and os.path.exists(os.path.join(path, "project.json")):
            from ..workflow.project import Project
            win._set_project(Project.load(path))
    return app.exec()
