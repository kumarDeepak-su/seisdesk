"""GUI smoke tests (pytest-qt; run with QT_QPA_PLATFORM=offscreen on a machine without a display)."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from seisdesk.core import synthetic  # noqa: E402
from seisdesk.processing import REGISTRY  # noqa: E402
from seisdesk.workflow.flow import FlowStep  # noqa: E402
from seisdesk.workflow.templates import templates  # noqa: E402

CLICK = {"left": True, "right": False, "shift": False, "ctrl": False, "double": False}


@pytest.fixture
def win(qtbot, monkeypatch, tmp_path):
    import pyqtgraph as pg
    from seisdesk.gui import common, dialogs, flow_editor, main_window, picking, velan, viewer

    def fail(_parent, title, error):
        raise AssertionError(f"error dialog '{title}': {error}")

    for mod in (common, dialogs, flow_editor, main_window, picking, velan, viewer):
        if hasattr(mod, "show_error"):
            monkeypatch.setattr(mod, "show_error", fail)
    monkeypatch.setattr(main_window, "default_work_dir", lambda: str(tmp_path))
    monkeypatch.setattr(main_window.MainWindow, "_ask_save_changes", lambda self: True)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False, background="w", foreground="k")
    w = main_window.MainWindow()
    qtbot.addWidget(w)
    w.show()
    return w


@pytest.fixture(scope="module")
def small_marine():
    return synthetic.marine_line(nshots=12, ngroups=24, nt=376, dt=0.004)


def test_viewer_navigation_and_modes(win, small_marine):
    name = win.add_dataset(small_marine.copy())
    v = win.viewer
    assert v.dataset_name == name and v.gather_key() == "fldr"
    assert v.idx.size == 24
    v.step(+1)
    assert v.current_gather_value() == v.gather_values[1]
    assert v.set_gather_key("cdp")
    assert v.idx.size >= 1
    for i in range(v.mode_combo.count()):
        v.mode_combo.setCurrentIndex(i)
    for i in range(v.gain_combo.count()):
        v.gain_combo.setCurrentIndex(i)
    v.key_combo.setCurrentIndex(0)                          # all traces
    assert v.idx.size == small_marine.ntr
    v._hover(float(v.xpos[3]), 0.5)
    assert "trace" in v.readout.text()
    d = v.displayed_data()
    from seisdesk.gui.dialogs import HeaderTableDialog, SpectrumDialog
    SpectrumDialog(d, win).close()
    HeaderTableDialog(d, win).close()


def test_picking_phases_and_fits(win, monkeypatch):
    land = synthetic.land_refraction_line(nshots=3, nrec=96, nt=601)
    win.add_dataset(land, "land")
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("fb", True))
    win.open_picking()
    v, panel = win.viewer, win.viewer.picking
    panel.new_set()
    ps = win.project.context.picks["fb"]
    assert ps.gather_key == "fldr" and ps.x_key == "offset"
    syn = land.meta["synthetic"]
    v1, (v2, t1) = syn["v1"], syn["refractors"][0]
    off = land.header("offset")[v.idx]
    near = [j for j in range(v.idx.size) if 5 < abs(off[j]) < 50]
    mid = [j for j in range(v.idx.size) if 80 < abs(off[j]) < 220]
    for j in near:
        panel.click(v, j, dict(CLICK, x=float(v.xpos[j]), y=abs(off[j]) / v1))
    panel.new_phase()                                          # the break
    for j in mid:
        panel.click(v, j, dict(CLICK, x=float(v.xpos[j]), y=abs(off[j]) / v2 + t1))
    assert len(ps.phase_order) == 2 and ps.count() == len(near) + len(mid)
    assert len(panel.fits) == 2
    assert abs(panel.fits[0]["velocity"] - v1) / v1 < 1e-3 and abs(panel.fits[1]["velocity"] - v2) / v2 < 1e-3
    assert len(panel.layers) == 1 and panel.layers[0]["thickness"] > 0
    assert v.overlay_items
    j = near[0]
    panel.click(v, j, dict(CLICK, left=False, right=True, x=float(v.xpos[j]), y=abs(off[j]) / v1))
    assert ps.count() == len(near) + len(mid) - 1
    panel.snap_combo.setCurrentText("peak")
    t = panel.snap(land, int(v.idx[mid[0]]), abs(off[mid[0]]) / v2 + t1 + 0.004)
    assert -0.002 < t - (abs(off[mid[0]]) / v2 + t1) < 0.02          # first peak of the causal wavelet
    panel.new_phase()
    panel.auto_pick(False)
    panel.auto_pick(True)
    assert ps.count() > land.ntr / 2
    panel.copy_tables()


def test_velocity_analysis_picking(win, small_marine):
    name = win.add_dataset(small_marine.copy())
    win.open_velan(name)
    va = win.velan
    assert va.spectrum is not None and va.spectrum.shape[0] == va.nv.value()
    va._click(dict(CLICK, x=2100.0, y=1.1))
    va._click(dict(CLICK, x=1700.0, y=0.6))
    assert len(win.project.context.velocity) == 2
    va._click(dict(CLICK, left=False, right=True, x=1700.0, y=0.6))
    assert len(win.project.context.velocity) == 1
    va.auto()
    assert len(win.project.context.velocity) >= 1
    va.mode_combo.setCurrentIndex(1)                            # eta scan
    assert va.axis.min() < 0.0 < va.axis.max()
    va._click(dict(CLICK, x=0.1, y=1.1))
    va.mode_combo.setCurrentIndex(0)
    va.step(+1)
    from seisdesk.gui.dialogs import VelocityFieldDialog
    d = win.project.datasets[name]
    VelocityFieldDialog(win.project.context.velocity, d.header("cdp"), d.times, win).close()


def test_param_forms_for_every_process(win):
    form = win.flow_editor.form
    for name in REGISTRY:
        form.set_step(FlowStep(name, {}))
        assert form.form.rowCount() == len(REGISTRY[name].params)
    step = FlowStep("agc", {})
    form.set_step(step)
    edit = form.form.itemAt(0, QtWidgets.QFormLayout.ItemRole.FieldRole).widget()
    edit.setText("0.75")
    edit.editingFinished.emit()
    assert step.params["window"] == 0.75
    edit.setText("-5")
    edit.editingFinished.emit()
    assert step.params["window"] == 0.75                        # rejected, the old value stays


def test_flow_editor_runs_a_template(win, qtbot, small_marine):
    name = win.add_dataset(small_marine.copy())
    ed = win.flow_editor
    ed.set_flow(templates()["Marine 2D: shots to migrated stack"])
    ed.refresh_inputs(name)
    results = []
    ed.sigResult.connect(results.append)
    ed.run(None)
    qtbot.waitUntil(lambda: ed._worker is None, timeout=300000)
    assert results
    result = win.project.datasets[results[-1]]
    assert result.ntr == np.unique(small_marine.header("cdp")).size
    assert np.all(np.isfinite(result.traces))
    assert win.project.context.velocity is not None and len(win.project.context.velocity) > 0
    win.show_dataset(name)                                      # the parameter test runs on a shot gather
    ed.preview_check.setChecked(True)
    ed.run(2)
    qtbot.waitUntil(lambda: ed._worker is None, timeout=300000)
    assert len(results) == 2


def test_project_save_and_open(win, small_marine, tmp_path, monkeypatch):
    win.add_dataset(small_marine.copy(), "a")
    folder = tmp_path / "p"
    folder.mkdir()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    assert win.save_project(ask=True)
    win.open_project()
    assert list(win.project.datasets) == ["a"]


def test_display_actually_paints(win, small_marine):
    """Density images, wiggles and overlay curves must put pixels on the screen (PySide6 6.9 painted nothing)."""
    import pyqtgraph as pg

    from seisdesk.gui.plot import SeismicPlot
    p = SeismicPlot()
    p.resize(500, 400)
    d = small_marine.gather("fldr", small_marine.gather_values("fldr")[0])
    p.show_traces(d.traces, d.t0, d.dt, d.header("offset"), "Density", "Gray (black = positive)",
                  float(np.percentile(np.abs(d.traces), 98)))
    p.reset_view()
    p.addItem(pg.PlotDataItem([-1200.0, -200.0], [0.3, 1.2], pen=pg.mkPen((255, 0, 0), width=5)), ignoreBounds=True)
    p.show()
    for _ in range(5):
        QtWidgets.QApplication.processEvents()
    img = p.grab().toImage()
    gray = red = 0
    for yy in range(0, img.height(), 3):
        for xx in range(0, img.width(), 3):
            c = img.pixelColor(xx, yy)
            if c.red() > 150 and c.green() < 90 and c.blue() < 90:
                red += 1
            elif max(c.red(), c.green(), c.blue()) < 235 and abs(c.red() - c.blue()) < 20:
                gray += 1
    p.close()
    assert gray > 2000, f"density image not painted ({gray} gray pixels)"
    assert red > 20, f"overlay curve not painted ({red} red pixels)"


def test_spectral_analysis_of_zoomed_window(win, small_marine):
    from seisdesk.gui.dialogs import SpectrumDialog
    win.add_dataset(small_marine.copy())
    v = win.viewer
    full = v.visible_data()
    v.plot.getPlotItem().vb.setRange(xRange=(float(v.xpos[2]), float(v.xpos[10])), yRange=(0.4, 1.0), padding=0)
    part = v.visible_data()
    assert part.ntr < full.ntr and part.nt < full.nt
    dlg = SpectrumDialog(part, win)
    r = dlg.result
    assert 5.0 < r["peak"] < 60.0 and r["band_20db"][0] <= r["peak"] <= r["band_20db"][1]
    assert dlg.tabs.count() == 3
    dlg.close()


def test_save_display_image_and_reference(win, small_marine, tmp_path, monkeypatch):
    win.add_dataset(small_marine.copy())
    target = tmp_path / "shot.png"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "PNG image (*.png)"))
    win.save_image()
    assert target.exists() and target.stat().st_size > 1000
    from seisdesk.processing.reference import process_reference_html
    assert "Kirchhoff" in process_reference_html()


def test_reflection_phase_picking_with_hyperbolas(win, monkeypatch):
    marine = synthetic.marine_line(nshots=4, ngroups=48, nt=751, dt=0.004)
    win.add_dataset(marine, "marine")
    win.open_picking()
    v, panel = win.viewer, win.viewer.picking
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("reflections", True))
    panel.new_set()
    panel.fit_combo.setCurrentIndex(1)                         # hyperbolas: reflection
    ps = win.project.context.picks["reflections"]
    off = marine.header("offset")[v.idx]
    names = []
    for k, (t0, vr) in enumerate(((0.6, 1700.0), (1.1, 2100.0))):
        if k:
            panel.new_phase()
        names.append(panel.current_phase())
        for j in range(0, v.idx.size, 4):
            panel.click(v, j, dict(CLICK, x=float(v.xpos[j]), y=float(np.sqrt(t0 ** 2 + (off[j] / vr) ** 2))))
    assert len(panel.fits) == 2
    assert abs(panel.fits[0]["velocity"] - 1700.0) < 5 and abs(panel.fits[1]["t0"] - 1.1) < 0.002
    assert len(panel.layers) == 2 and panel.layers[1]["velocity"] > panel.layers[0]["velocity"]
    target = []
    monkeypatch.setattr("seisdesk.gui.picking.write_pick_table", lambda rows, path: target.append(rows))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: ("x.csv", ""))
    panel.export_table()
    assert target and len(target[0]) == ps.count() and "sx" in target[0][0]


def test_mixed_event_types_keep_their_model(win, monkeypatch):
    marine = synthetic.marine_line(nshots=2, ngroups=24, nt=501, dt=0.004)
    win.add_dataset(marine, "mixed")
    win.open_picking()
    v, panel = win.viewer, win.viewer.picking
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("mixed", True))
    panel.new_set()
    ps = panel.current_set()
    direct = panel.current_phase()
    off = marine.header("offset")[v.idx]
    for j in range(0, len(v.idx), 3):
        panel.click(v, j, dict(CLICK, y=abs(off[j]) / 1500))
    panel.fit_combo.setCurrentIndex(1)
    assert ps.phase_type(direct) == "direct"  # changing the analysis never relabels an existing event
    panel.new_phase()
    reflector = panel.current_phase()
    for j in range(0, len(v.idx), 3):
        panel.click(v, j, dict(CLICK, y=float(np.hypot(0.6, off[j] / 2000))))
    assert ps.phase_type(reflector) == "reflection"
    assert [f["phase"] for f in panel.fits] == [reflector]
    panel.fit_combo.setCurrentIndex(0)
    assert [f["phase"] for f in panel.fits] == [direct]
    assert ps.phase_type(reflector) == "reflection" and not panel.layers


def test_survey_qc_and_refraction_statics_windows(win, small_marine):
    from seisdesk.gui.dialogs import RefractionStaticsDialog, SurveyQCDialog
    dlg = SurveyQCDialog(small_marine, win)
    assert dlg.tabs.count() == 3 and dlg.result["shots"] == 12
    dlg.close()
    res = {"x": [0.0, 10.0, 20.0], "elevation": [100.0, 101.0, 102.0], "delay_ms": [10.0, 11.0, 12.0],
           "thickness": [8.0, 9.0, 10.0], "base_elevation": [92.0, 92.0, 92.0], "static_ms": [-20.0, -21.0, -22.0],
           "picks": [5, 6, 7], "v1": 600.0, "v2": 2200.0, "vr": 2200.0, "datum": 80.0, "rms_ms": 0.3}
    r = RefractionStaticsDialog(res, win)
    assert r.table_text(True).splitlines()[1] == "0.00 -20.000"
    assert r.table_text(False).splitlines()[0].startswith("x,elevation,delay_ms")
    r.close()
