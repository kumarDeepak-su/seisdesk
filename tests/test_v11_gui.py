import json
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")
from PySide6 import QtCore, QtWidgets
from seisdesk.gui.tomography3d_dialog import Branches3DDialog, Tomography3DResultsDialog
from seisdesk.gui.interfaces3d_dialog import Interfaces3DResultsDialog
from seisdesk.gui.flow_editor import ParamForm
from seisdesk.workflow.flow import FlowStep
from seisdesk.workflow.project import Project


def test_shared_boundary_controls_and_flow_persistence(qtbot, monkeypatch):
    from seisdesk.core.picks import PickSet
    picks = PickSet(name="arrivals")
    picks.new_phase("P", "direct"); picks.new_phase("R", "reflection"); picks.new_phase("H", "refraction")
    definitions = {"P":dict(kind="first"), "R":dict(kind="reflection", depth=40), "H":dict(kind="head", depth=40)}
    project = Project(); project.context.picks["arrivals"] = picks
    step = FlowStep("joint_tomography_3d", dict(pickset="arrivals", branches=json.dumps(definitions), invert_interfaces=True))
    form = ParamForm(project); qtbot.addWidget(form); form.set_step(step)
    def accept(dialog):
        dialog.tabs.setCurrentIndex(1)
        assert not dialog.interfaces.cellWidget(0, 1).isEnabled()
        for row in (1, 2):
            dialog.interfaces.cellWidget(row, 1).setText("shared")
            dialog.interfaces.cellWidget(row, 3).setCurrentIndex(1)
            dialog.interfaces.cellWidget(row, 4).setValue(20)
            dialog.interfaces.cellWidget(row, 5).setValue(60)
        return QtWidgets.QDialog.DialogCode.Accepted
    monkeypatch.setattr(Branches3DDialog, "exec", accept)
    button = next(b for b in form.findChildren(QtWidgets.QPushButton) if b.text().startswith("Configure arrivals"))
    qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    saved = json.loads(step.params["branches"])
    assert saved["R"]["interface"] == saved["H"]["interface"] == "shared"
    assert saved["R"]["control_shape"] == [2, 2] and saved["H"]["depth_bounds"] == [20, 60]
    assert saved["H"]["invert_interface"] and step.build()["invert_interfaces"]


def ui_result():
    yy, xx = np.indices((5, 5)); surface = 15+yy+xx
    contacts = np.zeros((5, 5), int); contacts[1:4, 1:4] = 2
    r = dict(velocity=np.full((8, 5, 5), 2000), spacing=np.array([5,10,10]), origin=np.zeros(3), reference_elevation=0,
             axes=(np.arange(8)*5, np.arange(5)*10, np.arange(5)*10), interface_grid_step_m=5,
             interface_parameters={"boundary":dict(phases=["R"], shape=[2,2], active=True, max_shift_m=10,
                depth_bounds=[10,30], control_at_shift_bound=np.zeros((2,2), bool))},
             interface_surfaces={"boundary":surface}, starting_interface_surfaces={"boundary":np.full((5,5),20)},
             sampled_interfaces={"boundary":np.round(surface/5)*5}, starting_sampled_interfaces={"boundary":np.full((5,5),20)},
             interface_hits={"boundary":contacts}, interface_controls={"boundary":np.array([[-5,-1],[-1,3]])},
             stop_reason="iteration limit", interface_std={"boundary":np.where(contacts>0,2,np.nan)},
             interface_p05={"boundary":np.where(contacts>0,surface-2,np.nan)},
             interface_p95={"boundary":np.where(contacts>0,surface+2,np.nan)})
    return r


def test_interface_views_masked_fields_sections_and_exports(qtbot, monkeypatch, tmp_path):
    result = ui_result(); dialog = Interfaces3DResultsDialog(result); qtbot.addWidget(dialog); dialog.show()
    for key in ("sampled_interfaces", "change", "interface_hits", "interface_std", "interface_p95"):
        dialog.field.setCurrentIndex(dialog.field.findData(key))
        dialog.sliders[0].setValue(0); dialog.sliders[1].setValue(4)
        assert np.isfinite(dialog.map.image.getLevels()).all()
        assert dialog.bar.getAxis("left").labelText == dialog.field.currentText()
        assert dialog.bar.getAxis("right").labelText == ""
    assert "9 / 25" in dialog.summary.text()
    result["interface_std"]["boundary"][:] = 0
    dialog.field.setCurrentIndex(dialog.field.findData("interface_std"))
    assert dialog.map.image.getLevels()[0] == 0
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path/"surface.npy"), ""))
    dialog.save_surface()
    np.testing.assert_array_equal(np.load(tmp_path/"surface.npy", allow_pickle=False), result["interface_surfaces"]["boundary"])
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path/"all.npz"), ""))
    dialog.save_bundle()
    with np.load(tmp_path/"all.npz", allow_pickle=False) as data:
        assert json.loads(str(data["metadata"]))["interface_0"]["name"] == "boundary"


def test_embedded_surface_and_fixed_boundary_roundtrip(qtbot):
    surface = np.arange(9).reshape(3,3)+20
    dialog = Branches3DDialog({"R":dict(kind="reflection", surface=surface, interface="fixed", invert_interface=False,
                                       control_shape=[2,3], depth_bounds=[10,40])})
    qtbot.addWidget(dialog)
    definition = json.loads(dialog.json_text())["R"]
    assert definition["surface"] == surface.tolist() and not definition["invert_interface"]
    assert definition["control_shape"] == [2,3]
