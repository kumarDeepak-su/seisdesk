import json

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets

from seisdesk.core.picks import PickSet
from seisdesk.core.velocity import DepthModel
from seisdesk.gui.branch_editor import JointBranchDialog
from seisdesk.gui.dialogs import SEG2ImportDialog
from seisdesk.gui.flow_editor import ParamForm
from seisdesk.workflow.flow import FlowStep
from seisdesk.workflow.project import Project


def definitions():
    return {"D": dict(kind="first"), "R": dict(kind="reflection", interface="base", depth=30),
            "H": dict(kind="head", interface="base", depth=30),
            "R2": dict(kind="reflection", points=[[0, 50], [100, 60]])}


def test_editor_preserves_multiple_phases_and_updates_shared_geometry(qtbot):
    original = definitions()
    model = DepthModel(np.full((81, 51), 2000.), 1, 2)
    dialog = JointBranchDialog(original, model=model)
    qtbot.addWidget(dialog); dialog.show()
    dialog.table.setCurrentCell(1, 1)
    dialog.depth.setValue(35)
    result = dialog.values()
    assert result["R"]["depth"] == result["H"]["depth"] == 35
    assert original["R"]["depth"] == 30
    dialog._roi.setValue(38)
    dialog._roi.sigPositionChangeFinished.emit(dialog._roi)
    assert dialog.values()["R"]["depth"] == 38
    dialog.minimum.setText("32"); dialog.maximum.setText("45")
    dialog.maximum.editingFinished.emit()
    assert dialog.values()["H"]["depth_bounds"] == [32, 45]
    dialog.table.setCurrentCell(3, 1)
    dialog._roi.setPoints([[0, 55], [100, 65]])
    dialog._drag_points(dialog._roi)
    np.testing.assert_allclose(dialog.values()["R2"]["points"], [[0, 55], [100, 65]])
    dialog.table.item(2, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert list(dialog.values()) == ["D", "R", "R2"]
    assert json.loads(dialog.json_text()) == dialog.values()


def test_editor_rejects_invalid_depth_bounds(qtbot):
    dialog = JointBranchDialog(definitions())
    qtbot.addWidget(dialog)
    dialog.table.setCurrentCell(1, 1)
    dialog.minimum.setText("40"); dialog.maximum.setText("50")
    with pytest.raises(ValueError, match="outside their bounds"):
        dialog.values()
    dialog.maximum.setText("invalid")
    with pytest.raises(ValueError):
        dialog.values()


def test_flow_form_saves_editor_results_without_json_typing(qtbot, monkeypatch):
    project = Project()
    picks = PickSet(name="arrivals")
    for name, kind in (("R", "reflection"), ("H", "refraction")):
        picks.new_phase(name, kind)
    project.context.picks["arrivals"] = picks
    step = FlowStep("joint_tomography", dict(pickset="arrivals", branches=json.dumps(definitions())))
    form = ParamForm(project); qtbot.addWidget(form); form.set_step(step)
    def accept(dialog):
        dialog.table.setCurrentCell(1, 1)
        dialog.depth.setValue(42)
        return QtWidgets.QDialog.DialogCode.Accepted
    monkeypatch.setattr(JointBranchDialog, "exec", accept)
    button = next(b for b in form.findChildren(QtWidgets.QPushButton) if b.text().startswith("Configure arrivals"))
    qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    assert json.loads(step.params["branches"])["R"]["depth"] == 42


def test_seg2_timing_options_are_explicit(qtbot):
    dialog = SEG2ImportDialog(26)
    qtbot.addWidget(dialog)
    assert dialog.options() == dict(remove_delay=True, descale=True, apply_skew=False)
    dialog.skew.setChecked(True); dialog.align.setChecked(False)
    assert dialog.options() == dict(remove_delay=False, descale=True, apply_skew=True)
