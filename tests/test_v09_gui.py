import json

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets
from seisdesk.core.picks import PickSet
from seisdesk.core.volume import VelocityVolume
from seisdesk.gui.flow_editor import ParamForm
from seisdesk.gui.tomography3d_dialog import Branches3DDialog, Tomography3DResultsDialog
from seisdesk.gui.volume_dialog import VolumeSlicesDialog
from seisdesk.gui.reconstruction_dialog import ReconstructionResultsDialog
from seisdesk.workflow.flow import FlowStep
from seisdesk.workflow.project import Project


def test_3d_branch_controls_and_flow_roundtrip(qtbot, monkeypatch):
    picks = PickSet(name="arrivals")
    picks.new_phase("P", "direct"); picks.new_phase("R1", "reflection"); picks.new_phase("R2", "reflection")
    definitions = {"P": dict(kind="first"), "R1": dict(kind="reflection", depth=40),
                   "R2": dict(kind="reflection", depth=80)}
    project = Project(); project.context.picks["arrivals"] = picks
    step = FlowStep("joint_tomography_3d", dict(pickset="arrivals", branches=json.dumps(definitions)))
    form = ParamForm(project); qtbot.addWidget(form); form.set_step(step)
    def accept(dialog):
        assert not dialog.table.cellWidget(0, 4).isEnabled()
        dialog.table.cellWidget(1, 4).setValue(45)
        dialog.table.cellWidget(2, 5).setValue(.03)
        dialog.table.cellWidget(2, 3).setValue(.5)
        return QtWidgets.QDialog.DialogCode.Accepted
    monkeypatch.setattr(Branches3DDialog, "exec", accept)
    button = next(b for b in form.findChildren(QtWidgets.QPushButton) if b.text().startswith("Configure arrivals"))
    qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    saved = json.loads(step.params["branches"])
    assert saved["R1"]["depth"] == 45 and saved["R2"]["slope_x"] == .03
    assert saved["R2"]["uncertainty_ms"] == .5 and saved["P"]["kind"] == "first"
    assert definitions["R1"]["depth"] == 40


def test_masked_uncertainty_slices_and_arrival_table(qtbot):
    volume = VelocityVolume(np.full((3, 4, 5), 2000.))
    uncertainty = np.full(volume.values.shape, np.nan); uncertainty[1:, 1:, 1:] = 42
    dialog = VolumeSlicesDialog(volume, uncertainty, units="m/s", sequential=True)
    qtbot.addWidget(dialog); dialog.show()
    assert np.isfinite(dialog.velocity_limits).all()
    dialog.sliders[0].setValue(0); dialog.probe(0, 0, 0)
    assert "uncovered" in dialog.readout.text()
    result = dict(velocity=volume.values, spacing=volume.spacing, origin=volume.origin,
                  reference_elevation=0, observed=np.array([.01, .03]), t_pred=np.array([.011, .029]),
                  uncertainty=np.array([.001, .002]), residual=np.array([.001, -.001]),
                  standardized_residual=np.array([1., -.5]), outlier=np.array([False, False]),
                  trace_ids=np.array(["same", "same"]), phases=["P", "R"], rms=[.002, .001],
                  weighted_misfit=[2., .625], hits=np.ones(volume.values.shape), velocity_std=uncertainty)
    result_dialog = Tomography3DResultsDialog(result); qtbot.addWidget(result_dialog); result_dialog.show()
    model = result_dialog.table.model()
    assert model.rowCount() == 2 and model.data(model.index(1, 1)) == "R"
    assert result_dialog.field.findData("velocity_std") >= 0


def test_reconstruction_coverage_maps_and_controls(qtbot):
    project = Project(); form = ParamForm(project); qtbot.addWidget(form)
    form.set_step(FlowStep("survey_pocs"))
    result = dict(shape=(4, 5, 3, 3), origin=[200, 100, -20, -20], spacing=[20, 20, 20, 20],
                  occupied=np.arange(40), observed_fold=np.full((4, 5), 2), generated_fold=np.ones((4, 5)))
    dialog = ReconstructionResultsDialog(result); qtbot.addWidget(dialog); dialog.show()
    assert len(dialog.plots) == 2


@pytest.mark.parametrize("cancel", [False, True])
def test_survey_job_keeps_ui_responsive_and_publishes_only_success(qtbot, monkeypatch, cancel):
    import threading
    from seisdesk.core.dataset import SeismicData
    from seisdesk.gui.arrival_survey_job import ArrivalSurveyJobDialog
    from seisdesk.processing.survey_interpolation import SurveyPOCS
    entered, release = threading.Event(), threading.Event()
    def run(process, data, context):
        assert QtCore.QThread.currentThread() != QtWidgets.QApplication.instance().thread()
        context.results["isolated"] = dict(value=42)
        entered.set()
        while not release.wait(.01):
            context.report(.2)
        context.report(1)
        return data.copy()
    monkeypatch.setattr(SurveyPOCS, "run", run)
    project = Project(); name = project.add_dataset(SeismicData(np.zeros((4, 16)), .004))
    dialog = ArrivalSurveyJobDialog(project, "survey_pocs", name); qtbot.addWidget(dialog); dialog.show()
    try:
        dialog.accept(); qtbot.waitUntil(entered.is_set)
        ticks = []
        QtCore.QTimer.singleShot(0, lambda: ticks.append(1)); qtbot.waitUntil(lambda: bool(ticks))
        assert "isolated" not in project.context.results
        if cancel:
            dialog.reject()
        else:
            release.set()
        qtbot.waitUntil(lambda: dialog.worker is None, timeout=10000)
        assert ("isolated" in project.context.results) == (not cancel)
        assert (dialog.result_name is not None) == (not cancel)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not dialog.threads_running(), timeout=10000)
