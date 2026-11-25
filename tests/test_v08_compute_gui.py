import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from seisdesk.gui.compute_dialog import ComputeDialog
from seisdesk.gui.flow_editor import ParamForm
from seisdesk.workflow.flow import FlowStep
from seisdesk.workflow.project import Project


def test_device_defaults_persist_without_overriding_saved_flow_steps(qtbot, tmp_path):
    project = Project(str(tmp_path))
    dialog = ComputeDialog(project, discover=False); qtbot.addWidget(dialog)
    dialog.probe_finished(dict(available=True, message="Detected", devices=[dict(index=1, name="GPU 1",
                              free_bytes=8*1024**3, total_bytes=12*1024**3, tested=True)]))
    dialog.device.setCurrentIndex(dialog.device.findData("cuda")); dialog.index.setValue(1)
    dialog.save_choice()
    assert project.settings["compute_defaults"] == dict(device="cuda", device_index=1)
    assert project.settings["volume_job"]["workers"] == 1
    fresh = FlowStep("acoustic_model_3d")
    form = ParamForm(project); qtbot.addWidget(form); form.set_step(fresh)
    assert fresh.params["device"] == "cuda" and fresh.params["device_index"] == 1
    existing = FlowStep("acoustic_model_3d", dict(device="cpu", device_index=0))
    form.set_step(existing)
    assert existing.params["device"] == "cpu"
    remote = FlowStep("volume_file_job")
    form.local_compute_defaults = False; form.set_step(remote)
    assert "device" not in remote.params
    project.save()
    assert Project.load(str(tmp_path)).settings["compute_defaults"] == dict(device="cuda", device_index=1)


def test_unavailable_cuda_cannot_be_saved_as_detected(qtbot, tmp_path):
    project = Project(str(tmp_path))
    dialog = ComputeDialog(project, discover=False); qtbot.addWidget(dialog)
    dialog.device.setCurrentIndex(dialog.device.findData("cuda"))
    dialog.save_choice()
    assert "compute_defaults" not in project.settings
    assert "Detect" in dialog.status.text()
    dialog.device.setCurrentIndex(dialog.device.findData("auto")); dialog.save_choice()
    assert project.settings["compute_defaults"]["device"] == "auto"
