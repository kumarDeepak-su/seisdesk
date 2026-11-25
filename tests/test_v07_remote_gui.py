import copy
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore
from seisdesk.backends.remote import RemoteClient, RemoteProfile, save_record
from seisdesk.gui.remote_dialog import RemoteJobDialog
from seisdesk.workflow.project import Project


def saved_job(tmp_path):
    token = "a"*32
    profile = RemoteProfile(remote_root="/scratch/seisdesk", python="/venv/bin/python", modules=["Python/3.10.4"])
    record = dict(token=token, state="RUNNING", job_id="12345", created=1., profile=asdict(profile),
                  record_file=str(tmp_path/"remote-jobs"/token/"remote-job.json"),
                  remote_directory="/scratch/seisdesk/"+token)
    save_record(record)
    return profile, record


def test_gui_recovers_jobs_and_persists_host_and_status(qtbot, tmp_path, monkeypatch):
    profile, record = saved_job(tmp_path)
    project = Project(str(tmp_path))
    project.settings["remote_profile"] = asdict(profile)
    dialog = RemoteJobDialog(project)
    qtbot.addWidget(dialog); dialog.show()
    assert dialog.profile() == profile
    assert dialog.table.rowCount() == 1 and dialog.selected()["job_id"] == "12345"
    seen = []
    def status(client, item, **kwargs):
        seen.append(copy.deepcopy(item))
        item.update(state="FAILED", exit_code="2:0", log="A clear compute failure")
        save_record(item)
        return item
    monkeypatch.setattr(RemoteClient, "status", status)
    dialog.tabs.setCurrentIndex(2)
    qtbot.mouseClick(dialog.job_buttons[1], QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog.worker is None, timeout=10_000)
    assert len(seen) == 1 and dialog.selected()["state"] == "FAILED"
    assert "A clear compute failure" in dialog.log.toPlainText()
    project.save()
    restored = Project.load(str(tmp_path))
    assert restored.settings["remote_jobs"][0]["exit_code"] == "2:0"
    assert restored.settings["remote_profile"] == asdict(profile)


def test_gui_cancels_saved_job_on_its_original_host(qtbot, tmp_path, monkeypatch):
    profile, record = saved_job(tmp_path)
    project = Project(str(tmp_path))
    dialog = RemoteJobDialog(project)
    qtbot.addWidget(dialog); dialog.show()
    dialog.fields["host"].setText("different-host-for-next-job")
    calls = []
    def cancel(client, item, **kwargs):
        calls.append((client.profile.host, item["job_id"]))
        item["state"] = "CANCEL_REQUESTED"; save_record(item)
        return item
    monkeypatch.setattr(RemoteClient, "cancel", cancel)
    dialog.tabs.setCurrentIndex(2)
    qtbot.mouseClick(dialog.job_buttons[2], QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog.worker is None, timeout=10_000)
    assert calls == [("athena", "12345")]
    assert dialog.selected()["state"] == "CANCEL_REQUESTED"


def test_failed_connection_restores_controls(qtbot, tmp_path, monkeypatch):
    profile, record = saved_job(tmp_path)
    project = Project(str(tmp_path)); project.settings["remote_profile"] = asdict(profile)
    dialog = RemoteJobDialog(project); qtbot.addWidget(dialog)
    def fail(*args, **kwargs):
        raise RuntimeError("Authentication unavailable")
    monkeypatch.setattr(RemoteClient, "check", fail)
    dialog.check_connection()
    qtbot.waitUntil(lambda: dialog.worker is None, timeout=10_000)
    assert dialog.submit_button.isEnabled()
    assert "Authentication unavailable" in dialog.log.toPlainText()
    assert Path(dialog.selected()["record_file"]).is_file()
