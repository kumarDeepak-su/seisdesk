"""Interaction checks for the desktop workspace, executed on Windows CI."""
import json
from pathlib import Path
import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from seisdesk.core.dataset import SeismicData
from seisdesk.gui.main_window import MainWindow
from seisdesk.gui.workspace import CommandPalette
from seisdesk.gui.flow_editor import PROCESS_MIME
from seisdesk.workflow.flow import Flow, FlowStep
from seisdesk.workflow.project import Project


@pytest.fixture
def workspace(qtbot, tmp_path, monkeypatch):
    import pyqtgraph as pg
    from seisdesk.gui import common, main_window
    monkeypatch.setattr(common, "HEADLESS", True)
    pg.setConfigOptions(background="w", foreground="k", imageAxisOrder="row-major")
    monkeypatch.setattr(main_window, "default_work_dir", lambda: str(tmp_path))
    settings = QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    # Close after clearing dirty state ourselves. pytest-qt closes registered widgets
    # before fixture finalizers, which would otherwise open the real save prompt.
    window.show()
    try:
        yield window
    finally:
        window.dirty = False
        window.close()
        window.deleteLater()


def data():
    return SeismicData(np.arange(120, dtype=np.float32).reshape(3, 40), .004, dict(fldr=[1, 1, 1], offset=[0, 10, 20]))


def test_command_search_keyboard_and_real_action(workspace, qtbot):
    dialog = CommandPalette(workspace.command_entries(), workspace)
    qtbot.addWidget(dialog); dialog.show()
    qtbot.keyClicks(dialog.search, "add flow bandpass")
    assert dialog.results.count() > 0
    qtbot.keyClick(dialog.search, QtCore.Qt.Key.Key_Return)
    assert dialog.chosen and "Band" in dialog.chosen.title
    dialog.chosen.run()
    assert workspace.tabs.currentWidget() is workspace.flow_editor
    assert workspace.flow_editor.flow.steps[-1].process == "bandpass"
    assert workspace.dirty


def test_command_no_match_and_disabled_entries(workspace, qtbot):
    dialog = CommandPalette(workspace.command_entries(), workspace)
    qtbot.addWidget(dialog)
    dialog.search.setText("no-such-process-xyz")
    assert dialog.results.count() == 0
    dialog.activate()
    assert dialog.chosen is None


def test_dataset_filter_and_command_selection(workspace):
    workspace.add_dataset(data(), "Land line")
    workspace.add_dataset(data(), "Marine line")
    workspace.dataset_search.setText("land")
    visible = [workspace.dataset_list.item(i) for i in range(2) if not workspace.dataset_list.item(i).isHidden()]
    assert len(visible) == 1 and visible[0].data(QtCore.Qt.ItemDataRole.UserRole) == "Land line"
    next(c for c in workspace.command_entries() if c.title == "Show dataset: Land line").run()
    assert workspace.viewer.dataset_name == "Land line"
    workspace.dataset_search.setText("absent")
    assert workspace.dataset_empty.isVisible()
    assert workspace.home.metrics[1].text() == "6"


def test_reorder_bypass_and_saved_execution_order(workspace, tmp_path):
    editor = workspace.flow_editor
    editor.set_flow(Flow([FlowStep("bandpass", {}), FlowStep("agc", {}), FlowStep("stack", {})], "sequence"))
    editor.steps.item(1).setCheckState(QtCore.Qt.CheckState.Unchecked)
    # Qt's internal move changes the list model before the drop handler updates the flow.
    item = editor.steps.takeItem(2); editor.steps.insertItem(0, item)
    editor.steps.setCurrentRow(0); editor.steps.orderChanged.emit()
    assert [s.process for s in editor.flow.steps] == ["stack", "bandpass", "agc"]
    assert [s.enabled for s in editor.flow.steps] == [True, True, False]
    assert editor.form.step is editor.flow.steps[0]
    editor.flow.save(str(tmp_path / "flow.json"))
    restored = Flow.load(str(tmp_path / "flow.json"))
    assert [s.process for s in restored.steps] == ["stack", "bandpass", "agc"]
    assert not restored.steps[-1].enabled


def test_new_project_clears_previous_flow_and_tracks_unsaved_edits(workspace):
    workspace.flow_editor.add_process("agc")
    assert workspace.dirty and "Unsaved" in workspace.project_status.text()
    workspace._set_project(Project())
    assert not workspace.flow_editor.flow.steps and not workspace.dirty
    assert workspace.tabs.currentWidget() is workspace.home
    assert not workspace.flow_editor.run_btn.isEnabled()


def test_cleared_flow_save_and_recent_reopen(workspace, tmp_path, monkeypatch):
    workspace.add_dataset(data(), "line")
    workspace.flow_editor.add_process("agc")
    folder = tmp_path / "project"; folder.mkdir()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    assert workspace.save_project(ask=True)
    workspace.flow_editor.set_flow(Flow([], "empty"))
    assert workspace.dirty and workspace.save_project()
    assert not Project.load(str(folder)).flows
    assert json.loads(workspace.ui_settings.value("projects/recent")) == [str(folder)]
    workspace._set_project(Project())
    workspace.open_recent_project(str(folder))
    assert "line" in workspace.project.datasets and not workspace.dirty
    workspace.show_dataset("line")
    assert not workspace.dirty


def test_running_state_and_cancel_leave_editor_consistent(workspace):
    workspace.add_dataset(data(), "line")
    editor = workspace.flow_editor
    editor.add_process("agc")
    assert editor.run_btn.isEnabled()
    class Worker:
        cancelled = False
        def cancel(self): self.cancelled = True
    worker = Worker(); editor._worker = worker; editor._set_running(True)
    assert not editor.editor_panels.isEnabled() and not editor.run_btn.isEnabled()
    assert not workspace._ask_save_changes()
    editor.cancel()
    assert worker.cancelled and not editor.cancel_btn.isEnabled()
    editor._failed("Cancelled.")
    assert editor.editor_panels.isEnabled() and editor.run_btn.isEnabled()
    assert len(workspace.project.datasets) == 1 and "Cancelled" in editor.run_status.text()
    editor.undo_btn.click()
    assert "Add a process" in editor.run_status.text() and not editor.run_btn.isEnabled()


def test_theme_persistence_and_display_options(workspace, qtbot):
    assert workspace.theme == "light"
    workspace.theme_combo.setCurrentIndex(workspace.theme_combo.findData("dark"))
    assert workspace.ui_settings.value("appearance/theme") == "dark"
    workspace.theme_combo.setCurrentIndex(workspace.theme_combo.findData("light"))
    assert workspace.ui_settings.value("appearance/theme") == "light"
    assert QtWidgets.QApplication.instance().property("seisdeskTheme") == "light"
    workspace.add_dataset(data(), "line")
    before = workspace.project.datasets["line"].traces.copy()
    qtbot.mouseClick(workspace.viewer.display_options, QtCore.Qt.MouseButton.LeftButton)
    assert workspace.viewer.display_panel.isVisible()
    workspace.theme_combo.setCurrentIndex(workspace.theme_combo.findData("dark"))
    np.testing.assert_array_equal(before, workspace.project.datasets["line"].traces)
    workspace.viewer.display_options.click()
    assert not workspace.viewer.display_panel.isVisible()


def test_drop_process_then_undo_parameter_and_structure_edits(workspace):
    editor = workspace.flow_editor
    workspace.tabs.setCurrentWidget(editor)
    mime = QtCore.QMimeData(); mime.setData(PROCESS_MIME, b"agc")
    event = QtGui.QDropEvent(QtCore.QPointF(10, 10), QtCore.Qt.DropAction.CopyAction, mime,
                            QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    editor.steps.dropEvent(event)
    assert event.isAccepted() and editor.flow.steps[0].process == "agc"
    field = editor.form.form.itemAt(0, QtWidgets.QFormLayout.ItemRole.FieldRole).widget()
    initial = editor.flow.steps[0].params["window"]
    field.setText("0.75"); field.editingFinished.emit()
    assert editor.flow.steps[0].params["window"] == .75
    editor.undo_btn.click()
    assert editor.flow.steps[0].params["window"] == initial
    editor.undo_btn.click()
    assert not editor.flow.steps
    editor.redo_btn.click()
    assert editor.flow.steps[0].process == "agc"
    workspace._set_project(Project())
    assert not editor.undo_btn.isEnabled() and not editor.redo_btn.isEnabled()


def test_local_file_drop_uses_normal_import_path(workspace, tmp_path, monkeypatch):
    path = tmp_path / "shot.sgy"; path.write_bytes(b"fixture")
    mime = QtCore.QMimeData(); mime.setUrls([QtCore.QUrl.fromLocalFile(str(path))])
    captured = []
    monkeypatch.setattr(workspace, "_open_paths", captured.extend)
    event = QtGui.QDropEvent(QtCore.QPointF(20, 20), QtCore.Qt.DropAction.CopyAction, mime,
                            QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    workspace.dropEvent(event)
    assert event.isAccepted() and [Path(p) for p in captured] == [path]
    mime.setUrls([QtCore.QUrl("https://example.com/shot.sgy")])
    assert workspace._drop_paths(mime) == []


def test_workspaces_fit_a_compact_window(workspace, qtbot):
    workspace.resize(1280, 800)
    workspace.add_dataset(data(), "A long survey name " * 8)
    workspace.flow_editor.add_process("bandpass")
    for page in (workspace.home, workspace.viewer, workspace.flow_editor, workspace.velan):
        workspace.tabs.setCurrentWidget(page)
        QtWidgets.QApplication.processEvents()
        assert workspace.width() <= 1280, (type(page).__name__, workspace.minimumSizeHint().width())
