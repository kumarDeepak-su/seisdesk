"""Responsive single-step launch for the 3D arrival and survey reconstruction jobs."""
import copy
import dataclasses

from PySide6 import QtCore, QtWidgets

from .common import show_error, start_worker
from .dialogs import SingleStepDialog


class ArrivalSurveyJobDialog(SingleStepDialog):
    def __init__(self, project, process, dataset, parent=None):
        self.worker = None
        self.cancel_requested = False
        self.close_after_cancel = False
        super().__init__(project, process, dataset, parent)
        self.resize(900, 850)
        layout = self.layout(); layout.removeWidget(self.form)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.form)
        layout.insertWidget(0, scroll, 1)
        self.messages = QtWidgets.QPlainTextEdit(); self.messages.setReadOnly(True); self.messages.setMaximumHeight(120)
        self.progress = QtWidgets.QProgressBar()
        layout.insertWidget(2, self.messages); layout.insertWidget(3, self.progress)
        self.buttons = self.findChild(QtWidgets.QDialogButtonBox)
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Run")
        if process == "joint_tomography_3d" and project.settings.get("volume_model"):
            self.step.params["model"] = project.settings["volume_model"]
            self.form.set_step(self.step)

    def running(self, value):
        self.form.setEnabled(not value); self.replace.setEnabled(not value)
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(not value)
        button = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        button.setEnabled(True); button.setText("Cancel run" if value else "Close")

    def accept(self):
        if self.worker is not None:
            return
        try:
            data = self.project.datasets[self.dataset]
            process = self.step.build()
        except Exception as exc:
            show_error(self, self.windowTitle(), exc); return
        self.cancel_requested = self.close_after_cancel = False
        self.logs = []; self.messages.clear(); self.progress.setValue(0)
        base = dataclasses.replace(self.project.context, picks=copy.deepcopy(self.project.context.picks), results={})
        def execute(worker):
            context = dataclasses.replace(base, log=worker.message.emit, progress=worker.progress.emit,
                                          cancelled=worker.cancelled)
            return process(data, context), context
        self.running(True)
        self.worker = start_worker(self, execute, self.finished_job, self.failed_job, self.message, self.update_progress)

    def threads_running(self):
        for thread, _ in getattr(self, "_seisdesk_threads", []):
            try:
                if thread.isRunning():
                    return True
            except RuntimeError:
                pass
        return False

    @QtCore.Slot(object)
    def finished_job(self, payload):
        if self.threads_running():
            QtCore.QTimer.singleShot(10, lambda: self.finished_job(payload)); return
        if self.cancel_requested:
            self.failed_job("Cancelled."); return
        self.worker = None
        out, context = payload
        self.project.context.results.update(context.results)
        self.project.context.picks.update(context.picks)
        if self.replace.isChecked():
            out.name = self.dataset; self.project.datasets[self.dataset] = out; self.result_name = self.dataset
        else:
            self.result_name = self.project.add_dataset(out, f"{self.dataset} > {self.step.process}")
        QtWidgets.QDialog.accept(self)

    @QtCore.Slot(str)
    def failed_job(self, message):
        if self.threads_running():
            QtCore.QTimer.singleShot(10, lambda: self.failed_job(message)); return
        self.worker = None; self.running(False); self.message(message)
        if self.close_after_cancel:
            QtWidgets.QDialog.reject(self)

    @QtCore.Slot(str)
    def message(self, text):
        self.logs.append(text); self.messages.appendPlainText(text)

    @QtCore.Slot(float)
    def update_progress(self, fraction):
        self.progress.setValue(round(100*fraction))

    def reject(self):
        if self.worker is not None:
            self.cancel_requested = True; self.worker.cancel()
            self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setEnabled(False)
            self.message("Cancellation requested; waiting for the current operation to stop.")
        else:
            QtWidgets.QDialog.reject(self)

    def closeEvent(self, event):
        if self.worker is not None:
            self.close_after_cancel = True; self.reject(); event.ignore()
        else:
            super().closeEvent(event)
