"""Workstation CPU/CUDA selection and an explicit GPU runtime check."""
import json

from PySide6 import QtCore, QtWidgets

from ..core.compute import probe_cuda
from .common import start_worker


class ComputeDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None, *, discover=True):
        super().__init__(parent)
        self.project, self.worker, self.report = project, None, None
        self.setWindowTitle("Workstation compute devices")
        self.resize(850, 640)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("Choose the CPU or an NVIDIA CUDA GPU on this workstation. "
                               "CUDA accelerates full-volume 3D acoustic modelling, RTM and FWI. "
                               "Other processing steps keep their documented execution backend.")
        note.setWordWrap(True); layout.addWidget(note)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["GPU index", "Device", "Free VRAM", "Total VRAM", "CUDA check"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        form = QtWidgets.QFormLayout()
        self.device = QtWidgets.QComboBox()
        for label, value in (("CPU", "cpu"), ("NVIDIA GPU (CUDA)", "cuda"), ("Automatic — CUDA when usable", "auto")):
            self.device.addItem(label, value)
        self.index = QtWidgets.QSpinBox(); self.index.setRange(0, 63)
        defaults = project.settings.get("compute_defaults", {})
        self.device.setCurrentIndex(max(0, self.device.findData(defaults.get("device", "cpu"))))
        self.index.setValue(int(defaults.get("device_index", 0)))
        form.addRow("Default for new 3D processing steps", self.device)
        form.addRow("CUDA GPU index", self.index); layout.addLayout(form)
        info = QtWidgets.QLabel("Automatic selection reports when it uses CPU. Explicit CUDA selection stops with an "
                               "error if the GPU cannot run. Existing saved flows keep their own settings. GPU indices "
                               "refer to devices visible to this process; use one local shot worker for CUDA.")
        info.setWordWrap(True); layout.addWidget(info)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); layout.addWidget(self.log)
        self.status = QtWidgets.QLabel("Choose Detect devices to inspect this workstation.")
        self.status.setWordWrap(True); layout.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.detect_button = QtWidgets.QPushButton("Detect devices")
        self.test_button = QtWidgets.QPushButton("Run CUDA check")
        self.apply_button = QtWidgets.QPushButton("Save compute choice")
        self.detect_button.clicked.connect(lambda: self.refresh(False))
        self.test_button.clicked.connect(lambda: self.refresh(True))
        self.apply_button.clicked.connect(self.save_choice)
        for button in (self.detect_button, self.test_button, self.apply_button):
            row.addWidget(button)
        layout.addLayout(row)
        if discover:
            QtCore.QTimer.singleShot(0, lambda: self.refresh(False))

    def refresh(self, test=False):
        if self.worker:
            return
        self.status.setText("Checking CUDA kernels…" if test else "Detecting workstation devices…")
        for button in (self.detect_button, self.test_button, self.apply_button):
            button.setEnabled(False)
        index = self.index.value()
        self.worker = start_worker(self, lambda worker: probe_cuda(test=test, device_index=index if test else None),
                                   self.probe_finished, self.probe_failed)

    @QtCore.Slot(object)
    def probe_finished(self, report):
        self.report = report
        self.table.setRowCount(len(report["devices"]))
        for row, device in enumerate(report["devices"]):
            for col, value in enumerate((device["index"], device["name"], f"{device['free_bytes']/1024**3:.2f} GiB",
                                         f"{device['total_bytes']/1024**3:.2f} GiB",
                                         "Passed" if device.get("tested") else "Not run")):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))
        self.log.setPlainText(json.dumps(report, indent=2))
        self.status.setText(report["message"])
        self.restore_controls()

    @QtCore.Slot(str)
    def probe_failed(self, message):
        self.status.setText(message); self.restore_controls()

    def restore_controls(self):
        self.worker = None
        for button in (self.detect_button, self.test_button, self.apply_button):
            button.setEnabled(True)

    def save_choice(self):
        device, index = self.device.currentData(), self.index.value()
        if device == "cuda" and (not self.report or not self.report["available"] or
                                  not any(d["index"] == index for d in self.report["devices"])):
            self.status.setText("Detect the selected CUDA device successfully before saving this choice.")
            return
        defaults = dict(device=device, device_index=index)
        self.project.settings["compute_defaults"] = defaults
        job = self.project.settings.setdefault("volume_job", {})
        job.update(defaults)
        if device in ("cuda", "auto"):
            job["workers"] = 1
        self.status.setText("Saved for new local 3D steps and survey jobs. Existing flow steps retain their saved values.")

    def reject(self):
        if self.worker:
            self.status.setText("Wait for the device check to finish before closing.")
            return
        super().reject()

    def closeEvent(self, event):
        event.ignore() if self.worker else event.accept()
