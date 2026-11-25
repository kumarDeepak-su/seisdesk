"""Installed SU discovery, self documentation and streamed file jobs."""
from PySide6 import QtWidgets

from ..backends.su import SUBackend, run_pipeline
from .common import show_error, start_worker


class SUDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.worker = None
        self.setWindowTitle("Seismic Unix — programs and file jobs")
        self.resize(980, 740)
        self.backend = QtWidgets.QComboBox()
        self.backend.addItems(["local", "wsl"])
        self.bin_dir = QtWidgets.QLineEdit()
        self.distribution = QtWidgets.QLineEdit()
        settings = project.settings.get("su", {})
        self.backend.setCurrentText(settings.get("backend", "local"))
        self.bin_dir.setText(settings.get("bin_dir", ""))
        self.distribution.setText(settings.get("distribution", ""))
        self.input = QtWidgets.QLineEdit()
        self.output = QtWidgets.QLineEdit()
        self.pipeline = QtWidgets.QPlainTextEdit(settings.get("pipeline", "sugain tpow=2 | sufilter f=5,10,60,80"))
        self.pipeline.setMaximumHeight(120)
        self.timeout = QtWidgets.QSpinBox()
        self.timeout.setRange(1, 604800)
        self.timeout.setValue(3600)
        form = QtWidgets.QFormLayout()
        for label, field in (("Execution host", self.backend), ("SU bin directory", self.bin_dir),
                              ("WSL distribution", self.distribution)):
            form.addRow(label, field)
        for label, field, save in (("Input SU file", self.input, False), ("Output file", self.output, True)):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(field)
            button = QtWidgets.QPushButton("Browse…")
            button.clicked.connect(lambda checked=False, field=field, save=save: self.browse(field, save))
            row.addWidget(button)
            form.addRow(label, row)
        form.addRow("Pipeline / explicit bash script", self.pipeline)
        form.addRow("Maximum runtime (seconds)", self.timeout)
        self.programs = QtWidgets.QComboBox()
        self.refresh = QtWidgets.QPushButton("Find installed programs")
        self.help_button = QtWidgets.QPushButton("Show program help")
        self.help_button.clicked.connect(self.show_help)
        self.refresh.clicked.connect(self.discover)
        program_row = QtWidgets.QHBoxLayout()
        for widget in (self.refresh, self.programs, self.help_button):
            program_row.addWidget(widget)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.run_button = QtWidgets.QPushButton("Run file job")
        self.cancel_button = QtWidgets.QPushButton("Cancel job")
        self.cancel_button.setEnabled(False)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(lambda: self.worker.cancel() if self.worker else None)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel("File jobs stream data without loading the survey into the viewer. "
                               "Use Linux paths for script and parameter files when running in WSL. "
                               "To process the current dataset, add ‘Seismic Unix pipeline’ in the flow editor.")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addLayout(form)
        layout.addLayout(program_row)
        layout.addWidget(self.log, 1)
        layout.addLayout(buttons)

    def engine(self):
        return SUBackend(self.backend.currentText(), self.bin_dir.text().strip(), self.distribution.text().strip())

    def browse(self, field, save):
        chooser = QtWidgets.QFileDialog.getSaveFileName if save else QtWidgets.QFileDialog.getOpenFileName
        path, _ = chooser(self, "SU file", field.text(), "Seismic Unix (*.su);;All files (*)")
        if path:
            field.setText(path)

    def discover(self):
        try:
            engine = self.engine()
            programs = engine.programs()
            self.bin_dir.setText(engine.bin_dir)
            self.programs.clear()
            self.programs.addItems(programs)
            self.log.appendPlainText(f"Found {len(programs)} installed programs.")
        except Exception as exc:
            show_error(self, "SU discovery", exc)

    def show_help(self):
        try:
            self.log.setPlainText(self.engine().documentation(self.programs.currentText()))
        except Exception as exc:
            show_error(self, "SU documentation", exc)

    def run(self):
        try:
            engine = self.engine()
            source, target = self.input.text(), self.output.text()
            pipeline, timeout = self.pipeline.toPlainText(), self.timeout.value()
            if not source or not target:
                raise ValueError("choose the input and output files")
            self.project.settings["su"] = dict(backend=engine.kind, bin_dir=engine.bin_dir,
                                               distribution=engine.distribution, pipeline=pipeline)
            self.set_running(True)
            self.worker = start_worker(
                self, lambda worker: run_pipeline(pipeline, source, target, engine, timeout=timeout,
                                                   cancelled=worker.cancelled, log=worker.message.emit,
                                                   work_dir=self.project.context.work_dir),
                self.finished_job, self.failed_job, self.log.appendPlainText)
        except Exception as exc:
            self.set_running(False)
            show_error(self, "SU job", exc)

    def set_running(self, running):
        for widget in (self.run_button, self.refresh, self.help_button, self.pipeline, self.backend,
                       self.bin_dir, self.distribution, self.input, self.output):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def finished_job(self, report):
        self.worker = None
        self.set_running(False)
        self.project.context.results["su_pipeline"] = report
        self.log.appendPlainText(f"Completed: {report['output']}\nJob record: {report['record']}")

    def failed_job(self, error):
        self.worker = None
        self.set_running(False)
        self.log.appendPlainText(str(error))

    def reject(self):
        if self.worker is not None:
            self.worker.cancel()
            self.log.appendPlainText("Cancelling; close this window after the job stops.")
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            event.ignore()
        else:
            event.accept()
