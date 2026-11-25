"""Submit, recover, monitor and retrieve Slurm volume jobs without a terminal."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..backends.remote import RemoteClient, RemoteProfile, load_record, prepare_job
from ..backends.job_bundle import atomic_json, rebase_paths
from ..workflow.flow import FlowStep
from ..workflow.project import load_dataset
from .common import show_error, start_worker
from .flow_editor import ParamForm


class RemoteJobDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project, self.worker, self.result_name = project, None, None
        self.records = []
        self.setWindowTitle("Remote processing — SSH / Slurm")
        self.resize(1050, 850)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("Run 3D modelling, RTM or FWI on a compute host. Inputs and SeisDesk source are bundled; "
                               "jobs continue when this window closes. The host needs Python, NumPy, SciPy and segyio "
                               "(plus CuPy for CUDA), and an authenticated OpenSSH connection.")
        note.setWordWrap(True); layout.addWidget(note)
        self.tabs = QtWidgets.QTabWidget(); layout.addWidget(self.tabs, 1)
        self.step = FlowStep("volume_file_job", dict(project.settings.get("remote_volume_job", {})))
        if project.settings.get("volume_model"):
            self.step.params.setdefault("model", project.settings["volume_model"])
        self.form = ParamForm(project)
        self.form.local_compute_defaults = False
        self.form.set_step(self.step)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.form)
        self.tabs.addTab(scroll, "Job inputs")
        host = QtWidgets.QWidget(); host_form = QtWidgets.QFormLayout(host)
        self.fields = {}
        for name, label in (("host", "SSH host alias"), ("remote_root", "Remote job root (absolute path)"),
                            ("python", "Remote Python executable"), ("modules", "Environment modules (comma separated)"),
                            ("partition", "Slurm partition"), ("account", "Slurm account"),
                            ("cpus", "CPUs"), ("memory_gb", "Total memory (GiB)"),
                            ("gpus", "GPUs (0 or 1)"), ("walltime", "Wall time [days-]hh:mm:ss")):
            self.fields[name] = QtWidgets.QLineEdit()
            host_form.addRow(label, self.fields[name])
        self.set_profile(project.settings.get("remote_profile", asdict(RemoteProfile())))
        help_text = QtWidgets.QLabel("The SSH alias uses your existing OpenSSH configuration and keys. "
                                    "No passwords or private keys are stored in a project. This launcher requests one node; "
                                    "use one shot worker for CUDA. The local job-directory field is replaced by the remote job folder.")
        help_text.setWordWrap(True); host_form.addRow(help_text)
        row = QtWidgets.QHBoxLayout()
        for label, callback in (("Load profile…", self.load_profile), ("Save profile…", self.save_profile),
                                ("Check connection", self.check_connection)):
            button = QtWidgets.QPushButton(label); button.clicked.connect(callback); row.addWidget(button)
        host_form.addRow(row)
        self.tabs.addTab(host, "Compute host")
        jobs = QtWidgets.QWidget(); jobs_layout = QtWidgets.QVBoxLayout(jobs)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Host", "Slurm job", "State", "Local job record"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.selected_changed)
        jobs_layout.addWidget(self.table)
        actions = QtWidgets.QHBoxLayout(); self.job_buttons = []
        for label, callback in (("Open saved job…", self.open_record), ("Refresh status", self.refresh_status),
                                ("Cancel selected job", self.cancel_job), ("Download and open results", self.download)):
            button = QtWidgets.QPushButton(label); button.clicked.connect(callback); actions.addWidget(button)
            self.job_buttons.append(button)
        jobs_layout.addLayout(actions)
        self.tabs.addTab(jobs, "Jobs and results")
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(190)
        layout.addWidget(self.log)
        self.activity = QtWidgets.QLabel("Ready"); layout.addWidget(self.activity)
        self.submit_button = QtWidgets.QPushButton("Bundle inputs and submit")
        self.submit_button.clicked.connect(self.submit); layout.addWidget(self.submit_button)
        self.reload_records()
        self.timer = QtCore.QTimer(self); self.timer.setInterval(60_000)
        self.timer.timeout.connect(self.poll); self.timer.start()

    def set_profile(self, values):
        defaults = asdict(RemoteProfile())
        defaults.update(values)
        for name, field in self.fields.items():
            value = defaults[name]
            field.setText(", ".join(value) if name == "modules" else str(value))

    def profile(self):
        values = {name: field.text().strip() for name, field in self.fields.items()}
        values["modules"] = [v.strip() for v in values["modules"].split(",") if v.strip()]
        for name in ("cpus", "memory_gb", "gpus"):
            values[name] = int(values[name])
        return RemoteProfile(**values).validate()

    def load_profile(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Remote profile", "", "JSON (*.json)")
        if path:
            try:
                profile = RemoteProfile(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()
                self.set_profile(asdict(profile))
                self.project.settings["remote_profile"] = asdict(profile)
            except Exception as exc:
                show_error(self, "Remote profile", exc)

    def save_profile(self):
        try:
            profile = self.profile()
            self.project.settings["remote_profile"] = asdict(profile)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save remote profile", "remote-profile.json", "JSON (*.json)")
            if path:
                atomic_json(path, asdict(profile))
        except Exception as exc:
            show_error(self, "Remote profile", exc)

    def execute(self, label, fn):
        if self.worker:
            return
        self.activity.setText(label)
        self.form.setEnabled(False); self.submit_button.setEnabled(False)
        for button in self.job_buttons:
            button.setEnabled(False)
        self.worker = start_worker(self, fn, self.operation_finished, self.failed, self.log.appendPlainText)

    def check_connection(self):
        try:
            profile = self.profile()
            self.project.settings["remote_profile"] = asdict(profile)
            self.execute("Checking SSH and remote Python…",
                         lambda worker: ("check", RemoteClient(profile).check(worker.cancelled)))
        except Exception as exc:
            show_error(self, "Remote connection", exc)

    def submit(self):
        try:
            profile = self.profile()
            parameters = dict(self.step.build().values)
            self.project.settings.update(remote_profile=asdict(profile), remote_volume_job=parameters)
            work_dir = self.project.context.work_dir
            def task(worker):
                record = prepare_job(parameters, profile, Path(work_dir)/"remote-jobs", work_dir=work_dir,
                                     cancelled=worker.cancelled, report=worker.message.emit)
                RemoteClient(profile).submit(record, cancelled=worker.cancelled, report=worker.message.emit)
                return "submit", record
            self.execute("Bundling and submitting…", task)
        except Exception as exc:
            show_error(self, "Remote submission", exc)

    def selected(self):
        row = self.table.currentRow()
        return copy.deepcopy(self.records[row]) if 0 <= row < len(self.records) else None

    def selected_changed(self):
        record = self.selected()
        if record:
            details = {key: record[key] for key in ("state", "job_id", "reason", "exit_code", "elapsed", "nodes", "error",
                                                    "application", "local_results", "remote_directory") if key in record}
            self.log.setPlainText(json.dumps(details, indent=2)+"\n"+record.get("log", ""))

    def reload_records(self, extra=None):
        selected = extra or self.selected()
        paths = {r["record_file"] for r in self.project.settings.get("remote_jobs", [])}
        paths.update(str(p) for p in (Path(self.project.context.work_dir)/"remote-jobs").glob("*/remote-job.json"))
        if extra:
            paths.add(extra["record_file"])
        saved = {r["record_file"]: r for r in self.project.settings.get("remote_jobs", [])}
        records = []
        for path in sorted(paths):
            try:
                records.append(load_record(path) if Path(path).is_file() else saved[path])
            except (OSError, ValueError, KeyError):
                continue
        self.records = sorted(records, key=lambda r: r.get("created", 0), reverse=True)
        self.project.settings["remote_jobs"] = copy.deepcopy(self.records)
        self.table.blockSignals(True); self.table.setRowCount(len(self.records))
        selection = 0
        for row, record in enumerate(self.records):
            for col, value in enumerate((record["profile"]["host"], record.get("job_id", "—"),
                                         record["state"], record["record_file"])):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))
            if selected and record["token"] == selected["token"]:
                selection = row
        self.table.blockSignals(False)
        if self.records:
            self.table.selectRow(selection)
        self.table.resizeColumnsToContents()

    def open_record(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Recover remote job", "", "Job record (remote-job.json)")
        if path:
            try:
                self.reload_records(load_record(path))
            except Exception as exc:
                show_error(self, "Remote job", exc)

    def job_action(self, action, destination=None):
        record = self.selected()
        if not record or self.worker:
            return
        client = RemoteClient(RemoteProfile(**record["profile"]))
        def task(worker):
            if action == "fetch":
                client.fetch(record, destination, cancelled=worker.cancelled, report=worker.message.emit)
            else:
                getattr(client, action)(record, cancelled=worker.cancelled)
            return action, record
        self.execute({"status": "Refreshing…", "cancel": "Requesting cancellation…", "fetch": "Downloading results…"}[action], task)

    def refresh_status(self):
        self.job_action("status")

    def cancel_job(self):
        self.job_action("cancel")

    def download(self):
        record = self.selected()
        if record and record.get("local_results") and Path(record["local_results"]).is_dir():
            self.open_results(record)
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Download results into", self.project.context.work_dir)
        if directory:
            self.job_action("fetch", directory)

    def poll(self):
        record = self.selected()
        if not self.worker and record and record.get("state") in ("SUBMITTED", "PENDING", "RUNNING", "COMPLETING",
                                                                  "CANCEL_REQUESTED", "ACCOUNTING_PENDING"):
            self.refresh_status()

    def open_results(self, record):
        directory = Path(record["local_results"])
        preview, diagnostics = directory/"output/preview.npz", directory/"output/job.json"
        if (record.get("application") or {}).get("state") != "completed" or not preview.is_file():
            self.log.appendPlainText("Partial results and logs downloaded. No completed dataset is available to open.")
            return
        data = load_dataset(preview)
        mapping = json.loads((directory/"path-map.json").read_text(encoding="utf-8"))
        data.meta = rebase_paths(data.meta, mapping["remote_root"], directory)
        self.project.context.results["remote_"+record["token"]] = json.loads(diagnostics.read_text(encoding="utf-8"))
        self.result_name = self.project.add_dataset(data)
        self.log.appendPlainText("Opened "+self.result_name+". All predicted shots and diagnostics remain in "+str(directory))

    @QtCore.Slot(object)
    def operation_finished(self, value):
        action, result = value
        self.restore_controls()
        if action == "check":
            self.log.setPlainText(json.dumps(result, indent=2))
        else:
            self.reload_records(result); self.tabs.setCurrentIndex(2); self.selected_changed()
            if action == "fetch":
                try:
                    self.open_results(result)
                except Exception as exc:
                    self.log.appendPlainText("Results downloaded, but opening the dataset failed: "+str(exc))

    @QtCore.Slot(str)
    def failed(self, message):
        self.restore_controls(); self.reload_records()
        self.log.appendPlainText(message)
        self.activity.setText("Operation failed; saved jobs can be reopened and refreshed.")

    def restore_controls(self):
        self.worker = None
        self.form.setEnabled(True); self.submit_button.setEnabled(True)
        for button in self.job_buttons:
            button.setEnabled(True)
        self.activity.setText("Ready")

    def reject(self):
        if self.worker:
            self.worker.cancel()
            self.activity.setText("Stopping this connection; a submitted Slurm job continues. Close after the operation stops.")
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker:
            self.reject(); event.ignore()
        else:
            event.accept()
