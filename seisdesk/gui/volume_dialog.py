"""3D model creation, orthogonal slices and streamed survey job controls."""
import copy
import json

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core.volume import VelocityVolume
from ..workflow.flow import FlowStep
from .common import show_error, start_worker
from .flow_editor import ParamForm
from .plot import SeismicPlot
from .velocity_results import colorbar


class VolumeSlicesDialog(QtWidgets.QDialog):
    def __init__(self, volume, values=None, title="3D velocity", parent=None, *, units=None, sequential=False):
        super().__init__(parent)
        self.volume = volume
        self.values = volume.values if values is None else np.asarray(values)
        self.velocity = values is None
        self.sequential = self.velocity or sequential
        self.units = units or ("m/s" if self.velocity else "Relative amplitude")
        if self.values.shape != volume.values.shape:
            raise ValueError("slice values must have the same shape as the volume")
        self.setWindowTitle(title)
        self.resize(1150, 740)
        layout = QtWidgets.QGridLayout(self)
        self.plots = [SeismicPlot(y_label="Depth", y_units="m"),
                      SeismicPlot(y_label="Depth", y_units="m"),
                      SeismicPlot(y_label="Northing", y_units="m")]
        self.plots[2].getPlotItem().invertY(False)
        self.sliders, self.labels, self.bars = [], [], []
        finite = self.values[np.isfinite(self.values)]
        self.velocity_limits = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        if self.velocity_limits[0] == self.velocity_limits[1]:
            self.velocity_limits = (self.velocity_limits[0]-.5, self.velocity_limits[1]+.5)
        for index, (axis, label) in enumerate(((1, "y"), (2, "x"), (0, "z"))):
            plot = self.plots[index]
            plot.sigHover.connect(lambda horizontal, vertical, index=index: self.probe(index, horizontal, vertical))
            layout.addWidget(plot, 0, index)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, self.values.shape[axis]-1)
            slider.setValue(self.values.shape[axis]//2)
            slider.valueChanged.connect(self.update_slices)
            layout.addWidget(slider, 1, index)
            text = QtWidgets.QLabel(label)
            layout.addWidget(text, 2, index)
            self.sliders.append(slider); self.labels.append(text)
        self.readout = QtWidgets.QLabel("Move the pointer over a slice to inspect its position and value.")
        layout.addWidget(self.readout, 3, 0, 1, 3)
        self.update_slices()
        for plot in self.plots:
            self.bars.append(colorbar(plot, plot.image, plot.image.getLevels(),
                                      self.units, "Viridis" if self.sequential else "Seismic"))

    def update_slices(self):
        iy, ix, iz = [slider.value() for slider in self.sliders]
        z, y, x = self.volume.axes
        sections = (self.values[:, iy, :], self.values[:, :, ix], self.values[iz])
        extents = ((x, z), (y, z), (x, y))
        for index, (plot, values, (horizontal, vertical)) in enumerate(zip(self.plots, sections, extents)):
            if self.sequential:
                limits = self.velocity_limits
            else:
                finite = values[np.isfinite(values)]
                peak = max(float(np.percentile(np.abs(finite), 99)), 1e-20) if finite.size else 1.
                limits = (-peak, peak)
            dx, dy = horizontal[1]-horizontal[0], vertical[1]-vertical[0]
            plot.show_image(values, horizontal[0], dx, vertical[0], dy,
                            "Viridis" if self.sequential else "Seismic", limits)
            if self.bars:
                self.bars[index].setLevels(limits)
            plot.getPlotItem().setLabel("bottom", "Easting" if index != 1 else "Northing", units="m")
            plot.extent = (horizontal[0]-dx/2, horizontal[-1]+dx/2, vertical[0]-dy/2, vertical[-1]+dy/2)
            plot.reset_view()
        for label, text in zip(self.labels, (f"y = {y[iy]:g} m", f"x = {x[ix]:g} m", f"z = {z[iz]:g} m")):
            label.setText(text)

    def probe(self, index, horizontal, vertical):
        iy, ix, iz = [slider.value() for slider in self.sliders]
        coordinates = [iz, iy, ix]
        for axis, value in zip(((2, 0), (1, 0), (2, 1))[index], (horizontal, vertical)):
            coordinates[axis] = int(np.clip(round((value-self.volume.origin[axis])/self.volume.spacing[axis]),
                                            0, self.values.shape[axis]-1))
        z, y, x = [self.volume.origin[a]+coordinates[a]*self.volume.spacing[a] for a in range(3)]
        value = float(self.values[tuple(coordinates)])
        display = f"{value:.6g} {self.units}" if np.isfinite(value) else "uncovered / unavailable"
        self.readout.setText(f"Easting {x:g} m; northing {y:g} m; depth {z:g} m; value {display}")


class VolumeModelDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Create a 3D velocity model")
        self.resize(650, 650)
        form = QtWidgets.QFormLayout(self)
        self.fields = {}
        for name, default, label in (("shape", "101, 51, 101", "Nodes: nz, ny, nx"),
                                      ("spacing", "10, 10, 10", "Spacing: dz, dy, dx (m)"),
                                      ("origin", "0, 0, 0", "Origin: z0, y0, x0 (m)"),
                                      ("reference", "0", "Reference elevation (m)"),
                                      ("v0", "1500", "Velocity at z0 (m/s)"),
                                      ("gradient", "0.5", "Vertical velocity gradient (1/s)")):
            field = QtWidgets.QLineEdit(default)
            self.fields[name] = field
            form.addRow(label, field)
        self.boxes = QtWidgets.QPlainTextEdit("[]")
        self.boxes.setPlaceholderText('[{"z": [200, 400], "y": [100, 300], "x": [200, 500], "velocity": 2200}]')
        form.addRow("Optional velocity boxes (JSON, metres)", self.boxes)
        note = QtWidgets.QLabel("Boxes overwrite the gradient inside their bounds, in list order. "
                               "Coordinates must cover sources, receivers and the intended image. "
                               "The recording depth is reference elevation minus station elevation.")
        note.setWordWrap(True); form.addRow(note)
        buttons = QtWidgets.QDialogButtonBox()
        preview = buttons.addButton("Preview slices", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        save = buttons.addButton("Save model…", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        close = buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        preview.clicked.connect(self.preview); save.clicked.connect(self.save); close.clicked.connect(self.reject)
        form.addRow(buttons)

    def model(self):
        def vector(key, kind=float):
            values = tuple(kind(v.strip()) for v in self.fields[key].text().split(","))
            if len(values) != 3:
                raise ValueError(f"{key}: give three values in z,y,x order")
            return values
        shape, spacing, origin = vector("shape", int), vector("spacing"), vector("origin")
        if min(shape) < 3 or np.prod(shape) > 100_000_000:
            raise ValueError("model creation supports 3 to 100 million nodes; use a model file for larger volumes")
        if min(spacing) <= 0:
            raise ValueError("model spacings must be positive")
        z = np.arange(shape[0])*spacing[0]
        v0, gradient = float(self.fields["v0"].text()), float(self.fields["gradient"].text())
        values = np.broadcast_to((v0+gradient*z)[:, None, None], shape).astype(np.float32).copy()
        axes = [o+np.arange(n)*d for o, n, d in zip(origin, shape, spacing)]
        boxes = json.loads(self.boxes.toPlainText())
        if not isinstance(boxes, list):
            raise ValueError("boxes must be a JSON list")
        for box in boxes:
            masks = []
            for axis, key in zip(axes, ("z", "y", "x")):
                lo, hi = map(float, box[key])
                if lo > hi:
                    raise ValueError("box minimum must not exceed maximum")
                masks.append((axis >= lo) & (axis <= hi))
            values[np.ix_(*masks)] = float(box["velocity"])
        return VelocityVolume(values, spacing, origin, float(self.fields["reference"].text()))

    def preview(self):
        try:
            VolumeSlicesDialog(self.model(), parent=self).exec()
        except Exception as exc:
            show_error(self, "3D model", exc)

    def save(self):
        try:
            volume = self.model()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save 3D model", self.project.context.work_dir,
                                                           "3D velocity model (*.npz)")
            if path:
                if not path.lower().endswith(".npz"):
                    path += ".npz"
                volume.save(path)
                self.project.settings["volume_model"] = path
        except Exception as exc:
            show_error(self, "3D model", exc)


class VolumeJobDialog(QtWidgets.QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project, self.worker, self.result_name = project, None, None
        self.setWindowTitle("3D survey file job")
        self.resize(900, 850)
        layout = QtWidgets.QVBoxLayout(self)
        self.step = FlowStep("volume_file_job", dict(project.settings.get("volume_job", {})))
        if project.settings.get("volume_model"):
            self.step.params.setdefault("model", project.settings["volume_model"])
        self.form = ParamForm(project)
        self.form.set_step(self.step)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setWidget(self.form)
        layout.addWidget(scroll, 1)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True); self.log.setMaximumHeight(170)
        layout.addWidget(self.log)
        self.progress = QtWidgets.QProgressBar(); layout.addWidget(self.progress)
        row = QtWidgets.QHBoxLayout()
        self.run_button, self.cancel_button = QtWidgets.QPushButton("Run job"), QtWidgets.QPushButton("Cancel job")
        self.cancel_button.setEnabled(False)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(lambda: self.worker.cancel() if self.worker else None)
        row.addWidget(self.run_button); row.addWidget(self.cancel_button); layout.addLayout(row)

    def run(self):
        try:
            process = self.step.build()
            self.project.settings["volume_job"] = dict(process.values)
            self.form.setEnabled(False); self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
            def execute(worker):
                context = copy.copy(self.project.context)
                context.results = dict(context.results)
                context.log, context.progress, context.cancelled = worker.message.emit, worker.progress.emit, worker.cancelled
                output = process.run(None, context)
                output.add_history(process.describe())
                return output, context.results[process.name]
            self.worker = start_worker(self, execute, self.finished_job, self.failed_job,
                                        self.log.appendPlainText, self.update_progress)
        except Exception as exc:
            self.failed_job(str(exc))

    @QtCore.Slot(float)
    def update_progress(self, fraction):
        self.progress.setValue(round(100*fraction))

    def finished_job(self, result):
        output, diagnostics = result
        self.project.context.results["volume_file_job"] = diagnostics
        self.result_name = self.project.add_dataset(output)
        self.failed_job("Completed. Output is in the project; the job folder contains every predicted shot.")

    def failed_job(self, error):
        self.worker = None
        self.form.setEnabled(True); self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)
        self.log.appendPlainText(str(error))

    def reject(self):
        if self.worker is not None:
            self.worker.cancel()
            self.log.appendPlainText("Cancelling; close after the job stops.")
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel(); event.ignore()
        else:
            event.accept()
