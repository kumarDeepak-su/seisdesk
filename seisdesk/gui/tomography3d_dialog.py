"""3D arrival definitions and persisted model / per-pick diagnostic views."""
from copy import deepcopy
import json

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.volume import VelocityVolume
from ..processing.tomography3d import export_arrival_qc
from .common import show_error


class Branches3DDialog(QtWidgets.QDialog):
    def __init__(self, definitions, pickset=None, parent=None):
        super().__init__(parent)
        if not isinstance(definitions, dict):
            raise ValueError("arrival definitions must be a mapping")
        self.original = deepcopy(definitions)
        self.pickset = pickset
        self.names = list(dict.fromkeys([*definitions, *(pickset.phase_order if pickset else [])]))
        self.setWindowTitle("3D arrival paths and interface controls")
        self.resize(1200, 560)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("Select every phase to invert. A plane is defined by depth at the model's x0,y0 "
                               "and slopes dz/dx, dz/dy. A surface file overrides the plane: NumPy .npy, shape ny,nx, "
                               "depth in metres below the model reference elevation. Boundary updates are enabled "
                               "by Update selected interface depths in the processing form. "
                               "Head waves require faster velocity below the interface.")
        note.setWordWrap(True); layout.addWidget(note)
        self.table = QtWidgets.QTableWidget(len(self.names), 8)
        self.table.setHorizontalHeaderLabels(["Use", "Picked phase", "Path", "Uncertainty (ms)",
                                             "Plane depth (m)", "dz/dx", "dz/dy", "Surface file (optional)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(7, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for row, name in enumerate(self.names):
            definition = definitions.get(name, {})
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(QtCore.Qt.CheckState.Checked if name in definitions else QtCore.Qt.CheckState.Unchecked)
            if pickset is not None and name not in pickset.phase_order:
                use.setCheckState(QtCore.Qt.CheckState.Unchecked)
                use.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.table.setItem(row, 0, use)
            label = QtWidgets.QTableWidgetItem(name)
            label.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 1, label)
            kind = QtWidgets.QComboBox(); kind.addItems(["first", "reflection", "head"])
            default = "reflection" if pickset and pickset.phase_type(name) == "reflection" else "first"
            kind.setCurrentText(definition.get("kind", default)); self.table.setCellWidget(row, 2, kind)
            for col, key, value, bounds in ((3, "uncertainty_ms", 2, (.001, 10000)),
                                           (4, "depth", 50, (-1e6, 1e6)),
                                           (5, "slope_x", 0, (-1000, 1000)),
                                           (6, "slope_y", 0, (-1000, 1000))):
                field = QtWidgets.QDoubleSpinBox(); field.setRange(*bounds); field.setDecimals(4)
                field.setValue(float(definition.get(key, value))); self.table.setCellWidget(row, col, field)
            holder = QtWidgets.QWidget(); line = QtWidgets.QHBoxLayout(holder); line.setContentsMargins(0, 0, 0, 0)
            field = QtWidgets.QLineEdit(str(definition.get("surface_file", "")))
            field.setObjectName("surface_path")
            if "surface" in definition and "surface_file" not in definition:
                field.setText("[embedded surface array]"); field.setReadOnly(True)
            browse = QtWidgets.QToolButton(); browse.setText("…")
            def choose(_checked=False, editor=field):
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Interface depth surface", "", "Depth array (*.npy)")
                if path:
                    editor.setReadOnly(False); editor.setText(path)
            browse.clicked.connect(choose); line.addWidget(field); line.addWidget(browse)
            self.table.setCellWidget(row, 7, holder)
            kind.currentTextChanged.connect(lambda value, row=row: self.path_changed(row, value))
            self.path_changed(row, kind.currentText())
        self.tabs = QtWidgets.QTabWidget(); self.tabs.addTab(self.table, "Arrival paths and starting boundaries")
        self.interfaces = QtWidgets.QTableWidget(len(self.names), 6)
        self.interfaces.setHorizontalHeaderLabels(["Picked phase", "Shared interface ID", "Update", "Depth controls (y × x)",
                                                  "Minimum depth (m)", "Maximum depth (m)"])
        self.interfaces.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for row, name in enumerate(self.names):
            definition = definitions.get(name, {})
            label = QtWidgets.QTableWidgetItem(name)
            label.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.interfaces.setItem(row, 0, label)
            self.interfaces.setCellWidget(row, 1, QtWidgets.QLineEdit(str(definition.get("interface", name))))
            active = QtWidgets.QCheckBox(); active.setChecked(bool(definition.get("invert_interface", True)))
            self.interfaces.setCellWidget(row, 2, active)
            shape = QtWidgets.QComboBox()
            for n in range(1, 6):
                shape.addItem("Uniform shift" if n == 1 else f"{n} × {n}", [n, n])
            saved = list(definition.get("control_shape", [1, 1]))
            index = shape.findData(saved)
            if index < 0:
                shape.addItem(f"{saved[0]} × {saved[1]}", saved); index = shape.count()-1
            shape.setCurrentIndex(index); self.interfaces.setCellWidget(row, 3, shape)
            bounds = definition.get("depth_bounds", [-1e6, 1e6])
            for col, value in ((4, bounds[0]), (5, bounds[1])):
                field = QtWidgets.QDoubleSpinBox(); field.setRange(-1e6, 1e6); field.setDecimals(3)
                field.setValue(float(value)); self.interfaces.setCellWidget(row, col, field)
            self.path_changed(row, self.table.cellWidget(row, 2).currentText())
        self.tabs.addTab(self.interfaces, "Shared boundaries and depth bounds")
        layout.addWidget(self.tabs, 1)
        explanation = QtWidgets.QLabel("Phases with one shared ID must use identical geometry, controls, bounds and update selection. "
            "Controls shift the original surface: 1 × 1 translates it; larger grids permit bilinear deformation. "
            "Control shifts are bounded by the job's maximum shift. Boundary ordering and ray-grid depth resolution also apply.")
        explanation.setWordWrap(True); layout.addWidget(explanation)
        self.error = QtWidgets.QLabel(); self.error.setWordWrap(True); self.error.setStyleSheet("color: #b32020")
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)

    def path_changed(self, row, kind):
        for col in range(4, 8):
            self.table.cellWidget(row, col).setEnabled(kind != "first")
        if hasattr(self, "interfaces"):
            for col in range(1, 6):
                widget = self.interfaces.cellWidget(row, col)
                if widget is not None:
                    widget.setEnabled(kind != "first")

    def values(self):
        result = {}
        compatible = {"first": ("direct", "refraction", "unclassified"),
                      "reflection": ("reflection", "unclassified"), "head": ("refraction", "unclassified")}
        for row, name in enumerate(self.names):
            if self.table.item(row, 0).checkState() != QtCore.Qt.CheckState.Checked:
                continue
            kind = self.table.cellWidget(row, 2).currentText()
            if self.pickset and self.pickset.phase_type(name) not in compatible[kind]:
                raise ValueError(f"{name}: picked event type is incompatible with the selected path")
            definition = dict(kind=kind, uncertainty_ms=self.table.cellWidget(row, 3).value())
            if kind != "first":
                field = self.table.cellWidget(row, 7).findChild(QtWidgets.QLineEdit, "surface_path")
                filename = field.text().strip()
                if field.isReadOnly():
                    definition["surface"] = deepcopy(self.original[name]["surface"])
                elif filename:
                    definition["surface_file"] = filename
                else:
                    definition.update({key: self.table.cellWidget(row, col).value()
                                       for col, key in ((4, "depth"), (5, "slope_x"), (6, "slope_y"))})
                name_id = self.interfaces.cellWidget(row, 1).text().strip()
                low, high = (self.interfaces.cellWidget(row, col).value() for col in (4, 5))
                if not name_id or low >= high:
                    raise ValueError(f"{name}: give a shared interface ID and increasing depth bounds")
                definition.update(interface=name_id, invert_interface=self.interfaces.cellWidget(row, 2).isChecked(),
                                  control_shape=self.interfaces.cellWidget(row, 3).currentData(), depth_bounds=[low, high])
            result[name] = definition
        if not result:
            raise ValueError("select at least one arrival phase")
        return result

    def json_text(self):
        return json.dumps(self.values(), indent=2, default=lambda value: value.tolist())

    def accept(self):
        try:
            self.values()
        except (ValueError, KeyError) as exc:
            self.error.setText(str(exc)); return
        super().accept()


class ArrivalTableModel(QtCore.QAbstractTableModel):
    headers = ("Trace ID", "Phase", "Observed (ms)", "Predicted (ms)", "Sigma (ms)",
               "Residual (ms)", "Residual / sigma", "Over 3 sigma")

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.columns = (result["trace_ids"], result["phases"], 1000*np.asarray(result["observed"]),
                        1000*np.asarray(result["t_pred"]), 1000*np.asarray(result["uncertainty"]),
                        1000*np.asarray(result["residual"]), result["standardized_residual"], result["outlier"])

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.columns[0])

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self.headers[section] if orientation == QtCore.Qt.Orientation.Horizontal else str(section+1)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == QtCore.Qt.ItemDataRole.DisplayRole:
            value = self.columns[index.column()][index.row()]
            return f"{value:.5g}" if 2 <= index.column() <= 6 else str(value)


class Tomography3DResultsDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("3D tomography: model, coverage and arrival QC")
        self.resize(1150, 800)
        layout = QtWidgets.QVBoxLayout(self)
        text = (f"{len(result['observed'])} arrivals retained; RMS {result['rms'][0]*1000:.4g} → "
                f"{result['rms'][-1]*1000:.4g} ms; {int(np.count_nonzero(result['outlier']))} residuals over 3 sigma. "
                "A flag requests review; no picks are automatically removed. Residual = predicted − observed.")
        if "uncertainty_ensemble" in result:
            text += " Timing-noise ensemble: " + result["uncertainty_ensemble"]["assumption"] + ". Uncovered cells are masked."
        summary = QtWidgets.QLabel(text); summary.setWordWrap(True); layout.addWidget(summary)
        row = QtWidgets.QHBoxLayout(); self.field = QtWidgets.QComboBox()
        for name, key in (("Final velocity (m/s)", "velocity"), ("Starting velocity (m/s)", "starting_model"),
                          ("Arrival coverage (rays/node)", "hits"), ("Velocity standard deviation (m/s)", "velocity_std"),
                          ("Velocity 5th percentile (m/s)", "velocity_p05"), ("Velocity 95th percentile (m/s)", "velocity_p95")):
            if key in result:
                self.field.addItem(name, key)
        row.addWidget(self.field, 1)
        view = QtWidgets.QPushButton("Inspect orthogonal slices…"); view.clicked.connect(self.show_slices); row.addWidget(view)
        export = QtWidgets.QPushButton("Export arrival QC…"); export.clicked.connect(self.export_qc); row.addWidget(export)
        save = QtWidgets.QPushButton("Save velocity model…"); save.clicked.connect(self.save_model); row.addWidget(save)
        if "sampled_interfaces" in result:
            boundaries = QtWidgets.QPushButton("Inspect interfaces…")
            boundaries.clicked.connect(self.show_interfaces); row.addWidget(boundaries)
        layout.addLayout(row)
        graph = pg.GraphicsLayoutWidget(); layout.addWidget(graph, 1)
        residual = graph.addPlot(row=0, col=0, title="Arrival residuals by phase")
        residual.addLegend(); residual.setLabel("bottom", "Arrival index"); residual.setLabel("left", "Residual / sigma")
        for boundary in (-3, 3):
            residual.addItem(pg.InfiniteLine(pos=boundary, angle=0, pen=pg.mkPen("#777777", style=QtCore.Qt.PenStyle.DashLine)))
        phases = np.asarray(result["phases"])
        for number, phase in enumerate(dict.fromkeys(phases)):
            indices = np.flatnonzero(phases == phase)
            residual.plot(indices, np.asarray(result["standardized_residual"])[indices], pen=None, symbol="o",
                          symbolSize=5, symbolBrush=pg.intColor(number), name=str(phase))
        convergence = graph.addPlot(row=0, col=1, title="Weighted travel-time misfit")
        convergence.plot(result["weighted_misfit"], pen=pg.mkPen("#2474a8", width=2), symbol="o")
        convergence.setLabel("bottom", "Accepted update"); convergence.setLabel("left", "Mean squared residual / sigma")
        self.table = QtWidgets.QTableView(); self.table.setModel(ArrivalTableModel(result, self.table))
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

    def volume(self):
        r = self.result
        return VelocityVolume(r["velocity"], r["spacing"], r["origin"], r["reference_elevation"])

    def show_slices(self):
        from .volume_dialog import VolumeSlicesDialog
        key = self.field.currentData()
        VolumeSlicesDialog(self.volume(), values=self.result[key], title=self.field.currentText(), parent=self,
                           units="rays/node" if key == "hits" else "m/s", sequential=True).exec()

    def export_qc(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save all arrival residuals", "arrival-qc.csv", "CSV (*.csv)")
        if path:
            try:
                export_arrival_qc(self.result, path)
            except Exception as exc:
                show_error(self, "Arrival QC export", exc)

    def save_model(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save final 3D velocity", "velocity.npz", "Model (*.npz)")
        if path:
            try:
                self.volume().save(path if path.lower().endswith(".npz") else path+".npz")
            except Exception as exc:
                show_error(self, "3D model export", exc)

    def show_interfaces(self):
        from .interfaces3d_dialog import Interfaces3DResultsDialog
        Interfaces3DResultsDialog(self.result, self).exec()
