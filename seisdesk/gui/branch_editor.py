"""Named arrival paths and shared interface geometry, editable without writing JSON."""
from copy import deepcopy
import json

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


class JointBranchDialog(QtWidgets.QDialog):
    def __init__(self, definitions, pickset=None, model=None, parent=None, *, x_range=None, zmax=None):
        super().__init__(parent)
        if not isinstance(definitions, dict):
            raise ValueError("phase definitions must be a dictionary")
        self.setWindowTitle("Arrival paths and interfaces")
        self.resize(1160, 760)
        self.definitions = deepcopy(definitions)
        self.names = list(dict.fromkeys(list(definitions)+(list(pickset.phase_order) if pickset else [])))
        self._loading = True
        self._selected = None
        self._roi = None
        self._model = model
        self._xrange = (float(model.x[0]), float(model.x[-1])) if model is not None else (0., 1000.)
        self._zmax = float(model.z[-1]) if model is not None else 200.
        if x_range is not None:
            self._xrange = tuple(map(float, x_range))
        if zmax is not None:
            self._zmax = float(zmax)
        layout = QtWidgets.QVBoxLayout(self)
        explanation = QtWidgets.QLabel("Select the arrivals to use, then assign each one a path. "
                                       "Give phases the same interface name to move their boundary together. "
                                       "Edit depths in the table or drag the boundary handles below.")
        explanation.setWordWrap(True); layout.addWidget(explanation)
        self.table = QtWidgets.QTableWidget(len(self.names), 5)
        self.table.setHorizontalHeaderLabels(["Use", "Picked phase", "Path", "Uncertainty (ms)", "Interface"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMaximumHeight(200)
        for row, name in enumerate(self.names):
            default = "reflection" if pickset and pickset.phase_type(name) == "reflection" else "first"
            definition = self.definitions.setdefault(name, dict(kind=default, uncertainty_ms=2))
            definition.setdefault("uncertainty_ms", 2.)
            if definition.get("kind") != "first":
                definition.setdefault("interface", name)
                if "depth" not in definition and "points" not in definition:
                    definition["depth"] = .25*self._zmax
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(QtCore.Qt.CheckState.Checked if name in definitions else QtCore.Qt.CheckState.Unchecked)
            if pickset is not None and name not in pickset.phase_order:
                use.setCheckState(QtCore.Qt.CheckState.Unchecked)
                use.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                use.setToolTip("This phase is absent from the selected pick set")
            self.table.setItem(row, 0, use)
            phase = QtWidgets.QTableWidgetItem(name)
            phase.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 1, phase)
            kind = QtWidgets.QComboBox(); kind.addItems(["first", "reflection", "head"])
            kind.setCurrentText(definition.get("kind", "first"))
            kind.currentTextChanged.connect(lambda _v, r=row: self._row_changed(r))
            self.table.setCellWidget(row, 2, kind)
            sigma = QtWidgets.QDoubleSpinBox(); sigma.setRange(.001, 10000); sigma.setDecimals(3)
            sigma.setValue(definition.get("uncertainty_ms", 2))
            sigma.valueChanged.connect(lambda _v, r=row: self._row_changed(r))
            self.table.setCellWidget(row, 3, sigma)
            interface = QtWidgets.QLineEdit(str(definition.get("interface", name)))
            interface.editingFinished.connect(lambda r=row: self._row_changed(r))
            self.table.setCellWidget(row, 4, interface)
        layout.addWidget(self.table)
        splitter = QtWidgets.QSplitter()
        self.details = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.details)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["Flat depth", "Control points"])
        self.depth = QtWidgets.QDoubleSpinBox(); self.depth.setRange(0, 1_000_000); self.depth.setDecimals(3)
        self.depth.setSuffix(" m")
        self.update_depth = QtWidgets.QCheckBox("Allow this boundary to move")
        self.minimum = QtWidgets.QLineEdit(); self.minimum.setPlaceholderText("optional minimum depth, m")
        self.maximum = QtWidgets.QLineEdit(); self.maximum.setPlaceholderText("optional maximum depth, m")
        self.points = QtWidgets.QTableWidget(0, 2)
        self.points.setHorizontalHeaderLabels(["Line position (m)", "Depth (m)"])
        self.points.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.points.setMaximumHeight(190)
        form.addRow("Geometry", self.mode); form.addRow("Flat depth", self.depth)
        form.addRow("Depth updates", self.update_depth)
        form.addRow("Lower bound", self.minimum); form.addRow("Upper bound", self.maximum)
        form.addRow(self.points)
        buttons = QtWidgets.QWidget(); button_layout = QtWidgets.QHBoxLayout(buttons)
        add = QtWidgets.QPushButton("Insert midpoint"); remove = QtWidgets.QPushButton("Remove point")
        button_layout.addWidget(add); button_layout.addWidget(remove); form.addRow(buttons)
        note = QtWidgets.QLabel("Depth is below the highest ground elevation. Control points must span the ray grid. "
                              "Depth updates also need 'Update interface depths' enabled in the flow step.")
        note.setWordWrap(True); form.addRow(note)
        self.plot = pg.PlotWidget()
        self.plot.invertY(True)
        self.plot.setLabel("bottom", "Line position", units="m")
        self.plot.setLabel("left", "Depth", units="m")
        splitter.addWidget(self.details); splitter.addWidget(self.plot)
        splitter.setSizes([390, 720]); layout.addWidget(splitter, 1)
        self.error = QtWidgets.QLabel(); self.error.setStyleSheet("color: #b32020"); self.error.setWordWrap(True)
        layout.addWidget(self.error)
        controls = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                              QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(self.accept); controls.rejected.connect(self.reject); layout.addWidget(controls)
        self.table.currentCellChanged.connect(self._select)
        self.table.itemChanged.connect(lambda item: self._row_changed(item.row()))
        self.mode.currentTextChanged.connect(self._details_changed)
        self.depth.valueChanged.connect(self._details_changed)
        self.update_depth.toggled.connect(self._details_changed)
        self.minimum.editingFinished.connect(self._details_changed)
        self.maximum.editingFinished.connect(self._details_changed)
        self.points.itemChanged.connect(self._details_changed)
        add.clicked.connect(self._insert_point); remove.clicked.connect(self._remove_point)
        self._loading = False
        if self.names:
            self.table.setCurrentCell(0, 1)
        else:
            self.details.setEnabled(False)
            self.error.setText("Create named picks before configuring the inversion.")

    def _row_changed(self, row):
        if self._loading:
            return
        name = self.names[row]
        definition = self.definitions[name]
        kind = self.table.cellWidget(row, 2).currentText()
        interface = self.table.cellWidget(row, 4).text().strip() or name
        previous = definition.get("interface", name)
        definition.update(kind=kind, uncertainty_ms=self.table.cellWidget(row, 3).value())
        if kind != "first":
            definition["interface"] = interface
            if interface != previous:
                for phase, other in self.definitions.items():
                    if phase != name and other.get("kind") != "first" and other.get("interface", phase) == interface:
                        for key in ("depth", "points", "depth_bounds", "invert_depth"):
                            definition.pop(key, None)
                            if key in other:
                                definition[key] = deepcopy(other[key])
                        break
            if definition.get("depth") is None and "points" not in definition:
                definition["depth"] = .25*self._zmax
        if self._selected == row:
            self._load_details()
        self._redraw()

    def _select(self, row, _column, _oldrow, _oldcolumn):
        if row < 0:
            return
        self._selected = row
        self._load_details(); self._redraw()

    def _load_details(self):
        if self._selected is None:
            return
        definition = self.definitions[self.names[self._selected]]
        self._loading = True
        self.details.setEnabled(definition.get("kind") != "first")
        flat = definition.get("depth") is not None
        self.mode.setCurrentText("Flat depth" if flat else "Control points")
        self.depth.setValue(definition.get("depth", .25*self._zmax) or .25*self._zmax)
        self.depth.setEnabled(flat)
        self.update_depth.setChecked(definition.get("invert_depth", True))
        bounds = definition.get("depth_bounds", [None, None])
        self.minimum.setText("" if bounds[0] is None else str(bounds[0]))
        self.maximum.setText("" if bounds[1] is None else str(bounds[1]))
        points = definition.get("points", [[self._xrange[0], self.depth.value()], [self._xrange[1], self.depth.value()]])
        self.points.setRowCount(len(points)); self.points.setEnabled(not flat)
        for row, point in enumerate(points):
            for column, value in enumerate(point):
                self.points.setItem(row, column, QtWidgets.QTableWidgetItem(f"{value:.12g}"))
        self._loading = False

    def _point_values(self):
        try:
            points = np.array([[float(self.points.item(r, c).text()) for c in range(2)]
                               for r in range(self.points.rowCount())])
        except (AttributeError, ValueError):
            raise ValueError("Enter a number for every control-point coordinate") from None
        if len(points) < 2 or len(points) > 20 or not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) <= 0):
            raise ValueError("Use 2–20 finite control points with increasing line positions")
        if np.any(points[:, 1] <= 0):
            raise ValueError("Interface depths must be positive")
        return points.tolist()

    def _details_changed(self, *_):
        if self._loading or self._selected is None:
            return
        name = self.names[self._selected]
        definition = deepcopy(self.definitions[name])
        if definition.get("kind") == "first":
            return
        try:
            flat = self.mode.currentText() == "Flat depth"
            if flat:
                definition.pop("points", None); definition["depth"] = self.depth.value()
            else:
                definition.pop("depth", None); definition["points"] = self._point_values()
            low, high = self.minimum.text().strip(), self.maximum.text().strip()
            definition.pop("depth_bounds", None)
            if low or high:
                bounds = [float(low), float(high)]
                if not np.isfinite(bounds).all() or bounds[0] < 0 or bounds[0] >= bounds[1]:
                    raise ValueError("Depth bounds must be finite, nonnegative and increasing")
                definition["depth_bounds"] = bounds
            definition["invert_depth"] = self.update_depth.isChecked()
            self.definitions[name] = definition
            for phase, other in self.definitions.items():
                if phase == name or other.get("kind") == "first" or other.get("interface", phase) != definition.get("interface", name):
                    continue
                for key in ("depth", "points", "depth_bounds", "invert_depth"):
                    other.pop(key, None)
                    if key in definition:
                        other[key] = deepcopy(definition[key])
            self.depth.setEnabled(flat); self.points.setEnabled(not flat)
            self.error.clear(); self._redraw()
        except ValueError as exc:
            self.error.setText(str(exc))

    def _insert_point(self):
        if self._selected is None or self.mode.currentText() != "Control points":
            return
        try:
            points = self._point_values()
        except ValueError as exc:
            self.error.setText(str(exc)); return
        if len(points) >= 20:
            return
        row = max(0, min(self.points.currentRow(), len(points)-2))
        points.insert(row+1, np.mean(points[row:row+2], axis=0).tolist())
        self._set_points(points)

    def _remove_point(self):
        if self.points.rowCount() <= 2 or self.mode.currentText() != "Control points":
            return
        row = self.points.currentRow()
        if row >= 0:
            self._loading = True; self.points.removeRow(row); self._loading = False
            self._details_changed()

    def _set_points(self, points):
        self._loading = True
        self.points.setRowCount(len(points))
        for row, point in enumerate(points):
            for column, value in enumerate(point):
                self.points.setItem(row, column, QtWidgets.QTableWidgetItem(f"{value:.12g}"))
        self._loading = False
        self._details_changed()

    def _redraw(self):
        self.plot.clear(); self._roi = None
        if self._model is not None:
            image = pg.ImageItem(self._model.values, axisOrder="row-major")
            image.setRect(QtCore.QRectF(self._model.x[0]-.5*self._model.dx, self._model.z[0]-.5*self._model.dz,
                                       len(self._model.x)*self._model.dx, len(self._model.z)*self._model.dz))
            image.setOpacity(.25); self.plot.addItem(image)
        seen = set()
        selected = self.names[self._selected] if self._selected is not None else None
        target = self.definitions.get(selected, {}).get("interface", selected)
        for row, name in enumerate(self.names):
            definition = self.definitions[name]
            if definition.get("kind") == "first" or self.table.item(row, 0).checkState() != QtCore.Qt.CheckState.Checked:
                continue
            interface = definition.get("interface", name)
            if interface in seen:
                continue
            seen.add(interface)
            colour = pg.intColor(row)
            if definition.get("depth") is not None:
                if interface == target:
                    roi = pg.InfiniteLine(definition["depth"], angle=0, movable=True, pen=pg.mkPen(colour, width=3),
                                          label=interface+" {value:.1f} m", bounds=[.001, None])
                    roi.sigPositionChangeFinished.connect(lambda line: self.depth.setValue(line.value()))
                    self.plot.addItem(roi); self._roi = roi
                else:
                    self.plot.plot(self._xrange, [definition["depth"]]*2, pen=pg.mkPen(colour, width=2))
            else:
                points = np.asarray(definition["points"])
                if interface == target:
                    roi = pg.PolyLineROI(points, closed=False, pen=pg.mkPen(colour, width=3), movable=False)
                    roi.sigRegionChangeFinished.connect(self._drag_points)
                    self.plot.addItem(roi); self._roi = roi
                else:
                    self.plot.plot(points[:, 0], points[:, 1], pen=pg.mkPen(colour, width=2))
        self.plot.setXRange(*self._xrange, padding=.03)
        depths = [float(d.get("depth", 0) or 0) for d in self.definitions.values()]
        depths += [float(p[1]) for d in self.definitions.values() for p in d.get("points", [])]
        extent = max([self._zmax]+depths)
        self.plot.setYRange(-.03*extent, extent*1.05, padding=0)

    def _drag_points(self, roi):
        points = [roi.mapToParent(position) for _, position in roi.getLocalHandlePositions()]
        self._set_points([[p.x(), p.y()] for p in points])

    def values(self):
        self._details_changed()
        if self.error.text():
            raise ValueError(self.error.text())
        result = {}
        for row, name in enumerate(self.names):
            if self.table.item(row, 0).checkState() != QtCore.Qt.CheckState.Checked:
                continue
            definition = deepcopy(self.definitions[name])
            if definition.get("kind") == "first":
                definition = {k: definition[k] for k in ("kind", "uncertainty_ms")}
            else:
                depths = [definition["depth"]] if definition.get("depth") is not None else np.asarray(definition["points"])[:, 1]
                if not np.isfinite(depths).all() or min(depths) <= 0:
                    raise ValueError(f"{name}: interface depths must be positive and finite")
                if "depth_bounds" in definition:
                    low, high = definition["depth_bounds"]
                    if min(depths) < low or max(depths) > high:
                        raise ValueError(f"{name}: starting depths lie outside their bounds")
            result[name] = definition
        if not result:
            raise ValueError("Select at least one arrival phase")
        return result

    def accept(self):
        try:
            self.values()
        except ValueError as exc:
            self.error.setText(str(exc)); return
        super().accept()

    def json_text(self):
        return json.dumps(self.values(), indent=2)
