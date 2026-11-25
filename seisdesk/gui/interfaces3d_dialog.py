"""Depth maps, orthogonal interface sections and conditional uncertainty."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.interfaces3d import save_interfaces3d
from .common import show_error
from .plot import SeismicPlot
from .velocity_results import colorbar


class Interfaces3DResultsDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("3D interface depths, contacts and uncertainty")
        self.resize(1200, 800)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("Depths are positive down in the velocity model datum. Blue: final ray-grid boundary; "
            "grey: initial boundary; dashed: conditional 5th/95th percentiles where available. "
            "Uncertainty is masked where the final rays have no interface contacts. Zero spread may reflect grid resolution.")
        note.setWordWrap(True); layout.addWidget(note)
        row = QtWidgets.QHBoxLayout(); self.boundary = QtWidgets.QComboBox()
        self.boundary.addItems(list(result["sampled_interfaces"])); row.addWidget(self.boundary, 1)
        self.field = QtWidgets.QComboBox()
        for label, key in (("Final depth (m)", "sampled_interfaces"), ("Initial depth (m)", "starting_sampled_interfaces"),
                           ("Depth change (m)", "change"), ("Interface contacts", "interface_hits"),
                           ("Depth standard deviation (m)", "interface_std"), ("Depth 5th percentile (m)", "interface_p05"),
                           ("Depth 95th percentile (m)", "interface_p95")):
            if key in result or key == "change":
                self.field.addItem(label, key)
        row.addWidget(self.field, 1); layout.addLayout(row)
        self.map = SeismicPlot(y_label="Northing", y_units="m")
        self.map.getPlotItem().invertY(False); self.map.getPlotItem().setLabel("bottom", "Easting", units="m")
        layout.addWidget(self.map, 2)
        graph = pg.GraphicsLayoutWidget(); layout.addWidget(graph, 1)
        self.sections = [graph.addPlot(row=0, col=i) for i in range(2)]
        for plot, axis in zip(self.sections, ("Easting", "Northing")):
            plot.setLabel("left", "Depth", units="m"); plot.setLabel("bottom", axis, units="m"); plot.invertY(True)
        slider_row = QtWidgets.QHBoxLayout(); self.sliders, self.labels = [], []
        shape = np.asarray(result["velocity"]).shape
        for axis in (1, 2):
            label = QtWidgets.QLabel(); slider_row.addWidget(label); self.labels.append(label)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); slider.setRange(0, shape[axis]-1)
            slider.setValue(shape[axis]//2); slider.valueChanged.connect(self.refresh)
            slider_row.addWidget(slider); self.sliders.append(slider)
        layout.addLayout(slider_row)
        self.summary = QtWidgets.QLabel(); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        buttons = QtWidgets.QHBoxLayout()
        surface = QtWidgets.QPushButton("Save fitted surface for reuse (.npy)…")
        surface.clicked.connect(self.save_surface); buttons.addWidget(surface)
        bundle = QtWidgets.QPushButton("Export all interfaces and uncertainty (.npz)…")
        bundle.clicked.connect(self.save_bundle); buttons.addWidget(bundle)
        layout.addLayout(buttons)
        self.bar = None
        self.boundary.currentIndexChanged.connect(self.refresh); self.field.currentIndexChanged.connect(self.refresh)
        self.refresh()
        self.bar = colorbar(self.map, self.map.image, self.map.image.getLevels(), self.field.currentText(), "Viridis")

    def refresh(self):
        name, key = self.boundary.currentText(), self.field.currentData()
        result = self.result
        values = (np.asarray(result["sampled_interfaces"][name])-result["starting_sampled_interfaces"][name]
                  if key == "change" else np.asarray(result[key][name]))
        finite = values[np.isfinite(values)]
        levels = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        if key in ("interface_hits", "interface_std"):
            levels = (0., levels[1] if levels[1] > 0 else 1.)
        elif levels[0] == levels[1]:
            levels = (levels[0]-.5, levels[1]+.5)
        _, y, x = result["axes"]
        self.map.show_image(values, x[0], x[1]-x[0], y[0], y[1]-y[0], "Viridis", levels)
        self.map.getPlotItem().setTitle(f"{name}: {self.field.currentText()}"); self.map.reset_view()
        if self.bar is not None:
            self.bar.setLevels(levels)
            self.bar.getAxis("left").setLabel(self.field.currentText())
        iy, ix = [s.value() for s in self.sliders]
        for index, (plot, coordinate) in enumerate(zip(self.sections, (x, y))):
            plot.clear()
            section = lambda array: np.asarray(array)[iy, :] if index == 0 else np.asarray(array)[:, ix]
            plot.plot(coordinate, section(result["starting_sampled_interfaces"][name]), pen=pg.mkPen("#888888", width=2))
            plot.plot(coordinate, section(result["sampled_interfaces"][name]), pen=pg.mkPen("#087eaf", width=2))
            for bound in ("interface_p05", "interface_p95"):
                if bound in result:
                    plot.plot(coordinate, section(result[bound][name]), connect="finite",
                              pen=pg.mkPen("#b64e28", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.labels[0].setText(f"y = {y[iy]:g} m"); self.labels[1].setText(f"x = {x[ix]:g} m")
        params = result["interface_parameters"][name]
        contacts = np.asarray(result["interface_hits"][name])
        self.summary.setText(f"{name}: phases {', '.join(params['phases'])}; controls {tuple(params['shape'])}; "
            f"{'updated' if params['active'] else 'fixed'}; {np.count_nonzero(contacts)} / {contacts.size} nodes with boundary contacts; "
            f"{np.count_nonzero(params['control_at_shift_bound'])} controls at the shift bound. "
            f"Depth-grid step {result['interface_grid_step_m']:g} m. {result['stop_reason']}.")

    def save_surface(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save fitted nominal interface for reuse", "interface.npy", "NumPy surface (*.npy)")
        if path:
            try:
                np.save(path if path.lower().endswith(".npy") else path+".npy",
                        self.result["interface_surfaces"][self.boundary.currentText()], allow_pickle=False)
            except Exception as exc:
                show_error(self, "Interface export", exc)

    def save_bundle(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save all interface diagnostics", "interfaces.npz", "NumPy bundle (*.npz)")
        if path:
            try:
                save_interfaces3d(self.result, path if path.lower().endswith(".npz") else path+".npz")
            except Exception as exc:
                show_error(self, "Interface diagnostics export", exc)
