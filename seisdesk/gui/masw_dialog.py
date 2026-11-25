"""MASW window: phase-shift dispersion image of one shot, automatic and manual picking of the fundamental mode,
export of the picks and an approximate Vs profile (wavelength rule)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.masw import approximate_vs, dispersion_image, pick_curve
from .common import fmt, lookup_table

SIDES = ("positive offsets", "negative offsets", "both sides (|offset|)")


class MaswDialog(QtWidgets.QDialog):
    """Dispersion analysis of surface waves (Park, Miller & Xia 1998) on one shot gather of ``data``.

    Left click on the image adds a pick (replacing a pick at the same frequency); right click or shift + click removes
    the nearest pick."""

    def __init__(self, data, parent=None, project=None):
        super().__init__(parent)
        self.data = data
        self.project = project
        self.setWindowTitle(f"MASW dispersion analysis: {data.name}")
        self.resize(1250, 680)
        self.f = self.v = self.image = None
        self.picks: list[tuple[float, float]] = []
        self.key = "fldr" if data.has_header("fldr") else None
        off = data.header("offset")

        self.shot_combo = QtWidgets.QComboBox()
        self.shot_values = data.gather_values(self.key) if self.key else np.zeros(1)
        self.shot_combo.addItems([fmt(v) for v in self.shot_values] if self.key else ["all traces"])
        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItems(SIDES)
        self.side_combo.setCurrentIndex(0 if np.sum(off > 0) >= np.sum(off < 0) else 1)

        def spin(lo, hi, value, suffix, decimals=1, step=1.0):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setValue(value)
            s.setSuffix(suffix)
            return s

        amax = float(np.max(np.abs(off))) if off.size else 0.0
        nyq = 0.5 / data.dt
        self.omin = spin(0.0, 1e6, 0.0, " m")
        self.omax = spin(0.0, 1e6, max(amax, 1.0), " m")
        self.fmin = spin(0.1, nyq, min(2.0, nyq), " Hz")
        self.fmax = spin(0.5, nyq, min(60.0, nyq), " Hz")
        self.vmin = spin(1.0, 1e5, 50.0, " m/s", 0, 10.0)
        self.vmax = spin(2.0, 1e5, 1000.0, " m/s", 0, 50.0)
        b_compute = QtWidgets.QPushButton("Compute")
        b_auto = QtWidgets.QPushButton("Automatic picks")
        b_clear = QtWidgets.QPushButton("Clear picks")
        b_export = QtWidgets.QPushButton("Export picks (CSV)...")
        b_invert = QtWidgets.QPushButton("Invert layered Vs model…")
        b_invert.setEnabled(project is not None)
        b_invert.clicked.connect(self.invert_layers)
        b_compute.clicked.connect(self.compute)
        b_auto.clicked.connect(self.auto_pick)
        b_clear.clicked.connect(self.clear_picks)
        b_export.clicked.connect(lambda: self.export())
        self.info = QtWidgets.QLabel(" ")
        self.info.setWordWrap(True)

        form = QtWidgets.QFormLayout()
        form.addRow("Shot (FFID)", self.shot_combo)
        form.addRow("Traces", self.side_combo)
        form.addRow("Offset from", self.omin)
        form.addRow("Offset to", self.omax)
        form.addRow("Frequency from", self.fmin)
        form.addRow("Frequency to", self.fmax)
        form.addRow("Velocity from", self.vmin)
        form.addRow("Velocity to", self.vmax)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addLayout(form)
        for b in (b_compute, b_auto, b_clear, b_export, b_invert):
            lv.addWidget(b)
        hint = QtWidgets.QLabel("Click on the image to add a pick; right click (or Shift + click) removes the "
                                "nearest one.")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        lv.addWidget(self.info)
        lv.addStretch(1)
        left.setFixedWidth(270)

        self.image_plot = pg.PlotWidget(title="Dispersion image")
        pi = self.image_plot.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.setLabel("bottom", "frequency (Hz)")
        pi.setLabel("left", "phase velocity (m/s)")
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setLookupTable(lookup_table("Rainbow"))
        self.image_plot.addItem(self.image_item)
        self.pick_line = pg.PlotDataItem(pen=pg.mkPen("w", width=1.5))
        self.pick_dots = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(255, 255, 255), pen=pg.mkPen("k"))
        self.image_plot.addItem(self.pick_line)
        self.image_plot.addItem(self.pick_dots)
        self.image_plot.scene().sigMouseClicked.connect(self._clicked)

        self.vs_plot = pg.PlotWidget(title="Approximate Vs")
        vp = self.vs_plot.getPlotItem()
        vp.invertY(True)
        vp.setLabel("bottom", "Vs (m/s)")
        vp.setLabel("left", "depth (m)")
        vp.showGrid(x=True, y=True, alpha=0.3)
        self.vs_curve = self.vs_plot.plot([], [], pen=pg.mkPen((30, 60, 200), width=2), symbol="o", symbolSize=5)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self.image_plot)
        split.addWidget(self.vs_plot)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(left)
        lay.addWidget(split, 1)
        self.compute()

    def invert_layers(self):
        if not self.picks:
            self.info.setText("Pick a dispersion curve before starting layered inversion.")
            return
        from .surface_inversion_dialog import SurfaceInversionDialog
        picks=[dict(frequency_hz=f,velocity_ms=v,sigma_ms=10.,mode=0,wave="rayleigh",kind="phase") for f,v in self.picks]
        SurfaceInversionDialog(self.project,self.data.name,picks,self).exec()

    # ------------------------------------------------------------------ data
    def selected_traces(self) -> np.ndarray:
        """Trace indices of the chosen shot, side and offset range."""
        d = self.data
        if self.key:
            value = float(self.shot_values[max(0, self.shot_combo.currentIndex())])
            idx = d.gather_index(self.key, value, secondary=None)
        else:
            idx = np.arange(d.ntr)
        off = d.header("offset")[idx]
        side = self.side_combo.currentText()
        keep = off > 0 if side == SIDES[0] else (off < 0 if side == SIDES[1] else np.ones(idx.size, bool))
        a = np.abs(off)
        keep &= (a >= self.omin.value()) & (a <= self.omax.value())
        return idx[keep]

    def compute(self) -> bool:
        """Dispersion image of the selected traces; False (with a message) when it cannot be computed."""
        idx = self.selected_traces()
        if idx.size < 4:
            self.info.setText("Fewer than four traces in the selection: widen the offset range or change the side.")
            return False
        try:
            self.f, self.v, self.image = dispersion_image(
                self.data.traces[idx], self.data.header("offset")[idx], self.data.dt, self.fmin.value(),
                self.fmax.value(), self.vmin.value(), self.vmax.value(), 300)
        except ValueError as exc:
            self.info.setText(str(exc))
            return False
        df = float(self.f[1] - self.f[0]) if self.f.size > 1 else 1.0
        dv = float(self.v[1] - self.v[0])
        self.image_item.setImage(self.image.astype(np.float32), autoLevels=False, levels=(0.0, 1.0))
        self.image_item.setRect(QtCore.QRectF(float(self.f[0]) - df / 2, float(self.v[0]) - dv / 2,
                                              self.f.size * df, self.v.size * dv))
        self.image_plot.getPlotItem().vb.setRange(xRange=(float(self.f[0]), float(self.f[-1])),
                                                  yRange=(float(self.v[0]), float(self.v[-1])), padding=0.0)
        a = np.abs(self.data.header("offset")[idx])
        self.info.setText(f"{idx.size} traces, offsets {a.min():g}-{a.max():g} m, {self.f.size} frequencies.")
        self._redraw_picks()
        return True

    # ------------------------------------------------------------------ picks
    def auto_pick(self) -> int:
        if self.image is None and not self.compute():
            return 0
        f, v = pick_curve(self.image, self.f, self.v)
        self.picks = [(float(a), float(b)) for a, b in zip(f, v)]
        self._redraw_picks()
        return len(self.picks)

    def add_pick(self, f: float, v: float) -> None:
        half = 0.5 * float(self.f[1] - self.f[0]) if self.f is not None and self.f.size > 1 else 0.25
        self.picks = [p for p in self.picks if abs(p[0] - f) > half]
        self.picks.append((float(f), float(v)))
        self.picks.sort()
        self._redraw_picks()

    def remove_nearest(self, f: float, v: float) -> None:
        if not self.picks:
            return
        fr = max(float(self.fmax.value() - self.fmin.value()), 1e-9)
        vr = max(float(self.vmax.value() - self.vmin.value()), 1e-9)
        d = [((p[0] - f) / fr) ** 2 + ((p[1] - v) / vr) ** 2 for p in self.picks]
        self.picks.pop(int(np.argmin(d)))
        self._redraw_picks()

    def clear_picks(self) -> None:
        self.picks = []
        self._redraw_picks()

    def vs_profile(self):
        """Approximate (depth, Vs) from the picks."""
        if not self.picks:
            return np.zeros(0), np.zeros(0)
        p = np.array(self.picks)
        return approximate_vs(p[:, 0], p[:, 1])

    def _redraw_picks(self) -> None:
        if self.picks:
            p = np.array(self.picks)
            self.pick_line.setData(p[:, 0], p[:, 1])
            self.pick_dots.setData(p[:, 0], p[:, 1])
        else:
            self.pick_line.setData([], [])
            self.pick_dots.setData([], [])
        depth, vs = self.vs_profile()
        self.vs_curve.setData(vs, depth)

    def picks_table(self) -> str:
        lines = ["frequency_hz,phase_velocity_ms,wavelength_m,approx_depth_m,approx_vs_ms"]
        for f, v in self.picks:
            lam = v / f if f > 0 else float("nan")
            lines.append(f"{f:.3f},{v:.2f},{lam:.2f},{lam / 3.0:.2f},{1.09 * v:.2f}")
        return "\n".join(lines)

    def export(self, path: str | None = None) -> str | None:
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export dispersion picks", "dispersion_picks.csv",
                                                            "CSV table (*.csv)")
        if not path:
            return None
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.picks_table() + "\n")
        return path

    # ------------------------------------------------------------------ mouse
    def _clicked(self, ev) -> None:
        if self.image is None or ev.double():
            return
        vb = self.image_plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        shift = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if ev.button() == QtCore.Qt.MouseButton.RightButton or shift:
            self.remove_nearest(float(p.x()), float(p.y()))
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.add_pick(float(p.x()), float(p.y()))
