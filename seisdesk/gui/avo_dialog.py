"""AVO crossplot window: intercept against gradient for a time window, as a density image, with the background trend
and the class quadrants (Castagna & Swan 1997)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .common import lookup_table


def _panel(values, dt, t0):
    """(traces (ntr, nt), times) from a SeismicData or an array with its sample interval."""
    if hasattr(values, "traces"):
        return np.asarray(values.traces, dtype=np.float64), values.times
    arr = np.asarray(values, dtype=np.float64)
    if dt is None:
        raise ValueError("give the sample interval with array input")
    return arr, t0 + np.arange(arr.shape[1]) * dt


class AvoCrossplotDialog(QtWidgets.QDialog):
    """Intercept (x) against gradient (y) of every sample in a time window, coloured by point density."""

    BINS = 120

    def __init__(self, intercept, gradient, parent=None, dt: float | None = None, t0: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle("AVO crossplot")
        self.resize(900, 680)
        self.a, self.times = _panel(intercept, dt, t0)
        self.b, _ = _panel(gradient, dt, t0)
        if self.a.shape != self.b.shape:
            raise ValueError("intercept and gradient sections must have the same shape")
        self.result = None
        self.tmin = QtWidgets.QDoubleSpinBox()
        self.tmax = QtWidgets.QDoubleSpinBox()
        for s, v in ((self.tmin, float(self.times[0])), (self.tmax, float(self.times[-1]))):
            s.setRange(float(self.times[0]), float(self.times[-1]))
            s.setDecimals(3)
            s.setSingleStep(0.05)
            s.setSuffix(" s")
            s.setValue(v)
        self.update_btn = QtWidgets.QPushButton("Update")
        self.update_btn.clicked.connect(self.refresh)
        row = QtWidgets.QHBoxLayout()
        for label, w in (("Time from", self.tmin), ("to", self.tmax)):
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(w)
        row.addWidget(self.update_btn)
        row.addStretch(1)
        self.plot = pg.PlotWidget(title="A-B crossplot")
        self.plot.setLabel("bottom", "intercept A")
        self.plot.setLabel("left", "gradient B")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setLookupTable(lookup_table("Viridis"))
        self.plot.addItem(self.image)
        dash = pg.mkPen((120, 120, 120), style=QtCore.Qt.PenStyle.DashLine)
        self.plot.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=dash))
        self.plot.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=dash))
        self.trend = self.plot.plot([], [], pen=pg.mkPen((220, 40, 40), width=2))
        self.labels = []
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(self.plot, 1)
        lay.addWidget(self.info)
        self.refresh()

    # ------------------------------------------------------------------ computation
    def compute(self) -> dict:
        """Samples in the time window where intercept or gradient is non-zero: point count, the background trend
        B = slope A + offset (least squares) and the correlation coefficient."""
        lo, hi = sorted((self.tmin.value(), self.tmax.value()))
        cols = (self.times >= lo - 1e-9) & (self.times <= hi + 1e-9)
        a = self.a[:, cols].ravel()
        b = self.b[:, cols].ravel()
        keep = np.isfinite(a) & np.isfinite(b) & ((a != 0) | (b != 0))
        a, b = a[keep], b[keep]
        res = {"n": int(a.size), "slope": 0.0, "offset": 0.0, "r": 0.0, "a": a, "b": b}
        if a.size >= 3 and np.ptp(a) > 0:
            va = float(np.var(a))
            cov = float(np.mean((a - a.mean()) * (b - b.mean())))
            res["slope"] = cov / va
            res["offset"] = float(b.mean() - res["slope"] * a.mean())
            sb = float(np.std(b))
            res["r"] = cov / (np.sqrt(va) * sb) if sb > 0 else 0.0
        self.result = res
        return res

    def refresh(self) -> None:
        res = self.compute()
        for item in self.labels:
            self.plot.removeItem(item)
        self.labels = []
        a, b = res["a"], res["b"]
        if res["n"] < 3:
            self.image.clear()
            self.trend.setData([], [])
            self.info.setText("Fewer than three non-zero samples in the time window.")
            return
        ax = np.percentile(a, [0.5, 99.5])
        bx = np.percentile(b, [0.5, 99.5])
        ra = max(abs(ax[0]), abs(ax[1]), 1e-12) * 1.1
        rb = max(abs(bx[0]), abs(bx[1]), 1e-12) * 1.1
        h, _, _ = np.histogram2d(a, b, bins=self.BINS, range=[[-ra, ra], [-rb, rb]])
        img = np.log1p(h.T)                                     # rows = gradient (y), columns = intercept (x)
        self.image.setImage(img, autoLevels=False, levels=(0.0, max(float(img.max()), 1e-9)))
        self.image.setRect(QtCore.QRectF(-ra, -rb, 2 * ra, 2 * rb))
        xs = np.array([-ra, ra])
        self.trend.setData(xs, res["slope"] * xs + res["offset"])
        for text, x, y in (("I", 0.6, -0.6), ("III", -0.6, -0.6), ("IV", -0.6, 0.6), ("II", 0.05, -0.9)):
            t = pg.TextItem(text, color=(200, 40, 40), anchor=(0.5, 0.5))
            t.setPos(x * ra, y * rb)
            self.plot.addItem(t)
            self.labels.append(t)
        self.plot.setRange(xRange=(-ra, ra), yRange=(-rb, rb), padding=0.02)
        self.info.setText(
            f"{res['n']} samples between {min(self.tmin.value(), self.tmax.value()):.3f} and "
            f"{max(self.tmin.value(), self.tmax.value()):.3f} s. Background trend (red): B = {res['slope']:.3f} A "
            f"{res['offset']:+.4g}, correlation {res['r']:.2f}. Samples far from the trend are AVO anomalies: lower "
            "left (A and B negative) is class III, upper left (A negative, B positive) class IV, near A = 0 class II, "
            "lower right (A positive, B negative) class I.")
