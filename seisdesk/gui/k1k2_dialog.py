"""Fourier spectrum of a regularly sampled two-dimensional panel."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .common import lookup_table


def panel_spectrum(data, dx):
    if dx <= 0 or data.ntr < 2 or data.nt < 2:
        raise ValueError("a panel needs two or more traces/samples and positive spatial spacing")
    amplitude = np.abs(np.fft.fftshift(np.fft.fft2(data.traces)))
    k1 = np.fft.fftshift(np.fft.fftfreq(data.ntr, dx))
    k2 = np.fft.fftshift(np.fft.fftfreq(data.nt, data.dt))
    return k1, k2, amplitude


class WavenumberSpectrumDialog(QtWidgets.QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("2D panel spectrum")
        self.resize(900, 650)
        self.dx = QtWidgets.QDoubleSpinBox(); self.dx.setRange(.000001, 1e9); self.dx.setDecimals(6)
        self.dx.setValue(1)
        spacing_note = "Spacing was not inferred: enter the actual spacing before interpreting wavenumber values (initial value 1 m)."
        for key in ("gx", "cdpx", "offset"):
            if key in data.headers:
                diff = np.diff(data.header(key))
                if len(diff) and np.all(diff > 0) and np.allclose(diff, np.median(diff), rtol=.02):
                    self.dx.setValue(float(np.median(diff)))
                    spacing_note = f"Spacing inferred from {key}; override it for a different spatial axis."
                    break
        self.plot = pg.PlotWidget()
        go = QtWidgets.QPushButton("Compute spectrum"); go.clicked.connect(self.compute)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Spatial spacing (m, regularly ordered input panel)"))
        row.addWidget(self.dx); row.addWidget(go)
        layout = QtWidgets.QVBoxLayout(self); layout.addLayout(row); layout.addWidget(self.plot)
        note = QtWidgets.QLabel(spacing_note); note.setWordWrap(True); layout.addWidget(note)
        layout.addWidget(QtWidgets.QLabel("Use one gather or stacked section. Mixed gathers and irregular geometry must be binned first."))
        self.compute()

    def compute(self):
        k1, k2, amplitude = self.result = panel_spectrum(self.data, self.dx.value())
        db = 20 * np.log10(np.maximum(amplitude / max(amplitude.max(), 1e-30), 1e-6))
        self.plot.clear()
        item = pg.ImageItem(db.T, axisOrder="row-major", levels=(-60, 0))
        item.setLookupTable(lookup_table("Viridis"))
        dk1, dk2 = k1[1] - k1[0], k2[1] - k2[0]
        item.setRect(QtCore.QRectF(k1[0] - dk1/2, k2[0] - dk2/2, dk1 * len(k1), dk2 * len(k2)))
        self.plot.addItem(item)
        self.plot.setLabel("bottom", "Spatial wavenumber", units="1/m")
        self.plot.setLabel("left", "Frequency" if self.data.domain == "time" else "Depth wavenumber",
                           units="Hz" if self.data.domain == "time" else "1/m")
        return self.result
