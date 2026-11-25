"""Time-frequency analysis window: the S-transform, Gabor transform or Morlet wavelet transform of one trace, as an
amplitude image (time down, frequency across) next to the trace itself."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.timefreq import METHODS, cwt_morlet, gabor, stransform
from .plot import SeismicPlot


class TimeFrequencyDialog(QtWidgets.QDialog):
    """Time-frequency amplitude of one trace of a dataset; trace, method and parameters can be changed live."""

    NF = 128

    def __init__(self, data, parent=None, trace: int = 0):
        super().__init__(parent)
        self.data = data
        self.result = None
        self.setWindowTitle(f"Time-frequency analysis: {data.name}")
        self.resize(1100, 700)
        nyq = 0.5 / data.dt
        depth = data.domain == "depth"
        self.trace_spin = QtWidgets.QSpinBox()
        self.trace_spin.setRange(0, max(data.ntr - 1, 0))
        self.trace_spin.setValue(int(np.clip(trace, 0, max(data.ntr - 1, 0))))
        self.method = QtWidgets.QComboBox()
        self.method.addItems(METHODS)
        self.window = QtWidgets.QDoubleSpinBox()
        self.window.setDecimals(3)
        self.window.setRange(0.01, 5.0)
        self.window.setSingleStep(0.02)
        self.window.setValue(0.1)
        self.window.setSuffix(" m" if depth else " s")
        self.w0 = QtWidgets.QDoubleSpinBox()
        self.w0.setDecimals(1)
        self.w0.setRange(3.0, 20.0)
        self.w0.setValue(6.0)
        self.fmax = QtWidgets.QDoubleSpinBox()
        self.fmax.setDecimals(3 if depth else 1)
        self.fmax.setRange(0.001, nyq)
        self.fmax.setValue(min(nyq, 0.05 if depth else 100.0))
        self.fmax.setSuffix(" 1/m" if depth else " Hz")
        compute = QtWidgets.QPushButton("Compute")
        compute.clicked.connect(self.refresh)
        controls = QtWidgets.QHBoxLayout()
        for label, widget in (("Trace", self.trace_spin), ("Method", self.method), ("STFT window", self.window),
                              ("Morlet w0", self.w0), ("Highest frequency", self.fmax)):
            controls.addWidget(QtWidgets.QLabel(label))
            controls.addWidget(widget)
        controls.addWidget(compute)
        controls.addStretch(1)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        y_label, y_units = ("Depth", "m") if depth else ("Time", "s")
        self.trace_plot = pg.PlotWidget(title="Trace")
        self.trace_plot.invertY(True)
        self.trace_plot.setLabel("left", y_label, units=y_units)
        self.trace_plot.showGrid(x=False, y=True, alpha=0.25)
        self.tf_plot = SeismicPlot(y_label=y_label, y_units=y_units)
        self.tf_plot.getPlotItem().setLabel("bottom", "Wavenumber (1/m)" if depth else "Frequency (Hz)")
        self.tf_plot.setYLink(self.trace_plot)
        plots = QtWidgets.QHBoxLayout()
        plots.addWidget(self.trace_plot, 1)
        plots.addWidget(self.tf_plot, 4)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(controls)
        lay.addWidget(self.info)
        lay.addLayout(plots, 1)
        self.trace_spin.valueChanged.connect(lambda _v: self.refresh())
        self.method.currentIndexChanged.connect(lambda _i: self._method_changed())
        for spin in (self.window, self.w0, self.fmax):
            spin.editingFinished.connect(self.refresh)
        self._method_changed()

    def _method_changed(self) -> None:
        method = self.method.currentText()
        self.window.setEnabled(method.startswith("STFT"))
        self.w0.setEnabled(method.startswith("CWT"))
        self.refresh()

    @QtCore.Slot()
    def refresh(self) -> None:
        try:
            self.compute()
        except Exception as exc:                               # noqa: BLE001 - shown in the window, not raised
            self.info.setText(f"<b>Could not compute:</b> {exc}")

    def compute(self) -> dict:
        """Compute and display the map of the selected trace; returns times, freqs, amplitude (nt, nf) and the peak."""
        d = self.data
        if d.ntr == 0:
            raise ValueError("the dataset has no traces")
        i = int(self.trace_spin.value())
        tr = d.traces[i].astype(np.float64)
        method = self.method.currentText()
        fmax = min(float(self.fmax.value()), 0.5 / d.dt)
        if method.startswith("S-"):
            times, freqs, m = stransform(tr, d.dt, 0.0, fmax, self.NF, d.t0)
        elif method.startswith("STFT"):
            times, freqs, m = gabor(tr, d.dt, float(self.window.value()), 0.0, fmax, self.NF, d.t0)
        else:
            times, freqs, m = cwt_morlet(tr, d.dt, float(self.w0.value()), 0.0, fmax, self.NF, d.t0)
        amp = np.abs(m)
        top = float(amp.max()) if amp.size else 0.0
        df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
        self.tf_plot.show_image(amp, float(freqs[0]), df, float(times[0]), d.dt, "Rainbow",
                                (0.0, top if top > 0 else 1.0))
        self.tf_plot.reset_view()
        self.trace_plot.clear()
        self.trace_plot.plot(tr, times, pen=pg.mkPen((0, 0, 0), width=1))
        self.trace_plot.setYRange(float(times[0]), float(times[-1]), padding=0.0)
        it, jf = np.unravel_index(int(np.argmax(amp)), amp.shape)
        where = ", ".join(f"{k} {d.header(k)[i]:g}" for k in ("fldr", "cdp", "offset", "tracf") if d.has_header(k))
        unit = "m" if d.domain == "depth" else "s"
        funit = "1/m" if d.domain == "depth" else "Hz"
        self.info.setText(f"Trace {i}{' (' + where + ')' if where else ''}: {method}. Strongest amplitude at "
                          f"{times[it]:.3f} {unit}, {freqs[jf]:.3g} {funit}.")
        self.result = {"times": times, "freqs": freqs, "amplitude": amp,
                       "peak": (float(times[it]), float(freqs[jf]))}
        return self.result
