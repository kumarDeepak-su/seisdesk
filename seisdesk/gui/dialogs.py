"""Dialogs: import options, synthetic data, header table, spectra, velocity field, single processing step, text."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import synthetic
from ..core.spectra import amplitude_spectrum, autocorrelation, first_zero_crossing, spectrum_band, suggest_bandpass
from ..core.velocity import dix
from ..processing.base import REGISTRY
from ..workflow.flow import FlowStep
from .common import busy, fmt, show_error
from .flow_editor import ParamForm
from .plot import SeismicPlot


class ImportDialog(QtWidgets.QDialog):
    """File and reading options for SEG-Y / SU input."""

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import seismic data")
        self.path = QtWidgets.QLineEdit(path)
        self.order = QtWidgets.QComboBox()
        self.order.addItems(["big-endian (standard SEG-Y)", "little-endian"])
        if path.lower().endswith(".su"):
            self.order.setCurrentIndex(1)
        self.dt_ms = QtWidgets.QDoubleSpinBox()
        self.dt_ms.setRange(0.0, 1000.0)
        self.dt_ms.setDecimals(4)
        self.dt_ms.setSpecialValueText("from file")
        self.window = QtWidgets.QComboBox()
        self.window.addItems(["strict", "intersection", "union", "zero"])
        self.window.setToolTip("Strict requires a common input axis; intersection crops to shared time, union zero-pads, "
                               "zero starts at shot time zero. Alignment requires a target interval.")
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("dataset name (default: file name)")
        form = QtWidgets.QFormLayout(self)
        form.addRow("File", self.path)
        form.addRow("Byte order", self.order)
        form.addRow("Resample to interval (ms)", self.dt_ms)
        form.addRow("Time window", self.window)
        note = QtWidgets.QLabel("A nonzero interval resamples the samples with anti-alias filtering. "
                                "Original timing is retained. For mixed rates, delays or lengths, choose an alignment window.")
        note.setWordWrap(True)
        form.addRow(note)
        form.addRow("Name", self.name)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def options(self) -> dict:
        return {"path": self.path.text().strip(), "little": self.order.currentIndex() == 1,
                "dt": self.dt_ms.value() / 1000.0, "window": self.window.currentText(),
                "name": self.name.text().strip() or None}

    def accept(self):
        if self.window.currentText() != "strict" and self.dt_ms.value() <= 0:
            QtWidgets.QMessageBox.warning(self, "Target interval required", "Set a positive target sample interval for alignment.")
            return
        super().accept()


class SEG2ImportDialog(QtWidgets.QDialog):
    """Record timing choices are explicit because instrument SKEW conventions differ."""
    def __init__(self, count=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import {count} SEG-2 record{'s' if count != 1 else ''}")
        self.resize(510, 280)
        layout = QtWidgets.QVBoxLayout(self)
        self.align = QtWidgets.QCheckBox("Align each trace to shot time zero using DELAY")
        self.align.setChecked(True)
        self.descale = QtWidgets.QCheckBox("Apply DESCALING_FACTOR to samples")
        self.descale.setChecked(True)
        self.skew = QtWidgets.QCheckBox("Add SKEW to DELAY before alignment")
        explanation = QtWidgets.QLabel("Leave SKEW unchecked unless the instrument documentation says DELAY "
                                       "excludes the channel scan offset. Original timing and trace headers are "
                                       "retained in the project. Integer delays use exact sample shifts; fractional "
                                       "delays use windowed-sinc interpolation. Records end at the shortest trace.")
        explanation.setWordWrap(True)
        for widget in (self.align, self.descale, self.skew, explanation):
            layout.addWidget(widget)
        self.target_dt = QtWidgets.QDoubleSpinBox()
        self.target_dt.setDecimals(4); self.target_dt.setRange(0, 1000)
        self.target_dt.setSuffix(" ms"); self.target_dt.setSpecialValueText("original sample interval")
        self.window = QtWidgets.QComboBox()
        self.window.addItems(["zero", "intersection", "union"])
        options_form = QtWidgets.QFormLayout()
        options_form.addRow("Resample to", self.target_dt)
        options_form.addRow("Window when resampling", self.window)
        layout.addLayout(options_form)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)

    def options(self):
        out = dict(remove_delay=self.align.isChecked(), descale=self.descale.isChecked(),
                   apply_skew=self.skew.isChecked())
        if self.target_dt.value() > 0:
            out.update(target_dt=self.target_dt.value()/1000, window=self.window.currentText())
        return out


class SEGDImportDialog(ImportDialog):
    def __init__(self, path, parent=None):
        super().__init__(path, parent)
        self.setWindowTitle("Import SEG-D field records")
        self.order.setCurrentIndex(0); self.order.setEnabled(False)
        self.aux = QtWidgets.QCheckBox("Include auxiliary channels")
        self.descale = QtWidgets.QCheckBox("Apply MP descaling to millivolts")
        self.descale.setChecked(True)
        self.channel_set = QtWidgets.QSpinBox()
        self.channel_set.setRange(0, 65535); self.channel_set.setSpecialValueText("all seismic channel sets")
        form = self.layout()
        form.insertRow(form.rowCount()-1, "Channel set", self.channel_set)
        form.insertRow(form.rowCount()-1, self.descale)
        form.insertRow(form.rowCount()-1, self.aux)
        note = QtWidgets.QLabel("Rev 2.0/2.1 demultiplexed disk records with explicit sample counts. "
                                "Tape wrappers and proprietary extended descriptors need vendor conversion. "
                                "Assign surveyed coordinates using SPS after import.")
        note.setWordWrap(True)
        form.insertRow(form.rowCount()-1, note)
        self.resize(620, 440)

    def options(self):
        out = super().options()
        out["segd"] = dict(descale=self.descale.isChecked(), include_aux=self.aux.isChecked(),
                           channel_set=self.channel_set.value())
        return out


class SyntheticDialog(QtWidgets.QDialog):
    """Create a synthetic marine line (reflections, multiples, diffraction) or a land refraction line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Synthetic test data")
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Marine streamer line (reflections, multiples, diffractor)",
                            "Land refraction line (direct wave, refracted branches, statics)"])
        self.nshots = QtWidgets.QSpinBox()
        self.nshots.setRange(1, 2000)
        self.nshots.setValue(40)
        self.noise = QtWidgets.QDoubleSpinBox()
        self.noise.setRange(0.0, 5.0)
        self.noise.setDecimals(3)
        self.noise.setValue(0.02)
        self.statics = QtWidgets.QDoubleSpinBox()
        self.statics.setRange(0.0, 50.0)
        self.statics.setValue(8.0)
        self.statics.setSuffix(" ms")
        form = QtWidgets.QFormLayout(self)
        form.addRow("Type", self.kind)
        form.addRow("Shots", self.nshots)
        form.addRow("Noise (RMS)", self.noise)
        form.addRow("Random statics (land)", self.statics)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.kind.currentIndexChanged.connect(lambda i: self.nshots.setValue(40 if i == 0 else 12))

    def make(self):
        if self.kind.currentIndex() == 0:
            return synthetic.marine_line(nshots=self.nshots.value(), noise=self.noise.value(),
                                         diffractor=(3000.0, 900.0, 2000.0, 0.8))
        return synthetic.land_refraction_line(nshots=self.nshots.value(), noise=self.noise.value(),
                                              source_statics_ms=self.statics.value(),
                                              receiver_statics_ms=self.statics.value())


class HeaderModel(QtCore.QAbstractTableModel):
    def __init__(self, data, keys):
        super().__init__()
        self.d = data
        self.keys = keys

    def rowCount(self, parent=QtCore.QModelIndex()):          # noqa: N802 - Qt API
        return 0 if parent.isValid() else self.d.ntr

    def columnCount(self, parent=QtCore.QModelIndex()):       # noqa: N802
        return 0 if parent.isValid() else len(self.keys)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and index.isValid():
            return fmt(self.d.headers[self.keys[index.column()]][index.row()])
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):   # noqa: N802
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self.keys[section]
        return str(section + 1)


class HeaderTableDialog(QtWidgets.QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Trace headers: {data.name}")
        self.resize(1000, 600)
        self.d = data
        self.all_check = QtWidgets.QCheckBox("show headers that are zero everywhere")
        self.view = QtWidgets.QTableView()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.all_check)
        lay.addWidget(self.view, 1)
        self.all_check.toggled.connect(self._fill)
        self._fill(False)

    def _fill(self, show_all: bool) -> None:
        keys = [k for k in self.d.headers if show_all or np.any(self.d.headers[k] != 0)]
        self.model = HeaderModel(self.d, keys)
        self.view.setModel(self.model)


class SpectrumDialog(QtWidgets.QDialog):
    """Spectral analysis of the traces in view: average amplitude spectrum with its usable band and suggested
    band-pass corners, the autocorrelation (for deconvolution parameters) and the f-k spectrum."""

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Spectral analysis: {data.name}")
        self.resize(1100, 600)
        tabs = QtWidgets.QTabWidget()
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addWidget(tabs)
        self.tabs = tabs
        tr = data.traces.astype(np.float64)
        unit = "Hz" if data.domain == "time" else "cycles/m"
        f, db, _amp = amplitude_spectrum(tr, data.dt)
        peak = float(f[int(np.argmax(db))])
        b6 = spectrum_band(f, db, -6.0)
        b20 = spectrum_band(f, db, -20.0)
        corners = suggest_bandpass(f, db)
        self.result = {"peak": peak, "band_6db": b6, "band_20db": b20, "bandpass": corners}
        scale = 1000.0 if data.domain == "time" else 1.0
        tunit = "ms" if data.domain == "time" else "m"
        p1 = pg.PlotWidget()
        p1.plot(f, db, pen=pg.mkPen("b", width=1.5))
        for x, style in ((b20[0], QtCore.Qt.PenStyle.DashLine), (b20[1], QtCore.Qt.PenStyle.DashLine),
                         (peak, QtCore.Qt.PenStyle.SolidLine)):
            p1.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen((200, 0, 0), width=1, style=style)))
        p1.addItem(pg.InfiniteLine(pos=-20.0, angle=0, pen=pg.mkPen((150, 150, 150), style=QtCore.Qt.PenStyle.DotLine)))
        p1.setLabel("bottom", f"Frequency ({unit})")
        p1.setLabel("left", "Amplitude (dB re peak)")
        p1.showGrid(x=True, y=True, alpha=0.3)
        p1.setYRange(-80, 3)
        tabs.addTab(p1, "Amplitude spectrum")
        lags, acf = autocorrelation(tr, data.dt, max_lag=min(0.3 if data.domain == "time" else 300.0,
                                                             0.5 * (data.nt - 1) * data.dt))
        zc = first_zero_crossing(lags, acf)
        self.result["first_zero_crossing"] = zc
        p3 = pg.PlotWidget()
        p3.plot(lags * scale, acf, pen=pg.mkPen((0, 120, 0), width=1.5))
        p3.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen((150, 150, 150))))
        if zc is not None:
            p3.addItem(pg.InfiniteLine(pos=zc * scale, angle=90, pen=pg.mkPen((200, 0, 0),
                                                                                style=QtCore.Qt.PenStyle.DashLine)))
        p3.setLabel("bottom", f"Lag ({tunit})")
        p3.setLabel("left", "Autocorrelation (normalised)")
        p3.showGrid(x=True, y=True, alpha=0.3)
        tabs.addTab(p3, "Autocorrelation")
        zc_text = f"{zc * scale:.1f} {tunit}" if zc is not None else "none"
        self.info.setText(
            f"<b>{data.ntr} traces, {data.times[0] * scale:g}-{data.tmax * scale:g} {tunit}</b> (the traces and "
            f"window in view; zoom the viewer to analyse another part). Peak <b>{peak:.1f} {unit}</b>; "
            f"-6 dB band {b6[0]:.1f}-{b6[1]:.1f} {unit}; usable (-20 dB) band <b>{b20[0]:.1f}-{b20[1]:.1f} {unit}"
            f"</b>. Suggested band-pass corners <b>{', '.join(f'{c:g}' for c in corners)}</b>. Autocorrelation "
            f"first zero crossing {zc_text} (a predictive-deconvolution gap; operator length: where the "
            f"autocorrelation has died out).")
        if data.ntr >= 4:
            x = data.header("offset")
            if np.ptp(x) <= 0:
                x = data.header("cdpx")
            steps = np.abs(np.diff(x))
            steps = steps[steps > 0]
            dx = float(np.median(steps)) if steps.size else 1.0
            nk = int(2 ** np.ceil(np.log2(data.ntr)))
            nfft = int(2 ** np.ceil(np.log2(max(data.nt, 2))))
            fk_f = np.fft.rfftfreq(nfft, data.dt)
            fk = np.fft.fftshift(np.abs(np.fft.fft(np.fft.rfft(tr, nfft, axis=1), nk, axis=0)), axes=0)
            fk_db = 20 * np.log10(np.maximum(fk, 1e-12) / max(float(fk.max()), 1e-12))
            k = np.fft.fftshift(np.fft.fftfreq(nk, dx))
            p2 = SeismicPlot(y_label="Frequency", y_units="Hz")
            p2.getPlotItem().invertY(False)
            p2.show_image(fk_db.T, float(k[0]), float(k[1] - k[0]), 0.0, float(fk_f[1] - fk_f[0]), "Viridis",
                          (-60.0, 0.0))
            p2.getPlotItem().setLabel("bottom", "Wavenumber (cycles per metre)")
            p2.reset_view()
            tabs.addTab(p2, "f-k spectrum (dB)")


class VelocityFieldDialog(QtWidgets.QDialog):
    """RMS and Dix interval velocity fields interpolated from the picks over the CMPs of a dataset."""

    def __init__(self, picks, cdps, times, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Velocity field")
        self.resize(1100, 600)
        tabs = QtWidgets.QTabWidget()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs)
        cdps = np.unique(cdps)
        if cdps.size > 600:
            cdps = cdps[:: int(np.ceil(cdps.size / 600))]
        vrms = picks.field(cdps, times)
        vint = np.stack([dix(times, v) for v in vrms])
        dc = float(cdps[1] - cdps[0]) if cdps.size > 1 else 1.0
        for title, field in (("RMS velocity", vrms), ("Interval velocity (Dix)", vint)):
            lo, hi = (float(v) for v in np.percentile(field, [1.0, 99.0]))
            hi = hi if hi > lo else lo + 1.0
            p = SeismicPlot()
            p.show_image(field.T, float(cdps[0]), dc, float(times[0]), float(times[1] - times[0]), "Rainbow", (lo, hi))
            p.getPlotItem().setLabel("bottom", "CMP")
            try:                                          # colour bar: pyqtgraph >= 0.13
                bar = pg.ColorBarItem(values=(lo, hi), colorMap=pg.ColorMap(np.linspace(0, 1, 6), _rainbow()),
                                      label="m/s", interactive=False)
                bar.setImageItem(p.image, insert_in=p.getPlotItem())
            except Exception:                             # noqa: BLE001 - the field is still shown without it
                pass
            for c in picks.cdps:
                if cdps[0] <= c <= cdps[-1]:
                    p.addItem(pg.InfiniteLine(pos=c, angle=90, pen=pg.mkPen(0, 0, 0, 90)), ignoreBounds=True)
            p.reset_view()
            tabs.addTab(p, title)


def _rainbow():
    return np.array([(0, 0, 143), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)], dtype=np.ubyte)


class SingleStepDialog(QtWidgets.QDialog):
    """Run one processing step on a dataset; the result replaces it or becomes a new dataset."""

    def __init__(self, project, process: str, dataset: str, parent=None):
        super().__init__(parent)
        cls = REGISTRY[process]
        self.setWindowTitle(cls.label)
        self.project = project
        self.dataset = dataset
        self.step = FlowStep(process, {p.name: p.default for p in cls.params})
        self.form = ParamForm(project)
        self.form.input_data = lambda: self.project.datasets.get(self.dataset)
        data = project.datasets.get(dataset)
        self.form.header_keys = sorted(data.headers) if data is not None else []
        self.form.set_step(self.step)
        self.replace = QtWidgets.QCheckBox(f"replace '{dataset}' (otherwise a new dataset)")
        self.replace.setChecked(cls.category == "Geometry")
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.form, 1)
        lay.addWidget(self.replace)
        lay.addWidget(buttons)
        self.result_name = None

    def accept(self) -> None:
        data = self.project.datasets.get(self.dataset)
        if data is None:
            return super().reject()
        try:
            proc = self.step.build()
            logs = []
            ctx = self.project.context
            old_log = ctx.log
            ctx.log = logs.append
            try:
                with busy():
                    out = proc(data, ctx)
            finally:
                ctx.log = old_log
        except Exception as exc:                          # noqa: BLE001
            show_error(self, self.windowTitle(), exc)
            return
        if self.replace.isChecked():
            out.name = self.dataset
            self.project.datasets[self.dataset] = out
            self.result_name = self.dataset
        else:
            self.result_name = self.project.add_dataset(out, f"{self.dataset} > {proc.name}")
        self.logs = logs
        super().accept()


class TextDialog(QtWidgets.QDialog):
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 520)
        edit = QtWidgets.QPlainTextEdit(text)
        edit.setReadOnly(True)
        edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(edit)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)


class SurveyQCDialog(QtWidgets.QDialog):
    """Survey quality control: geometry map or elevation profile and stacking chart, fold and offset coverage per
    CMP, and trace RMS amplitude by shot and channel with the average per shot and per receiver."""

    def __init__(self, data, parent=None, tmin: float = 0.0, tmax: float = 0.0):
        super().__init__(parent)
        from ..core.qc import fold_per_cmp, grid_by, mean_by, trace_rms
        self.setWindowTitle(f"Survey QC: {data.name}")
        self.resize(1150, 700)
        tabs = QtWidgets.QTabWidget()
        self.tabs = tabs
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs)
        sx, gx = data.header("sx"), data.header("gx")
        red, blue = pg.mkBrush(200, 30, 30), pg.mkBrush(30, 60, 200)

        geo = QtWidgets.QWidget()
        gl = QtWidgets.QHBoxLayout(geo)
        p = pg.PlotWidget(title="Sources (red) and receivers (blue)")
        if np.ptp(data.header("sy")) > 0 or np.ptp(data.header("gy")) > 0:
            p.addItem(pg.ScatterPlotItem(gx, data.header("gy"), size=4, brush=blue, pen=None))
            p.addItem(pg.ScatterPlotItem(sx, data.header("sy"), size=7, brush=red, pen=None))
            p.setLabel("bottom", "x (m)")
            p.setLabel("left", "y (m)")
            p.setAspectLocked(True)
        else:
            p.addItem(pg.ScatterPlotItem(gx, data.header("gelev"), size=4, brush=blue, pen=None))
            p.addItem(pg.ScatterPlotItem(sx, data.header("selev"), size=7, brush=red, pen=None))
            p.setLabel("bottom", "position along the line (m)")
            p.setLabel("left", "elevation (m)")
        p.showGrid(x=True, y=True, alpha=0.3)
        gl.addWidget(p)
        p2 = pg.PlotWidget(title="Stacking chart (one point per trace)")
        p2.addItem(pg.ScatterPlotItem(gx, sx, size=3, brush=pg.mkBrush(0, 0, 0, 120), pen=None))
        p2.setLabel("bottom", "receiver x (m)")
        p2.setLabel("left", "source x (m)")
        p2.showGrid(x=True, y=True, alpha=0.3)
        gl.addWidget(p2)
        tabs.addTab(geo, "Geometry")

        fold_w = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(fold_w)
        if data.has_header("cdp"):
            cdp, fold = fold_per_cmp(data)
            p3 = pg.PlotWidget(title=f"Fold per CMP (max {int(fold.max())})")
            p3.plot(cdp, fold, pen=pg.mkPen((0, 100, 0), width=1.5))
            p3.setLabel("bottom", "CMP")
            p3.setLabel("left", "fold")
            p3.showGrid(x=True, y=True, alpha=0.3)
            fl.addWidget(p3)
            p4 = pg.PlotWidget(title="Offset coverage")
            p4.addItem(pg.ScatterPlotItem(data.header("cdp"), data.header("offset"), size=3,
                                          brush=pg.mkBrush(0, 0, 0, 120), pen=None))
            p4.setLabel("bottom", "CMP")
            p4.setLabel("left", "offset (m)")
            p4.showGrid(x=True, y=True, alpha=0.3)
            fl.addWidget(p4)
        else:
            fl.addWidget(QtWidgets.QLabel("No CMP numbers yet: run a geometry step."))
        tabs.addTab(fold_w, "Fold and offsets")

        amp_w = QtWidgets.QWidget()
        al = QtWidgets.QGridLayout(amp_w)
        rms = trace_rms(data, tmin, tmax)
        db = 20.0 * np.log10(np.maximum(rms, 1e-12) / max(float(np.median(rms[rms > 0])) if (rms > 0).any()
                                                         else 1.0, 1e-12))
        db[rms <= 0] = np.nan
        rows, cols, grid = grid_by(db, data.header("fldr"), data.header("tracf"))

        def grid_axis(keys):
            # the header values themselves when they are evenly spaced (FFID 1, 2, 3 ...), otherwise a running index
            step = np.diff(keys)
            if keys.size == 1 or (step.size and step[0] > 0 and np.allclose(step, step[0])):
                return float(keys[0]), float(step[0]) if step.size else 1.0, True
            return 0.0, 1.0, False

        y0, dy, y_keys = grid_axis(rows)
        x0, dx, x_keys = grid_axis(cols)
        p5 = SeismicPlot(y_label="FFID" if y_keys else "FFID (row number)", y_units="")
        p5.getPlotItem().invertY(True)
        # a fixed +/-20 dB scale: normal traces sit in the middle, dead ones (no amplitude) at the cold end,
        # noisy ones at the hot end
        p5.show_image(np.nan_to_num(np.clip(grid, -20.0, 20.0), nan=-20.0), x0, dx, y0, dy, "Rainbow",
                      (-20.0, 20.0))
        # a short title: a long one sets a minimum plot width and pushes the right of the grid out of view
        p5.getPlotItem().setTitle("Trace RMS, dB re median (blue dead, red noisy)")
        p5.getPlotItem().setLabel("bottom", "channel" if x_keys else "channel (column number)")
        ny, nx = grid.shape
        p5.extent = (x0 - dx / 2, x0 + (nx - 0.5) * dx, y0 - dy / 2, y0 + (ny - 0.5) * dy)   # whole edge cells
        p5.reset_view()
        al.addWidget(p5, 0, 0, 2, 1)
        shots, shot_db = mean_by(np.nan_to_num(db, nan=-120.0), data.header("fldr"))
        p6 = pg.PlotWidget(title="Average RMS per shot (dB)")
        p6.plot(shots, shot_db, pen=pg.mkPen((200, 30, 30), width=1.5), symbol="o", symbolSize=4)
        p6.setLabel("bottom", "FFID")
        p6.showGrid(x=True, y=True, alpha=0.3)
        al.addWidget(p6, 0, 1)
        recs, rec_db = mean_by(np.nan_to_num(db, nan=-120.0), gx)
        p7 = pg.PlotWidget(title="Average RMS per receiver position (dB)")
        p7.plot(recs, rec_db, pen=pg.mkPen((30, 60, 200), width=1.5), symbol="o", symbolSize=4)
        p7.setLabel("bottom", "receiver x (m)")
        p7.showGrid(x=True, y=True, alpha=0.3)
        al.addWidget(p7, 1, 1)
        tabs.addTab(amp_w, "Amplitudes")
        self.result = {"traces": data.ntr, "shots": int(np.unique(data.header("fldr")).size)}


class RefractionStaticsDialog(QtWidgets.QDialog):
    """Results of the delay-time refraction statics: surface, base of weathering and datum; delays and statics."""

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Refraction statics")
        self.resize(1050, 680)
        self.result = result
        x = np.asarray(result["x"])
        info = QtWidgets.QLabel(
            f"V1 (weathering) <b>{result['v1']:.0f} m/s</b>, V2 (refractor) <b>{result['v2']:.0f} m/s</b>, "
            f"replacement velocity {result['vr']:.0f} m/s, datum {result['datum']:g} m. Misfit of the first "
            f"breaks {result['rms_ms']:.2f} ms. Weathering thickness {min(result['thickness']):.1f}-"
            f"{max(result['thickness']):.1f} m; statics {min(result['static_ms']):.1f} to "
            f"{max(result['static_ms']):.1f} ms.")
        info.setWordWrap(True)
        p1 = pg.PlotWidget(title="Surface (black), base of weathering (blue), datum (dashed)")
        p1.plot(x, result["elevation"], pen=pg.mkPen("k", width=2))
        p1.plot(x, result["base_elevation"], pen=pg.mkPen((30, 60, 200), width=2))
        p1.addItem(pg.InfiniteLine(pos=result["datum"], angle=0,
                                   pen=pg.mkPen((120, 120, 120), style=QtCore.Qt.PenStyle.DashLine)))
        p1.setLabel("left", "elevation (m)")
        p1.showGrid(x=True, y=True, alpha=0.3)
        p2 = pg.PlotWidget(title="Statics to datum (red) and delay times (green), ms")
        p2.plot(x, result["static_ms"], pen=pg.mkPen((200, 30, 30), width=2))
        p2.plot(x, result["delay_ms"], pen=pg.mkPen((0, 130, 0), width=1.5))
        p2.setLabel("bottom", "station x (m)")
        p2.setLabel("left", "ms")
        p2.showGrid(x=True, y=True, alpha=0.3)
        p2.setXLink(p1)
        b_table = QtWidgets.QPushButton("Export station table (CSV)...")
        b_static = QtWidgets.QPushButton("Export x-static table...")
        b_table.clicked.connect(lambda: self.export(False))
        b_static.clicked.connect(lambda: self.export(True))
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(b_table)
        row.addWidget(b_static)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(p1, 1)
        lay.addWidget(p2, 1)
        lay.addLayout(row)

    def table_text(self, statics_only: bool) -> str:
        r = self.result
        if statics_only:
            return "x static_ms\n" + "\n".join(f"{x:.2f} {s:.3f}" for x, s in zip(r["x"], r["static_ms"]))
        lines = ["x,elevation,delay_ms,thickness,base_elevation,static_ms,picks"]
        for v in zip(r["x"], r["elevation"], r["delay_ms"], r["thickness"], r["base_elevation"], r["static_ms"],
                     r["picks"]):
            lines.append(",".join(f"{a:.3f}" for a in v[:-1]) + f",{int(v[-1])}")
        return "\n".join(lines)

    def export(self, statics_only: bool) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export", "statics.txt" if statics_only else "refraction_statics.csv",
            "Text (*.txt)" if statics_only else "CSV table (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.table_text(statics_only) + "\n")
