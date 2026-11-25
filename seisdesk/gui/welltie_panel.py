"""Wavelet, time-tie, deviation and AVO controls for the LAS dialog."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import deviation as dev, welltie as wt, welllog as wl
from ..core.synthetic import ricker
from .common import lookup_table


def spin(value, low, high):
    box = QtWidgets.QDoubleSpinBox()
    box.setDecimals(3); box.setRange(low, high); box.setValue(value)
    return box


def image_panel(plot, values, x0, dx, t0, dt):
    plot.clear()
    clip = max(float(np.percentile(np.abs(values), 99)), 1e-30)
    item = pg.ImageItem(values.T, axisOrder="row-major", levels=(-clip, clip))
    item.setLookupTable(lookup_table("Gray (black = positive)"))
    item.setRect(QtCore.QRectF(x0-dx/2, t0-dt/2, dx*values.shape[0], dt*values.shape[1]))
    plot.addItem(item); plot.invertY(True); plot.setLabel("left", "Two-way time", units="s")


class WellTieExtensions:
    def tie_button(self, text, fn):
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(lambda checked=False: self._guard(text, fn))
        return b

    def init_tie_controls(self):
        self.deviation = self.deviation_survey = None
        self.wavelet_mode = QtWidgets.QComboBox(); self.wavelet_mode.addItems(["statistical + phase scan", "deterministic"])
        self.wavelet_length = spin(.08, .004, 1)
        self.prewhitening = spin(.1, 0, 100)
        self.strain_limit = spin(.1, .001, .5)
        self.manual_points = QtWidgets.QLineEdit()
        self.manual_points.setPlaceholderText("Bulk-shifted synthetic : seismic times (seconds), e.g. 1.10:1.12, 1.30:1.34")
        self.tie_summary = QtWidgets.QLabel(); self.tie_summary.setWordWrap(True)
        tab = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(tab); form = QtWidgets.QGridLayout()
        for j, (label, box) in enumerate((("Extraction", self.wavelet_mode), ("Wavelet length (s)", self.wavelet_length),
                                          ("Prewhitening (%)", self.prewhitening), ("Maximum strain", self.strain_limit))):
            form.addWidget(QtWidgets.QLabel(label), 0, 2*j); form.addWidget(box, 0, 2*j+1)
        for j, (label, fn) in enumerate((("Extract wavelet", self.extract_wavelet), ("Automatic stretch / squeeze", self.auto_warp),
                                        ("Apply manual tie points", self.manual_warp), ("Reset to bulk shift", self.reset_warp))):
            form.addWidget(self.tie_button(label, fn), 1, 2*j, 1, 2)
        layout.addLayout(form); layout.addWidget(self.manual_points)
        graph = pg.GraphicsLayoutWidget()
        self.wavelet_plot = graph.addPlot(row=0, col=0, title="Wavelet")
        self.amplitude_plot = graph.addPlot(row=0, col=1, title="Amplitude spectrum")
        self.phase_plot = graph.addPlot(row=0, col=2, title="Phase (degrees)")
        self.shift_plot = graph.addPlot(row=1, col=0, colspan=3, title="Stretch / squeeze shift (ms)")
        self.wavelet_plot.setLabel('bottom', 'Time', units='s')
        self.wavelet_plot.setLabel('left', 'Amplitude')
        for p in (self.amplitude_plot, self.phase_plot):
            p.setLabel('bottom', 'Frequency', units='Hz')
        self.phase_plot.setLabel('left', 'Phase', units='deg')
        self.shift_plot.setLabel('bottom', 'Synthetic time', units='s')
        self.shift_plot.setLabel('left', 'Shift', units='ms')
        layout.addWidget(graph); layout.addWidget(self.tie_summary); self.tabs.addTab(tab, "Wavelet and time tie")
        self.panel_plot = pg.PlotWidget(title="Seismic and five repeated synthetics")
        self.tabs.addTab(self.panel_plot, "Multi-trace tie")

        tab = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(self.tie_button("Load deviation survey (MD metres, inclination, azimuth)…", self.ask_deviation))
        self.use_deviation = QtWidgets.QCheckBox("Use TVD below seismic datum for logs"); self.use_deviation.setChecked(True)
        layout.addWidget(self.use_deviation); form = QtWidgets.QGridLayout()
        self.head_x, self.head_y, self.kb, self.seismic_datum = (spin(0, -1e8, 1e8) for _ in range(4))
        self.project_md = spin(1000, 0, 1e5)
        for j, (label, box) in enumerate((("Wellhead east (m)", self.head_x), ("Wellhead north (m)", self.head_y),
                                          ("KB elevation (m)", self.kb), ("Seismic datum elevation (m)", self.seismic_datum),
                                          ("Project at MD (m)", self.project_md))):
            form.addWidget(QtWidgets.QLabel(label), j//3, 2*(j%3)); form.addWidget(box, j//3, 2*(j%3)+1)
        layout.addLayout(form)
        layout.addWidget(self.tie_button("Update path and choose nearest seismic trace", self.update_deviation))
        self.path_graph = pg.GraphicsLayoutWidget(); layout.addWidget(self.path_graph)
        self.path_info = QtWidgets.QLabel("Checkshots must use the same depth datum as the converted logs.")
        self.path_info.setWordWrap(True); layout.addWidget(self.path_info); self.tabs.addTab(tab, "Deviated well")

        tab = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(tab); row = QtWidgets.QHBoxLayout()
        self.avo_method = QtWidgets.QComboBox(); self.avo_method.addItems(list(wt.AVO_METHODS))
        self.avo_depth = spin(1000, -1e5, 1e5)
        for box in (self.avo_method, QtWidgets.QLabel("Interface depth (m)"), self.avo_depth,
                    self.tie_button("Compute angle gather and interface A/B", self.compute_avo)):
            row.addWidget(box)
        layout.addLayout(row); self.avo_plot = pg.PlotWidget(); layout.addWidget(self.avo_plot)
        self.avo_info = QtWidgets.QLabel(); self.avo_info.setWordWrap(True); layout.addWidget(self.avo_info)
        self.tabs.addTab(tab, "AVO synthetic")

    def tie_result(self):
        if self.result is None:
            self.compute()
        if self.result["seismic"] is None:
            raise ValueError("choose a seismic dataset for the well tie")
        return self.result

    def extract_wavelet(self):
        r = self.tie_result(); s = r["seismic"]; dt = s["dt"]
        times = r["times"] + r["tie"]["shift"]
        if self.wavelet_mode.currentText().startswith("deterministic"):
            result = wt.deterministic_wavelet(times, r["reflectivity"], s["times"], s["values"],
                                               self.wavelet_length.value(), self.prewhitening.value())
            wavelet, centre = result["wavelet"], result["centre"]
        else:
            data = self._seismic_data(); i = s["index"]
            mask = (data.times >= times[0]) & (data.times <= times[-1])
            wavelet, centre = wt.statistical_wavelet(data.traces[max(0, i-5):i+6, mask], dt, self.wavelet_length.value())
            syn = wl.convolve_wavelet(r["reflectivity"], wavelet, centre)
            target = np.interp(times, s["times"], s["values"], left=0, right=0)
            wavelet = wt.rotate(wavelet, wt.constant_phase_scan(syn, target)["angle"])
        r["wavelet"], r["wavelet_centre"] = wavelet, centre
        r["synthetic"] = wl.convolve_wavelet(r["reflectivity"], wavelet, centre)
        r.pop("warp_shifts", None)
        spectrum = wt.wavelet_spectrum(wavelet, centre, dt)
        for p in (self.wavelet_plot, self.amplitude_plot, self.phase_plot):
            p.clear()
        pen = pg.mkPen((30, 70, 160), width=1.5)
        self.wavelet_plot.plot((np.arange(len(wavelet))-centre)*dt, wavelet, pen=pen)
        self.amplitude_plot.plot(spectrum["freq"], spectrum["amplitude"], pen=pen)
        self.phase_plot.plot(spectrum["freq"], spectrum["phase"], connect="finite", pen=pen)
        self.update_tie_display()
        return wavelet

    def auto_warp(self):
        r = self.tie_result(); s = r["seismic"]
        times = r["times"] + r["tie"]["shift"]
        if times[0] < s["times"][0] or times[-1] > s["times"][-1]:
            raise ValueError("the complete synthetic must lie within the seismic record for automatic warping")
        target = np.interp(times, s["times"], s["values"])
        result = wt.dynamic_warp(r["synthetic"], target, s["dt"], self.max_shift.value()/1000,
                                 strain=self.strain_limit.value())
        r["warp_shifts"] = result["shifts"]
        self.update_tie_display()
        return result

    def manual_warp(self):
        r = self.tie_result()
        r["warp_shifts"] = wt.tie_point_shifts(r["times"] + r["tie"]["shift"], wt.parse_tie_points(self.manual_points.text()))
        self.update_tie_display()

    def reset_warp(self):
        self.tie_result().pop("warp_shifts", None)
        self.update_tie_display()

    def tied_times(self):
        r = self.result
        return r["times"] + (r["tie"]["shift"] if r["tie"] else 0) + r.get("warp_shifts", 0)

    def tied_depth_times(self):
        r = self.result; bulk = r["tie"]["shift"] if r["tie"] else 0
        shifts = r.get("warp_shifts")
        return wt.tied_time_depth(r["depth"], r["twt"], bulk,
                                  r["times"] + bulk if shifts is not None else None, shifts)

    def update_tie_display(self):
        r = self.tie_result(); times = self.tied_times(); s = r["seismic"]
        if np.any(np.diff(times) <= 0):
            raise ValueError("tie must keep time increasing")
        span = (s["times"] >= times[0]) & (s["times"] <= times[-1])
        pred = np.interp(s["times"][span], times, r["synthetic"])
        correlation = wt._corr(pred, s["values"][span])
        strain = wt.strain_of(r["times"], r.get("warp_shifts", np.zeros_like(times)))
        change = wt.velocity_change(r["depth"], r["twt"], self.tied_depth_times())
        self.tie_summary.setText(f"Correlation after tie: {correlation:.3f}. Maximum strain: {strain:.3f}. "
                                 f"Largest implied interval-velocity change: {change:.1f}%. " +
                                 ("Strain exceeds the selected limit; revise the manual tie points." if strain > self.strain_limit.value()+1e-6 else ""))
        self.shift_plot.clear()
        self.shift_plot.plot(r["times"], 1000*(times-r["times"]-r["tie"]["shift"]),
                              pen=pg.mkPen((30, 70, 160), width=1.5))
        panel = self.tie_panel_data = wt.tie_panel(self._seismic_data(), s["index"], times, r["synthetic"])
        image_panel(self.panel_plot, panel.traces, 1, 1, panel.t0, panel.dt)
        self.panel_plot.setLabel("bottom", "Panel trace")
        self._draw()

    def ask_deviation(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Deviation survey, MD in metres", "", "Tables (*.csv *.txt *.dat)")
        if path:
            self.load_deviation(path)

    def load_deviation(self, path):
        self.deviation_survey = dev.read_deviation(path)
        self.update_deviation()

    def update_deviation(self):
        if self.deviation_survey is None:
            raise ValueError("load a deviation survey first")
        self.deviation = dev.minimum_curvature(*self.deviation_survey, self.head_x.value(), self.head_y.value())
        p = self.deviation; self.path_graph.clear()
        track = self.path_graph.addPlot(row=0, col=0, title="Map path")
        track.plot(p["x"], p["y"], symbol="o", pen=pg.mkPen((30, 70, 160), width=1.5))
        track.setLabel("bottom", "East (m)"); track.setLabel("left", "North (m)")
        if np.ptp(p['x']) < 1:
            track.setXRange(float(np.mean(p['x']))-.5, float(np.mean(p['x']))+.5, padding=0)
        if np.ptp(p['y']) < 1:
            track.setYRange(float(np.mean(p['y']))-.5, float(np.mean(p['y']))+.5, padding=0)
        track = self.path_graph.addPlot(row=0, col=1, title="Displacement versus TVD")
        track.plot(p["displacement"], p["tvd"], pen=pg.mkPen((30, 70, 160), width=1.5)); track.invertY(True)
        track.setLabel("bottom", "Displacement (m)"); track.setLabel("left", "TVD below KB (m)")
        data = self._seismic_data(); text = "Checkshots must use the same TVD and datum as the converted logs. "
        if data is not None:
            xkey, ykey = ("cdpx", "cdpy") if "cdpx" in data.headers else ("gx", "gy")
            if not all(k in data.headers for k in (xkey, ykey)):
                text += "Projection needs both map coordinates on the seismic traces."
            elif not p["md"][0] <= self.project_md.value() <= p["md"][-1]:
                text += "Projection MD is outside the deviation survey."
            else:
                x, y = (np.interp(self.project_md.value(), p["md"], p[k]) for k in ("x", "y"))
                distance = np.hypot(data.header(xkey)-x, data.header(ykey)-y)
                i = int(np.argmin(distance)); self.x_pos.setValue(float(self._positions(data)[i]))
                text += f"Nearest trace {i+1}; map distance {distance[i]:.1f} m."
        self.path_info.setText(text); self.invalidate_synthetic()

    def vertical_depth(self, md):
        if self.deviation is None or not self.use_deviation.isChecked():
            return md
        z = dev.md_to_tvd(md, self.deviation) - self.kb.value() + self.seismic_datum.value()
        if np.any(np.diff(z) <= 1e-8):
            raise ValueError("a 1D well synthetic needs increasing TVD; horizontal/upgoing log intervals are unsupported")
        return z

    def compute_avo(self):
        if self.result is None:
            self.compute()
        r = self.result; dt = float(r["times"][1]-r["times"][0])
        _, z, vp, rho, vs = self._valid()
        wavelet, centre = (r["wavelet"], r["wavelet_centre"]) if "wavelet" in r else ricker(self.f0.value(), dt)
        result = self.avo_result = wt.avo_synthetic(self.tied_depth_times(), vp, vs, rho, np.arange(0, 41, 2),
                                                    dt, wavelet, centre, self.avo_method.currentText())
        image_panel(self.avo_plot, result["gather"], 0, 2, result["times"][0], dt)
        self.avo_plot.setLabel("bottom", "Incidence angle", units="deg")
        try:
            props = wt.layer_contrast(z, vp, vs, rho, self.avo_depth.value())
            a, b = wt.intercept_gradient(*props)
            self.avo_info.setText(f"Interface A={a:.4f}, B={b:.4f}, class {wt.avo_class(a, b)} (5 m averaging each side).")
        except ValueError as exc:
            self.avo_info.setText(str(exc))
        return result
