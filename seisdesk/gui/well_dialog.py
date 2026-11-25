"""Well-log window: LAS logs, time-depth relation (integrated sonic, optional checkshots), synthetic seismogram and its
tie to a seismic trace, Backus averaging with Thomsen parameters, and Gassmann fluid substitution."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from ..core import welllog as wl
from ..processing.base import trace_positions
from .common import show_error
from .welltie_panel import WellTieExtensions

NONE = "(none)"
MUDROCK = "(mudrock line from Vp)"
GARDNER = "(Gardner from Vp)"
CONSTANT = "(constant)"
BLUE, RED, GREEN, GREY = (30, 60, 200), (200, 30, 30), (0, 130, 0), (120, 120, 120)


def _track(title: str, xlabel: str, ylabel: str) -> pg.PlotWidget:
    p = pg.PlotWidget(title=title)
    p.getPlotItem().invertY(True)
    p.setLabel("bottom", xlabel)
    p.setLabel("left", ylabel)
    p.showGrid(x=True, y=True, alpha=0.3)
    return p


def _spin(lo: float, hi: float, value: float, decimals: int = 1, suffix: str = "",
          step: float | None = None) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    if step:
        s.setSingleStep(step)
    return s


def _pen(colour, width: float = 1.2):
    return pg.mkPen(colour, width=width)


class WellLogDialog(QtWidgets.QDialog, WellTieExtensions):
    """Logs of one well (LAS 2.0), its synthetic seismogram tied to a seismic trace, Backus/Thomsen and Gassmann."""

    def __init__(self, project=None, parent=None, las_path: str | None = None):
        super().__init__(parent)
        self.project = project
        self.log = None
        self.checkshots = None                      # (depth m, two-way time s)
        self.result = None
        self.backus = None
        self.gassmann_result = None
        self.setWindowTitle("Well logs and synthetic seismogram")
        self.resize(1300, 780)

        load = QtWidgets.QPushButton("Load LAS...")
        load.clicked.connect(lambda: self._guard("Load LAS", self._ask_las))
        cs = QtWidgets.QPushButton("Load checkshots...")
        cs.clicked.connect(lambda: self._guard("Checkshots", self._ask_checkshots))
        self.info = QtWidgets.QLabel("No well loaded: load a LAS 2.0 file.")
        self.info.setWordWrap(True)
        self.sonic, self.density, self.shear, self.porosity = (QtWidgets.QComboBox() for _ in range(4))
        self.v_above = _spin(300.0, 8000.0, 2000.0, 0, " m/s", 50.0)
        self.f0 = _spin(2.0, 200.0, 30.0, 1, " Hz")
        self.f0.valueChanged.connect(lambda _v: self._update_hint())
        self.dt_ms = _spin(0.25, 16.0, 2.0, 2, " ms", 0.5)
        self.dataset = QtWidgets.QComboBox()
        self.x_pos = _spin(-1e7, 1e7, 0.0, 1, " m", 25.0)
        self.max_shift = _spin(0.0, 500.0, 100.0, 0, " ms", 10.0)
        go = QtWidgets.QPushButton("Compute synthetic and tie")
        go.clicked.connect(lambda: self._guard("Synthetic seismogram", self.compute))
        self.refresh_datasets()
        self.dataset.currentTextChanged.connect(self._dataset_changed)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(load, 0, 0)
        grid.addWidget(cs, 0, 1)
        grid.addWidget(self.info, 0, 2, 1, 8)
        row = [("Sonic or Vp", self.sonic), ("Density", self.density), ("Shear (Vs)", self.shear),
               ("Velocity above the log", self.v_above)]
        for k, (text, w) in enumerate(row):
            grid.addWidget(QtWidgets.QLabel(text), 1, 2 * k)
            grid.addWidget(w, 1, 2 * k + 1)
        row = [("Ricker peak", self.f0), ("dt (no seismic)", self.dt_ms), ("Seismic", self.dataset),
               ("Trace nearest x", self.x_pos), ("Max shift", self.max_shift)]
        for k, (text, w) in enumerate(row):
            grid.addWidget(QtWidgets.QLabel(text), 2, 2 * k)
            grid.addWidget(w, 2, 2 * k + 1)
        grid.addWidget(go, 3, 8, 1, 2)

        # ---- logs and synthetic
        self.p_vp = _track("Vp", "m/s", "depth (m)")
        self.p_rho = _track("Density", "kg/m3", "depth (m)")
        self.p_imp = _track("Impedance", "kg/(m2 s)", "depth (m)")
        self.p_rc = _track("Reflectivity", "", "two-way time (s)")
        self.p_syn = _track("Synthetic (tied)", "", "two-way time (s)")
        self.p_seis = _track("Seismic", "", "two-way time (s)")
        for p in (self.p_rho, self.p_imp):
            p.setYLink(self.p_vp)
        for p in (self.p_syn, self.p_seis):
            p.setYLink(self.p_rc)
        self.readout = QtWidgets.QLabel("")
        self.readout.setWordWrap(True)
        tab1 = QtWidgets.QWidget()
        lay1 = QtWidgets.QVBoxLayout(tab1)
        tracks = QtWidgets.QHBoxLayout()
        for p in (self.p_vp, self.p_rho, self.p_imp, self.p_rc, self.p_syn, self.p_seis):
            tracks.addWidget(p)
        lay1.addLayout(tracks, 1)
        lay1.addWidget(self.readout)

        # ---- Backus / Thomsen
        self.backus_len = _spin(0.1, 1000.0, 10.0, 1, " m")
        self.backus_hint = QtWidgets.QLabel("")
        avg = QtWidgets.QPushButton("Average")
        avg.clicked.connect(lambda: self._guard("Backus average", self.compute_backus))
        self.p_thomsen = _track("Thomsen epsilon (red), delta (green), gamma (blue)", "", "depth (m)")
        self.p_v0 = _track("Vertical Vp0 (red) and Vs0 (blue) of the averaged medium", "m/s", "depth (m)")
        self.p_v0.setYLink(self.p_thomsen)
        tab2 = QtWidgets.QWidget()
        lay2 = QtWidgets.QVBoxLayout(tab2)
        top2 = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Averaging length"), self.backus_len, avg, self.backus_hint):
            top2.addWidget(w)
        top2.addStretch(1)
        lay2.addLayout(top2)
        plots2 = QtWidgets.QHBoxLayout()
        plots2.addWidget(self.p_thomsen)
        plots2.addWidget(self.p_v0)
        lay2.addLayout(plots2, 1)

        # ---- Gassmann
        self.phi_const = _spin(0.0, 0.6, 0.2, 3)
        self.k_min = _spin(1.0, 100.0, 36.6, 2, " GPa")
        self.k_fl1 = _spin(0.001, 10.0, 2.8, 3, " GPa")
        self.rho_fl1 = _spin(0.01, 2.0, 1.09, 3, " g/cc")
        self.k_fl2 = _spin(0.001, 10.0, 0.04, 3, " GPa")
        self.rho_fl2 = _spin(0.01, 2.0, 0.25, 3, " g/cc")
        self.zone_top = _spin(-1e5, 1e5, 0.0, 1, " m")
        self.zone_bottom = _spin(-1e5, 1e5, 0.0, 1, " m")
        sub = QtWidgets.QPushButton("Substitute")
        sub.clicked.connect(lambda: self._guard("Gassmann substitution", self.compute_gassmann))
        self.p_gvp = _track("Vp", "m/s", "depth (m)")
        self.p_gvs = _track("Vs", "m/s", "depth (m)")
        self.p_grho = _track("Density", "kg/m3", "depth (m)")
        self.p_gsyn = _track("Synthetic", "", "two-way time (s)")
        for p in (self.p_gvs, self.p_grho):
            p.setYLink(self.p_gvp)
        self.g_readout = QtWidgets.QLabel("")
        tab3 = QtWidgets.QWidget()
        lay3 = QtWidgets.QVBoxLayout(tab3)
        form = QtWidgets.QGridLayout()
        items = [("Porosity", self.porosity), ("constant", self.phi_const), ("Mineral K", self.k_min),
                 ("Zone top", self.zone_top), ("Zone bottom (0 = whole log)", self.zone_bottom),
                 ("Fluid 1 K", self.k_fl1), ("Fluid 1 density", self.rho_fl1), ("Fluid 2 K", self.k_fl2),
                 ("Fluid 2 density", self.rho_fl2)]
        for k, (text, w) in enumerate(items):
            form.addWidget(QtWidgets.QLabel(text), k // 5, 2 * (k % 5))
            form.addWidget(w, k // 5, 2 * (k % 5) + 1)
        form.addWidget(sub, 1, 8, 1, 2)
        lay3.addLayout(form)
        plots3 = QtWidgets.QHBoxLayout()
        for p in (self.p_gvp, self.p_gvs, self.p_grho, self.p_gsyn):
            plots3.addWidget(p)
        lay3.addLayout(plots3, 1)
        lay3.addWidget(self.g_readout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(tab1, "Logs and synthetic")
        self.tabs.addTab(tab2, "Backus / Thomsen")
        self.tabs.addTab(tab3, "Gassmann")
        self.init_tie_controls()
        b_syn = QtWidgets.QPushButton("Export synthetic (CSV)...")
        b_syn.clicked.connect(lambda: self._guard("Export", self.export_synthetic))
        b_td = QtWidgets.QPushButton("Export time-depth table (CSV)...")
        b_td.clicked.connect(lambda: self._guard("Export", self.export_time_depth))
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        for b in (b_syn, b_td, close):
            bottom.addWidget(b)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(self.tabs, 1)
        lay.addLayout(bottom)
        for combo in (self.sonic, self.density, self.shear, self.dataset):
            combo.currentIndexChanged.connect(self.invalidate_synthetic)
        for box in (self.v_above, self.f0, self.dt_ms, self.x_pos, self.max_shift,
                    self.kb, self.seismic_datum):
            box.valueChanged.connect(self.invalidate_synthetic)
        self.use_deviation.toggled.connect(self.invalidate_synthetic)
        if las_path:
            self.load_las(las_path)

    # ------------------------------------------------------------------ input
    def invalidate_synthetic(self, *_args) -> None:
        self.result = self.backus = self.gassmann_result = None
        for p in (self.p_vp, self.p_rho, self.p_imp, self.p_rc, self.p_syn, self.p_seis,
                  self.wavelet_plot, self.amplitude_plot, self.phase_plot, self.shift_plot,
                  self.panel_plot, self.avo_plot, self.p_thomsen, self.p_v0,
                  self.p_gvp, self.p_gvs, self.p_grho, self.p_gsyn):
            p.clear()
        self.readout.setText("Inputs changed: compute the synthetic and tie again.")
        self.tie_summary.clear(); self.avo_info.clear(); self.g_readout.clear()

    def _guard(self, title: str, fn) -> None:
        try:
            fn()
        except Exception as exc:                            # noqa: BLE001 - shown to the user
            show_error(self, title, exc)

    def _ask_las(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open LAS file", "",
                                                        "LAS well logs (*.las *.LAS);;All files (*)")
        if path:
            self.load_las(path)

    def _ask_checkshots(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open checkshot table (depth m, time)", "",
                                                        "Text tables (*.txt *.csv *.dat);;All files (*)")
        if not path:
            return
        kinds = ["one-way time in ms", "two-way time in ms", "one-way time in s", "two-way time in s"]
        kind, ok = QtWidgets.QInputDialog.getItem(self, "Checkshots", "The second column holds:", kinds, 0, False)
        if ok:
            self.load_checkshots(path, kind.startswith("one"), "ms" if kind.endswith("ms") else "s")

    def load_las(self, path: str) -> None:
        log = wl.read_las(path)
        self.log = log
        names = log.names[1:]
        self._fill(self.sonic, names, [wl.find_curve(log, "sonic"), wl.find_curve(log, "vp")])
        self._fill(self.density, names, [wl.find_curve(log, "density")], (GARDNER,))
        self._fill(self.shear, names, [wl.find_curve(log, "shear sonic"), wl.find_curve(log, "vs")], (MUDROCK,))
        self._fill(self.porosity, names, [wl.find_curve(log, "porosity")], (CONSTANT,))
        z = log.depth_m
        self.info.setText(f"<b>{log.name}</b>: {z.size} samples, depth {np.nanmin(z):.1f}-{np.nanmax(z):.1f} m "
                          f"(file unit {log.depth_unit}); curves {', '.join(names)}")
        self.checkshots = None
        self.deviation = self.deviation_survey = None
        self.path_graph.clear()
        self.invalidate_synthetic()
        self._update_hint()

    def load_checkshots(self, path: str, one_way: bool = True, time_unit: str = "ms") -> None:
        self.checkshots = wl.read_checkshots(path, one_way, time_unit)
        self.invalidate_synthetic()
        self.info.setText(self.info.text() + f"; {self.checkshots[0].size} checkshots")

    @staticmethod
    def _fill(combo: QtWidgets.QComboBox, names, preferred, extra=()) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(list(extra) + list(names))
        choice = next((p for p in preferred if p), None)
        combo.setCurrentText(choice or (extra[0] if extra else (names[0] if names else "")))
        combo.blockSignals(False)

    def refresh_datasets(self) -> None:
        self.dataset.blockSignals(True)
        self.dataset.clear()
        self.dataset.addItem(NONE)
        if self.project is not None:
            for name, d in self.project.datasets.items():
                if d.domain == "time":
                    self.dataset.addItem(name)
        self.dataset.blockSignals(False)

    def set_seismic(self, name: str, x: float) -> None:
        """Choose the seismic dataset and the position of the well (the nearest trace is used)."""
        self.dataset.setCurrentText(name)
        self.x_pos.setValue(float(x))

    @staticmethod
    def _positions(d) -> np.ndarray:
        try:
            return trace_positions(d)
        except ValueError:
            return np.arange(1, d.ntr + 1, dtype=np.float64)

    def _seismic_data(self):
        name = self.dataset.currentText()
        if self.project is None or not name or name == NONE:
            return None
        return self.project.datasets.get(name)

    def _dataset_changed(self, _name: str) -> None:
        d = self._seismic_data()
        if d is not None:
            self.x_pos.setValue(float(np.median(self._positions(d))))

    def seismic_trace(self) -> dict | None:
        """The trace of the chosen dataset nearest to the well position, or None."""
        d = self._seismic_data()
        if d is None:
            return None
        x = self._positions(d)
        i = int(np.argmin(np.abs(x - self.x_pos.value())))
        return {"times": d.times, "values": d.traces[i].astype(np.float64), "dt": d.dt, "index": i,
                "x": float(x[i]), "name": self.dataset.currentText()}

    def _velocity_from(self, name: str) -> np.ndarray:
        values = np.asarray(self.log.curve(name), dtype=np.float64)
        unit = self.log.units.get(name, "").lower()
        mnem = name.upper().split(":")[0]
        if "/s" in unit or mnem in wl.ALIASES["vp"] + wl.ALIASES["vs"]:
            scale = 1000.0 if "km" in unit else (wl.FT if "f" in unit else 1.0)
            v = values * scale
            v[~(v > 0)] = np.nan
            return v
        return wl.sonic_to_velocity(values, unit)

    def curves(self):
        """Depth (m) and the Vp, density and Vs logs (SI units) of the chosen curves."""
        if self.log is None:
            raise ValueError("load a LAS file first")
        if not self.sonic.currentText():
            raise ValueError("choose the sonic (or Vp) curve")
        z = self.vertical_depth(self.log.depth_m)
        vp = self._velocity_from(self.sonic.currentText())
        dn = self.density.currentText()
        rho = wl.gardner_density(vp) if dn == GARDNER else wl.density_to_si(self.log.curve(dn),
                                                                             self.log.units.get(dn, ""))
        sn = self.shear.currentText()
        vs = wl.mudrock_vs(vp) if sn == MUDROCK else self._velocity_from(sn)
        return z, vp, rho, vs

    def _valid(self):
        """(slice into the log, depth, vp, rho, vs) over the common valid span of Vp and density, gaps filled."""
        z, vp, rho, vs = self.curves()
        vp, rho, vs = (wl.fill_gaps(z, c) for c in (vp, rho, vs))
        sl = wl.valid_span(vp, rho)
        return sl, z[sl], vp[sl], rho[sl], vs[sl]

    def _update_hint(self) -> None:
        try:
            vs = self._valid()[4]
            vmin = float(np.nanmin(vs))
            f0 = self.f0.value()
            self.backus_hint.setText(f"Suggested at most {wl.backus_length(vmin, f0):.1f} m: a third of the shortest "
                                     f"S wavelength ({vmin:.0f} m/s at {f0:g} Hz).")
        except (ValueError, KeyError):
            self.backus_hint.setText("")

    # ------------------------------------------------------------------ computations
    def compute(self) -> bool:
        """Synthetic seismogram of the chosen logs and, with a seismic dataset chosen, its tie to the nearest trace."""
        _sl, z, vp, rho, _vs = self._valid()
        seis = self.seismic_trace()
        dt = seis["dt"] if seis is not None else self.dt_ms.value() / 1000.0
        twt = wl.two_way_time(z, vp, v_above=self.v_above.value())
        if self.checkshots is not None:
            twt = wl.apply_checkshots(z, twt, *self.checkshots)
        res = wl.synthetic_seismogram(z, vp, rho, dt, f0=self.f0.value(), twt=twt)
        res["seismic"] = seis
        res["tie"] = None
        if seis is not None:
            res["tie"] = wl.well_tie(res["times"], res["synthetic"], seis["times"], seis["values"],
                                     self.max_shift.value() / 1000.0)
        self.result = res
        self._draw()
        if seis is not None:
            self.update_tie_display()
        return True

    def _draw(self) -> None:
        r = self.result
        for p in (self.p_vp, self.p_rho, self.p_imp, self.p_rc, self.p_syn, self.p_seis):
            p.clear()
        self.p_vp.plot(r["vp"], r["depth"], pen=_pen(BLUE))
        self.p_rho.plot(r["rho"], r["depth"], pen=_pen(GREEN))
        self.p_imp.plot(r["impedance"], r["depth"], pen=_pen(RED))
        t = r["times"]
        self.p_rc.plot(r["reflectivity"], t, pen=_pen(GREY))
        tie = r["tie"]
        shift = tie["shift"] if tie else 0.0
        self.p_syn.plot(r["synthetic"], self.tied_times(), pen=_pen(BLUE, 1.5))
        wavelet_label = "extracted wavelet" if "wavelet" in r else f"Ricker {self.f0.value():g} Hz"
        text = (f"Synthetic over {t[0]:.3f}-{t[-1]:.3f} s two-way time: {wavelet_label}, "
                f"dt {1000 * (t[1] - t[0]) if t.size > 1 else 0:g} ms"
                f"{', checkshot-corrected' if self.checkshots is not None else ''}.")
        s = r["seismic"]
        if s is not None:
            self.p_seis.plot(s["values"], s["times"], pen=_pen("k"))
            text += (f" Tie with {s['name']}, trace {s['index'] + 1} (x {s['x']:.0f} m): best shift "
                     f"<b>{shift * 1000:+.0f} ms</b>, correlation <b>{tie['r']:.2f}</b>.")
            if -tie["r_negative"] > tie["r"]:
                text += " The reversed polarity correlates better: check the polarity convention."
        self.p_rc.setYRange(t[0] - 0.05, t[-1] + 0.05, padding=0.0)
        self.readout.setText(text)

    def compute_backus(self) -> dict:
        _sl, z, vp, rho, vs = self._valid()
        res = wl.backus_average(z, vp, vs, rho, self.backus_len.value())
        res["depth"] = z
        self.backus = res
        self.p_thomsen.clear()
        self.p_v0.clear()
        for key, colour in (("epsilon", RED), ("delta", GREEN), ("gamma", BLUE)):
            self.p_thomsen.plot(res[key], z, pen=_pen(colour, 1.5), connect="finite")
        self.p_v0.plot(res["vp0"], z, pen=_pen(RED, 1.5), connect="finite")
        self.p_v0.plot(res["vs0"], z, pen=_pen(BLUE, 1.5), connect="finite")
        return res

    def compute_gassmann(self) -> dict:
        sl, z, vp, rho, vs = self._valid()
        name = self.porosity.currentText()
        if name == CONSTANT or not name:
            phi = np.full(z.size, self.phi_const.value())
        else:
            phi = wl.fill_gaps(self.log.depth_m, self.log.curve(name))[sl]
            if np.any(np.isfinite(phi)) and float(np.nanmedian(phi)) > 1.0:
                phi = phi / 100.0                                   # porosity in percent
        top, bottom = self.zone_top.value(), self.zone_bottom.value()
        if bottom > top:
            phi = np.where((z >= top) & (z <= bottom), phi, 0.0)
        phi = np.nan_to_num(phi, nan=0.0)
        vp2, vs2, rho2 = wl.gassmann(vp, vs, rho, phi, self.k_min.value() * 1e9, self.k_fl1.value() * 1e9,
                                     self.rho_fl1.value() * 1000.0, self.k_fl2.value() * 1e9,
                                     self.rho_fl2.value() * 1000.0)
        seis = self.seismic_trace()
        dt = seis["dt"] if seis is not None else self.dt_ms.value() / 1000.0
        before = wl.synthetic_seismogram(z, vp, rho, dt, self.f0.value(), v_above=self.v_above.value())
        after = wl.synthetic_seismogram(z, vp2, rho2, dt, self.f0.value(), v_above=self.v_above.value())
        res = {"depth": z, "porosity": phi, "vp": vp, "vs": vs, "rho": rho, "vp2": vp2, "vs2": vs2, "rho2": rho2,
               "before": before, "after": after}
        self.gassmann_result = res
        for p in (self.p_gvp, self.p_gvs, self.p_grho, self.p_gsyn):
            p.clear()
        for p, a, b in ((self.p_gvp, vp, vp2), (self.p_gvs, vs, vs2), (self.p_grho, rho, rho2)):
            p.plot(a, z, pen=_pen(GREY), connect="finite")
            p.plot(b, z, pen=_pen(RED), connect="finite")
        self.p_gsyn.plot(before["synthetic"], before["times"], pen=_pen(GREY, 1.5))
        self.p_gsyn.plot(after["synthetic"], after["times"], pen=_pen(RED, 1.5))
        changed = (phi > 0.01) & np.isfinite(vp)
        if changed.any():
            dv = 100.0 * (vp2[changed] / vp[changed] - 1.0)
            self.g_readout.setText(f"{int(changed.sum())} samples substituted: Vp changes {dv.min():+.1f} to "
                                   f"{dv.max():+.1f} %, density by {np.nanmean(rho2[changed] - rho[changed]):+.0f} "
                                   "kg/m3 on average. Grey: before, red: after.")
        else:
            self.g_readout.setText("No sample with porosity above 1 % in the zone: nothing substituted.")
        return res

    # ------------------------------------------------------------------ export
    def synthetic_csv(self) -> str:
        if self.result is None:
            self.compute()
        r = self.result
        lines = ["time_s,reflectivity,synthetic,time_tied_s"]
        lines += [f"{t:.6f},{a:.6f},{s:.6f},{tied:.6f}"
                  for t, a, s, tied in zip(r["times"], r["reflectivity"], r["synthetic"], self.tied_times())]
        return "\n".join(lines)

    def time_depth_csv(self) -> str:
        if self.result is None:
            self.compute()
        return wl.time_depth_table(self.result["depth"], self.tied_depth_times())

    def _save(self, path: str | None, default: str, text_fn) -> None:
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", default, "CSV table (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text_fn() + "\n")

    def export_synthetic(self, path: str | None = None) -> None:
        self._save(path, "synthetic.csv", self.synthetic_csv)

    def export_time_depth(self, path: str | None = None) -> None:
        self._save(path, "time_depth.csv", self.time_depth_csv)
