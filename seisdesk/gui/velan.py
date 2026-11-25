"""Interactive velocity analysis: semblance spectrum, CMP (super)gather with moveout curves, NMO-corrected preview,
Dix interval velocities, automatic picking, and an eta scan for VTI (non-hyperbolic) moveout."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from scipy.ndimage import uniform_filter1d

from ..core.velocity import VelocityPicks, dix
from ..processing.moveout import gate_by_energy, moveout_time, sample_at
from ..processing.nmo import nmo_correct
from ..processing.velan import auto_pick, semblance, supergather
from .common import DISPLAY_GAINS, busy, clip_value, display_gain, fmt, show_error
from .plot import SeismicPlot


def eta_semblance(gather: np.ndarray, offsets: np.ndarray, times: np.ndarray, dt: float, vfun: np.ndarray,
                  etas: np.ndarray, window: float) -> np.ndarray:
    """Semblance S(eta, t0) with the NMO velocity fixed to the picked function v(t0)."""
    t0 = times[None, :]
    x = np.asarray(offsets, dtype=np.float64)[:, None]
    nwin = max(1, int(round(window / dt)))
    g = gather.astype(np.float64)
    out = np.zeros((etas.size, times.size))
    energy = np.zeros((etas.size, times.size))
    for k, eta in enumerate(etas):
        tt = moveout_time(t0, x, vfun[None, :], eta)
        vals, valid = sample_at(g, tt, times[0], dt)
        num = uniform_filter1d(vals.sum(axis=0) ** 2, nwin, mode="nearest")
        den = uniform_filter1d(valid.sum(axis=0) * (vals ** 2).sum(axis=0), nwin, mode="nearest")
        out[k] = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
        energy[k] = uniform_filter1d((vals ** 2).sum(axis=0), nwin, mode="nearest")
    return gate_by_energy(out, energy)


def offset_binned(traces: np.ndarray, offsets: np.ndarray, max_cols: int = 400):
    """Average the traces of a (super)gather in regular |offset| bins: (binned traces, bin offsets)."""
    a = np.abs(np.asarray(offsets, dtype=np.float64))
    u = np.unique(np.round(a, 3))
    if u.size < 2:
        return traces, a
    dx = float(np.median(np.diff(u)))
    span = float(u[-1] - u[0])
    ncol = int(round(span / dx)) + 1
    if ncol > max_cols:
        ncol = max_cols
        dx = span / (ncol - 1)
    col = np.clip(np.rint((a - u[0]) / dx).astype(np.int64), 0, ncol - 1)
    out = np.zeros((ncol, traces.shape[1]), dtype=np.float64)
    np.add.at(out, col, traces)
    count = np.bincount(col, minlength=ncol)
    live = count > 0
    out[live] /= count[live, None]
    return out[live].astype(np.float32), (u[0] + np.arange(ncol) * dx)[live]


def _dspin(lo, hi, value, step, decimals, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    w.setSuffix(suffix)
    return w


class VelocityAnalysis(QtWidgets.QWidget):
    sigStatus = QtCore.Signal(str)
    sigPicksChanged = QtCore.Signal()

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.data = None
        self.dataset_name = None
        self.cdps = np.zeros(0)
        self.spectrum = None
        self.axis = np.zeros(0)            # velocities, or etas in eta mode
        self.gather_idx = np.zeros(0, dtype=np.int64)
        self.overlays: list = []
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.dataset_label = QtWidgets.QLabel("<i>choose a CMP-sorted prestack dataset</i>")
        self.dataset_label.setWordWrap(True)
        self.dataset_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.cdp_combo = QtWidgets.QComboBox()
        self.cdp_combo.setMinimumWidth(80)
        self.prev_btn = QtWidgets.QPushButton("< Prev")
        self.next_btn = QtWidgets.QPushButton("Next >")
        self.step_spin = QtWidgets.QSpinBox()
        self.step_spin.setRange(1, 100000)
        self.step_spin.setPrefix("step ")
        self.super_spin = QtWidgets.QSpinBox()
        self.super_spin.setRange(0, 50)
        self.super_spin.setValue(2)
        self.super_spin.setPrefix("+/- ")
        self.super_spin.setSuffix(" CMPs")
        self.super_spin.setToolTip("Supergather: neighbouring CMPs added on each side")
        self.vmin = _dspin(10, 20000, 1400, 100, 0, " m/s")
        self.vmax = _dspin(20, 30000, 4500, 100, 0, " m/s")
        self.nv = QtWidgets.QSpinBox()
        self.nv.setRange(10, 1000)
        self.nv.setValue(120)
        self.window = _dspin(1, 500, 30, 2, 0, " ms")
        self.eta_spin = _dspin(-0.2, 0.6, 0.0, 0.01, 3)
        self.eta_spin.setToolTip("Anisotropy parameter eta used for the semblance and new picks (0 = hyperbolic)")
        self.stretch = _dspin(0, 500, 50, 5, 0, " %")
        self.stretch.setToolTip("NMO stretch mute limit")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["velocity scan", "eta scan (VTI)"])
        self.gain_combo = QtWidgets.QComboBox()
        self.gain_combo.addItems(DISPLAY_GAINS)
        self.gain_combo.setCurrentText("AGC 0.5 s")
        b_compute = QtWidgets.QPushButton("Compute")
        b_compute.setProperty("primary", True)
        b_auto = QtWidgets.QPushButton("Auto-pick")
        b_clear = QtWidgets.QPushButton("Clear CMP")
        b_load = QtWidgets.QPushButton("Load picks...")
        b_save = QtWidgets.QPushButton("Save picks...")

        row1 = QtWidgets.QHBoxLayout()
        for label, w in (("CMP", self.cdp_combo), (None, self.prev_btn), (None, self.next_btn),
                         (None, self.step_spin), ("Supergather", self.super_spin), ("Mode", self.mode_combo)):
            if label:
                row1.addWidget(QtWidgets.QLabel(label))
            row1.addWidget(w)
        row1.addStretch(1)
        row2 = QtWidgets.QHBoxLayout()
        self.scan_options = QtWidgets.QPushButton("Scan settings")
        self.scan_options.setCheckable(True)
        row2.addWidget(self.scan_options)
        for b in (b_compute, b_auto, b_clear, b_load, b_save):
            row2.addWidget(b)
        row2.addStretch(1)
        self.scan_panel = QtWidgets.QWidget()
        options = QtWidgets.QGridLayout(self.scan_panel)
        options.setContentsMargins(0, 4, 0, 8)
        for i, (title, widget) in enumerate((("Minimum velocity", self.vmin), ("Maximum velocity", self.vmax),
                                             ("Samples", self.nv), ("Window", self.window), ("Eta", self.eta_spin),
                                             ("Stretch mute", self.stretch), ("Display gain", self.gain_combo))):
            row, col = divmod(i, 3)
            options.addWidget(QtWidgets.QLabel(title), row, col * 2)
            options.addWidget(widget, row, col * 2 + 1)
        self.scan_panel.hide()
        self.scan_options.toggled.connect(self.scan_panel.setVisible)

        self.semb_plot = SeismicPlot(self)
        self.semb_plot.getPlotItem().setLabel("bottom", "Velocity", units="m/s")
        self.semb_plot.getPlotItem().setTitle("Semblance (click: pick, right click: delete)")
        self.gather_plot = SeismicPlot(self)
        self.gather_plot.getPlotItem().setLabel("bottom", "Offset", units="m")
        self.gather_plot.getPlotItem().setTitle("CMP gather with moveout of the picks")
        self.nmo_plot = SeismicPlot(self)
        self.nmo_plot.getPlotItem().setLabel("bottom", "Offset", units="m")
        self.nmo_plot.getPlotItem().setTitle("After NMO with the picks")
        for p in (self.gather_plot, self.nmo_plot):
            p.setYLink(self.semb_plot)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        for p in (self.semb_plot, self.gather_plot, self.nmo_plot):
            split.addWidget(p)
        self.picks_label = QtWidgets.QLabel(" ")
        self.picks_label.setWordWrap(True)
        self.readout = QtWidgets.QLabel(" ")
        self.readout.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.dataset_label)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self.scan_panel)
        layout.addWidget(split, 1)
        layout.addWidget(self.picks_label)
        layout.addWidget(self.readout)

        self.cdp_combo.currentIndexChanged.connect(lambda _i: self.compute())
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn.clicked.connect(lambda: self.step(+1))
        b_compute.clicked.connect(self.compute)
        b_auto.clicked.connect(self.auto)
        b_clear.clicked.connect(self.clear_cdp)
        b_load.clicked.connect(self.load_picks)
        b_save.clicked.connect(self.save_picks)
        self.mode_combo.currentIndexChanged.connect(lambda _i: self.compute())
        self.gain_combo.currentIndexChanged.connect(lambda _i: self.draw_gathers())
        self.stretch.valueChanged.connect(lambda _v: self.draw_gathers())
        self.semb_plot.sigClick.connect(self._click)
        self.semb_plot.sigHover.connect(self._hover)

    # ------------------------------------------------------------------ data
    @property
    def picks(self) -> VelocityPicks:
        if self.project.context.velocity is None:
            self.project.context.velocity = VelocityPicks()
        return self.project.context.velocity

    def set_dataset(self, name: str | None) -> None:
        self.dataset_name = name
        self.data = self.project.datasets.get(name) if name else None
        self.cdp_combo.blockSignals(True)
        self.cdp_combo.clear()
        if self.data is None or not self.data.has_header("cdp") or not self.data.has_header("offset"):
            self.cdps = np.zeros(0)
            self.spectrum = None
            self.gather_idx = np.zeros(0, dtype=np.int64)
            for p in (self.semb_plot, self.gather_plot, self.nmo_plot):
                p.clear_display()
            self.draw_overlays()
            self.dataset_label.setText("<i>the dataset needs CMP numbers and offsets (run a geometry step)</i>")
            self.cdp_combo.blockSignals(False)
            return
        self.cdps = np.unique(self.data.header("cdp"))
        self.cdp_combo.addItems([fmt(c) for c in self.cdps])
        self.step_spin.setValue(max(1, self.cdps.size // 20))
        mid = self.cdps.size // 2
        self.cdp_combo.setCurrentIndex(mid)
        self.cdp_combo.blockSignals(False)
        self.dataset_label.setText(f"<b>{name}</b>  ({self.cdps.size} CMPs)")
        self.compute()

    def current_cdp(self) -> float | None:
        i = self.cdp_combo.currentIndex()
        return float(self.cdps[i]) if 0 <= i < self.cdps.size else None

    def step(self, direction: int) -> None:
        if self.cdps.size < 2:
            return
        i = self.cdp_combo.currentIndex() + direction * self.step_spin.value()
        self.cdp_combo.setCurrentIndex(int(min(max(i, 0), self.cdps.size - 1)))

    def show_cdp(self, cdp: float) -> None:
        if self.cdps.size:
            self.cdp_combo.setCurrentIndex(int(np.argmin(np.abs(self.cdps - cdp))))

    def _function_here(self, cdp: float, times: np.ndarray):
        """(v, eta) functions of t0 at this CMP: its own picks, else interpolated from neighbours, else None."""
        vp = self.picks
        if len(vp) == 0:
            return None
        return vp.function(cdp, times), vp.field([cdp], times, column=2)[0]

    # ------------------------------------------------------------------ computation
    def compute(self) -> None:
        d, cdp = self.data, self.current_cdp()
        if d is None or cdp is None:
            return
        if self.vmax.value() <= self.vmin.value():
            show_error(self, "Velocity analysis", "v max must exceed v min")
            return
        try:
            with busy():
                idx = supergather(d, cdp, self.super_spin.value())
                self.gather_idx = idx
                if idx.size < 2:
                    self.sigStatus.emit(f"CMP {fmt(cdp)} has fewer than two traces")
                    return
                g = d.traces[idx]
                off = d.header("offset")[idx]
                times = d.times
                win = self.window.value() / 1000.0
                if self.mode_combo.currentIndex() == 0:
                    self.axis = np.linspace(self.vmin.value(), self.vmax.value(), self.nv.value())
                    self.spectrum = semblance(g, off, times, d.dt, self.axis, win, self.eta_spin.value(),
                                              self.stretch.value() / 100.0)
                    self.semb_plot.getPlotItem().setLabel("bottom", "Velocity", units="m/s")
                else:
                    f = self._function_here(cdp, times)
                    if f is None:
                        self.mode_combo.blockSignals(True)
                        self.mode_combo.setCurrentIndex(0)
                        self.mode_combo.blockSignals(False)
                        self.sigStatus.emit("pick velocities first; the eta scan keeps the picked velocity fixed")
                        return self.compute()
                    self.axis = np.linspace(-0.05, 0.35, 81)
                    self.spectrum = eta_semblance(g, off, times, d.dt, f[0], self.axis, win)
                    self.semb_plot.getPlotItem().setLabel("bottom", "eta")
            da = self.axis[1] - self.axis[0]
            top = float(np.percentile(self.spectrum, 99.8)) or 1.0
            self.semb_plot.show_image(self.spectrum.T, float(self.axis[0]), float(da), d.t0, d.dt, "Rainbow",
                                      (0.0, top))
            self.semb_plot.reset_view()
            self.draw_gathers()
            self.sigStatus.emit(f"CMP {fmt(cdp)}: {idx.size} traces in the supergather")
        except Exception as exc:                          # noqa: BLE001
            show_error(self, "Velocity analysis", exc)

    def draw_gathers(self) -> None:
        d, cdp = self.data, self.current_cdp()
        if d is None or cdp is None or self.gather_idx.size < 2:
            return
        idx = self.gather_idx
        off = d.header("offset")[idx]
        g = d.traces[idx]
        mode = self.gain_combo.currentText()
        binned, xpos = offset_binned(g, off)
        disp = display_gain(binned, d.times, d.dt, mode)
        self.gather_plot.show_traces(disp, d.t0, d.dt, xpos, "Density", "Gray (black = positive)",
                                     clip_value(disp, 98))
        f = self._function_here(cdp, d.times)
        if f is not None:
            vf = np.broadcast_to(f[0][None, :], g.shape)
            ef = np.broadcast_to(f[1][None, :], g.shape)
            nmo = nmo_correct(g, d.times, off, vf, ef, self.stretch.value() / 100.0)
            nb, nx = offset_binned(nmo, off)
            nd = display_gain(nb, d.times, d.dt, mode)
            self.nmo_plot.show_traces(nd, d.t0, d.dt, nx, "Density", "Gray (black = positive)", clip_value(nd, 98))
        else:
            self.nmo_plot.clear_display()
        for p in (self.gather_plot, self.nmo_plot):
            p.reset_view(x_only=True)
        self.draw_overlays()

    # ------------------------------------------------------------------ overlays
    def _add(self, plot, item) -> None:
        plot.addItem(item, ignoreBounds=True)
        self.overlays.append((plot, item))

    def draw_overlays(self) -> None:
        for plot, item in self.overlays:
            plot.removeItem(item)
        self.overlays = []
        d, cdp = self.data, self.current_cdp()
        if d is None or cdp is None:
            return
        vp = self.picks
        here = vp.picks_at(cdp)
        eta_mode = self.mode_combo.currentIndex() == 1
        # neighbours as a guide
        others = vp.cdps[vp.cdps != cdp]
        if others.size and not eta_mode:
            near = others[np.argmin(np.abs(others - cdp))]
            p = vp.picks_at(near)
            self._add(self.semb_plot, pg.PlotDataItem(p[:, 1], p[:, 0], pen=pg.mkPen(255, 255, 255, 140, width=1,
                                                                                    style=QtCore.Qt.PenStyle.DashLine)))
        if here.shape[0]:
            xs = here[:, 2] if eta_mode else here[:, 1]
            self._add(self.semb_plot, pg.PlotDataItem(xs, here[:, 0], pen=pg.mkPen("w", width=2), symbol="o",
                                                      symbolSize=8, symbolBrush=pg.mkBrush("w"),
                                                      symbolPen=pg.mkPen("k")))
            if not eta_mode and here.shape[0] >= 2:
                t = d.times
                vrms = np.interp(t, here[:, 0], here[:, 1])
                vint = dix(t, vrms)
                keep = (t >= here[0, 0]) & (t <= here[-1, 0])
                self._add(self.semb_plot, pg.PlotDataItem(vint[keep], t[keep],
                                                          pen=pg.mkPen(255, 0, 255, width=1.5)))
            off = np.abs(d.header("offset")[self.gather_idx])
            xline = np.linspace(0.0, float(off.max()), 60)
            for t0, v, eta in here:
                tt = moveout_time(np.full(xline.size, t0), xline, np.full(xline.size, v), eta)
                self._add(self.gather_plot, pg.PlotDataItem(xline, tt, pen=pg.mkPen(255, 60, 60, width=1.5)))
        lines = [f"CMP {fmt(cdp)}: {here.shape[0]} picks"]
        if here.shape[0]:
            lines.append("  ".join(f"{t * 1000:.0f} ms {v:.0f} m/s" + (f" eta {e:.3f}" if e else "")
                                   for t, v, e in here))
        lines.append(f"project: {len(vp)} picks at {vp.cdps.size} CMPs. Magenta = Dix interval velocity; white dashed "
                     "= nearest analysed CMP.")
        self.picks_label.setText("   ".join(lines))

    # ------------------------------------------------------------------ mouse
    def _hover(self, x: float, y: float) -> None:
        d = self.data
        if d is None or self.spectrum is None or self.axis.size < 2:
            return
        ia = int(round((x - self.axis[0]) / (self.axis[1] - self.axis[0])))
        it = int(round((y - d.t0) / d.dt))
        s = self.spectrum[ia, it] if 0 <= ia < self.axis.size and 0 <= it < self.spectrum.shape[1] else float("nan")
        name = "eta" if self.mode_combo.currentIndex() == 1 else "v"
        self.readout.setText(f"t = {y:.3f} s   {name} = {x:.4g}   semblance {s:.3f}")

    def _click(self, info: dict) -> None:
        d, cdp = self.data, self.current_cdp()
        if d is None or cdp is None:
            return
        vp = self.picks
        t = info["y"]
        if self.mode_combo.currentIndex() == 1:                 # eta scan: set eta of the nearest pick in time
            here = vp.picks_at(cdp)
            if here.shape[0] == 0:
                self.sigStatus.emit("no velocity picks at this CMP to attach eta to")
                return
            k = int(np.argmin(np.abs(here[:, 0] - t)))
            vp.add(cdp, here[k, 0], here[k, 1], 0.0 if (info["right"] or info["shift"]) else info["x"])
        elif info["right"] or info["shift"]:
            vrange = self.vmax.value() - self.vmin.value()
            vp.remove_nearest(cdp, t, info["x"], t_scale=0.05 * (d.tmax - d.t0), v_scale=0.05 * vrange)
        elif info["left"] and info["x"] > 0:
            vp.add(cdp, t, info["x"], self.eta_spin.value())
        self.draw_gathers()
        self.sigPicksChanged.emit()

    # ------------------------------------------------------------------ actions
    def auto(self) -> None:
        d, cdp = self.data, self.current_cdp()
        if d is None or cdp is None or self.spectrum is None or self.mode_combo.currentIndex() == 1:
            return
        p = auto_pick(self.spectrum, d.times, self.axis, max_dv_per_s=2500.0, pick_interval=0.2)
        self.picks.clear(cdp)
        for t, v in p:
            self.picks.add(cdp, t, v, self.eta_spin.value())
        self.draw_gathers()
        self.sigPicksChanged.emit()

    def clear_cdp(self) -> None:
        cdp = self.current_cdp()
        if cdp is not None:
            self.picks.clear(cdp)
            self.draw_gathers()
            self.sigPicksChanged.emit()

    def load_picks(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load velocity picks", "",
                                                        "Velocity picks (*.txt *.vel);;All files (*)")
        if not path:
            return
        try:
            self.project.context.velocity = VelocityPicks.load(path)
        except Exception as exc:                          # noqa: BLE001
            show_error(self, "Load velocity picks", exc)
            return
        self.draw_gathers()
        self.sigPicksChanged.emit()

    def save_picks(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save velocity picks", "velocity.txt",
                                                        "Velocity picks (*.txt)")
        if path:
            try:
                self.picks.save(path)
            except Exception as exc:                      # noqa: BLE001
                show_error(self, "Save velocity picks", exc)
