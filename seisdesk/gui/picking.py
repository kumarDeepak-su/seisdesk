"""The picking tool of the gather viewer.

Picks live in named pick sets; inside a set each gather holds any number of phases. "New phase" is the break: the
picker stops the current phase and starts another, e.g. the direct wave near zero offset, then each refracted branch
as it takes over farther out, or a top mute and a bottom mute. Straight lines fitted to the phases give apparent
velocities, intercept times and, by the intercept-time method, the thickness and depth of each layer.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.picks import PickSet, dix_layers, pick_table, refraction_layers, write_pick_table
from ..processing.firstbreak import mer_picks
from .common import busy, show_error

ROLE = QtCore.Qt.ItemDataRole.UserRole
HELP = ("<b>Left click</b>: pick on the active phase.<br><b>Right click</b> or <b>Shift+click</b>: delete the "
        "nearest pick.<br><b>New phase / break</b> (key <b>N</b>): stop this phase and start the next one, e.g. "
        "direct wave, then each refracted branch as it overtakes the previous one away from zero offset.<br>"
        "Page Up / Page Down: previous / next gather. Drag to pan, wheel to zoom.")


def _color_icon(color: str) -> QtGui.QIcon:
    pix = QtGui.QPixmap(14, 14)
    pix.fill(QtGui.QColor(color))
    return QtGui.QIcon(pix)


def _spin(lo, hi, value, step=1.0, decimals=0, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    return w


class PickingPanel(QtWidgets.QWidget):
    sigChanged = QtCore.Signal()

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.project = viewer.project
        self.fits: list[dict] = []
        self.layers: list[dict] = []
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        lay = QtWidgets.QVBoxLayout(self)

        box = QtWidgets.QGroupBox("Pick set")
        g = QtWidgets.QGridLayout(box)
        self.set_combo = QtWidgets.QComboBox()
        self.set_info = QtWidgets.QLabel("")
        self.set_info.setWordWrap(True)
        b_new, b_load, b_save, b_del = (QtWidgets.QPushButton(t) for t in ("New...", "Load...", "Save...", "Delete"))
        g.addWidget(self.set_combo, 0, 0, 1, 4)
        for i, b in enumerate((b_new, b_load, b_save, b_del)):
            g.addWidget(b, 1, i)
        g.addWidget(self.set_info, 2, 0, 1, 4)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Phases")
        g = QtWidgets.QGridLayout(box)
        self.phase_list = QtWidgets.QListWidget()
        self.phase_list.setMaximumHeight(130)
        b_phase = QtWidgets.QPushButton("New phase / break  [N]")
        b_phase.setToolTip("Stop the current phase and start a new one (e.g. the next refracted branch)")
        b_ren, b_delp, b_clear = (QtWidgets.QPushButton(t) for t in ("Rename...", "Delete phase", "Clear gather"))
        b_clear.setToolTip("Delete every pick of this gather")
        self.enable_check = QtWidgets.QCheckBox("Pick with the mouse")
        self.enable_check.setChecked(True)
        self.snap_combo = QtWidgets.QComboBox()
        self.snap_combo.addItems(["none", "peak", "trough", "zero crossing", "energy onset"])
        self.snap_ms = _spin(1, 500, 20, 1, 0, " ms")
        self.snap_ms.setToolTip("Search window (+/-) for snapping")
        g.addWidget(self.phase_list, 0, 0, 1, 3)
        g.addWidget(b_phase, 1, 0, 1, 3)
        g.addWidget(b_ren, 2, 0)
        g.addWidget(b_delp, 2, 1)
        g.addWidget(b_clear, 2, 2)
        g.addWidget(self.enable_check, 3, 0, 1, 3)
        g.addWidget(QtWidgets.QLabel("Snap to"), 4, 0)
        g.addWidget(self.snap_combo, 4, 1)
        g.addWidget(self.snap_ms, 4, 2)
        help_label = QtWidgets.QLabel(HELP)
        help_label.setWordWrap(True)
        help_label.setProperty("muted", True)
        self.phase_type_combo = QtWidgets.QComboBox()
        for kind in PickSet.PHASE_TYPES:
            self.phase_type_combo.addItem(kind, kind)
        self.phase_type_combo.setToolTip("Each event keeps its own type; only compatible events enter layer analysis")
        g.addWidget(QtWidgets.QLabel("Event type"), 5, 0)
        g.addWidget(self.phase_type_combo, 5, 1, 1, 2)
        g.addWidget(help_label, 6, 0, 1, 3)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Automatic first breaks")
        g = QtWidgets.QGridLayout(box)
        self.vfast = _spin(100, 20000, 6000, 100, 0, " m/s")
        self.vslow = _spin(50, 10000, 300, 50, 0, " m/s")
        self.fb_window = _spin(2, 200, 20, 1, 0, " ms")
        self.fb_margin = _spin(0, 2000, 50, 5, 0, " ms")
        for r, (label, w) in enumerate((("Fastest velocity", self.vfast), ("Slowest velocity", self.vslow),
                                        ("Energy window", self.fb_window), ("Margin", self.fb_margin))):
            g.addWidget(QtWidgets.QLabel(label), r, 0)
            g.addWidget(w, r, 1)
        b_auto1 = QtWidgets.QPushButton("Pick this gather")
        b_auto_all = QtWidgets.QPushButton("Pick all gathers")
        g.addWidget(b_auto1, 4, 0)
        g.addWidget(b_auto_all, 4, 1)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Fits and layers")
        g = QtWidgets.QGridLayout(box)
        self.fit_combo = QtWidgets.QComboBox()
        self.fit_combo.addItem("straight lines: refraction (intercept time)", "line")
        self.fit_combo.addItem("hyperbolas: reflection (NMO velocity, Dix)", "hyperbola")
        self.fit_combo.setToolTip("Refraction branches and the direct wave are straight lines in t-x; reflections "
                                  "are hyperbolas t^2 = t0^2 + x^2/v^2")
        self.side_combo = QtWidgets.QComboBox()
        for text, data in (("|offset|, both sides", "abs"), ("positive offsets", "positive"),
                           ("negative offsets", "negative")):
            self.side_combo.addItem(text, data)
        self.pooled_check = QtWidgets.QCheckBox("Pool all gathers")
        self.show_fits = QtWidgets.QCheckBox("Draw fitted lines")
        self.show_fits.setChecked(True)
        self.fit_table = self._table(("Phase", "Velocity (m/s)", "Intercept (ms)", "RMS (ms)", "Picks"))
        self.layer_table = self._table(("Layer", "Velocity (m/s)", "Thickness (m)", "Base depth (m)",
                                        "Crossover (m)"))
        self.fit_note = QtWidgets.QLabel("")
        self.fit_note.setWordWrap(True)
        b_copy = QtWidgets.QPushButton("Copy tables")
        b_export = QtWidgets.QPushButton("Export picks (CSV)...")
        b_export.setToolTip("Every pick with its gather, phase, x and time, plus source/receiver positions and "
                            "elevations (input for refraction tomography, horizon files)")
        g.addWidget(QtWidgets.QLabel("Analyse"), 0, 0)
        g.addWidget(self.fit_combo, 0, 1)
        g.addWidget(QtWidgets.QLabel("Offsets"), 1, 0)
        g.addWidget(self.side_combo, 1, 1)
        g.addWidget(self.pooled_check, 2, 0)
        g.addWidget(self.show_fits, 2, 1)
        g.addWidget(self.fit_table, 3, 0, 1, 2)
        g.addWidget(self.layer_table, 4, 0, 1, 2)
        g.addWidget(self.fit_note, 5, 0, 1, 2)
        g.addWidget(b_copy, 6, 0)
        g.addWidget(b_export, 6, 1)
        lay.addWidget(box)
        lay.addStretch(1)

        self.set_combo.currentIndexChanged.connect(self._set_changed)
        b_new.clicked.connect(self.new_set)
        b_load.clicked.connect(self.load_set)
        b_save.clicked.connect(self.save_set)
        b_del.clicked.connect(self.delete_set)
        self.phase_list.currentRowChanged.connect(self._phase_selected)
        self.phase_type_combo.currentIndexChanged.connect(self._phase_type_changed)
        b_phase.clicked.connect(self.new_phase)
        b_ren.clicked.connect(self.rename_phase)
        b_delp.clicked.connect(self.delete_phase)
        b_clear.clicked.connect(self.clear_gather)
        b_auto1.clicked.connect(lambda: self.auto_pick(False))
        b_auto_all.clicked.connect(lambda: self.auto_pick(True))
        self.side_combo.currentIndexChanged.connect(lambda _i: self._edited())
        self.fit_combo.currentIndexChanged.connect(self._fit_mode_changed)
        self.pooled_check.toggled.connect(lambda _b: self._edited())
        self.show_fits.toggled.connect(lambda _b: self.sigChanged.emit())
        b_copy.clicked.connect(self.copy_tables)
        b_export.clicked.connect(self.export_table)

    @staticmethod
    def _table(headers) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(list(headers))
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        t.setMinimumHeight(110)
        return t

    # ------------------------------------------------------------------ state
    def current_set(self) -> PickSet | None:
        return self.project.context.picks.get(self.set_combo.currentText())

    def current_phase(self) -> str | None:
        item = self.phase_list.currentItem()
        return item.data(ROLE) if item is not None else None

    def _gather_key(self) -> str:
        return self.viewer.gather_key() or "none"

    def _gather(self) -> float:
        value = self.viewer.current_gather_value()
        return 0.0 if value is None else value

    def _ready(self) -> PickSet | None:
        ps = self.current_set()
        if ps is None:
            self.viewer.sigStatus.emit("Create or load a pick set first (Picking panel, Pick set: New...)")
            return None
        if ps.gather_key != self._gather_key():
            self.viewer.sigStatus.emit(f"Pick set '{ps.name}' is organised by '{ps.gather_key}'; choose that gather "
                                       f"type to pick")
            return None
        return ps

    def refresh_sets(self, select: str | None = None) -> None:
        current = select or self.set_combo.currentText()
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        self.set_combo.addItems(list(self.project.context.picks))
        i = self.set_combo.findText(current)
        if i >= 0:
            self.set_combo.setCurrentIndex(i)
        self.set_combo.blockSignals(False)
        self._set_changed()

    def refresh_phases(self, select: str | None = None) -> None:
        ps = self.current_set()
        current = select or self.current_phase()
        self.phase_list.blockSignals(True)
        self.phase_list.clear()
        if ps is not None:
            for ph in ps.phase_order:
                item = QtWidgets.QListWidgetItem(_color_icon(ps.colors.get(ph, "#000000")), ph)
                item.setData(ROLE, ph)
                self.phase_list.addItem(item)
            names = list(ps.phase_order)
            if names:
                self.phase_list.setCurrentRow(names.index(current) if current in names else len(names) - 1)
        self.phase_list.blockSignals(False)
        self._phase_selected()
        self._edited()

    def _phase_selected(self, _row=None) -> None:
        ps, ph = self.current_set(), self.current_phase()
        self.phase_type_combo.blockSignals(True)
        self.phase_type_combo.setCurrentIndex(self.phase_type_combo.findData(ps.phase_type(ph))
                                             if ps is not None and ph is not None else 0)
        self.phase_type_combo.blockSignals(False)
        self.sigChanged.emit()

    def _phase_type_changed(self, _index=None) -> None:
        ps, ph = self.current_set(), self.current_phase()
        if ps is not None and ph is not None:
            ps.set_phase_type(ph, self.phase_type_combo.currentData())
            self._edited()

    def _fit_mode_changed(self, _index=None) -> None:
        # An empty event may follow the chosen analysis; populated events never change type implicitly.
        ps, ph = self.current_set(), self.current_phase()
        if ps is not None and ph is not None and ps.gathers(ph).size == 0:
            ps.set_phase_type(ph, "reflection" if self.fit_combo.currentData() == "hyperbola" else "refraction")
            self._phase_selected()
        self._edited()

    def _new_event_type(self, ps) -> str:
        if ps.gather_key == "none":
            return "horizon"
        if self.fit_combo.currentData() == "hyperbola":
            return "reflection"
        return "direct" if not ps.phase_order else "refraction"

    def _set_changed(self, _i=None) -> None:
        ps = self.current_set()
        if ps is not None and ps.gather_key != self._gather_key() and self.viewer.data is not None:
            if ps.gather_key == "none":
                self.viewer.key_combo.setCurrentIndex(0)
            elif not self.viewer.set_gather_key(ps.gather_key):
                self.viewer.sigStatus.emit(f"this dataset has no '{ps.gather_key}' header for pick set '{ps.name}'")
        self.refresh_phases()

    def dataset_changed(self) -> None:
        self.refresh_sets()

    def gather_changed(self) -> None:
        self.update_fits()

    def _edited(self) -> None:
        self.update_fits()
        self.sigChanged.emit()

    # ------------------------------------------------------------------ pick sets
    def new_set(self) -> None:
        d = self.viewer.data
        if d is None:
            return
        k = len(self.project.context.picks) + 1
        name, ok = QtWidgets.QInputDialog.getText(self, "New pick set", "Name:", text=f"picks {k}")
        name = name.strip()
        if not ok or not name:
            return
        if name in self.project.context.picks:
            show_error(self, "New pick set", f"A pick set named '{name}' exists already.")
            return
        if self._gather_key() == "none":            # a stacked or migrated section: picks are horizons along CMP
            x_key = self.viewer.x_key() or ("cdp" if "cdp" in d.headers else "tracl")
        else:
            x_key = "offset" if "offset" in d.headers else (self.viewer.x_key() or "tracf")
        ps = PickSet(self._gather_key(), x_key, name)
        ps.new_phase(kind=self._new_event_type(ps))
        self.project.context.picks[name] = ps
        self.refresh_sets(select=name)
        self.viewer.sigStatus.emit(f"pick set '{name}': gathers by {ps.gather_key}, x = {x_key}")

    def load_set(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load pick set", "", "Pick sets (*.json);;All files (*)")
        if not path:
            return
        try:
            ps = PickSet.load(path)
        except Exception as exc:                          # noqa: BLE001
            show_error(self, "Load pick set", exc)
            return
        name, k = ps.name, 2
        while name in self.project.context.picks:
            name = f"{ps.name} ({k})"
            k += 1
        ps.name = name
        self.project.context.picks[name] = ps
        self.refresh_sets(select=name)

    def save_set(self) -> None:
        ps = self.current_set()
        if ps is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save pick set", f"{ps.name}.json",
                                                        "Pick sets (*.json)")
        if path:
            try:
                ps.save(path)
            except Exception as exc:                      # noqa: BLE001
                show_error(self, "Save pick set", exc)

    def delete_set(self) -> None:
        ps = self.current_set()
        if ps is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete pick set", f"Delete pick set '{ps.name}' and its "
                                                f"{ps.count()} picks?")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.project.context.picks.pop(ps.name, None)
            self.refresh_sets()

    # ------------------------------------------------------------------ phases
    def new_phase(self) -> None:
        ps = self.current_set()
        if ps is None:
            self.viewer.sigStatus.emit("Create a pick set first")
            return
        name = ps.new_phase(kind=self._new_event_type(ps))
        self.refresh_phases(select=name)
        self.viewer.sigStatus.emit(f"new phase '{name}' started; picks now go to it")

    def rename_phase(self) -> None:
        ps, ph = self.current_set(), self.current_phase()
        if ps is None or ph is None:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename phase", "New name (e.g. direct, refractor 1):", text=ph)
        if ok and new.strip() and new.strip() != ph:
            try:
                ps.rename_phase(ph, new.strip())
            except ValueError as exc:
                show_error(self, "Rename phase", exc)
                return
            self.refresh_phases(select=new.strip())

    def delete_phase(self) -> None:
        ps, ph = self.current_set(), self.current_phase()
        if ps is None or ph is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete phase", f"Delete phase '{ph}' in every gather?")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            ps.delete_phase(ph)
            self.refresh_phases()

    def clear_gather(self) -> None:
        ps = self._ready()
        if ps is not None:
            ps.clear_gather(self._gather())
            self._edited()

    # ------------------------------------------------------------------ mouse picking
    def click(self, viewer, j: int, info: dict) -> None:
        if not self.enable_check.isChecked():
            return
        ps = self._ready()
        if ps is None:
            return
        d = viewer.data
        tr = int(viewer.idx[j])
        gather = self._gather()
        xval = float(d.header(ps.x_key)[tr])
        if info["right"] or info["shift"]:
            xs = np.sort(d.header(ps.x_key)[viewer.idx])
            gaps = np.diff(xs)
            gaps = gaps[gaps > 0]
            x_scale = 3.0 * float(np.median(gaps)) if gaps.size else 1.0
            y_range = viewer.plot.getPlotItem().vb.viewRange()[1]
            t_scale = 0.04 * abs(y_range[1] - y_range[0]) or 0.02
            phase = self.current_phase()
            if not ps.remove_nearest(gather, xval, info["y"], phase, x_scale, t_scale, max_distance=1.0, trace_id=d.trace_ids[tr]):
                ps.remove_nearest(gather, xval, info["y"], None, x_scale, t_scale, max_distance=1.0, trace_id=d.trace_ids[tr])
        elif info["left"]:
            phase = self.current_phase()
            if phase is None:
                phase = ps.new_phase(kind=self._new_event_type(ps))
                self.refresh_phases(select=phase)
            ps.add(gather, phase, xval, self.snap(d, tr, info["y"]), trace_id=d.trace_ids[tr])
        self._edited()

    def snap(self, d, tr: int, t: float) -> float:
        mode = self.snap_combo.currentText()
        if mode == "none":
            return float(t)
        win = self.snap_ms.value() / 1000.0 if d.domain == "time" else self.snap_ms.value()
        times = d.times
        trace = d.traces[tr].astype(np.float64)
        i0 = max(0, int(np.floor((t - win - d.t0) / d.dt)))
        i1 = min(d.nt, int(np.ceil((t + win - d.t0) / d.dt)) + 1)
        if i1 - i0 < 3:
            return float(t)
        seg = trace[i0:i1]
        if mode == "zero crossing":
            s = np.flatnonzero(seg[:-1] * seg[1:] < 0)
            if s.size == 0:
                return float(t)
            tz = times[i0 + s] + d.dt * seg[s] / (seg[s] - seg[s + 1])
            return float(tz[np.argmin(np.abs(tz - t))])
        if mode == "energy onset":
            return float(mer_picks(trace[None, :], times, max(3 * d.dt, win / 2), np.array([times[i0]]),
                                   np.array([times[i1 - 1]]))[0])
        k = int(np.argmax(seg) if mode == "peak" else np.argmin(seg))
        i = i0 + k
        if 0 < i < d.nt - 1:
            y0, y1, y2 = trace[i - 1], trace[i], trace[i + 1]
            den = y0 - 2.0 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            if abs(frac) <= 1.0:
                return float(times[i] + frac * d.dt)
        return float(times[i])

    def auto_pick(self, all_gathers: bool) -> None:
        d = self.viewer.data
        ps = self._ready()
        if d is None or ps is None:
            return
        if "offset" not in d.headers:
            show_error(self, "First breaks", "The dataset has no offsets; run a geometry step first.")
            return
        phase = self.current_phase() or ps.new_phase()
        key = self.viewer.gather_key()
        if all_gathers and key is not None:
            groups = list(d.iter_gathers(key, None))
        else:
            groups = [(self._gather(), self.viewer.idx)]
        x = np.abs(d.header("offset"))
        vfast, vslow = self.vfast.value(), self.vslow.value()
        margin, window = self.fb_margin.value() / 1000.0, self.fb_window.value() / 1000.0
        xkey = d.header(ps.x_key)
        with busy():
            for g, idx in groups:
                tmin = np.maximum(x[idx] / vfast - margin, d.t0)
                tmax = np.minimum(x[idx] / vslow + margin, d.tmax)
                fb = mer_picks(d.traces[idx], d.times, window, tmin, tmax)
                for tr, xv, tv in zip(idx, xkey[idx], fb):
                    ps.add(g, phase, float(xv), float(tv), trace_id=d.trace_ids[tr])
        self.refresh_phases(select=phase)
        self.viewer.sigStatus.emit(f"first breaks picked in {len(groups)} gather(s) into phase '{phase}'")

    # ------------------------------------------------------------------ fits
    def update_fits(self) -> None:
        self.fits, self.layers = [], []
        ps = self.current_set()
        self.fit_table.setRowCount(0)
        self.layer_table.setRowCount(0)
        if ps is None:
            self.set_info.setText("No pick set. Press New... to start picking.")
            self.fit_note.setText("")
            return
        self.set_info.setText(f"Gathers by <b>{ps.gather_key}</b>, x = <b>{ps.x_key}</b>; {ps.count()} picks in "
                              f"{len(ps.gathers())} gathers, {len(ps.phase_order)} phases")
        kind = self.fit_combo.currentData()
        if kind == "hyperbola":
            self.fit_table.setHorizontalHeaderLabels(["Phase", "NMO velocity (m/s)", "t0 (ms)", "RMS (ms)", "Picks"])
            self.layer_table.setHorizontalHeaderLabels(["Layer", "Interval velocity (m/s)", "Thickness (m)",
                                                        "Base depth (m)", "Reflector t0 (ms)"])
        else:
            self.fit_table.setHorizontalHeaderLabels(["Phase", "Velocity (m/s)", "Intercept (ms)", "RMS (ms)", "Picks"])
            self.layer_table.setHorizontalHeaderLabels(["Layer", "Velocity (m/s)", "Thickness (m)", "Base depth (m)",
                                                        "Crossover (m)"])
        if ps.gather_key == "none":
            self.fit_note.setText("Picks on a section are horizons: no fits. Export them with Export picks.")
            return
        gather = None if self.pooled_check.isChecked() else self._gather()
        side = self.side_combo.currentData()
        eligible = {"reflection"} if kind == "hyperbola" else {"direct", "refraction"}
        for ph in ps.phase_order:
            if ps.phase_type(ph) not in eligible:
                continue
            try:
                self.fits.append(ps.fit_hyperbola(gather, ph, side) if kind == "hyperbola"
                                 else ps.fit_line(gather, ph, side))
            except ValueError:
                continue
        self.fit_table.setRowCount(len(self.fits))
        for r, f in enumerate(self.fits):
            vel = "inf" if not np.isfinite(f["velocity"]) else f"{f['velocity']:.0f}"
            for c, text in enumerate((f["phase"], vel, f"{f['intercept'] * 1000:.1f}", f"{f['rms'] * 1000:.2f}",
                                      str(f["n"]))):
                self.fit_table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        if kind == "hyperbola":
            self._reflection_layers()
            return
        ordered = sorted((f for f in self.fits if np.isfinite(f["velocity"]) and f["velocity"] > 0),
                         key=lambda f: f["velocity"])
        if len(ordered) < 2:
            self.fit_note.setText("Pick the direct wave and each refracted branch as separate phases (use the "
                                  "break) to get layer thicknesses.")
            return
        try:
            self.layers = refraction_layers(ordered)
        except ValueError as exc:
            self.fit_note.setText(str(exc))
            return
        self.fit_note.setText("Phases ordered by velocity: " + " < ".join(f["phase"] for f in ordered) +
                              ". Horizontal layers assumed; use both shooting directions for dipping refractors.")
        self.layer_table.setRowCount(len(self.layers))
        for r, L in enumerate(self.layers):
            for c, text in enumerate((str(L["layer"]), f"{L['velocity']:.0f}", f"{L['thickness']:.1f}",
                                      f"{L['depth_to_base']:.1f}", f"{L['crossover_distance']:.0f}")):
                self.layer_table.setItem(r, c, QtWidgets.QTableWidgetItem(text))

    def _reflection_layers(self) -> None:
        if not self.fits:
            self.fit_note.setText("Pick each reflection as its own phase (at least three picks across offsets) to "
                                  "get its NMO velocity and t0.")
            return
        try:
            self.layers = dix_layers(self.fits)
        except ValueError as exc:
            self.fit_note.setText(str(exc))
            return
        self.fit_note.setText("Reflections ordered by t0; interval velocities by Dix's equation, flat layers assumed. "
                              "Use a CMP gather (or a shot over flat layers) for NMO velocities.")
        self.layer_table.setRowCount(len(self.layers))
        for r, L in enumerate(self.layers):
            for c, text in enumerate((f"{L['layer']} (to {L['reflector']})", f"{L['velocity']:.0f}",
                                      f"{L['thickness']:.1f}", f"{L['depth_to_base']:.1f}", f"{L['t0'] * 1000:.1f}")):
                self.layer_table.setItem(r, c, QtWidgets.QTableWidgetItem(text))

    def copy_tables(self) -> None:
        lines = []
        for table in (self.fit_table, self.layer_table):
            if table.rowCount() == 0:
                continue
            heads = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
            lines.append("\t".join(heads))
            for r in range(table.rowCount()):
                lines.append("\t".join((table.item(r, c).text() if table.item(r, c) else "")
                                       for c in range(table.columnCount())))
            lines.append("")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.viewer.sigStatus.emit("fit and layer tables copied to the clipboard")

    def export_table(self) -> None:
        ps = self.current_set()
        if ps is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export picks", f"{ps.name}.csv", "CSV table (*.csv)")
        if not path:
            return
        try:
            write_pick_table(pick_table(ps, self.viewer.data), path)
        except Exception as exc:                          # noqa: BLE001
            show_error(self, "Export picks", exc)
            return
        self.viewer.sigStatus.emit(f"{ps.count()} picks written to {path}")

    # ------------------------------------------------------------------ drawing
    def draw(self, viewer) -> None:
        ps = self.current_set()
        if ps is None or ps.gather_key != self._gather_key() or ps.x_key not in viewer.data.headers:
            return
        gather = self._gather()
        active = self.current_phase()
        for ph in ps.phase_order:
            records = ps.records(gather, ph)
            displayed = {str(viewer.data.trace_ids[i]): j for j, i in enumerate(viewer.idx)}
            valid = [(x, t, tid) for x, t, tid in records if tid is None or tid in displayed]
            if not valid:
                continue
            pts = np.array([(x, t) for x, t, _ in valid])
            color = ps.colors.get(ph, "#000000")
            big = ph == active
            # An identity-bearing pick maps directly to the displayed trace, even for equal offsets.
            locations = [viewer.xpos[displayed[tid]] if tid is not None else
                         viewer.display_x(ps.x_key, np.array([x]))[0] for x, _, tid in valid]
            for (x, t, _), px in zip(valid, locations):
                viewer.add_overlay(pg.PlotDataItem([px], [t], pen=None, symbol="o",
                                                   symbolSize=7 if big else 5, symbolBrush=pg.mkBrush(color),
                                                   symbolPen=pg.mkPen(color)))
            # each side of the shot separately, so the line never jumps across the gap at the source
            sides = ((pts[pts[:, 0] < 0], pts[pts[:, 0] >= 0]) if ps.gather_key != "none" else (pts,))
            if np.unique(viewer.data.header(ps.x_key)[viewer.idx]).size != viewer.idx.size:
                sides = ()
            for seg in sides:
                if seg.shape[0] == 0:
                    continue
                px = viewer.display_x(ps.x_key, seg[:, 0])
                order = np.argsort(px, kind="stable")
                viewer.add_overlay(pg.PlotDataItem(px[order], seg[order, 1],
                                                   pen=pg.mkPen(color, width=3 if big else 1.5), symbol="o",
                                                   symbolSize=7 if big else 5, symbolBrush=pg.mkBrush(color),
                                                   symbolPen=pg.mkPen(color)))
        if (not self.show_fits.isChecked() or
                np.unique(viewer.data.header(ps.x_key)[viewer.idx]).size != viewer.idx.size):
            return
        side = self.side_combo.currentData()
        for f in self.fits:
            pts = ps.points(gather, f["phase"])
            if pts.shape[0] == 0:
                continue
            xs = pts[:, 0]
            if side == "positive":
                xs = xs[xs >= 0]
            elif side == "negative":
                xs = xs[xs <= 0]
            if xs.size == 0:
                continue
            color = ps.colors.get(f["phase"], "#000000")
            end = None
            for seg in (xs[xs < 0], xs[xs >= 0]):         # each side of the shot over its own picked range
                if seg.size == 0:
                    continue
                xline = np.linspace(seg.min(), seg.max(), 32)
                if f.get("kind") == "hyperbola":
                    tline = np.sqrt(f["t0"] ** 2 + (xline / f["velocity"]) ** 2)
                else:
                    tline = f["intercept"] + f["slowness"] * np.abs(xline)
                px = viewer.display_x(ps.x_key, xline)
                order = np.argsort(px, kind="stable")
                viewer.add_overlay(pg.PlotDataItem(px[order], tline[order],
                                                   pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine)))
                if end is None or abs(xline[-1]) > abs(end[0]):
                    end = (xline[-1], float(px[order][-1]), float(tline[order][-1]))
            if end is not None:
                vel = "inf" if not np.isfinite(f["velocity"]) else f"{f['velocity']:.0f}"
                what = "t0" if f.get("kind") == "hyperbola" else "ti"
                label = pg.TextItem(f"{f['phase']}: {vel} m/s, {what} {f['intercept'] * 1000:.1f} ms", color=color,
                                    anchor=(0, 1))
                label.setPos(end[1], end[2])
                viewer.add_overlay(label)
