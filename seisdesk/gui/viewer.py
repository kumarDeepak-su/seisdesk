"""Gather viewer: navigate a dataset by FFID, shot, receiver, CMP, offset, inline, crossline or any header; density and
wiggle display; trace read-out; and the multi-phase picking tool."""

from __future__ import annotations

import numpy as np
import html
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.headers import DESCRIPTIONS, GATHER_KEYS
from .common import DISPLAY_GAINS, SEISMIC_MAPS, clip_value, display_gain, fmt, show_error
from .picking import PickingPanel
from .plot import DISPLAY_MODES, SeismicPlot

ALL_TRACES = "__all__"
TRACE_AXIS = "__trace__"
ABS_OFFSET = "__absoffset__"
MAX_DISPLAY_TRACES = 6000


def reduce_display(traces: np.ndarray, dt: float, shifts: np.ndarray) -> np.ndarray:
    """Traces moved earlier by ``shifts`` seconds (reduced time t - |x| / v) by linear interpolation, zero beyond the
    record; for display only."""
    ntr, nt = traces.shape
    if nt < 2 or ntr == 0:
        return traces.copy()
    pos = np.arange(nt)[None, :] + (np.asarray(shifts, dtype=np.float64) / dt)[:, None]
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    ok = (i0 >= 0) & (i0 < nt - 1)
    i0 = np.clip(i0, 0, nt - 2)
    rows = np.arange(ntr)[:, None]
    out = (1.0 - frac) * traces[rows, i0] + frac * traces[rows, i0 + 1]
    return np.where(ok, out, 0.0).astype(np.float32)


class GatherViewer(QtWidgets.QWidget):
    sigStatus = QtCore.Signal(str)
    sigPicksChanged = QtCore.Signal()

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.data = None
        self.dataset_name = None
        self.gather_values = np.zeros(0)
        self.idx = np.zeros(0, dtype=np.int64)
        self.xpos = np.zeros(0)
        self.disp = None
        self.x_is_header = False
        self.overlay_items = []
        self.reduced_shift = None             # per displayed trace |offset| / v while reduced time is shown
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.dataset_label = QtWidgets.QLabel("<i>no dataset</i>")
        self.dataset_label.setWordWrap(True)
        self.dataset_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.key_combo = QtWidgets.QComboBox()
        self.key_combo.setToolTip("Gather type: the header that defines a gather")
        self.prev_btn = QtWidgets.QPushButton("< Prev")
        self.next_btn = QtWidgets.QPushButton("Next >")
        self.prev_btn.setToolTip("Previous gather (Page Up)")
        self.next_btn.setToolTip("Next gather (Page Down)")
        self.value_combo = QtWidgets.QComboBox()
        self.value_combo.setMinimumWidth(90)
        self.value_combo.setToolTip("Gather to display")
        self.step_spin = QtWidgets.QSpinBox()
        self.step_spin.setRange(1, 100000)
        self.step_spin.setPrefix("step ")
        self.step_spin.setToolTip("Gathers skipped by Prev / Next")
        self.sort_combo = QtWidgets.QComboBox()
        self.sort_combo.setToolTip("Order of the traces inside the gather")
        self.xaxis_combo = QtWidgets.QComboBox()
        self.xaxis_combo.setToolTip("Horizontal axis")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(DISPLAY_MODES)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(SEISMIC_MAPS)
        self.clip_spin = QtWidgets.QDoubleSpinBox()
        self.clip_spin.setRange(50.0, 100.0)
        self.clip_spin.setDecimals(1)
        self.clip_spin.setValue(98.0)
        self.clip_spin.setSuffix(" %")
        self.clip_spin.setToolTip("Amplitude clip: percentile of |amplitude| shown at full colour")
        self.gain_combo = QtWidgets.QComboBox()
        self.gain_combo.addItems(DISPLAY_GAINS)
        self.gain_combo.setToolTip("Gain for display only; the data are not changed")
        self.reduce_spin = QtWidgets.QDoubleSpinBox()
        self.reduce_spin.setRange(0.0, 20000.0)
        self.reduce_spin.setDecimals(0)
        self.reduce_spin.setSingleStep(100.0)
        self.reduce_spin.setSuffix(" m/s")
        self.reduce_spin.setSpecialValueText("off")
        self.reduce_spin.setToolTip("Reduction velocity for display: time t - |offset| / v, so that arrivals at this "
                                    "velocity are flat (off = 0). Display only; picking pauses while it is on.")
        self.keep_zoom = QtWidgets.QCheckBox("Keep zoom")
        self.pick_btn = QtWidgets.QPushButton("Picking")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip("Show the picking tool: first breaks, refraction branches, mutes")
        self.reset_btn = QtWidgets.QPushButton("Reset view")
        self.display_options = QtWidgets.QPushButton("Display options")
        self.display_options.setCheckable(True)
        self.display_options.setToolTip("Colour map, amplitude clip, display gain, axes and reduction velocity")

        row1 = QtWidgets.QHBoxLayout()
        for label, w in (("Gather", self.key_combo), (None, self.prev_btn), (None, self.value_combo),
                         (None, self.next_btn), (None, self.step_spin)):
            if label:
                row1.addWidget(QtWidgets.QLabel(label))
            row1.addWidget(w)
        row1.addStretch(1)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Display"))
        row2.addWidget(self.mode_combo)
        row2.addWidget(self.display_options)
        row2.addWidget(self.keep_zoom)
        row2.addWidget(self.reset_btn)
        row2.addStretch(1)
        row2.addWidget(self.pick_btn)
        self.display_panel = QtWidgets.QWidget()
        options = QtWidgets.QGridLayout(self.display_panel)
        options.setContentsMargins(0, 4, 0, 8)
        for i, (title, widget) in enumerate((("Colours", self.cmap_combo), ("Gain", self.gain_combo),
                                            ("Clip", self.clip_spin), ("Sort", self.sort_combo),
                                            ("X axis", self.xaxis_combo), ("Reduce", self.reduce_spin))):
            row, col = divmod(i, 3)
            options.addWidget(QtWidgets.QLabel(title), row, col * 2)
            options.addWidget(widget, row, col * 2 + 1)
        self.display_panel.hide()
        self.display_options.toggled.connect(self.display_panel.setVisible)

        self.plot = SeismicPlot(self)
        self.picking = PickingPanel(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.picking)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        scroll.hide()
        self.picking_scroll = scroll
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self.plot)
        split.addWidget(scroll)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        self.readout = QtWidgets.QLabel(" ")
        self.readout.setWordWrap(True)
        self.readout.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.dataset_label)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self.display_panel)
        layout.addWidget(split, 1)
        layout.addWidget(self.readout)

        self.key_combo.currentIndexChanged.connect(self._key_changed)
        self.value_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn.clicked.connect(lambda: self.step(+1))
        for combo in (self.sort_combo, self.xaxis_combo, self.gain_combo):
            combo.currentIndexChanged.connect(lambda _i: self.refresh())
        self.mode_combo.currentIndexChanged.connect(lambda _i: self.refresh(keep_view=True))
        self.cmap_combo.currentIndexChanged.connect(lambda _i: self.refresh(keep_view=True))
        self.clip_spin.valueChanged.connect(lambda _v: self.refresh(keep_view=True))
        self.reduce_spin.valueChanged.connect(self._reduction_changed)
        self.reset_btn.clicked.connect(lambda: self.plot.reset_view())
        self.pick_btn.toggled.connect(self._toggle_picking)
        self.plot.sigHover.connect(self._hover)
        self.plot.sigClick.connect(self._click)
        self.picking.sigChanged.connect(self._picks_changed)
        for seq, fn in (("PgDown", lambda: self.step(+1)), ("PgUp", lambda: self.step(-1)),
                        ("N", self.picking.new_phase), ("B", self.picking.new_phase)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ------------------------------------------------------------------ dataset
    def set_dataset(self, name: str | None) -> None:
        self.dataset_name = name
        self.data = self.project.datasets.get(name) if name else None
        if self.data is None:
            self.dataset_label.setText("<i>no dataset</i>")
            self.plot.clear_display()
            self._clear_overlays()
            return
        d = self.data
        self.dataset_label.setText(f"<b>{html.escape(name)}</b>  ·  {d.ntr:,} traces  ·  {d.nt:,} samples")
        self.plot.getPlotItem().setLabel("left", "Time" if d.domain == "time" else "Depth",
                                         units="s" if d.domain == "time" else "m")
        stacked = d.has_header("cdp") and np.unique(d.header("cdp")).size == d.ntr
        for combo in (self.key_combo, self.sort_combo, self.xaxis_combo):
            combo.blockSignals(True)
            combo.clear()
        self.key_combo.addItem("All traces (section)", ALL_TRACES)
        present = [k for k in d.headers if np.unique(d.headers[k]).size > 1]
        for key, text in GATHER_KEYS.items():
            if key in present:
                self.key_combo.addItem(f"{text} ({key})", key)
        for key in sorted(present):
            if key not in GATHER_KEYS:
                self.key_combo.addItem(key, key)
        self.sort_combo.addItem("file order", "")
        self.sort_combo.addItem("offset (signed)", "offset")
        self.sort_combo.addItem("|offset|", ABS_OFFSET)
        for key in ("gx", "sx", "tracf", "cdp", "fldr", "ep", "iline", "xline", "cdpx"):
            if key in present:
                self.sort_combo.addItem(key, key)
        self.xaxis_combo.addItem("trace number", TRACE_AXIS)
        for key in ("offset", "gx", "sx", "cdp", "cdpx", "tracf", "fldr", "ep", "iline", "xline"):
            if key in present:
                self.xaxis_combo.addItem(key, key)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText("Density" if (stacked or d.domain == "depth" or "fldr" not in present)
                                       else "Density + wiggle")
        self.mode_combo.blockSignals(False)
        if stacked or d.domain == "depth" or "fldr" not in present:
            self._select(self.key_combo, ALL_TRACES)
            self._select(self.sort_combo, "cdp" if "cdp" in present else "")
            self._select(self.xaxis_combo, "cdp" if "cdp" in present else TRACE_AXIS)
        else:
            self._select(self.key_combo, "fldr")
            self._select(self.sort_combo, "offset" if "offset" in present else "")
            self._select(self.xaxis_combo, "offset" if "offset" in present else TRACE_AXIS)
        for combo in (self.key_combo, self.sort_combo, self.xaxis_combo):
            combo.blockSignals(False)
        self._fill_values()
        blocked = self.picking.blockSignals(True)
        try:
            self.picking.dataset_changed()
        finally:
            self.picking.blockSignals(blocked)
        self.refresh(keep_view=False)

    def reload(self) -> None:
        """The dataset object changed under the same name (e.g. after a header edit)."""
        self.set_dataset(self.dataset_name)

    @staticmethod
    def _select(combo: QtWidgets.QComboBox, data) -> None:
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def gather_key(self) -> str | None:
        key = self.key_combo.currentData()
        return None if key in (None, ALL_TRACES) else key

    def set_gather_key(self, key: str) -> bool:
        i = self.key_combo.findData(key)
        if i < 0:
            return False
        self.key_combo.setCurrentIndex(i)
        return True

    def x_key(self) -> str | None:
        key = self.xaxis_combo.currentData()
        return None if key in (None, TRACE_AXIS) else key

    def _key_changed(self, _i=None) -> None:
        self._fill_values()
        if self.gather_key() == "cdp":
            for combo, value in ((self.sort_combo, ABS_OFFSET), (self.xaxis_combo, "offset")):
                combo.blockSignals(True)
                self._select(combo, value)
                combo.blockSignals(False)
        self.refresh(keep_view=False)

    def _fill_values(self) -> None:
        self.value_combo.blockSignals(True)
        self.value_combo.clear()
        key = self.gather_key()
        if self.data is None or key is None:
            self.gather_values = np.zeros(0)
            self.value_combo.setEnabled(False)
        else:
            self.gather_values = self.data.gather_values(key)
            self.value_combo.addItems([fmt(v) for v in self.gather_values])
            self.value_combo.setEnabled(True)
        self.value_combo.blockSignals(False)
        for w in (self.prev_btn, self.next_btn, self.step_spin):
            w.setEnabled(self.gather_values.size > 1)

    def current_gather_value(self) -> float | None:
        if self.gather_key() is None or self.gather_values.size == 0:
            return None
        i = max(0, self.value_combo.currentIndex())
        return float(self.gather_values[min(i, self.gather_values.size - 1)])

    def show_gather(self, value: float) -> None:
        if self.gather_values.size == 0:
            return
        i = int(np.argmin(np.abs(self.gather_values - value)))
        self.value_combo.setCurrentIndex(i)

    def step(self, direction: int) -> None:
        n = self.gather_values.size
        if n < 2:
            return
        i = self.value_combo.currentIndex() + direction * self.step_spin.value()
        self.value_combo.setCurrentIndex(int(min(max(i, 0), n - 1)))

    # ------------------------------------------------------------------ display
    def _current_indices(self) -> np.ndarray:
        d = self.data
        key = self.gather_key()
        if key is None:
            idx = np.arange(d.ntr)
        else:
            value = self.current_gather_value()
            idx = d.gather_index(key, value, secondary=None)
        sort = self.sort_combo.currentData()
        if sort and idx.size > 1:
            vals = np.abs(d.header("offset")[idx]) if sort == ABS_OFFSET else d.header(sort)[idx]
            idx = idx[np.argsort(vals, kind="stable")]
        if idx.size > MAX_DISPLAY_TRACES:
            step = int(np.ceil(idx.size / MAX_DISPLAY_TRACES))
            idx = idx[::step]
            self.sigStatus.emit(f"showing every {step}th trace of {d.ntr} (choose a gather type to see all traces)")
        return idx

    def refresh(self, keep_view: bool = True) -> None:
        d = self.data
        if d is None:
            return
        try:
            idx = self._current_indices()
            if idx.size == 0:
                self.plot.clear_display()
                return
            self.idx = idx
            disp = display_gain(d.traces[idx], d.times, d.dt, self.gain_combo.currentText(), d.domain)
            v_red = self.reduction_velocity()
            if v_red > 0:
                self.reduced_shift = np.abs(d.header("offset")[idx]) / v_red
                disp = reduce_display(disp, d.dt, self.reduced_shift)
            else:
                self.reduced_shift = None
            reduced = self.reduced_shift is not None
            if reduced and self.pick_btn.isChecked():
                self.pick_btn.setChecked(False)               # picks live in true time: picking pauses
            self.pick_btn.setEnabled(not reduced)
            self._time_label()
            self.disp = disp
            xkey = self.x_key()
            xpos = d.header(xkey)[idx].astype(np.float64) if xkey else np.arange(1, idx.size + 1, dtype=np.float64)
            clip = clip_value(disp, self.clip_spin.value())
            self.xpos = self.plot.show_traces(disp, d.t0, d.dt, xpos, self.mode_combo.currentText(),
                                              self.cmap_combo.currentText(), clip)
            self.x_is_header = xkey is not None and self.xpos is xpos
            self.plot.getPlotItem().setLabel("bottom", (xkey or "trace number") if self.xpos is xpos or xkey is None
                                             else "trace number (header not usable as an axis)")
            if not (keep_view or self.keep_zoom.isChecked()):
                self.plot.reset_view()
            elif not self.keep_zoom.isChecked():
                self.plot.reset_view(x_only=True)
            self.draw_overlays()
            self.picking.gather_changed()
        except Exception as exc:                          # noqa: BLE001
            show_error(self, "Display", exc)

    def displayed_data(self):
        """The displayed traces as a SeismicData (for spectra, header views, ...)."""
        if self.data is None or self.idx.size == 0:
            return None
        return self.data.subset(self.idx)

    def visible_data(self):
        """The traces and the time (depth) window currently in view: zoom in to analyse part of a gather."""
        d = self.displayed_data()
        if d is None:
            return None
        (x0, x1), (y0, y1) = self.plot.getPlotItem().vb.viewRange()
        keep = (self.xpos >= min(x0, x1)) & (self.xpos <= max(x0, x1))
        if keep.sum() >= 1:
            d = d.subset(np.flatnonzero(keep))
        lo, hi = max(min(y0, y1), d.t0), min(max(y0, y1), d.tmax)
        if hi - lo >= 8 * d.dt and self.reduced_shift is None:        # reduced times are not the traces' times
            d = d.time_window(lo, hi)
        return d

    # ------------------------------------------------------------------ reduced time
    def reduction_velocity(self) -> float:
        """The display reduction velocity (0 = off; always off for depth data)."""
        if self.data is None or self.data.domain != "time":
            return 0.0
        return float(self.reduce_spin.value())

    def _time_label(self) -> None:
        d = self.data
        if d is None:
            return
        if self.reduced_shift is not None:
            self.plot.getPlotItem().setLabel("left", f"Reduced time t - |offset| / {self.reduction_velocity():g} m/s",
                                             units="s")
        else:
            self.plot.getPlotItem().setLabel("left", "Time" if d.domain == "time" else "Depth",
                                             units="s" if d.domain == "time" else "m")

    def _reduction_changed(self, _value=None) -> None:
        self.refresh(keep_view=False)

    # ------------------------------------------------------------------ overlays (picks)
    def _clear_overlays(self) -> None:
        for item in self.overlay_items:
            self.plot.removeItem(item)
        self.overlay_items = []

    def add_overlay(self, item) -> None:
        self.plot.addItem(item, ignoreBounds=True)
        self.overlay_items.append(item)

    def display_x(self, x_key: str, values: np.ndarray) -> np.ndarray:
        """Display positions of header values ``values`` of ``x_key`` (via the nearest displayed trace)."""
        values = np.asarray(values, dtype=np.float64)
        if self.data is None or self.idx.size == 0:
            return values
        if x_key == self.x_key() and self.x_is_header:
            return values
        xk = self.data.header(x_key)[self.idx]
        j = np.argmin(np.abs(xk[None, :] - values[:, None]), axis=1)
        return self.xpos[j]

    def draw_overlays(self) -> None:
        self._clear_overlays()
        if self.data is None or self.picking_scroll.isHidden():
            return
        self.picking.draw(self)

    # ------------------------------------------------------------------ mouse
    def _hover(self, x: float, y: float) -> None:
        d = self.data
        if d is None or self.idx.size == 0:
            return
        j = self.plot.nearest_trace(x)
        if j is None:
            return
        tr = int(self.idx[j])
        it = int(round((y - d.t0) / d.dt))
        amp = float(self.disp[j, it]) if self.disp is not None and 0 <= it < d.nt else float("nan")
        parts = [f"trace {tr + 1}"]
        for key in ("fldr", "ep", "tracf", "cdp", "offset", "sx", "gx", "iline", "xline"):
            if key in d.headers:
                parts.append(f"{key} {fmt(d.headers[key][tr])}")
        nano = d.domain == "time" and d.meta.get("display_time_unit") == "ns"
        scale, unit = (1e9, "ns") if nano else (1, "s")
        if self.reduced_shift is not None and j < self.reduced_shift.size:
            parts.append(f"reduced t = {scale*y:.4f} {unit}, t = {scale*(y + float(self.reduced_shift[j])):.4f} {unit}")
        else:
            parts.append(f"t = {scale*y:.4f} {unit}" if d.domain == "time"
                         else f"z = {y:.1f} m")
        parts.append(f"amp {amp:.4g}")
        self.readout.setText("   ".join(parts))

    def _click(self, info: dict) -> None:
        if self.data is None or self.picking_scroll.isHidden():
            return
        j = self.plot.nearest_trace(info["x"])
        if j is None:
            return
        self.picking.click(self, j, info)

    def _toggle_picking(self, on: bool) -> None:
        self.picking_scroll.setVisible(on)
        if on:
            self.picking.dataset_changed()
        self.draw_overlays()

    def _picks_changed(self) -> None:
        self.draw_overlays()
        self.sigPicksChanged.emit()

    def header_names(self) -> list[str]:
        return sorted(self.data.headers) if self.data is not None else []

    @staticmethod
    def describe_key(key: str) -> str:
        return DESCRIPTIONS.get(key, key)
