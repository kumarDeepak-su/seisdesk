"""The seismic display widget: variable density image and wiggle traces on a time (depth) axis pointing down."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .common import lookup_table

DISPLAY_MODES = ("Density", "Wiggle", "Density + wiggle")


def regular_columns(xpos: np.ndarray, max_factor: int = 4):
    """Columns of a regular grid for traces at positions ``xpos``.

    Returns (trace index per column, -1 for an empty column; x of the first column; column spacing), or None when the
    positions are all equal or too irregular for a grid (the caller then uses the trace number)."""
    n = xpos.size
    if n == 0:
        return None
    if n == 1:
        return np.zeros(1, np.int64), float(xpos[0]), 1.0
    order = np.argsort(xpos, kind="stable")
    xs = xpos[order]
    span = float(xs[-1] - xs[0])
    if span <= 0:
        return None
    d = np.diff(xs)
    d = d[d > 1e-9 * span]
    dx = float(np.median(d))
    ncol = int(round(span / dx)) + 1
    if ncol > max_factor * n + 16:
        return None
    grid = xs[0] + np.arange(ncol) * dx
    j = np.clip(np.searchsorted(xs, grid), 1, n - 1)
    nearest = np.where(np.abs(grid - xs[j - 1]) <= np.abs(xs[j] - grid), j - 1, j)
    cols = order[nearest]
    cols = np.where(np.abs(xs[nearest] - grid) > 0.5 * dx * 1.0001, -1, cols)
    return cols.astype(np.int64), float(xs[0]), dx


class SeismicPlot(pg.PlotWidget):
    """A pyqtgraph plot showing traces (ntr, nt) at positions x; emits hover and click positions in data units."""
    sigHover = QtCore.Signal(float, float)
    sigClick = QtCore.Signal(object)          # dict(x, y, left, right, shift, ctrl, double)

    def __init__(self, parent=None, y_label: str = "Time", y_units: str = "s"):
        super().__init__(parent)
        pi = self.getPlotItem()
        pi.invertY(True)
        pi.showGrid(x=False, y=True, alpha=0.25)
        pi.setMenuEnabled(False)
        pi.setLabel("left", y_label, units=y_units)
        pi.hideButtons()
        self.image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self.image)
        self.fill_item = QtWidgets.QGraphicsPathItem()
        self.fill_item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.fill_item.setBrush(pg.mkBrush(0, 0, 0))
        self.line_item = QtWidgets.QGraphicsPathItem()
        self.line_item.setPen(pg.mkPen(0, 0, 0, width=1))
        self.addItem(self.fill_item)
        self.addItem(self.line_item)
        self.xpos = np.zeros(0)
        self.extent = None                    # (xmin, xmax, ymin, ymax)
        self.scene().sigMouseMoved.connect(self._moved)
        self.scene().sigMouseClicked.connect(self._clicked)

    # ------------------------------------------------------------------ drawing
    def clear_display(self) -> None:
        self.image.clear()
        self.fill_item.setPath(QtGui.QPainterPath())
        self.line_item.setPath(QtGui.QPainterPath())
        self.xpos = np.zeros(0)
        self.extent = None

    def show_traces(self, traces: np.ndarray, t0: float, dt: float, xpos: np.ndarray, mode: str = "Density",
                    cmap: str = "Gray (black = positive)", clip: float = 1.0, wiggle_limit: int = 1000,
                    levels: tuple | None = None) -> np.ndarray:
        """Draw; returns the x positions actually used (the trace number when ``xpos`` cannot be gridded)."""
        ntr, nt = traces.shape
        xpos = np.asarray(xpos, dtype=np.float64)
        reg = regular_columns(xpos)
        if reg is None:
            xpos = np.arange(1, ntr + 1, dtype=np.float64)
            reg = regular_columns(xpos)
        cols, x0, dx = reg
        self.xpos = xpos
        times = t0 + np.arange(nt) * dt
        if mode in ("Density", "Density + wiggle"):
            img = np.zeros((nt, cols.size), dtype=np.float32)
            live = cols >= 0
            img[:, live] = traces[cols[live]].T
            lv = levels if levels is not None else (-clip, clip)
            self.image.setImage(img, autoLevels=False, levels=lv)
            self.image.setLookupTable(lookup_table(cmap))
            self.image.setRect(QtCore.QRectF(x0 - dx / 2, t0 - dt / 2, cols.size * dx, nt * dt))
            self.image.show()
        else:
            self.image.hide()
        if mode in ("Wiggle", "Density + wiggle") and ntr <= wiggle_limit:
            line, fill = wiggle_paths(traces, times, xpos, dx, clip, fill=(mode == "Wiggle"))
            self.line_item.setPath(line)
            self.fill_item.setPath(fill if fill is not None else QtGui.QPainterPath())
        else:
            self.line_item.setPath(QtGui.QPainterPath())
            self.fill_item.setPath(QtGui.QPainterPath())
        self.extent = (float(xpos.min()) - dx / 2, float(xpos.max()) + dx / 2, t0, t0 + (nt - 1) * dt)
        return xpos

    def show_image(self, values: np.ndarray, x0: float, dx: float, y0: float, dy: float, cmap: str,
                   levels: tuple) -> None:
        """A plain image (rows = y, columns = x), e.g. a semblance panel or a velocity field."""
        self.fill_item.setPath(QtGui.QPainterPath())
        self.line_item.setPath(QtGui.QPainterPath())
        ny, nx = values.shape
        self.image.setImage(np.asarray(values, dtype=np.float32), autoLevels=False, levels=levels)
        self.image.setLookupTable(lookup_table(cmap))
        self.image.setRect(QtCore.QRectF(x0 - dx / 2, y0 - dy / 2, nx * dx, ny * dy))
        self.image.show()
        self.xpos = x0 + np.arange(nx) * dx
        self.extent = (x0 - dx / 2, x0 + (nx - 0.5) * dx, y0, y0 + (ny - 1) * dy)

    def reset_view(self, x_only: bool = False) -> None:
        if self.extent is None:
            return
        x0, x1, y0, y1 = self.extent
        if x_only:
            self.getPlotItem().vb.setXRange(x0, x1, padding=0.0)
        else:
            self.getPlotItem().vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.0)

    # ------------------------------------------------------------------ mouse
    def _to_data(self, scene_pos):
        vb = self.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        p = vb.mapSceneToView(scene_pos)
        return float(p.x()), float(p.y())

    def _moved(self, scene_pos) -> None:
        pos = self._to_data(scene_pos)
        if pos is not None:
            self.sigHover.emit(*pos)

    def _clicked(self, ev) -> None:
        if ev.double():                       # pyqtgraph repeats a double click; one action per click only
            return
        pos = self._to_data(ev.scenePos())
        if pos is None:
            return
        mods = ev.modifiers()
        self.sigClick.emit({"x": pos[0], "y": pos[1],
                            "left": ev.button() == QtCore.Qt.MouseButton.LeftButton,
                            "right": ev.button() == QtCore.Qt.MouseButton.RightButton,
                            "shift": bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier),
                            "ctrl": bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier),
                            "double": False})

    def nearest_trace(self, x: float) -> int | None:
        if self.xpos.size == 0:
            return None
        return int(np.argmin(np.abs(self.xpos - x)))


def wiggle_paths(traces: np.ndarray, times: np.ndarray, xpos: np.ndarray, dx: float, clip: float, fill: bool = True):
    """Wiggle lines and (optionally) positive variable-area fill as two QPainterPaths."""
    ntr, nt = traces.shape
    a = np.clip(traces / (clip if clip > 0 else 1.0), -1.5, 1.5) * (0.9 * dx)
    sep = np.full((ntr, 1), np.nan)
    tt = np.broadcast_to(times[None, :], (ntr, nt))
    xl = np.hstack([xpos[:, None] + a, sep]).ravel()
    yl = np.hstack([tt, sep]).ravel()
    line = pg.arrayToQPath(xl, yl, connect="finite")
    if not fill:
        return line, None
    base = xpos[:, None]
    xf = np.hstack([base, base + np.maximum(a, 0.0), base, sep]).ravel()
    yf = np.hstack([np.full((ntr, 1), times[0]), tt, np.full((ntr, 1), times[-1]), sep]).ravel()
    area = pg.arrayToQPath(xf, yf, connect="finite")
    area.setFillRule(QtCore.Qt.FillRule.WindingFill)
    return line, area
