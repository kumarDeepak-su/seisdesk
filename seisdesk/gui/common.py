"""Shared GUI pieces: colour maps, display gain, busy cursor, error boxes and a background worker thread."""

from __future__ import annotations

import traceback
from contextlib import contextmanager

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..processing.gain import agc

# ---------------------------------------------------------------------------------------------- colour maps
_MAPS = {
    "Gray (black = positive)": ([0.0, 1.0], [(255, 255, 255), (0, 0, 0)]),
    "Gray (white = positive)": ([0.0, 1.0], [(0, 0, 0), (255, 255, 255)]),
    "Red-white-blue": ([0.0, 0.5, 1.0], [(0, 0, 255), (255, 255, 255), (255, 0, 0)]),
    "Seismic": ([0.0, 0.25, 0.5, 0.75, 1.0], [(0, 0, 77), (0, 0, 255), (255, 255, 255), (255, 0, 0), (128, 0, 0)]),
    "Rainbow": ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                [(0, 0, 143), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)]),
    "Viridis": ([0.0, 0.25, 0.5, 0.75, 1.0],
                [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]),
}
SEISMIC_MAPS = ("Gray (black = positive)", "Gray (white = positive)", "Red-white-blue", "Seismic")
VELOCITY_MAPS = ("Rainbow", "Viridis")


def lookup_table(name: str, n: int = 256) -> np.ndarray:
    pos, colors = _MAPS.get(name, _MAPS["Gray (black = positive)"])
    cmap = pg.ColorMap(np.array(pos), np.array(colors, dtype=np.ubyte))
    return cmap.getLookupTable(0.0, 1.0, n)


def map_names() -> list[str]:
    return list(_MAPS)


# ---------------------------------------------------------------------------------------------- display gain
DISPLAY_GAINS = ("None", "AGC 0.25 s", "AGC 0.5 s", "AGC 1 s", "t^2", "Trace normalise")


def display_gain(traces: np.ndarray, times: np.ndarray, dt: float, mode: str, domain: str = "time") -> np.ndarray:
    """Gain applied for display only (the data are not changed). In depth sections the AGC window is in metres,
    1000 m per second of window (0.5 s -> 500 m), and t^2 uses depth in km."""
    if mode.startswith("AGC"):
        window = float(mode.split()[1]) * (1000.0 if domain == "depth" else 1.0)
        return agc(traces, dt, window)
    if mode == "t^2" and domain == "depth":
        return (traces * (np.maximum(times, 0.0) / 1000.0) ** 2).astype(np.float32)
    if mode == "t^2":
        return (traces * (np.maximum(times, 0.0) ** 2)[None, :]).astype(np.float32)
    if mode == "Trace normalise":
        peak = np.abs(traces).max(axis=1, keepdims=True)
        return (traces / np.where(peak > 0, peak, 1.0)).astype(np.float32)
    return traces


def clip_value(traces: np.ndarray, percentile: float) -> float:
    a = np.abs(traces[np.isfinite(traces)])
    if a.size == 0:
        return 1.0
    if a.size > 2_000_000:
        a = a[:: a.size // 2_000_000 + 1]
    c = float(np.percentile(a, percentile))
    return c if c > 0 else (float(a.max()) or 1.0)


# ---------------------------------------------------------------------------------------------- small helpers
HEADLESS = False            # set by the self-test: errors raise instead of opening a message box nobody can close


@contextmanager
def busy():
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


def show_error(parent, title: str, error) -> None:
    if HEADLESS:
        raise RuntimeError(f"{title}: {error}")
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    if isinstance(error, BaseException):
        box.setText(f"{type(error).__name__}: {error}")
        box.setDetailedText("".join(traceback.format_exception(type(error), error, error.__traceback__)))
    else:
        text = str(error)
        box.setText(text.strip().splitlines()[-1] if text.strip() else title)
        box.setDetailedText(text)
    box.exec()


def fmt(value: float) -> str:
    """Compact number formatting for forms and read-outs."""
    if value is None:
        return ""
    v = float(value)
    if not np.isfinite(v):
        return str(v)
    if v == int(v) and abs(v) < 1e12:
        return str(int(v))
    return f"{v:.6g}"


# ---------------------------------------------------------------------------------------------- worker thread
class Worker(QtCore.QObject):
    """Runs ``fn(worker)`` in a QThread; ``fn`` may call worker.message.emit / worker.progress.emit and should poll
    ``worker.cancelled()``. Results and errors come back through signals on the GUI thread."""
    message = QtCore.Signal(str)
    progress = QtCore.Signal(float)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def cancelled(self) -> bool:
        return self._cancel

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._fn(self)
        except InterruptedError:
            self.failed.emit("Cancelled.")
        except Exception:                                  # noqa: BLE001 - reported to the user
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)


def start_worker(owner: QtCore.QObject, fn, on_finished, on_failed, on_message=None, on_progress=None) -> Worker:
    """Start ``fn`` on a new thread owned by ``owner``. The callbacks must be methods of QObjects living in the GUI
    thread so that Qt queues the calls back to it."""
    thread = QtCore.QThread(owner)
    worker = Worker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    if on_message is not None:
        worker.message.connect(on_message)
    if on_progress is not None:
        worker.progress.connect(on_progress)
    worker.finished.connect(thread.quit, QtCore.Qt.ConnectionType.DirectConnection)   # QThread.quit is thread-safe
    worker.failed.connect(thread.quit, QtCore.Qt.ConnectionType.DirectConnection)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    owner._seisdesk_threads = [t for t in getattr(owner, "_seisdesk_threads", []) if _alive(t)] + [(thread, worker)]
    thread.start()
    return worker


def _alive(pair) -> bool:
    thread, _ = pair
    try:
        return thread.isRunning()
    except RuntimeError:                                   # already deleted on the C++ side
        return False
