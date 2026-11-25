"""Rendering diagnostic: does pyqtgraph draw an ImageItem and a curve with this Qt binding? Prints measured facts."""

import os
import sys

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import PySide6  # noqa: E402
import pyqtgraph as pg  # noqa: E402
import shiboken6  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "diag"
os.makedirs(out, exist_ok=True)
print(f"PySide6 {PySide6.__version__}, shiboken6 {shiboken6.__version__}, pyqtgraph {pg.__version__}, "
      f"numpy {np.__version__}, platform {QtWidgets.QApplication.platformName() if QtWidgets.QApplication.instance() else '-'}")
pg.setConfigOptions(imageAxisOrder="row-major", background="w", foreground="k")
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
print("platform plugin:", app.platformName())

# low level 1: arrayToQPath with connect='all' must reproduce the points
x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([0.0, 2.0, 1.0, 3.0])
path = pg.arrayToQPath(x, y)
pts = [(path.elementAt(i).x, path.elementAt(i).y) for i in range(path.elementCount())]
print("arrayToQPath(all) elements:", pts)
path_f = pg.arrayToQPath(np.append(x, np.nan), np.append(y, np.nan), connect="finite")
print("arrayToQPath(finite+nan) elements:", [(path_f.elementAt(i).x, path_f.elementAt(i).y)
                                             for i in range(path_f.elementCount())])

# low level 2: ndarray_to_qimage must keep the pixel values
arr = np.zeros((4, 6, 4), dtype=np.ubyte)
arr[..., 0] = 200
arr[..., 3] = 255
try:
    qimg = pg.functions.ndarray_to_qimage(arr, QtGui.QImage.Format.Format_RGBA8888)
    c = qimg.pixelColor(2, 1)
    print("ndarray_to_qimage pixel (2,1):", c.red(), c.green(), c.blue(), c.alpha())
except Exception as exc:                                   # noqa: BLE001
    print("ndarray_to_qimage failed:", type(exc).__name__, exc)


def count(img: QtGui.QImage, test) -> int:
    n = 0
    for yy in range(0, img.height(), 2):
        for xx in range(0, img.width(), 2):
            if test(img.pixelColor(xx, yy)):
                n += 1
    return n


def check(name, add):
    w = pg.PlotWidget()
    w.resize(400, 300)
    add(w)
    w.show()
    for _ in range(5):
        app.processEvents()
    img = w.grab().toImage()
    img.save(os.path.join(out, f"{name}.png"))
    red = count(img, lambda c: c.red() > 150 and c.green() < 90 and c.blue() < 90)
    dark = count(img, lambda c: c.red() < 100 and c.green() < 100 and c.blue() < 100)
    print(f"{name}: red pixels {red}, dark pixels {dark}")
    w.close()


def image_item(w):
    item = pg.ImageItem(axisOrder="row-major")
    data = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (80, 1))
    item.setImage(data, autoLevels=False, levels=(0.0, 1.0))
    item.setLookupTable(pg.ColorMap(np.array([0.0, 1.0]), np.array([(255, 0, 0), (255, 0, 0)], dtype=np.ubyte))
                        .getLookupTable(0.0, 1.0, 256))
    item.setRect(QtCore.QRectF(0, 0, 10, 8))
    w.addItem(item)


def image_item_default(w):
    item = pg.ImageItem()
    data = np.full((80, 100, 3), (255, 0, 0), dtype=np.ubyte)
    item.setImage(data)
    w.addItem(item)


def curve(w):
    w.addItem(pg.PlotDataItem(np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50)), pen=pg.mkPen((255, 0, 0), width=4)))


def curve_nan(w):
    xx = np.append(np.linspace(0, 10, 50), np.nan)
    yy = np.append(np.sin(np.linspace(0, 10, 50)), np.nan)
    w.addItem(pg.PlotDataItem(xx, yy, pen=pg.mkPen((255, 0, 0), width=4), connect="finite"))


def scatter(w):
    w.addItem(pg.ScatterPlotItem(np.arange(10.0), np.arange(10.0), size=12, brush=pg.mkBrush(255, 0, 0)))


def path_item(w):
    p = pg.arrayToQPath(np.append(np.linspace(0, 10, 50), np.nan), np.append(np.cos(np.linspace(0, 10, 50)), np.nan),
                        connect="finite")
    item = QtWidgets.QGraphicsPathItem(p)
    item.setPen(pg.mkPen((255, 0, 0), width=4))
    w.addItem(item)
    w.setXRange(0, 10)
    w.setYRange(-1, 1)


for name, fn in (("image_item_float_lut", image_item), ("image_item_rgb", image_item_default), ("curve", curve),
                 ("curve_nan_finite", curve_nan), ("scatter", scatter), ("qgraphicspathitem", path_item)):
    try:
        check(name, fn)
    except Exception as exc:                               # noqa: BLE001
        print(f"{name}: EXCEPTION {type(exc).__name__}: {exc}")
