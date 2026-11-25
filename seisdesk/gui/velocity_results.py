"""Interactive diagnostics for velocity scans, Fowler panels and MVA updates."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .common import lookup_table


def colorbar(plot, image, levels, label, cmap='Viridis'):
    lo, hi = map(float, levels)
    hi = max(hi, lo + max(abs(lo)*1e-6, 1e-12))
    lut = lookup_table(cmap)
    bar = pg.ColorBarItem(values=(lo, hi), colorMap=pg.ColorMap(np.linspace(0, 1, len(lut)), lut),
                          label=label, interactive=False)
    bar.setImageItem(image, insert_in=plot.getPlotItem() if isinstance(plot, pg.PlotWidget) else plot)
    return bar


def panel(plot, values, x, y, cmap='Viridis', levels=None):
    plot.clear()
    values, x, y = map(np.asarray, (values, x, y))
    dx = (x[-1]-x[0])/max(len(x)-1, 1) if len(x) > 1 else 1
    dy = (y[-1]-y[0])/max(len(y)-1, 1) if len(y) > 1 else 1
    image = pg.ImageItem(values, axisOrder='row-major')
    image.setLookupTable(lookup_table(cmap))
    if levels is not None:
        image.setLevels(levels)
    image.setRect(QtCore.QRectF(x[0]-dx/2, y[0]-dy/2, dx*len(x), dy*len(y)))
    plot.addItem(image); plot.invertY(True)
    return image


class VelocitySliceDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent); self.result = result
        self.setWindowTitle(f"Velocity semblance at {result['time']:g} s"); self.resize(900, 650)
        layout = QtWidgets.QVBoxLayout(self); self.plot = pg.PlotWidget()
        self.plot.setTitle(f"Semblance at {result['time']:g} s")
        item = panel(self.plot, result['semblance'].T, np.arange(len(result['cdp'])), result['velocity'], levels=(0, 1))
        colorbar(self.plot, item, (0, 1), 'Semblance')
        self.plot.setLabel('bottom', 'CMP index'); self.plot.setLabel('left', 'Velocity', units='m/s')
        self.info = QtWidgets.QLabel('Click a cell to inspect its CMP, velocity and semblance.')
        self.plot.scene().sigMouseClicked.connect(self.clicked)
        layout.addWidget(self.plot); layout.addWidget(self.info)

    def clicked(self, event):
        point = self.plot.getViewBox().mapSceneToView(event.scenePos())
        r = self.result; ix = int(np.clip(round(point.x()), 0, len(r['cdp'])-1))
        iv = int(np.argmin(np.abs(r['velocity']-point.y())))
        self.info.setText(f"CMP {r['cdp'][ix]:g}; velocity {r['velocity'][iv]:g} m/s; semblance {r['semblance'][ix, iv]:.3f}")


class FowlerResultsDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent); self.result = result
        self.setWindowTitle('Fowler constant-velocity panels'); self.resize(1100, 730)
        self.trial = QtWidgets.QComboBox(); self.trial.addItems([f'{v:g} m/s' for v in result['velocities']])
        self.trial.currentIndexChanged.connect(self.refresh)
        graph = pg.GraphicsLayoutWidget()
        self.stack = graph.addPlot(row=0, col=0, title='DMO stack at selected trial velocity')
        self.velocity = graph.addPlot(row=0, col=1, title='Selected velocity (m/s)')
        x = np.arange(len(result['cdp']))
        item = panel(self.velocity, result['selected_velocity'].T, x, result['times'])
        colorbar(self.velocity, item, (result['velocities'][0], result['velocities'][-1]), 'm/s')
        for p in (self.stack, self.velocity):
            p.setLabel('bottom', 'CMP index'); p.setLabel('left', 'Time', units='s')
        layout = QtWidgets.QVBoxLayout(self); layout.addWidget(self.trial); layout.addWidget(graph)
        note = QtWidgets.QLabel('Velocity selection follows smoothed coherence. Check the panels and original gathers before accepting automatic picks.')
        note.setWordWrap(True); layout.addWidget(note); self.refresh()

    def refresh(self):
        values = self.result['stacks'][max(self.trial.currentIndex(), 0)].T
        clip = max(float(np.percentile(np.abs(values), 99)), 1e-30)
        panel(self.stack, values, np.arange(values.shape[1]), self.result['times'], 'Gray (black = positive)', (-clip, clip))


class MVAResultsDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent); self.result = result
        self.setWindowTitle('Residual-moveout model update'); self.resize(1050, 700)
        graph = pg.GraphicsLayoutWidget()
        models = [np.asarray(result[k]) for k in ('starting_model', 'updated_model')]
        levels = (min(m.min() for m in models), max(m.max() for m in models))
        for i, (title, model) in enumerate(zip(('Migration velocity', 'Updated velocity'), models)):
            p = graph.addPlot(row=0, col=i, title=title)
            item = panel(p, model, result['x'], result['z'], levels=levels)
            colorbar(p, item, levels, 'm/s')
            p.setLabel('bottom', 'Position', units='m'); p.setLabel('left', 'Depth', units='m')
        p = graph.addPlot(row=1, col=0, colspan=2, title='Picked moveout coefficient a')
        p.plot([q['depth'] for q in result['picks']], [q['a'] for q in result['picks']], pen=None, symbol='o')
        p.setLabel('bottom', 'Analysis depth', units='m'); p.setLabel('left', 'Coefficient a')
        p.getAxis('left').enableAutoSIPrefix(False)
        p.setXRange(result['z'][0], result['z'][-1], padding=.02)
        coefficients = [q['a'] for q in result['picks']]
        p.setYRange(min(0, min(coefficients))-.02, max(0, max(coefficients))+.02, padding=0)
        note = QtWidgets.QLabel(result['next_step']); note.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self); layout.addWidget(graph); layout.addWidget(note)


class ShearRotationResultsDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent); self.setWindowTitle('Sliding shear rotation: angles modulo 90 degrees'); self.resize(900, 650)
        plot = pg.PlotWidget(); layout = QtWidgets.QVBoxLayout(self); layout.addWidget(plot)
        for i, angle in enumerate(result['angles'][:100]):
            plot.plot(result['times'], angle % 90, pen=pg.mkPen(pg.intColor(i)))
        plot.setLabel('bottom', 'Time', units='s'); plot.setLabel('left', 'Rotation angle', units='deg')
        layout.addWidget(QtWidgets.QLabel('Up to 100 stations shown. The 90-degree fast/slow ambiguity is not resolved automatically.'))
