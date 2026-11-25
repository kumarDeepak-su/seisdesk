"""Inversion convergence and model comparison; no computation is launched by the dialog."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .common import lookup_table
from .velocity_results import colorbar


class FwiResultsDialog(QtWidgets.QDialog):
    def __init__(self, result, model=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Full-waveform inversion results")
        self.resize(1000, 740)
        self.result = result
        lay = QtWidgets.QVBoxLayout(self)
        graph = pg.GraphicsLayoutWidget()
        lay.addWidget(graph)
        x, z = np.asarray(result["x"]), np.asarray(result["z"])
        values = [np.asarray(result[k]) for k in ("starting_model", "final_model")]
        levels = (min(v.min() for v in values), max(v.max() for v in values))
        self.model_plots = []
        for col, (title, v) in enumerate(zip(("Starting velocity", "Inverted velocity"), values)):
            plot = graph.addPlot(row=0, col=col, title=title)
            item = pg.ImageItem(v, axisOrder="row-major")
            item.setLookupTable(lookup_table("Viridis")); item.setLevels(levels)
            item.setRect(QtCore.QRectF(x[0], z[0], x[-1] - x[0], z[-1] - z[0]))
            plot.addItem(item); plot.invertY(True)
            colorbar(plot, item, levels, 'm/s')
            plot.setLabel("bottom", "Line position", units="m"); plot.setLabel("left", "Depth", units="m")
            self.model_plots.append(plot)
        misfit = graph.addPlot(row=1, col=0, colspan=2, title="Objective by accepted iteration")
        self.histories = [list(s["history"]) for s in result["stages"]]
        for i, h in enumerate(self.histories):
            misfit.plot(np.arange(len(h)), h, pen=pg.mkPen(pg.intColor(i), width=2), symbol="o")
        misfit.setLabel("bottom", "Iteration within stage"); misfit.setLabel("left", "Normalised objective")
        info = QtWidgets.QLabel("\n".join(f"Stage {i + 1} ({s['frequencies']} Hz): {s['message']}"
                                         for i, s in enumerate(result["stages"])))
        info.setWordWrap(True); lay.addWidget(info)
