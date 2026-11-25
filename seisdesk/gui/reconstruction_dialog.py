"""Recorded and reconstructed fold maps in midpoint coordinates."""
import numpy as np
from PySide6 import QtWidgets

from .plot import SeismicPlot
from .velocity_results import colorbar


class ReconstructionResultsDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Survey reconstruction coverage")
        self.resize(1050, 620)
        layout = QtWidgets.QVBoxLayout(self)
        generated = int(np.sum(result["generated_fold"]))
        text = QtWidgets.QLabel(f"{len(result['occupied'])} occupied bins; {generated} reconstructed traces; "
                               f"grid {tuple(result['shape'])}. Maps show trace counts per midpoint bin. "
                               "Predicted traces carry interp=1; these maps describe coverage, not accuracy. "
                               "Assess accuracy by withholding representative recorded traces before reconstruction.")
        text.setWordWrap(True); layout.addWidget(text)
        maps = QtWidgets.QHBoxLayout(); layout.addLayout(maps, 1)
        self.plots, self.bars = [], []
        for title, key in (("Recorded bins", "observed_fold"), ("Reconstructed bins", "generated_fold")):
            values = np.asarray(result[key])
            plot = SeismicPlot(y_label="Northing", y_units="m")
            plot.getPlotItem().invertY(False); plot.getPlotItem().setTitle(title)
            plot.getPlotItem().setLabel("bottom", "Easting", units="m")
            levels = (0, max(1, float(values.max())))
            plot.show_image(values, result["origin"][1], result["spacing"][1],
                             result["origin"][0], result["spacing"][0], "Viridis", levels)
            plot.reset_view()
            self.bars.append(colorbar(plot, plot.image, levels, "Traces", "Viridis"))
            self.plots.append(plot); maps.addWidget(plot)
