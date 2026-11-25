"""Tomography coverage/misfit and reciprocal-method diagnostic plots."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .common import lookup_table
from .velocity_results import colorbar


class TomographyResultsDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("Joint tomography: model, coverage and misfit" if "branches" in result else
                            "First-arrival tomography: model, coverage and misfit")
        self.resize(1000, 720)
        graph = pg.GraphicsLayoutWidget()
        layout = QtWidgets.QVBoxLayout(self); layout.addWidget(graph)
        x, z = np.asarray(result["x"]), np.asarray(result["z"])
        for col, (key, title) in enumerate((("velocity", "Velocity (m/s)"), ("hits", "Ray coverage"))):
            plot = graph.addPlot(row=0, col=col, title=title)
            values = np.asarray(result[key])
            item = pg.ImageItem(values, axisOrder="row-major")
            item.setLookupTable(lookup_table("Viridis"))
            dx, dz = x[1]-x[0], z[1]-z[0]
            item.setRect(QtCore.QRectF(x[0]-.5*dx, z[0]-.5*dz, len(x)*dx, len(z)*dz))
            plot.addItem(item); plot.invertY(True)
            colorbar(plot, item, (values.min(), values.max()), 'm/s' if key == 'velocity' else 'Hits')
            plot.plot(x, result["surface"], pen=pg.mkPen("w", width=2))
            if "starting_branches" in result:
                seen = set()
                for phase, branch in result["branches"].items():
                    interface = branch.get("interface", phase)
                    if branch.get("kind") == "first" or interface in seen:
                        continue
                    seen.add(interface)
                    phase_order = list(dict.fromkeys(result.get("phases", result["branches"])))
                    colour = pg.intColor(phase_order.index(phase))
                    for initial, definitions in ((True, result["starting_branches"]), (False, result["branches"])):
                        definition = definitions[phase]
                        depth = (np.full_like(x, definition["depth"], dtype=float) if definition.get("depth") is not None else
                                 np.interp(x, np.asarray(definition["points"])[:, 0], np.asarray(definition["points"])[:, 1]))
                        sampled = result.get("starting_sampled_interfaces" if initial else "sampled_interfaces", {})
                        if interface in sampled:
                            depth = np.asarray(sampled[interface])
                        pen = pg.mkPen(colour, width=2, style=QtCore.Qt.PenStyle.DashLine if initial else
                                       QtCore.Qt.PenStyle.SolidLine)
                        plot.plot(x, depth, pen=pen)
            plot.setLabel("bottom", "Line position", units="m"); plot.setLabel("left", "Depth", units="m")
        p = graph.addPlot(row=1, col=0, title="Traveltime RMS")
        p.plot(np.arange(len(result["rms"])), 1000 * np.array(result["rms"]), symbol="o",
               pen=pg.mkPen((30, 70, 160), width=1.5))
        p.setLabel("bottom", "Accepted update" if "interface_history" in result else "Iteration")
        p.setLabel("left", "RMS", units="ms")
        p = graph.addPlot(row=1, col=1, title="Observed and predicted arrivals")
        if "phases" in result:
            p.addLegend()
            phases = np.asarray(result["phases"])
            for i, phase in enumerate(dict.fromkeys(phases)):
                mask = phases == phase
                p.plot(np.asarray(result["observed"])[mask]*1000, np.asarray(result["t_pred"])[mask]*1000,
                       pen=None, symbol="o", symbolBrush=pg.intColor(i), name=str(phase))
        else:
            p.plot(np.asarray(result["observed"]) * 1000, np.asarray(result["t_pred"]) * 1000, pen=None, symbol="o")
        p.setLabel("bottom", "Observed", units="ms"); p.setLabel("left", "Predicted", units="ms")
        note = QtWidgets.QLabel("Velocity in cells with no rays is unconstrained. Depth is below the highest ground elevation.")
        if "interface_history" in result:
            note.setText(note.text()+" Dashed interfaces: starting depths; solid: final depths. "
                         "Velocity-depth trade-offs require independent checks. "+result.get("stop_reason", ""))
        note.setWordWrap(True); layout.addWidget(note)


class GRMResultsDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("GRM candidate XY diagnostics")
        self.resize(1000, 650)
        graph = pg.GraphicsLayoutWidget()
        layout = QtWidgets.QVBoxLayout(self); layout.addWidget(graph)
        plots = [graph.addPlot(row=0, col=i, title=title)
                 for i, title in enumerate(("Velocity function", "Time-depth function", "Refractor depth"))]
        for i, candidate in enumerate(result["candidates"]):
            for plot, key, scale in zip(plots, ("tv", "tg", "depth"), (1000, 1000, 1)):
                plot.plot(result["x"], np.asarray(candidate[key]) * scale, pen=pg.mkPen(pg.intColor(i), width=2))
                plot.setLabel("bottom", "Position", units="m")
        plots[0].setLabel("left", "tV", units="ms"); plots[1].setLabel("left", "tG", units="ms")
        plots[2].setLabel("left", "Depth", units="m"); plots[2].invertY(True)
        # A constant synthetic refractor must not magnify floating-point roundoff into apparent structure.
        for plot, key, scale in zip(plots, ('tv', 'tg', 'depth'), (1000, 1000, 1)):
            values = np.concatenate([np.asarray(c[key])*scale for c in result['candidates']])
            lo, hi = float(values.min()), float(values.max())
            margin = max(.05*(hi-lo), .02*max(abs(lo), abs(hi)), .001)
            plot.setYRange(lo-margin, hi+margin, padding=0)
        note = QtWidgets.QLabel("\n".join(f"XY {c['xy']:g} m: Vr {c['velocity']:.1f} m/s; score {c['score']:.6g} s"
                                         for c in result["candidates"]))
        layout.addWidget(note)
