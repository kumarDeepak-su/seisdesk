"""Amplitude statistics and a bounded trace-similarity view."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets


def trace_statistics(data, bins=100):
    a = data.traces.astype(float)
    if not np.all(np.isfinite(a)):
        raise ValueError("statistics require finite samples; edit invalid traces first")
    energy = np.sum(a * a, axis=1)
    centroid = np.divide((a * a) @ np.arange(data.nt), energy,
                         out=np.full(data.ntr, np.nan), where=energy > 0)
    count, edges = np.histogram(a, bins=bins)
    quantiles = np.quantile(a, [0, .01, .25, .5, .75, .99, 1])
    # Similarity is explicitly limited; the histogram and trace statistics use every sample.
    index = np.unique(np.linspace(0, data.ntr - 1, min(data.ntr, 256)).astype(int))
    b = a[index] - a[index].mean(axis=1, keepdims=True)
    norm = np.linalg.norm(b, axis=1)
    b = np.divide(b, norm[:, None], out=np.zeros_like(b), where=norm[:, None] > 0)
    return dict(count=count, edges=edges, quantiles=quantiles, mean=a.mean(axis=1),
                rms=np.sqrt(energy / data.nt), maximum=np.max(np.abs(a), axis=1),
                centroid_sample=centroid, similarity=b @ b.T, similarity_indices=index)


class TraceStatisticsDialog(QtWidgets.QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle("Trace statistics")
        self.resize(1000, 700)
        self.graph = pg.GraphicsLayoutWidget()
        self.readout = QtWidgets.QLabel(); self.readout.setWordWrap(True)
        export = QtWidgets.QPushButton("Export per-trace statistics (CSV)…")
        export.clicked.connect(self.export_csv)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.graph); layout.addWidget(self.readout); layout.addWidget(export)
        self.compute()

    def compute(self):
        r = self.result = trace_statistics(self.data)
        self.graph.clear()
        p = self.graph.addPlot(row=0, col=0, title="Amplitude histogram: all samples")
        p.plot(r["edges"], r["count"], stepMode="center", fillLevel=0, brush=(70, 140, 200, 90))
        p.setLabel("bottom", "Amplitude"); p.setLabel("left", "Sample count")
        p = self.graph.addPlot(row=0, col=1, title="Per-trace RMS (blue) and peak magnitude (orange)")
        x = np.arange(1, self.data.ntr + 1)
        p.plot(x, r["rms"], pen="b"); p.plot(x, r["maximum"], pen=(220, 120, 20))
        p.setLabel("bottom", "Trace number")
        p = self.graph.addPlot(row=1, col=0, title="Energy centroid (zero-based sample)")
        p.plot(x, r["centroid_sample"], connect="finite", pen=pg.mkPen((30, 70, 160), width=1.5))
        p.setLabel("bottom", "Trace number")
        p = self.graph.addPlot(row=1, col=1, title="Trace similarity: up to 256 evenly selected traces")
        p.addItem(pg.ImageItem(r["similarity"], axisOrder="row-major", levels=(-1, 1)))
        p.setLabel("bottom", "Selected trace index"); p.setLabel("left", "Selected trace index")
        self.readout.setText("Quantiles 0, 1, 25, 50, 75, 99, 100%: " +
                             ", ".join(f"{q:.5g}" for q in r["quantiles"]))
        return r

    def csv_text(self):
        r = self.result
        lines = ["trace,trace_id,mean,rms,maximum_absolute,energy_centroid_sample"]
        for i in range(self.data.ntr):
            lines.append(f"{i+1},{self.data.trace_ids[i]}," + ",".join(
                f"{r[k][i]:.10g}" for k in ("mean", "rms", "maximum", "centroid_sample")))
        return "\n".join(lines) + "\n"

    def export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export statistics", "trace_statistics.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.csv_text())
