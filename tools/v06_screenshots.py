"""Render v0.6 interface controls and diagnostics on the Windows CI runner."""
from pathlib import Path
import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

from seisdesk.core.interface_inversion import invert_interfaces
from seisdesk.core.joint_tomography import joint_arrivals
from seisdesk.core.tomography import NodeGrid
from seisdesk.core.velocity import DepthModel
from seisdesk.gui.branch_editor import JointBranchDialog
from seisdesk.gui.dialogs import SEG2ImportDialog
from seisdesk.gui.tomo_dialog import TomographyResultsDialog


def main(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    if os.name == "nt":
        font = Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"/"arial.ttf"
        families = QtGui.QFontDatabase.applicationFontFamilies(QtGui.QFontDatabase.addApplicationFont(str(font)))
        if not families:
            raise RuntimeError("preview font unavailable")
        app.setFont(QtGui.QFont(families[0], 9))
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="w", foreground="k")

    def capture(widget, name):
        widget.show()
        for _ in range(8):
            app.processEvents()
        if not widget.grab().save(str(destination/name)):
            raise RuntimeError("could not save "+name)
        widget.close(); print("rendered", name, flush=True)

    branches = {"direct": dict(kind="first", uncertainty_ms=1),
                "reflection 1": dict(kind="reflection", interface="upper boundary", depth=30),
                "refractor 1": dict(kind="head", interface="upper boundary", depth=30),
                "reflection 2": dict(kind="reflection", interface="lower boundary", points=[[0, 50], [100, 60]])}
    model = DepthModel(1200+20*np.arange(81)[:, None]+np.zeros((81, 51)), 1, 2)
    editor = JointBranchDialog(branches, model=model)
    editor.table.setCurrentCell(3, 1)
    capture(editor, "v06_branch_editor.png")
    capture(SEG2ImportDialog(26), "v06_seg2_import.png")
    grid = NodeGrid(0, 5, 21, 2, 46, radius=5)
    source, receiver, phases = [], [], []
    for sx in (0, 50, 100):
        for gx in (10, 30, 70, 90):
            for phase in ("direct", "R1", "R2"):
                source.append(grid.surface_node(sx)); receiver.append(grid.surface_node(gx)); phases.append(phase)
    truth = {"direct": dict(kind="first"), "R1": dict(kind="reflection", depth=30),
             "R2": dict(kind="reflection", depth=60)}
    observed = joint_arrivals(grid, np.full((grid.nz, grid.nx), 2000.), source, receiver, phases, truth)[0]
    start = {"direct": dict(kind="first"), "R1": dict(kind="reflection", depth=20),
             "R2": dict(kind="reflection", depth=70)}
    result = invert_interfaces(grid, np.full((grid.nz, grid.nx), 1900.), source, receiver, observed, phases,
                               start, iterations=5, max_step=8, max_shift=20)
    result.update(x=grid.x, z=grid.z, surface=grid.surface)
    capture(TomographyResultsDialog(result), "v06_interface_inversion.png")


if __name__ == "__main__":
    main(sys.argv[1])
