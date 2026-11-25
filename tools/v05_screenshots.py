"""Render the new GUI controls on the Windows runner, never on the user's Mac."""
from pathlib import Path
import os
import sys
import tempfile

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

from seisdesk.core.joint_tomography import invert_joint, joint_arrivals
from seisdesk.core.tomography import NodeGrid
from seisdesk.gui.su_dialog import SUDialog
from seisdesk.gui.tomo_dialog import TomographyResultsDialog
from seisdesk.gui.volume_dialog import VolumeJobDialog, VolumeModelDialog, VolumeSlicesDialog
from seisdesk.workflow.project import Project


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
        widget.close()
        print("rendered", name, flush=True)
    with tempfile.TemporaryDirectory() as directory:
        project = Project(directory)
        model_dialog = VolumeModelDialog(project)
        model_dialog.fields["shape"].setText("61,41,81")
        model_dialog.boxes.setPlainText('[{"z":[180,320],"y":[100,300],"x":[250,550],"velocity":2200}]')
        volume = model_dialog.model()
        capture(model_dialog, "v05_model_builder.png")
        slices = VolumeSlicesDialog(volume)
        slices.probe(0, 400, 250)
        capture(slices, "v05_volume_slices.png")
        capture(VolumeJobDialog(project), "v05_volume_job.png")
        capture(SUDialog(project), "v05_su_workbench.png")
    grid = NodeGrid(0, 10, 21, 10, 16, radius=3)
    source, receiver, phases = [], [], []
    for sx in (0, 100, 200):
        for gx in (20, 60, 140, 180):
            for phase in ("direct", "R1", "R2"):
                source.append(grid.surface_node(sx)); receiver.append(grid.surface_node(gx)); phases.append(phase)
    branches = {"direct": {"kind": "first"}, "R1": {"kind": "reflection", "depth": 50},
                "R2": {"kind": "reflection", "depth": 100}}
    observed = joint_arrivals(grid, np.full((16, 21), 2200.), source, receiver, phases, branches)[0]
    result = invert_joint(grid, np.full((16, 21), 1900.), source, receiver, observed, phases, branches, iterations=4)
    result.update(x=grid.x, z=grid.z, surface=grid.surface)
    capture(TomographyResultsDialog(result), "v05_joint_tomography.png")


if __name__ == "__main__":
    main(sys.argv[1])
