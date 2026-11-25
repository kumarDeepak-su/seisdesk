"""Render workstation controls on Windows CI; the populated view is labelled as an example."""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6 import QtGui, QtWidgets
from seisdesk.gui.compute_dialog import ComputeDialog
from seisdesk.workflow.project import Project


def main(destination):
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    if os.name == "nt":
        font = Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"/"arial.ttf"
        families = QtGui.QFontDatabase.applicationFontFamilies(QtGui.QFontDatabase.addApplicationFont(str(font)))
        if not families:
            raise RuntimeError("preview font unavailable")
        app.setFont(QtGui.QFont(families[0], 9))
        if not QtGui.QFontMetrics(app.font()).inFont("A"):
            raise RuntimeError("preview font has no Latin glyphs")
    dialog = ComputeDialog(Project(), discover=False)
    dialog.probe_finished(dict(available=True, message="Illustrative GPU information for the manual.",
                               driver_version="12.8", runtime_version="12.8", cupy_version="14.2.0",
                               devices=[dict(index=0, name="NVIDIA CUDA workstation GPU (example)",
                                             free_bytes=10*1024**3, total_bytes=12*1024**3, tested=True)]))
    dialog.device.setCurrentIndex(dialog.device.findData("cuda")); dialog.show()
    for _ in range(8):
        app.processEvents()
    if not dialog.grab().save(str(root/"v08_workstation_devices.png")):
        raise RuntimeError("could not render workstation device controls")
    dialog.close()


if __name__ == "__main__":
    main(sys.argv[1])
