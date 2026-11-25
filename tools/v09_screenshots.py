"""Render tested v0.9 controls and synthetic result windows on Windows CI."""
from pathlib import Path
import sys

from v09_example import examples
from PySide6 import QtGui, QtWidgets
import os
import pyqtgraph as pg
from seisdesk.core.volume import VelocityVolume
from seisdesk.gui.tomography3d_dialog import Branches3DDialog, Tomography3DResultsDialog
from seisdesk.gui.volume_dialog import VolumeSlicesDialog
from seisdesk.gui.reconstruction_dialog import ReconstructionResultsDialog


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
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="w", foreground="k")
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        example = examples(directory)
    result = example["tomography"]
    volume = VelocityVolume(result["velocity"], result["spacing"], result["origin"])
    windows = [(Branches3DDialog(result["branches"]), "v09_arrival_controls.png"),
               (Tomography3DResultsDialog(result), "v09_arrival_qc.png"),
               (VolumeSlicesDialog(volume, title="Synthetic 3D tomography — final velocity"), "v09_velocity_slices.png"),
               (VolumeSlicesDialog(volume, result["velocity_std"], "Synthetic timing-noise uncertainty", units="m/s", sequential=True),
                "v09_uncertainty_slices.png"),
               (ReconstructionResultsDialog(example["reconstruction"]), "v09_reconstruction_coverage.png")]
    for widget, name in windows:
        widget.show()
        for _ in range(8):
            app.processEvents()
        if not widget.grab().save(str(root/name)):
            raise RuntimeError("could not render "+name)
        widget.close()
        print("rendered", name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
