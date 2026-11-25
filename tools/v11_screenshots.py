"""Render interface controls and diagnostics on Windows CI."""
import os
from pathlib import Path
import sys
from PySide6 import QtGui, QtWidgets
import pyqtgraph as pg
from v11_example import example
from seisdesk.gui.tomography3d_dialog import Branches3DDialog, Tomography3DResultsDialog
from seisdesk.gui.interfaces3d_dialog import Interfaces3DResultsDialog


def main(destination):
    root=Path(destination); root.mkdir(parents=True,exist_ok=True)
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([]); app.setStyle("Fusion")
    if os.name == "nt":
        path=Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts/arial.ttf"
        families=QtGui.QFontDatabase.applicationFontFamilies(QtGui.QFontDatabase.addApplicationFont(str(path)))
        app.setFont(QtGui.QFont(families[0],9))
    pg.setConfigOptions(background="w",foreground="k",antialias=True)
    result,_=example(bootstrap=3)
    controls=Branches3DDialog(result["starting_branches"]); controls.tabs.setCurrentIndex(1)
    surfaces=Interfaces3DResultsDialog(result)
    uncertainty=Interfaces3DResultsDialog(result)
    uncertainty.field.setCurrentIndex(uncertainty.field.findData("interface_std"))
    windows=[(controls,"v11_interface_controls.png"),(surfaces,"v11_interface_depths.png"),
             (uncertainty,"v11_interface_uncertainty.png"),(Tomography3DResultsDialog(result),"v11_joint_arrival_qc.png")]
    for widget,name in windows:
        widget.show()
        for _ in range(8):app.processEvents()
        if not widget.grab().save(str(root/name)):raise RuntimeError("could not render "+name)
        widget.close(); print("rendered",name,flush=True)


if __name__ == "__main__":main(sys.argv[1])
