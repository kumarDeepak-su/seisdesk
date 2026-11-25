"""Windows-only previews of field import, SPS form and layered inversion QC."""
from pathlib import Path
import sys
import os
from PySide6 import QtWidgets,QtGui
import pyqtgraph as pg
from seisdesk.gui.dialogs import ImportDialog,SEG2ImportDialog,SEGDImportDialog
from seisdesk.gui.flow_editor import ParamForm
from seisdesk.gui.surface_inversion_dialog import SurfaceInversionDialog
from seisdesk.workflow.project import Project
from seisdesk.workflow.flow import FlowStep
from v10_example import example


def main(destination):
    root=Path(destination); root.mkdir(parents=True,exist_ok=True)
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([]); app.setStyle("Fusion")
    if os.name == "nt":
        font=Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts/arial.ttf"
        families=QtGui.QFontDatabase.applicationFontFamilies(QtGui.QFontDatabase.addApplicationFont(str(font)))
        app.setFont(QtGui.QFont(families[0],9))
    pg.setConfigOptions(background="w",foreground="k",antialias=True)
    project=Project(); result=example(bootstrap=8)
    project.context.results["surface_wave:synthetic"]=result
    project.settings["surface_wave:synthetic"]=dict(layers=[[13,460,230,1800,100,350,5,20],[0,920,460,2000,250,650,0,0]],
                                                    picks=result["picks"],fit_h=True,starts=3,bootstrap=8)
    generic=ImportDialog("mixed-rate-survey.sgy"); generic.dt_ms.setValue(2); generic.window.setCurrentText("intersection")
    segd=SEGDImportDialog("field-record.segd")
    sps=ParamForm(project); sps.set_step(FlowStep("sps_geometry")); sps.resize(900,480)
    surface=SurfaceInversionDialog(project,"synthetic"); surface.tabs.setCurrentIndex(2)
    windows=[(generic,"v10_resample_import.png"),(segd,"v10_segd_import.png"),(sps,"v10_sps_geometry.png"),
             (surface,"v10_surface_inversion.png")]
    for widget,name in windows:
        widget.show()
        for _ in range(8):app.processEvents()
        if not widget.grab().save(str(root/name)):raise RuntimeError("could not render "+name)
        widget.close(); print("rendered",name,flush=True)


if __name__ == "__main__":main(sys.argv[1])
