"""Render the real desktop workspace and record layout dimensions on Windows."""
import json
import sys
import tempfile
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from seisdesk.core import synthetic
from seisdesk.gui.main_window import MainWindow
from seisdesk.gui.workspace import CommandPalette
from seisdesk.workflow.flow import Flow, FlowStep


def main(destination):
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(background="w", foreground="k", imageAxisOrder="row-major", antialias=False)
    with tempfile.TemporaryDirectory() as folder:
        settings = QtCore.QSettings(str(Path(folder) / "preview.ini"), QtCore.QSettings.Format.IniFormat)
        window = MainWindow(settings=settings)
        window.resize(1500, 920); window.show()
        report = []
        def capture(widget, name):
            for _ in range(8): app.processEvents()
            if not QtGui.QFontMetrics(window.home.metrics[0].font()).inFont("A"):
                raise RuntimeError("Screenshot font is missing Latin glyphs")
            if not widget.grab().save(str(root / name)):
                raise RuntimeError("Could not save " + name)
            report.append(dict(image=name, width=widget.width(), height=widget.height(),
                               minimum_width=widget.minimumSizeHint().width(),
                               requested_width=1280 if "compact" in name else 740 if widget is not window else 1500,
                               workspace_minimums={key:w.minimumSizeHint().width() for key,w in
                                   (("viewer",window.viewer),("flow",window.flow_editor),("velocity",window.velan),
                                    ("project",window.project_dock))}))
            print("Rendered", name, flush=True)
        capture(window, "v12_home_light.png")
        window.theme_combo.setCurrentIndex(window.theme_combo.findData("dark"))
        capture(window, "v12_home_dark.png")
        data = synthetic.marine_line(nshots=5, ngroups=40, nt=501, dt=.004)
        window.add_dataset(data, "Marine survey · shot gathers")
        capture(window, "v12_viewer_dark.png")
        editor = window.flow_editor
        editor.set_flow(Flow([FlowStep("bandpass", {}), FlowStep("agc", {}), FlowStep("nmo", {}),
                             FlowStep("stack", {})], "Marine processing"))
        editor.search.setText("filter")
        editor.steps.setCurrentRow(0)
        window.tabs.setCurrentWidget(editor)
        capture(window, "v12_flow_dark.png")
        dialog = CommandPalette(window.command_entries(), window)
        dialog.search.setText("tomography"); dialog.show()
        capture(dialog, "v12_command_search.png"); dialog.close()
        window.theme_combo.setCurrentIndex(window.theme_combo.findData("light"))
        capture(window, "v12_flow_light.png")
        window.tabs.setCurrentWidget(window.viewer)
        window.viewer.display_options.setChecked(True)
        capture(window, "v12_viewer_light.png")
        window.theme_combo.setCurrentIndex(window.theme_combo.findData("dark"))
        window.resize(1280, 800)
        window.tabs.setCurrentWidget(window.home)
        capture(window, "v12_home_compact.png")
        window.tabs.setCurrentWidget(editor)
        capture(window, "v12_flow_compact.png")
        window.dirty = False; window.close()
        (root / "v12_layout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        oversized = [item for item in report if item["width"] > item["requested_width"]]
        if oversized:
            raise RuntimeError("Workspace exceeds requested window size: " + str(oversized))


if __name__ == "__main__":
    main(sys.argv[1])
