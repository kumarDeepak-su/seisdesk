"""Render remote controls on Windows CI; no network connection or job submission."""
from dataclasses import asdict
from pathlib import Path
import os
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6 import QtGui, QtWidgets
from seisdesk.backends.remote import RemoteProfile, save_record
from seisdesk.gui.remote_dialog import RemoteJobDialog
from seisdesk.workflow.project import Project


def main(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    if os.name == "nt":
        font = Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"/"arial.ttf"
        families = QtGui.QFontDatabase.applicationFontFamilies(QtGui.QFontDatabase.addApplicationFont(str(font)))
        if families:
            app.setFont(QtGui.QFont(families[0], 9))
    with tempfile.TemporaryDirectory() as directory:
        project = Project(directory)
        profile = RemoteProfile(remote_root="/scratch/user/seisdesk-jobs", python="/scratch/user/venv/bin/python",
                                modules=["Python/3.10.4"], partition="compute", account="research-project")
        project.settings["remote_profile"] = asdict(profile)
        token = "d"*32
        record = dict(token=token, state="COMPLETED", job_id="12345", created=1., profile=asdict(profile),
                      record_file=str(Path(directory)/"remote-jobs"/token/"remote-job.json"), exit_code="0:0",
                      remote_directory=profile.remote_root+"/"+token,
                      application=dict(state="completed", host="compute001"),
                      log="Illustrative job record for the manual. Status and compute logs appear here.")
        save_record(record)
        dialog = RemoteJobDialog(project); dialog.show(); dialog.timer.stop()
        for index, name in ((1, "v07_remote_host.png"), (2, "v07_remote_jobs.png")):
            dialog.tabs.setCurrentIndex(index)
            for _ in range(8):
                app.processEvents()
            if not dialog.grab().save(str(destination/name)):
                raise RuntimeError("could not render "+name)
        dialog.close()


if __name__ == "__main__":
    main(sys.argv[1])
