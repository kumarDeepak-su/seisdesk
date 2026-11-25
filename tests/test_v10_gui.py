import pytest
import numpy as np
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
from seisdesk.gui.dialogs import ImportDialog, SEG2ImportDialog, SEGDImportDialog
from seisdesk.gui.flow_editor import ParamForm
from seisdesk.workflow.flow import FlowStep
from seisdesk.workflow.project import Project


def test_import_options_select_real_resampling(qtbot):
    dialog=ImportDialog("survey.su"); qtbot.addWidget(dialog)
    dialog.dt_ms.setValue(2); dialog.window.setCurrentText("union")
    assert dialog.options()["dt"] == .002
    assert dialog.options()["window"] == "union"
    assert dialog.options()["little"]
    seg2=SEG2ImportDialog(3); qtbot.addWidget(seg2)
    seg2.target_dt.setValue(4); seg2.window.setCurrentText("intersection")
    assert seg2.options()["target_dt"] == .004
    assert seg2.options()["window"] == "intersection"


def test_segd_options_and_sps_flow_persistence(qtbot):
    dialog=SEGDImportDialog("field.segd"); qtbot.addWidget(dialog)
    assert dialog.options()["segd"] == dict(descale=True,include_aux=False,channel_set=0)
    dialog.channel_set.setValue(2)
    assert dialog.options()["segd"]["channel_set"] == 2
    step=FlowStep("sps_geometry",dict(source_file="s.sps",receiver_file="r.rps",relation_file="x.xps"))
    form=ParamForm(Project()); qtbot.addWidget(form); form.set_step(step)
    assert step.params["source_file"] == "s.sps"


def test_surface_inversion_runs_and_persists_result(qtbot,tmp_path):
    from seisdesk.gui.surface_inversion_dialog import SurfaceInversionDialog
    from test_v10_surface import example
    from seisdesk.core.dataset import SeismicData
    truth,_,picks=example(); project=Project()
    project.add_dataset(SeismicData(np.zeros((2,20)),.001,name="surface"))
    dialog=SurfaceInversionDialog(project,"surface",picks); qtbot.addWidget(dialog)
    dialog.starts.setValue(1); dialog.run()
    qtbot.waitUntil(lambda:dialog.worker is None,timeout=60000)
    assert dialog.result is not None
    assert dialog.result["rms_ms"] < .01
    project.save(str(tmp_path/"p")); back=Project.load(str(tmp_path/"p"))
    assert back.context.results["surface_wave:surface"]["picks"] == picks
    assert dialog.qc.rowCount() == len(picks)


def test_surface_cancel_keeps_prior_result_and_waits_for_thread(qtbot,monkeypatch):
    import time
    from seisdesk.gui import surface_inversion_dialog as module
    from test_v10_surface import example
    project=Project(); _,_,picks=example()
    dialog=module.SurfaceInversionDialog(project,"surface",picks); qtbot.addWidget(dialog)
    def fake(*args,**kwargs):
        while not kwargs["cancelled"]():time.sleep(.01)
        raise InterruptedError("cancelled")
    monkeypatch.setattr(module,"invert_dispersion",fake)
    project.context.results[dialog.result_key]="previous"
    dialog.run(); dialog.reject()
    qtbot.waitUntil(lambda:dialog.worker is None,timeout=5000)
    assert project.context.results[dialog.result_key] == "previous"
    assert not dialog.threads_running()
