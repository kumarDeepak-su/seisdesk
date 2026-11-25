import json
from pathlib import Path

import numpy as np
import pytest

from seisdesk.backends.volume_jobs import run_volume_job, stage_shots
from seisdesk.core.acoustic3d import Acoustic3D
from seisdesk.core.dataset import SeismicData
from seisdesk.core.segy_io import write_su
from seisdesk.core.streaming import iter_file_gathers
from seisdesk.core.volume import VelocityVolume
from seisdesk.processing import Context, create


def survey():
    volume = VelocityVolume(np.full((9, 8, 10), 1900., dtype=np.float32), (10, 10, 10))
    params = dict(frequency=45, source_delay=.025, source_amplitude=1e6, padding=3, block=11,
                  vmax=2400, vmin=1000, memory_gb=.2, iterations=2, smooth_cells=1, max_update=.03,
                  allow_coarse_grid=True)  # Small algebra/job-control tests, not a physical dispersion benchmark.
    nt, dt = 85, .001
    t = np.arange(nt)*dt
    a = np.pi*45*(t-.025)
    wavelet = 1e6*(1-2*a*a)*np.exp(-a*a)
    solver = Acoustic3D(volume, dt, padding=3, block=11)
    shots = []
    for index, x in enumerate((20, 60)):
        source = [[20, 30, x]]
        receiver = [[20, 20, 10], [20, 35, 50], [20, 50, 70]]
        traces = solver.forward(source, receiver, wavelet).T
        headers = dict(sx=x, sy=30, selev=-20, sdepth=0, gx=[10, 50, 70], gy=[20, 35, 50], gelev=-20,
                       fldr=index+1, tracf=[1, 2, 3], offset=[10-x, 50-x, 70-x])
        shots.append((np.arange(3)+index*3, SeismicData(traces, dt, headers)))
    return volume, params, shots


def test_3d_file_job_modelling_rtm_fwi_and_parallel_shots(tmp_path):
    volume, params, gathers = survey()
    shots = stage_shots(gathers, volume, tmp_path/"shots")
    model = run_volume_job(shots, volume, tmp_path/"model", params, mode="model", log=lambda s: None)
    assert model["objective"][0] < 1e-6
    parallel = run_volume_job(shots, volume, tmp_path/"parallel", dict(params, workers=2), mode="model", log=lambda s: None)
    for left, right in zip(model["shot_results"], parallel["shot_results"]):
        np.testing.assert_array_equal(np.load(left["files"]["predicted"]), np.load(right["files"]["predicted"]))
    starting = VelocityVolume(np.full(volume.values.shape, 1800., dtype=np.float32), volume.spacing)
    image = run_volume_job(shots, starting, tmp_path/"rtm", params, mode="rtm", log=lambda s: None)
    assert np.isfinite(np.load(image["image"])).all() and np.linalg.norm(np.load(image["image"])) > 0
    fit = run_volume_job(shots, starting, tmp_path/"fwi", params, mode="fwi", log=lambda s: None)
    assert len(fit["objective"]) >= 2
    assert fit["objective"][-1] < .8*fit["objective"][0]
    assert not list(tmp_path.rglob("checkpoints-*"))


def test_streamed_input_and_process_adapter(tmp_path):
    volume, params, gathers = survey()
    data = SeismicData(np.concatenate([g.traces for _, g in gathers]), .001,
                       {key: np.concatenate([g.header(key) for _, g in gathers]) for key in gathers[0][1].headers})
    data = data.subset([0, 3, 1, 4, 2, 5])  # Deliberately interleaved shots.
    source = tmp_path/"input.su"
    write_su(data, str(source))
    chunks = list(iter_file_gathers(source))
    assert [idx.tolist() for idx, _ in chunks] == [[0, 2, 4], [1, 3, 5]]
    for indices, gather in chunks:
        np.testing.assert_array_equal(gather.traces, data.traces[indices])
    volume.save(tmp_path/"velocity.npz")
    context = Context(work_dir=str(tmp_path), log=lambda s: None)
    preview = create("volume_file_job", input_file="input.su", model="velocity.npz", mode="model", **params).run(None, context)
    assert preview.ntr == 3 and "preview" in preview.name
    assert len(context.results["volume_file_job"]["shot_results"]) == 2
    assert Path(preview.meta["volume_job"]).is_file()
    # Exercise the headless entry point used by Slurm, including relative model
    # paths and the saved preview that can be opened on the desktop.
    from seisdesk.cli import main
    (tmp_path/"parameters.json").write_text(json.dumps(dict(params, model="velocity.npz")), encoding="utf-8")
    assert main(["volume-job", str(tmp_path/"parameters.json"), str(source), str(tmp_path/"cli-job"),
                 "--mode", "model"]) == 0
    assert (tmp_path/"cli-job"/"preview.npz").is_file()
    assert json.loads((tmp_path/"cli-job"/"job.json").read_text())["state"] == "completed"


def test_volume_job_cancellation_and_geometry_validation(tmp_path):
    volume, params, gathers = survey()
    shots = stage_shots(gathers, volume, tmp_path/"shots")
    with pytest.raises(InterruptedError):
        run_volume_job(shots, volume, tmp_path/"cancelled", params, mode="rtm", cancelled=lambda: True)
    bad = gathers[0][1].copy()
    bad.set_header("sy", 1000)
    with pytest.raises(ValueError, match="outside"):
        stage_shots([(np.arange(3), bad)], volume, tmp_path/"bad")
    with pytest.raises(ValueError, match="points per shortest source wavelength"):
        run_volume_job(shots, volume, tmp_path/"under-resolved", dict(params, allow_coarse_grid=False), mode="model")
