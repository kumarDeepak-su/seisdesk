import numpy as np
import pytest

from seisdesk.backends.volume_jobs import run_volume_job, stage_shots
from test_v05_volume_jobs import survey


def test_distributed_shots_match_serial_results(tmp_path):
    distributed = pytest.importorskip("distributed")
    volume, params, gathers = survey()
    shots = stage_shots(gathers, volume, tmp_path/"shots")
    serial = run_volume_job(shots, volume, tmp_path/"serial", params, mode="model", log=lambda s: None)
    with distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None) as cluster:
        with distributed.Client(cluster) as client:
            address = client.scheduler.address
            parallel = run_volume_job(shots, volume, tmp_path/"distributed", dict(params, scheduler=address),
                                      mode="model", log=lambda s: None)
    for a, b in zip(serial["shot_results"], parallel["shot_results"]):
        np.testing.assert_array_equal(np.load(a["files"]["predicted"]), np.load(b["files"]["predicted"]))
