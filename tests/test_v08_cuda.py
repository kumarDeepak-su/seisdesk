"""Actual workstation-style CUDA selection, executed inside an Athena allocation."""
import json
import os

import pytest

from seisdesk.core.compute import probe_cuda

pytestmark = pytest.mark.skipif(os.environ.get("SEISDESK_TEST_CUDA") != "1", reason="requires a CUDA allocation")


def test_gpu_detection_kernel_check_and_automatic_volume_job(tmp_path):
    from test_v05_volume_jobs import survey
    from seisdesk.backends.volume_jobs import run_volume_job, stage_shots
    report = probe_cuda(test=True, device_index=0)
    assert report["available"], report["message"]
    assert report["devices"][0]["tested"] and report["devices"][0]["total_bytes"] > 0
    volume, params, gathers = survey()
    shots = stage_shots(gathers, volume, tmp_path/"shots")
    result = run_volume_job(shots, volume, tmp_path/"auto", dict(params, device="auto", device_index=0),
                            mode="model", log=lambda s: None)
    assert result["objective"][0] < 1e-6
    saved = json.loads((tmp_path/"auto/job.json").read_text())
    assert saved["params"]["device"] == "cuda"
    assert saved["compute"]["device"]["name"] == report["devices"][0]["name"]
