import json

import pytest

from seisdesk.core import compute


def ready():
    return dict(available=True, devices=[dict(index=1, name="Selected GPU", tested=True)],
                driver_version="12.8", runtime_version="12.8", cupy_version="14.2.0")


def test_cpu_does_not_initialise_cuda_and_explicit_cuda_never_falls_back(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("CPU selection must not probe CUDA")
    monkeypatch.setattr(compute, "probe_cuda", unexpected)
    assert compute.resolve_device("cpu")[0] == "cpu"
    missing = dict(available=False, message="NVRTC missing", devices=[])
    monkeypatch.setattr(compute, "probe_cuda", lambda **kwargs: missing)
    with pytest.raises(RuntimeError, match="NVRTC missing"):
        compute.resolve_device("cuda")
    messages = []
    device, details = compute.resolve_device("auto", log=messages.append)
    assert device == "cpu" and details["reason"] == "NVRTC missing"
    assert "CPU" in messages[0]


def test_selected_gpu_is_checked_and_recorded(monkeypatch):
    calls = []
    def probe(**kwargs):
        calls.append(kwargs)
        return ready()
    monkeypatch.setattr(compute, "probe_cuda", probe)
    device, details = compute.resolve_device("cuda", 1)
    assert device == "cuda" and details["device"]["name"] == "Selected GPU"
    assert calls == [dict(test=True, device_index=1)]
    with pytest.raises(ValueError):
        compute.resolve_device("cuda", -1)


def test_missing_cuda_package_is_a_report_not_an_import_crash(monkeypatch):
    def unavailable(name):
        raise ImportError("No CuPy installation")
    monkeypatch.setattr(compute.importlib, "import_module", unavailable)
    report = compute.probe_cuda(test=True)
    assert report["cpu"] and not report["available"] and not report["package_available"]
    assert "No CuPy installation" in report["message"]


def test_automatic_cpu_job_retains_actual_backend_and_fallback_reason(tmp_path, monkeypatch):
    from test_v05_volume_jobs import survey
    from seisdesk.backends.volume_jobs import run_volume_job, stage_shots
    monkeypatch.setattr(compute, "probe_cuda", lambda **kw: dict(available=False, devices=[], message="no visible GPU"))
    volume, params, gathers = survey()
    shots = stage_shots(gathers, volume, tmp_path/"shots")
    result = run_volume_job(shots, volume, tmp_path/"job", dict(params, device="auto"), mode="model", log=lambda s: None)
    assert result["params"]["device"] == "cpu"
    saved = json.loads((tmp_path/"job/job.json").read_text())
    assert saved["compute"] == dict(requested="auto", selected="cpu", reason="no visible GPU")


def test_cuda_failure_is_persisted_in_job_record(tmp_path, monkeypatch):
    from test_v05_volume_jobs import survey
    from seisdesk.backends.volume_jobs import run_volume_job, stage_shots
    monkeypatch.setattr(compute, "probe_cuda", lambda **kw: dict(available=False, devices=[], message="driver unavailable"))
    volume, params, gathers = survey()
    shots = stage_shots(gathers, volume, tmp_path/"shots")
    with pytest.raises(RuntimeError, match="driver unavailable"):
        run_volume_job(shots, volume, tmp_path/"job", dict(params, device="cuda"), mode="model", log=lambda s: None)
    saved = json.loads((tmp_path/"job/job.json").read_text())
    assert saved["state"] == "failed" and "driver unavailable" in saved["error"]
