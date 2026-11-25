"""Actual GPU checks; run on Athena with SEISDESK_TEST_CUDA=1."""
import os

import numpy as np
import pytest

from seisdesk.core.acoustic3d import Acoustic3D
from seisdesk.core.volume import VelocityVolume

pytestmark = pytest.mark.skipif(os.environ.get("SEISDESK_TEST_CUDA") != "1", reason="requires a CUDA allocation")


def test_cuda_wavefield_source_adjoint_and_gradient_agree_with_cpu(tmp_path):
    values = np.full((13, 12, 14), 1800., dtype=np.float64)
    values[6:, 4:, 5:] = 2100
    volume = VelocityVolume(values, (10, 12, 8))
    source = [[25, 35, 24]]
    receiver = [[20, 25, 10], [25, 35, 45], [40, 55, 65]]
    times = np.arange(95)*.001
    phase = np.pi*35*(times-.03)
    wavelet = 1e6*(1-2*phase**2)*np.exp(-phase**2)
    cpu = Acoustic3D(volume, .001, padding=4, dtype="float64", block=13)
    gpu = Acoustic3D(volume, .001, padding=4, dtype="float64", block=13, device="cuda")
    cpu_data = cpu.forward(source, receiver, wavelet, checkpoints=tmp_path/"cpu")
    gpu_data = gpu.forward(source, receiver, wavelet, checkpoints=tmp_path/"gpu")
    np.testing.assert_allclose(gpu_data, cpu_data, rtol=1e-9, atol=1e-10)
    residual = np.random.default_rng(72).normal(size=cpu_data.shape)
    cpu_adj = cpu.adjoint(source, receiver, wavelet, residual, tmp_path/"cpu", image=True)
    gpu_adj = gpu.adjoint(source, receiver, wavelet, residual, tmp_path/"gpu", image=True)
    for key in ("gradient", "source_adjoint", "image", "illumination"):
        np.testing.assert_allclose(gpu_adj[key], cpu_adj[key], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(np.vdot(gpu_data, residual), np.vdot(wavelet, gpu_adj["source_adjoint"][:, 0]),
                               rtol=1e-9, atol=1e-9)
    direction = np.random.default_rng(39).normal(size=values.shape)
    direction *= 100/np.linalg.norm(direction)
    step = .001
    responses = []
    for sign in (-1, 1):
        perturbed = Acoustic3D(VelocityVolume(values+sign*step*direction, volume.spacing), .001,
                               padding=4, dtype="float64", block=13, device="cuda")
        responses.append(np.vdot(perturbed.forward(source, receiver, wavelet), residual))
    np.testing.assert_allclose(np.vdot(gpu_adj["gradient"], direction), (responses[1]-responses[0])/(2*step),
                               rtol=3e-5, atol=1e-8)


def test_cuda_full_modelling_rtm_fwi_jobs(tmp_path):
    from seisdesk.backends.volume_jobs import run_volume_job, stage_shots
    from test_v05_volume_jobs import survey
    volume, params, gathers = survey()
    params["device"] = "cuda"
    shots = stage_shots(gathers, volume, tmp_path/"shots")
    model = run_volume_job(shots, volume, tmp_path/"model", params, mode="model", log=lambda s: None)
    assert model["objective"][0] < 1e-6
    starting = VelocityVolume(np.full(volume.values.shape, 1800., dtype=np.float32), volume.spacing)
    migration = run_volume_job(shots, starting, tmp_path/"rtm", params, mode="rtm", log=lambda s: None)
    assert np.isfinite(np.load(migration["image"])).all() and np.linalg.norm(np.load(migration["image"])) > 0
    result = run_volume_job(shots, starting, tmp_path/"fwi", params, mode="fwi", log=lambda s: None)
    assert len(result["objective"]) >= 2 and result["objective"][-1] < .8*result["objective"][0]
