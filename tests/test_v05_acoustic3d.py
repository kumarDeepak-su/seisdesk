import numpy as np

from seisdesk.core.acoustic3d import Acoustic3D, Sampling, laplacian
from seisdesk.core.volume import VelocityVolume


def setup_model():
    values = np.full((9, 8, 10), 1800.)
    values[5:, 3:, 4:] = 2150
    volume = VelocityVolume(values, (10, 12, 8))
    engine = Acoustic3D(volume, .001, padding=3, dtype="float64", block=9)
    source = [[20, 30, 24]]
    receivers = [[20, 18, 10], [20, 30, 38], [35, 45, 45]]
    t = np.arange(65)*.001
    phase = np.pi*40*(t-.025)
    wavelet = (1-2*phase**2)*np.exp(-phase**2)*1e6
    return volume, engine, source, receivers, wavelet


def test_laplacian_and_interpolation_transposes():
    rng = np.random.default_rng(31)
    a, b = rng.normal(size=(7, 8, 9)), rng.normal(size=(7, 8, 9))
    spacing = (2, 3, 4)
    np.testing.assert_allclose(np.vdot(a, laplacian(b, spacing)), np.vdot(laplacian(a, spacing), b), rtol=1e-12)
    sample = Sampling([[1.1, 3.7, 6.1], [2.3, 2.1, 3.9]], (3, 4, 5), spacing, (0, 0, 0), 2)
    field = rng.normal(size=(7, 8, 9)); values = rng.normal(size=2)
    transpose = np.zeros_like(field)
    sample.add(transpose, values)
    np.testing.assert_allclose(np.vdot(sample.take(field), values), np.vdot(field, transpose), rtol=1e-12)


def test_checkpointed_source_adjoint_and_velocity_gradient(tmp_path):
    volume, engine, src, rec, wavelet = setup_model()
    predicted = engine.forward(src, rec, wavelet, checkpoints=tmp_path)
    rng = np.random.default_rng(42)
    residual = rng.normal(size=predicted.shape)
    adj = engine.adjoint(src, rec, wavelet, residual, tmp_path)
    np.testing.assert_allclose(np.vdot(predicted, residual), np.vdot(wavelet, adj["source_adjoint"][:, 0]),
                               rtol=1e-9, atol=1e-9)
    direction = rng.normal(size=volume.values.shape)
    direction /= np.linalg.norm(direction)
    direction *= 100
    step = .001
    def objective(offset):
        perturbed = VelocityVolume(volume.values+offset*direction, volume.spacing)
        solver = Acoustic3D(perturbed, .001, padding=3, dtype="float64", block=9)
        return np.vdot(solver.forward(src, rec, wavelet), residual)
    finite_difference = (objective(step)-objective(-step))/(2*step)
    np.testing.assert_allclose(np.vdot(adj["gradient"], direction), finite_difference, rtol=2e-5, atol=1e-8)
    # Replay block size must not change the adjoint.
    other = Acoustic3D(volume, .001, padding=3, dtype="float64", block=17)
    folder = tmp_path/"other"
    other.forward(src, rec, wavelet, checkpoints=folder)
    other_adjoint = other.adjoint(src, rec, wavelet, residual, folder)
    np.testing.assert_allclose(adj["gradient"], other_adjoint["gradient"], rtol=1e-12, atol=1e-12)


def test_volume_storage_and_crossline_geometry(tmp_path):
    volume, _, _, _, _ = setup_model()
    volume.save(tmp_path/"volume")
    back = VelocityVolume.load(tmp_path/"volume")
    np.testing.assert_array_equal(back.values, volume.values)
    assert back.spacing == (10, 12, 8)
    dataset = back.dataset()
    assert dataset.ntr == 8*10 and dataset.nt == 9 and dataset.domain == "depth"
    np.testing.assert_array_equal(dataset.traces.reshape(8, 10, 9).transpose(2, 0, 1), volume.values)
