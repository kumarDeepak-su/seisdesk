import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.processing import Context, create
from seisdesk.processing.multiples import adaptive_subtract, hyperbolic_operator, srme_predict
from seisdesk.processing.tvdecon import blend_weights, parse_windows


def test_hyperbolic_radon_exact_adjoint():
    times = np.arange(101) * 0.004
    op = hyperbolic_operator(np.array([0., 50., 100.]), times, 0, 0.004, 101, 1 / np.array([1000., 2000.]) ** 2)
    rng = np.random.default_rng(7)
    m, d = rng.normal(size=op.shape[1]), rng.normal(size=op.shape[0])
    assert np.dot(op @ m, d) == pytest.approx(np.dot(m, op.T @ d), rel=1e-10)
    pulse = np.zeros((2, 101)); pulse[1, 25] = 1
    gather = (op @ pulse.ravel()).reshape(3, 101)
    expected = np.hypot(0.1, np.array([0, 50, 100]) / 2000)
    assert np.allclose(np.argmax(gather, axis=1) * 0.004, expected, atol=0.004)


def test_srme_convolution_predicts_later_event_without_input_overlap():
    sx, gx = np.meshgrid([0., 10.], [0., 10.], indexing="ij")
    traces = np.zeros((4, 128)); traces[:, 20] = 1
    data = SeismicData(traces, 0.004, {"sx": sx.ravel(), "gx": gx.ravel(), "fldr": [1, 1, 2, 2]})
    model, _ = srme_predict(data, grid=10, max_shift=0, near_traces=0, fmin=0, fmax=0, taper=0)
    assert np.all(np.argmax(np.abs(model), axis=1) == 40)
    assert np.allclose(model[:, 40], -20, atol=1e-4)
    out = create("srme", grid=10, max_shift=0, near_traces=0, fmin=0, fmax=124,
                 taper=0, output="multiple model")(data, Context())
    assert np.max(np.abs(out.traces[:, 38:43])) > 10


def test_adaptive_subtraction_matches_scale_without_shifting_event():
    rng = np.random.default_rng(42)
    model = rng.normal(size=(3, 200))
    target = 2.5 * model
    matched = adaptive_subtract(target, model, 0.004, filter_length=0.008, window_time=0.4,
                                window_traces=3, damping=1e-5)
    assert np.linalg.norm(target - matched) / np.linalg.norm(target) < 1e-3


def test_multiwindow_blending_partition_and_validation():
    weights = blend_weights(np.linspace(0, 3, 101), np.array([0.5, 1.5, 2.5]))
    assert np.all(weights >= 0) and np.allclose(weights.sum(axis=0), 1)
    assert parse_windows("1-2,0-1") == [(0, 1), (1, 2)]
    with pytest.raises(ValueError):
        parse_windows("2-1")


def test_multiwindow_decon_adapts_to_late_wavelet():
    from scipy.signal import lfilter
    white = np.random.default_rng(81).normal(size=(6, 4000))
    early = lfilter([1], [1, -.2], white, axis=1)
    late = lfilter([1], [1, -.97], white, axis=1)
    data = SeismicData(np.concatenate([early[:, :2000], late[:, 2000:]], axis=1), .002)
    params = dict(operator_length=.06, gap=.002, prewhitening=.01)
    stationary = create("tv_decon", windows="0.5-3.5", **params)(data, Context())
    adaptive = create("tv_decon", windows="0.5-3.5,4.5-7.5", **params)(data, Context())
    def lag1(a):
        a = a[:, 3000:3800]
        return abs(np.sum(a[:, 1:]*a[:, :-1])/np.sum(a*a))
    assert lag1(adaptive.traces) < .2*lag1(stationary.traces)


def test_gabor_decon_broadens_attenuated_band_and_keeps_dead_traces():
    dt = .002; n = 2048
    white = np.random.default_rng(12).normal(size=(4, n))
    f = np.fft.rfftfreq(n, dt)
    attenuated = np.fft.irfft(np.fft.rfft(white, axis=1)*np.exp(-np.pi*f*1.5/40), n, axis=1)
    data = SeismicData(np.vstack([attenuated, np.zeros(n)]), dt)
    out = create("gabor_decon", smoothing="boxcar", phase="zero", stabilisation=.1,
                 band=[5, 10, 80, 100])(data, Context())
    def centroid(a):
        power = np.abs(np.fft.rfft(a[:4, 256:-256], axis=1))**2
        freq = np.fft.rfftfreq(a.shape[1]-512, dt)
        return np.sum(power*freq)/np.sum(power)
    assert centroid(out.traces) > 1.5*centroid(data.traces)
    assert np.isfinite(out.traces).all() and np.all(out.traces[-1] == 0)


def test_v04_processes_are_registered():
    from seisdesk.processing import REGISTRY
    names = {"tv_decon", "gabor_decon", "hyperbolic_radon", "taup_decon", "srme", "adaptive_subtraction",
             "offset_to_angle", "angle_stack", "avo_attributes", "avo_modelling", "ps_binning", "ps_nmo",
             "ps_dmo", "vpvs_analysis", "ps_synthetics", "refraction_tomography", "grm", "fwi", "rtm"}
    assert names <= REGISTRY.keys()
