"""Discrete adjoint verification and an independent homogeneous Green-function check."""
import numpy as np
import pytest
from scipy.special import hankel2

from seisdesk.core.helmholtz import Helmholtz, objective_gradient
from seisdesk.processing.fwi import frequency_groups, invert_frequency, smooth_penalty


def experiment(v, frequencies=(8.0,)):
    shots = [(np.array([20., 30.]), np.column_stack([np.full(8, 200.), np.linspace(20, 160, 8)])),
             (np.array([100., 20.]), np.column_stack([np.linspace(20, 210, 8), np.full(8, 170.)]))]
    obs = []
    for f in frequencies:
        op = Helmholtz(v, 10, 10, f, nabs=10)
        obs.append([op.sampling(rec) @ op.solve(op.source(src)) for src, rec in shots])
    return shots, obs


def test_helmholtz_homogeneous_green_phase():
    v = np.full((81, 81), 2000.)
    op = Helmholtz(v, 5, 5, 10, nabs=24)
    receivers = [[250, 200], [275, 200], [300, 200]]
    response = op.sampling(receivers) @ op.solve(op.source([200, 200]))
    analytic = hankel2(0, 2 * np.pi * 10 * np.array([50, 75, 100]) / 2000)
    ratio = (response / response[0]) / (analytic / analytic[0])
    assert np.max(np.abs(np.angle(ratio))) < 0.08
    assert np.max(np.abs(np.abs(ratio) - 1)) < 0.12


@pytest.mark.parametrize("estimate_source,alpha", [(False, 0.0), (True, 0.0), (False, 1.0)])
def test_velocity_adjoint_directional_gradient(estimate_source, alpha):
    shape = (20, 24)
    truth = np.full(shape, 2100.)
    shots, observations = experiment(truth)
    start = np.full(shape, 2000.)
    rng = np.random.default_rng(42)
    direction = rng.normal(size=shape) * 30
    kwargs = dict(nabs=10, estimate_source=estimate_source, alpha=alpha)
    _, gradient, _ = objective_gradient(start, 10, 10, [8.], shots, observations, **kwargs)
    eps = 1e-3
    plus = objective_gradient(start + eps * direction, 10, 10, [8.], shots, observations, **kwargs)[0]
    minus = objective_gradient(start - eps * direction, 10, 10, [8.], shots, observations, **kwargs)[0]
    fd = (plus - minus) / (2 * eps)
    adjoint = float(np.sum(gradient * direction))
    assert adjoint == pytest.approx(fd, rel=0.005, abs=1e-9)


def test_fwi_reduces_independent_held_start_residual():
    start = np.full((20, 24), 2000.)
    truth = start.copy(); truth[5:15, 8:16] = 1850.
    shots, obs = experiment(truth, (6., 9.))
    initial = objective_gradient(start, 10, 10, [6., 9.], shots, obs, nabs=10, estimate_source=False)[0]
    final, info = invert_frequency(start, 10, 10, [6., 9.], shots, obs, iterations=5,
                                   vmin=1500, vmax=2500, smoothing=0.001, nabs=10, estimate_source=False)
    assert info["data_misfit"] < initial * 0.7
    assert np.mean(final[5:15, 8:16]) < np.mean(start[5:15, 8:16])
    assert np.all((final >= 1500) & (final <= 2500))


def test_smooth_penalty_gradient_and_frequency_validation():
    rng = np.random.default_rng(2)
    q, direction = rng.normal(size=(8, 9)), rng.normal(size=(8, 9))
    _, grad = smooth_penalty(q)
    eps = 1e-5
    fd = (smooth_penalty(q + eps * direction)[0] - smooth_penalty(q - eps * direction)[0]) / (2 * eps)
    assert np.sum(grad * direction) == pytest.approx(fd, rel=1e-6)
    assert frequency_groups("3,4;4,5") == [[3, 4], [4, 5]]
    with pytest.raises(ValueError):
        frequency_groups("5,6;2,3")


def test_inversion_honours_cancellation_during_first_gradient():
    start = np.full((20, 24), 2000.)
    shots, observations = experiment(start)
    updates = []
    def cancel(fraction):
        updates.append(fraction)
        if fraction > 0:
            raise RuntimeError("test cancellation")
    with pytest.raises(RuntimeError, match="test cancellation"):
        invert_frequency(start, 10, 10, [8.], shots, observations, nabs=10, report=cancel)
    assert updates[0] == 0 and updates[-1] < .01


def test_rtm_workflow_focuses_a_discrete_born_scatterer():
    """Exercise time-data Fourier scaling, geometry and RTM together. The Green test above is independent of this operator."""
    from seisdesk.core.dataset import SeismicData
    from seisdesk.core.velocity import DepthModel
    from seisdesk.processing import Context, create
    dx, dt, nt = 10., .004, 256
    velocity = np.full((41, 51), 2000.)
    sx = np.repeat([100., 250., 400.], 17); gx = np.tile(np.linspace(50, 450, 17), 3)
    frequencies = np.fft.rfftfreq(nt, dt)
    spectra = np.zeros((len(sx), len(frequencies)), complex)
    for fi in np.flatnonzero((frequencies >= 4) & (frequencies <= 15)):
        f = frequencies[fi]; op = Helmholtz(velocity, dx, dx, f, nabs=12)
        scatterer = op.sampling([[250, 200]])
        q = (f/10)**2*np.exp(-(f/10)**2)
        for shot in np.unique(sx):
            idx = np.flatnonzero(sx == shot)
            source = op.solve(op.source([shot, 0]))*q
            rhs = scatterer.T.toarray().ravel()*(scatterer @ source)[0]*(2*np.pi*f)**2
            scattered = op.solve(rhs)
            spectra[idx, fi] = op.sampling(np.column_stack([gx[idx], np.zeros(len(idx))])) @ scattered
    traces = np.fft.irfft(spectra/dt, nt, axis=1)
    data = SeismicData(traces, dt, {'sx': sx, 'gx': gx, 'fldr': sx, 'offset': gx-sx})
    context = Context(depth_model=DepthModel(velocity, dx, dx))
    out = create('rtm', fmin=4, fmax=15, f0=10, nabs=12, normalise=False, laplacian=False)(data, context)
    interior = np.abs(out.traces[:, 8:])
    ix, iz = np.unravel_index(np.argmax(interior), interior.shape)
    assert abs(out.header('cdpx')[ix]-250) <= 20
    assert abs(out.times[iz+8]-200) <= 20
