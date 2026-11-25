import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from seisdesk.core.dataset import SeismicData
from seisdesk.core.models import model_from_text
from seisdesk.processing import Context, create
from seisdesk.processing.converted import (asymptotic_ccp, conversion_distance, gamma_from_times,
                                           ps_flat_reflections, ps_moveout_time, ps_nmo_correct)
from seisdesk.processing.modelling import elastic_shot, flat_reflections
from seisdesk.processing.modelling_aniso import chebyshev_propagators, ti_flat_reflections
from seisdesk.processing.migration_time import rho_filter


def test_chebyshev_large_step_matches_exact_fourier_mode():
    n, spacing, dt, speed = 64, 5., .01, 2000.
    mode = np.broadcast_to(np.cos(2*np.pi*2*np.arange(n)/n), (n, n)).copy()
    omega = speed*2*np.pi*2/(n*spacing)
    advance = chebyshev_propagators(np.full((n, n), speed), spacing, dt)
    u, velocity = advance(mode, np.zeros_like(mode))
    assert np.allclose(u, np.cos(omega*dt)*mode, atol=1e-8)
    assert np.allclose(velocity, -omega*np.sin(omega*dt)*mode, atol=1e-6)
    assert speed*dt/spacing > 3  # well beyond the explicit FD stability limit


def test_vti_elastic_direct_arrival_velocities():
    n, spacing, dt = 161, 5., .001
    vp = np.full((n, n), 2000.); vs = vp/1.8; rho = np.full_like(vp, 2200)
    recordings = elastic_shot(vp, vs, rho, spacing, dt, 220, (80, 80), np.array([80, 120]),
                              np.array([120, 80]), 35, 20, component='pressure', epsilon=.15, delta=.05)
    corrected = rho_filter(recordings, dt, modelling=True)
    expected = np.array([200/(2000*np.sqrt(1.3)), 200/2000])
    for trace, time in zip(corrected, expected):
        start, end = int((time-.025)/dt), int((time+.025)/dt)
        pick = (start+np.argmax(np.abs(trace[start:end])))*dt
        assert pick == pytest.approx(time, rel=.03)


def test_ti_ray_zero_anisotropy_and_vertical_limit():
    model = model_from_text('2000,2600 / 400 / 1100,1400 / 2200,2400')
    sx = np.zeros(4); gx = np.array([0., 100, 300, 600]); z = np.zeros(4)
    isotropic = flat_reflections(model, sx, z, gx, z)[0]
    assert np.array_equal(ti_flat_reflections(model, sx, z, gx, z, 0, 0)[0], isotropic)
    anisotropic = ti_flat_reflections(model, sx, z, gx, z, .1, .05)[0]
    assert anisotropic[0, 0] == pytest.approx(.4, abs=1e-8)
    assert anisotropic[-1, 0] < isotropic[-1, 0]


def test_ps_ccp_moveout_and_independent_ray_minimum():
    offset = np.array([0., 100, 300, 600])
    assert np.allclose(conversion_distance(offset, 1e6, 2000, 1000), asymptotic_ccp(0, offset, 2), atol=1e-4)
    model = model_from_text('2000,2600 / 400 / 1000,1400 / 2200,2400')
    times = ps_flat_reflections(model, np.zeros(4), np.zeros(4), offset, np.zeros(4))[0][:, 0]
    for off, actual in zip(offset, times):
        f = lambda x: np.hypot(x, 400)/2000 + np.hypot(off-x, 400)/1000
        minimum = minimize_scalar(f, bounds=(0, max(off, 1e-6)), method='bounded').fun
        assert actual == pytest.approx(minimum, abs=1e-6)
    t = np.arange(1001)*.001
    moved = ps_moveout_time(.5, offset, np.sqrt(2000*1000), 2)
    traces = np.exp(-.5*((t[None]-moved[:, None])/.005)**2)
    corrected = ps_nmo_correct(traces, t, offset, np.full(traces.shape, np.sqrt(2e6)), 2, stretch_limit=0)
    assert np.allclose(t[np.argmax(corrected, axis=1)], .5, atol=.001)
    avg, interval = gamma_from_times(np.array([[.2, .4]]), np.array([[.3, .65]]))
    assert np.allclose(avg, [[2, 2.25]]) and np.allclose(interval, [[2, 2.5]])


def test_sliding_shear_rotation_recovers_two_angles():
    nt, dt = 600, .002; t = np.arange(nt)*dt
    fast = np.exp(-.5*((t-.25)/.025)**2)+np.exp(-.5*((t-.85)/.025)**2)
    slow = .7*(np.exp(-.5*((t-.30)/.025)**2)+np.exp(-.5*((t-.90)/.025)**2))
    angle = np.deg2rad(np.where(t < .6, 20, 40)); c, s = np.cos(angle), np.sin(angle)
    rows = np.stack([c*c*fast+s*s*slow, c*s*(fast-slow), c*s*(fast-slow), s*s*fast+c*c*slow])
    d = SeismicData(rows, dt, {'comp': [1, 2, 3, 4], 'sx': 0, 'gx': 100})
    ctx = Context(); out = create('ltt_rotation', window=.06)(d, ctx)
    angles = ctx.results['ltt_rotation']['angles'][0] % 90
    assert angles[int(.25/dt)] == pytest.approx(20, abs=1)
    assert angles[int(.85/dt)] == pytest.approx(40, abs=1)
    assert np.linalg.norm(out.traces[1:3]) < .01*np.linalg.norm(rows)
