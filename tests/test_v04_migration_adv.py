import numpy as np
import pytest
from scipy.signal import hilbert

from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.processing import Context, create
from seisdesk.processing.migration_3d_depth import layered_time_table
from seisdesk.processing.migration_wave import _trace_beams_vti


def volume(flat=False):
    x, y = np.meshgrid(np.arange(11)*20., np.arange(11)*20., indexing='ij')
    x, y = x.ravel(), y.ravel(); nt, dt = 251, .002
    times = np.full(len(x), .18) if flat else 2*np.sqrt((x-100)**2+(y-100)**2+180**2)/2000
    w, c = ricker(30, dt)
    values = events_to_traces(times[:, None], np.ones((len(x), 1)), nt, dt, w, c)
    return SeismicData(values, dt, {'sx': x, 'gx': x, 'sy': y, 'gy': y, 'offset': 0, 'cdpx': x, 'cdpy': y,
                                   'iline': x/20+1, 'xline': y/20+1, 'cdp': np.arange(len(x))+1})


def test_layered_ray_table_homogeneous_and_vertical():
    depth = np.arange(0, 501, 10.); radius = np.arange(0, 801, 10.)
    table = layered_time_table(depth, np.full(len(depth), 2000.), radius)
    assert np.allclose(table[1:], np.hypot(depth[1:, None], radius[None])/2000, rtol=1e-8)
    velocity = 1500+.8*depth
    table = layered_time_table(depth, velocity, [0, 100])
    exact = np.cumsum(np.diff(depth)/(.5*(velocity[:-1]+velocity[1:])))
    assert np.allclose(table[1:, 0], exact)


def test_3d_depth_migration_focus_and_flat_reflector_phase():
    d = volume()
    out = create('kirchhoff_psdm_3d', velocity='2000', dz=5, zmax=300, aperture=250)(d, Context())
    i, j = np.unravel_index(np.argmax(np.abs(out.traces)), out.traces.shape)
    assert abs(out.header('cdpx')[i]-100) <= 20 and abs(out.header('cdpy')[i]-100) <= 20
    assert abs(out.times[j]-180) <= 10
    flat = create('kirchhoff_psdm_3d', velocity='2000', dz=5, zmax=300, aperture=250)(volume(True), Context())
    central = (flat.header('cdpx') == 100) & (flat.header('cdpy') == 100)
    analytic = hilbert(flat.traces[central][0]); peak = np.argmax(np.abs(analytic))
    assert abs(flat.times[peak]-180) <= 10
    assert abs(np.rad2deg(np.angle(analytic[peak]))) < 40


def test_3d_dmo_preserves_flat_event():
    d = volume(True)
    d.set_header('sx', d.header('cdpx')-100); d.set_header('gx', d.header('cdpx')+100)
    d.set_header('offset', 200)
    out = create('dmo_3d')(d, Context())
    assert np.linalg.norm(out.traces-d.traces) < .02*np.linalg.norm(d.traces)


def test_vti_hamiltonian_ray_obeys_group_slowness():
    # Compare a constant-medium ray against an independent phase-slowness derivative.
    u, eta, p = 1000., .15, .0004
    field = np.zeros((6, 31, 201)); field[0] = u
    table, valid = _trace_beams_vti(np.array([100.]), np.array([p]), field, 5, 5, 201, 31, eta)
    q = lambda pp: np.sqrt((1/u**2-(1+2*eta)*pp**2)/(1-2*eta*u*u*pp**2))
    dq = (q(p+1e-8)-q(p-1e-8))/2e-8
    assert valid[-1, 0]
    assert table['x'][-1, 0] == pytest.approx(100-150*dq, rel=1e-5)
    assert table['tau'][-1, 0] == pytest.approx(150*(q(p)-p*dq), rel=1e-5)
    assert table['px'][-1, 0] == p


def test_fowler_finds_velocity_on_dipping_reflector():
    cmp = np.repeat(np.arange(21), 6); x = (cmp-10)*20.
    offset = np.tile(np.arange(6)*100., 21); dt = .004; nt = 201
    dip = np.deg2rad(15); velocity = 2000.
    zero = .35+2*np.sin(dip)*x/velocity
    arrival = np.sqrt(zero**2+(offset*np.cos(dip)/velocity)**2)
    w, c = ricker(25, dt)
    values = events_to_traces(arrival[:, None], np.ones((len(cmp), 1)), nt, dt, w, c)
    d = SeismicData(values, dt, {'cdp': cmp+1, 'cdpx': x, 'offset': offset, 'sx': x-offset/2, 'gx': x+offset/2})
    ctx = Context()
    create('fowler', vmin=1700, vmax=2300, nv=7, offset_bin=50, fmax=60, stretch=0, write_picks=False)(d, ctx)
    selected = ctx.results['fowler']['selected_velocity'][10, int(round(.35/dt))]
    assert selected == pytest.approx(velocity, abs=150)
