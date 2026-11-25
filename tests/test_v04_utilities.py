import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.gpr import read_dt1
from seisdesk.core.segy_io import write_segy, write_su
from seisdesk.core.synthetic import ricker, events_to_traces
from seisdesk.core.velocity import DepthModel, vint_to_vrms
from seisdesk.processing import create, Context
from seisdesk.processing.geometry_tools import utm_wgs84
from seisdesk.processing.modelling_tools import goupillaud_response
from seisdesk.processing.velocity_tools import log_offset_continuation, ti_nmo_velocity


def test_dt1_import_preserves_nanoseconds_and_per_trace_zero(tmp_path):
    path = tmp_path/'profile.DT1'; ns = 101
    (tmp_path/'profile.HD').write_text('NUMBER OF TRACES = 2\nNUMBER OF PTS/TRC = 101\n'
                                     'TIMEZERO AT POINT = 10\nTOTAL TIME WINDOW = 100\n'
                                     'POSITION UNITS = metres\nANTENNA SEPARATION = 0.2\n')
    with path.open('wb') as f:
        for i, adjustment in enumerate((0, 2)):
            h = np.zeros(32, '<f4'); h[0] = i+1; h[1] = i*.5; h[2] = ns; h[5] = 2; h[20] = adjustment
            a = np.zeros(ns, '<i2'); a[30+adjustment] = 100
            f.write(h.tobytes()); f.write(a.tobytes())
    d = read_dt1(path)
    assert d.dt == pytest.approx(1e-9) and d.t0 == pytest.approx(-12e-9)
    assert d.nt == 103
    assert np.allclose(d.times[np.argmax(d.traces, axis=1)], 20e-9, rtol=0, atol=1e-15)
    assert np.allclose(d.header('offset'), .2)
    for writer, extension in ((write_su, 'su'), (write_segy, 'sgy')):
        with pytest.raises(ValueError, match='GPR'):
            writer(d, str(tmp_path/f'bad.{extension}'))


def test_utm_reference_and_ne_azimuth():
    e, n, zone, south = utm_wgs84([2.2945], [48.8583])
    assert zone == 31 and not south
    assert e[0] == pytest.approx(448251.9, abs=1)
    assert n[0] == pytest.approx(5411943.8, abs=1)
    d = SeismicData(np.zeros((1, 100)), .004, {'sx': 0, 'sy': 0, 'gx': 100, 'gy': 100})
    out = create('azimuth')(d, Context())
    assert out.header('azimuth')[0] == pytest.approx(45)
    assert out.header('offset')[0] == pytest.approx(np.sqrt(20000))


def test_flex_binning_and_floating_datum_sign():
    d = SeismicData(np.zeros((6, 200)), .002, {'cdp': [1, 2, 2, 3, 3, 3], 'cdpx': [0, 10, 10, 20, 20, 20],
                    'sx': np.arange(6)*10, 'gx': np.arange(6)*10+100, 'selev': 100, 'gelev': 80})
    flex = create('flex_binning', target_fold=3)(d, Context())
    assert np.all(np.unique(flex.header('cdp'), return_counts=True)[1] >= 3)
    assert len(set(flex.trace_ids)) == flex.ntr
    flat = create('floating_datum', mode='surface to flat', velocity=2000, apply=False)(d, Context())
    assert np.allclose(flat.header('sstat'), -50) and np.allclose(flat.header('gstat'), -40)


def test_streamer_motion_inverse_kinematics():
    v, u, offset, t0 = 1500., 20., 1000., .4
    fixed = np.hypot(t0, offset/v)
    a, b = offset*u/v**2, 1-(u/v)**2
    moving = (a+np.sqrt(a*a+b*fixed*fixed))/b
    time = np.arange(1501)*.001
    traces = np.exp(-.5*((time-moving)/.006)**2)[None]
    d = SeismicData(traces, .001, {'sx': 0, 'gx': offset})
    out = create('streamer_motion', speed=u, velocity=v)(d, Context())
    assert out.times[np.argmax(out.traces[0])] == pytest.approx(fixed, abs=.001)


def test_interval_model_to_rms_and_taup_ellipse():
    ctx = Context(depth_model=DepthModel(np.tile([1500., 2000., 2500., 3000.], (2, 1)).T, 100, 10))
    d = SeismicData(np.zeros((2, 1000)), .001, {'cdp': [1, 2], 'cdpx': [0, 10]})
    create('interval_to_stacking')(d, ctx)
    row = ctx.results['interval_to_stacking'][0]
    assert np.allclose(row['vrms'], vint_to_vrms(row['time'], ctx.depth_model.values[:, 0]))
    p = np.linspace(0, .0004, 11); t0 = .5; v = 2000.
    tau = t0*np.sqrt(1-(p*v)**2)
    traces = np.exp(-.5*((d.times[None]-tau[:, None])/.006)**2)
    gather = SeismicData(traces, d.dt, {'p': p})
    out = create('taup_nmo', velocity=str(v))(gather, Context())
    assert np.allclose(out.times[np.argmax(out.traces, axis=1)], t0, atol=.001)
    assert np.allclose(ti_nmo_velocity(2000, 0, 0, np.array([0, 30, 60])), 2000/np.cos(np.deg2rad([0, 30, 60])))


def test_goupillaud_internal_multiple_coefficients_and_impedance_roundtrip():
    r = [.2, .3]; response = goupillaud_response(r, .02, .001, 300)
    assert response[20] == pytest.approx(.2, abs=1e-6)
    assert response[40] == pytest.approx((1-.2**2)*.3, abs=1e-6)
    assert response[60] == pytest.approx(-(1-.2**2)*.2*.3**2, abs=1e-6)
    primary = goupillaud_response(r, .02, .001, 300, True)
    assert primary[20] == .2 and primary[40] == .3 and np.count_nonzero(primary) == 2
    imp = np.array([[4e6, 5e6, 3e6, 6e6]], np.float32)
    d = SeismicData(imp, .001)
    rc = create('impedance_reflectivity')(d, Context())
    recovered = create('impedance_reflectivity', mode='reflectivity to impedance', initial=4e6)(rc, Context())
    assert np.allclose(recovered.traces, imp, rtol=1e-6)


def test_scatterer_time_and_offset_continuation_flat_event():
    d = SeismicData(np.zeros((3, 1001)), .001, {'sx': [0, 100, 200], 'gx': [200, 300, 400]})
    out = create('scatterer_synthetics', scatterers='200,0,400,1')(d, Context())
    exact = (np.hypot(d.header('sx')-200, 400)+np.hypot(d.header('gx')-200, 400))/2000
    assert np.allclose(out.times[np.argmax(out.traces, axis=1)], exact, atol=.0011)
    flat = np.tile(np.exp(-.5*((d.times-.5)/.01)**2), (32, 1))
    continued = log_offset_continuation(flat, d.times, 10, 200, 0)
    assert np.allclose(continued, flat, atol=1e-10)


def test_offset_continuation_dipping_event_kinematics():
    x = (np.arange(80)-40)*10; time = np.arange(1001)*.001
    sine = np.sin(np.deg2rad(20)); velocity = 2000.; h = 200
    zero = .5+2*sine*x/velocity
    nmo_time = np.sqrt(zero**2-(2*h*sine/velocity)**2)
    wavelet, centre = ricker(30, .001)
    traces = events_to_traces(nmo_time[:, None], np.ones((len(x), 1)), len(time), .001, wavelet, centre)
    continued = log_offset_continuation(traces, time, 10, h, 0)
    picks = time[np.argmax(continued, axis=1)]
    assert np.max(np.abs(picks[20:60]-zero[20:60])) < .004
