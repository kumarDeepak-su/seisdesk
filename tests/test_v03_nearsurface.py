"""Tests of the version 0.3 near-surface tools: linear moveout (one velocity and dynamic), the refraction convolution
section, Kirchhoff wave-equation datuming (prestack and zero offset), MASW dispersion analysis and the viewer's
reduced-time display."""

import numpy as np
import pytest

from seisdesk.core import masw
from seisdesk.core.dataset import SeismicData
from seisdesk.core.picks import PickSet
from seisdesk.core.synthetic import causal_wavelet, events_to_traces, ricker
from seisdesk.processing import REGISTRY, Context, create
from seisdesk.processing import datuming, nearsurface  # noqa: F401  (importing registers the steps)

NEW = ("lmo", "refraction_convolution", "datum_kirchhoff", "datum_zero_offset")
F_C = np.array([4.0, 10.0, 20.0, 30.0, 50.0])            # a normally dispersive Rayleigh-wave curve (Hz, m/s)
V_C = np.array([450.0, 330.0, 240.0, 210.0, 190.0])


def test_near_surface_steps_registered():
    for name in NEW:
        assert name in REGISTRY


def linear_events(offsets, t0, v, nt=500, dt=0.002, f0=30.0):
    w, c = ricker(f0, dt)
    times = (t0 + np.abs(offsets) / v)[:, None]
    tr = events_to_traces(times, np.ones_like(times), nt, dt, w, c)
    return SeismicData(tr.astype(np.float32), dt, {"offset": offsets, "fldr": 1.0})


# ---------------------------------------------------------------------------------------------- linear moveout
def test_lmo_flattens_and_inverse_restores():
    off = np.linspace(-400.0, 400.0, 33)
    d = linear_events(off, 0.05, 2000.0)
    out = create("lmo", velocity=2000.0, shift=0.02)(d)
    peaks = d.times[np.argmax(np.abs(out.traces), axis=1)]
    assert np.all(np.abs(peaks - 0.07) <= 1.5 * d.dt)
    assert out.meta["reduced_time"]["velocity"] == 2000.0
    back = create("lmo", velocity=2000.0, shift=0.02, direction="inverse")(out)
    assert np.abs(back.traces - d.traces).max() < 0.02 * np.abs(d.traces).max()


def test_dynamic_lmo_collapses_a_dispersed_surface_wave():
    off = np.arange(10.0, 101.0, 5.0)
    tr = masw.surface_wave_record(off, 1000, 0.001, F_C, V_C, f0=15.0, t_delay=0.1)
    d = SeismicData(tr.astype(np.float32), 0.001, {"offset": off, "fldr": 1.0})
    curve = ", ".join(f"{a:g}:{b:g}" for a, b in zip(F_C, V_C))
    out = create("lmo", mode="dynamic (dispersion curve)", curve=curve, shift=0.2)(d)
    peaks = out.times[np.argmax(np.abs(out.traces), axis=1)]
    assert np.all(np.abs(peaks - 0.3) <= 0.003)
    raw_peaks = d.times[np.argmax(np.abs(d.traces), axis=1)]
    assert np.ptp(raw_peaks) > 0.2                         # the input really was dispersed over offset


def test_parse_curve_errors():
    with pytest.raises(ValueError):
        nearsurface.parse_curve("10-300")
    f, v = nearsurface.parse_curve("20:200; 5:400")
    assert list(f) == [5.0, 20.0] and list(v) == [400.0, 200.0]


# ---------------------------------------------------------------------------------------------- convolution section
def time_depth(x):
    return 0.015 + 0.005 * np.sin(2 * np.pi * np.asarray(x) / 600.0)


def rcs_line(wavelet="ricker", v2=2000.0, dt=0.001, nt=700):
    """Two shots (x = 0 and 600 m) recorded by receivers every 10 m: one head wave t = tg(s) + tg(g) + |g - s| / V2."""
    rec = np.arange(0.0, 601.0, 10.0)
    sx = np.repeat([0.0, 600.0], rec.size)
    gx = np.tile(rec, 2)
    fl = np.repeat([1.0, 2.0], rec.size)
    t = time_depth(sx) + time_depth(gx) + np.abs(gx - sx) / v2
    w, c = ricker(40.0, dt) if wavelet == "ricker" else causal_wavelet(40.0, dt)
    tr = events_to_traces(t[:, None], np.ones((t.size, 1)), nt, dt, w, c)
    heads = {"fldr": fl, "sx": sx, "gx": gx, "offset": gx - sx, "tracf": np.tile(np.arange(1, rec.size + 1), 2)}
    return SeismicData(tr.astype(np.float32), dt, heads), t


def test_refraction_convolution_images_twice_the_time_depth():
    d, _t = rcs_line()
    t_ab = 2 * 0.015 + 600.0 / 2000.0
    ctx = Context()
    out = create("refraction_convolution", pairing="two shots", forward_ffid=2, reverse_ffid=1, reciprocal="value",
                 reciprocal_time=t_ab)(d, ctx)
    gx = out.header("gx")
    assert gx.min() > 0 and gx.max() < 600 and out.ntr == 59
    peaks = out.times[np.argmax(np.abs(out.traces), axis=1)]
    assert np.all(np.abs(peaks - 2 * time_depth(gx)) <= 0.0015)
    res = ctx.results["refraction_convolution"]
    assert res["forward"][0] == 1.0 and res["reverse"][0] == 2.0


def test_refraction_convolution_reciprocal_from_picks_and_first_breaks():
    d, t = rcs_line()
    ps = PickSet("fldr", "offset", "fb")
    ps.new_phase("refractor 1")
    for i in range(d.ntr):
        ps.add(d.header("fldr")[i], "refractor 1", d.header("offset")[i], t[i])
    ctx = Context()
    ctx.picks["fb"] = ps
    out = create("refraction_convolution", min_offset=0.0, reciprocal="picks", pickset="fb")(d, ctx)
    peaks = out.times[np.argmax(np.abs(out.traces), axis=1)]
    assert np.all(np.abs(peaks - 2 * time_depth(out.header("gx"))) <= 0.0015)
    assert np.allclose(ctx.results["refraction_convolution"]["reciprocal_ms"], 330.0, atol=0.01)
    dc, _ = rcs_line("causal")
    ctx2 = Context()
    create("refraction_convolution", min_offset=0.0)(dc, ctx2)              # first breaks measured on the traces
    assert np.allclose(ctx2.results["refraction_convolution"]["reciprocal_ms"], 330.0, atol=6.0)


# ---------------------------------------------------------------------------------------------- datuming
def point_source_records(v=1000.0, dt=0.001, nt=600, surface=140.0, depth_point=-100.0):
    """Arrivals from a point at (0, depth_point) recorded on a flat surface 140 m high, receivers every 5 m."""
    x = np.arange(-400.0, 401.0, 5.0)
    r = np.hypot(x, surface - depth_point)
    w, c = ricker(25.0, dt)
    tr = events_to_traces((r / v)[:, None], (1.0 / np.sqrt(r))[:, None], nt, dt, w, c)
    return x, r, tr.astype(np.float32)


def test_kirchhoff_receiver_datuming_follows_oblique_rays():
    v, datum = 1000.0, 100.0
    x, r, tr = point_source_records(v)
    d = SeismicData(tr, 0.001, {"fldr": 1.0, "sx": 0.0, "gx": x, "offset": x, "gelev": 140.0, "selev": -100.0})
    out = create("datum_kirchhoff", datum=datum, velocity=v, side="receivers only", aperture=200.0, max_angle=75.0,
                 fmax=50.0)(d)
    t_true = np.hypot(x, datum + 100.0) / v
    t_static = (r - 40.0) / v
    inner = np.abs(x) <= 200.0
    peaks = out.times[np.argmax(np.abs(out.traces), axis=1)]
    assert np.all(np.abs(peaks[inner] - t_true[inner]) <= 0.002)
    far = inner & (np.abs(x) >= 180.0)
    assert np.all(np.abs(t_static[far] - t_true[far]) > 0.006)           # a vertical-ray static would be wrong here
    assert np.allclose(out.header("gelev"), datum) and np.allclose(out.header("selev"), -100.0)


def test_kirchhoff_source_datuming_and_headers():
    v, datum = 1000.0, 100.0
    x, _r, tr = point_source_records(v)
    heads = {"fldr": np.arange(1, x.size + 1), "sx": x, "gx": 0.0, "offset": -x, "selev": 140.0, "gelev": -100.0}
    d = SeismicData(tr, 0.001, heads)
    out = create("datum_kirchhoff", datum=datum, velocity=v, side="sources only", aperture=200.0, max_angle=75.0,
                 fmax=50.0)(d)
    inner = np.abs(x) <= 200.0
    peaks = out.times[np.argmax(np.abs(out.traces), axis=1)]
    assert np.all(np.abs(peaks[inner] - np.hypot(x, datum + 100.0)[inner] / v) <= 0.002)
    both = create("datum_kirchhoff", datum=datum, velocity=v, aperture=100.0, fmax=40.0)(d)
    assert np.all(np.isfinite(both.traces))
    assert np.allclose(both.header("selev"), datum) and np.allclose(both.header("gelev"), datum)
    assert np.allclose(both.header("sdepth"), 0.0) and both.meta["datum"]["elevation"] == datum


def test_zero_offset_datuming_flattens_a_reflector_under_topography():
    v, dt = 2000.0, 0.002
    x = np.arange(0.0, 1000.0, 10.0)
    elev = 110.0 + 10.0 * np.sin(2 * np.pi * x / 1000.0)
    datum, z_ref = 80.0, -300.0
    w, c = ricker(20.0, dt)
    tr = events_to_traces((2.0 * (elev - z_ref) / v)[:, None], np.ones((x.size, 1)), 400, dt, w, c)
    d = SeismicData(tr.astype(np.float32), dt, {"cdp": np.arange(1, x.size + 1), "cdpx": x, "gelev": elev})
    out = create("datum_zero_offset", datum=datum, velocity=v, aperture=200.0, fmax=40.0)(d)
    peaks = out.times[np.argmax(np.abs(out.traces), axis=1)]
    inner = (x > 150.0) & (x < 850.0)
    assert np.all(np.abs(peaks[inner] - 2.0 * (datum - z_ref) / v) <= 1.5 * dt)
    assert np.allclose(out.header("gelev"), datum)


def test_spacing_weights():
    w = datuming.spacing_weights(np.array([0.0, 10.0, 10.0, 30.0]))
    assert np.allclose(w, [10.0, 7.5, 7.5, 20.0])
    assert np.all(datuming.spacing_weights(np.array([5.0, 5.0])) == 0.0)


# ---------------------------------------------------------------------------------------------- MASW
def test_dispersion_image_and_picks_follow_the_curve():
    off = np.arange(5.0, 96.0, 2.0)
    tr = masw.surface_wave_record(off, 1000, 0.001, F_C, V_C, f0=18.0, t_delay=0.05)
    f, v, img = masw.dispersion_image(tr, off, 0.001, 5.0, 45.0, 100.0, 600.0, 251)
    assert img.shape == (v.size, f.size) and np.allclose(img.max(axis=0), 1.0)
    fp, vp = masw.pick_curve(img, f, v)
    band = (fp >= 8.0) & (fp <= 40.0)
    assert band.sum() >= 20
    true = np.interp(fp[band], F_C, V_C)
    assert np.all(np.abs(vp[band] - true) <= 0.05 * true)
    depth, vs = masw.approximate_vs(fp, vp)
    assert np.all(np.diff(depth) >= 0) and np.allclose(np.sort(vs), np.sort(1.09 * vp))
    assert np.allclose(np.sort(depth), np.sort(vp / fp / 3.0))


def test_masw_dialog(qtbot, tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from seisdesk.gui import common
    from seisdesk.gui.masw_dialog import MaswDialog
    common.HEADLESS = True
    off = np.concatenate([-np.arange(95.0, 4.0, -2.0), np.arange(5.0, 96.0, 2.0)])      # split spread
    tr = masw.surface_wave_record(off, 1000, 0.001, F_C, V_C, f0=18.0, t_delay=0.05)
    d = SeismicData(tr.astype(np.float32), 0.001, {"fldr": 1.0, "offset": off}, name="surface waves")
    dlg = MaswDialog(d)
    qtbot.addWidget(dlg)
    assert dlg.image is not None and dlg.selected_traces().size == 46
    n = dlg.auto_pick()
    assert n >= 10
    depth, _vs = dlg.vs_profile()
    assert depth.size == n
    f0, v0 = dlg.picks[0]
    dlg.add_pick(f0, v0 + 5.0)                                   # replaces the pick at that frequency
    assert len(dlg.picks) == n
    dlg.remove_nearest(f0, v0)
    assert len(dlg.picks) == n - 1
    assert dlg.picks_table().splitlines()[0].startswith("frequency_hz,phase_velocity_ms")
    path = dlg.export(str(tmp_path / "picks.csv"))
    assert path and (tmp_path / "picks.csv").read_text().count("\n") == n
    dlg.omax.setValue(3.0)
    assert not dlg.compute()                                     # too few traces: a message, no error


# ---------------------------------------------------------------------------------------------- viewer
def test_viewer_reduced_time_display(qtbot):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from seisdesk.gui import common
    from seisdesk.gui.viewer import GatherViewer, reduce_display
    from seisdesk.workflow.project import Project
    common.HEADLESS = True
    spike = np.zeros((1, 10), np.float32)
    spike[0, 5] = 1.0
    moved = reduce_display(spike, 0.004, np.array([0.008]))
    assert moved[0, 3] == pytest.approx(1.0) and moved[0, 5] == 0.0
    d = linear_events(np.linspace(-400.0, 400.0, 41), 0.05, 2000.0)
    project = Project()
    name = project.add_dataset(d, "linear events")
    v = GatherViewer(project)
    qtbot.addWidget(v)
    v.set_dataset(name)
    v.reduce_spin.setValue(2000.0)
    assert v.reduced_shift is not None and not v.pick_btn.isEnabled()
    rows = np.argmax(np.abs(v.disp), axis=1)
    assert np.all(np.abs(d.times[rows] - 0.05) <= 1.5 * d.dt)
    v._hover(float(v.xpos[3]), 0.05)
    assert "reduced t" in v.readout.text()
    v.reduce_spin.setValue(0.0)
    assert v.reduced_shift is None and v.pick_btn.isEnabled()


def test_refraction_convolution_uses_the_refraction_before_a_stronger_direct_wave():
    # the direct wave (amplitude 1) arrives inside the record behind the weaker refraction (0.6) at short offsets;
    # the section must still be built from the refraction: onset at twice the time-depth of every receiver
    v1, v2, dt, nt = 600.0, 2200.0, 0.001, 400
    rec = np.arange(0.0, 601.0, 5.0)
    shot_x = np.array([0.0, 600.0])

    def time_depth(x):
        return 0.010 + 0.006 * np.sin(2 * np.pi * x / 450.0)

    sx, gx = np.repeat(shot_x, rec.size), np.tile(rec, shot_x.size)
    x = np.abs(gx - sx)
    times = np.stack([x / v1, time_depth(sx) + time_depth(gx) + x / v2], axis=1)
    amps = np.stack([np.ones(x.size), np.full(x.size, 0.6)], axis=1)
    w, c = causal_wavelet(40.0, dt)
    traces = events_to_traces(times, amps, nt, dt, w, c)
    heads = {"fldr": np.repeat([1.0, 2.0], rec.size), "sx": sx, "gx": gx, "offset": gx - sx,
             "tracf": np.tile(np.arange(1, rec.size + 1), 2)}
    d = SeismicData(traces.astype(np.float32), dt, heads)
    out = create("refraction_convolution", pairing="two shots", forward_ffid=1.0, reverse_ffid=2.0, keep=0.04,
                 tmax=0.08)(d, Context())
    ww = np.convolve(w, w)
    checked = 0
    for k in range(out.ntr):
        g = float(out.header("gx")[k])
        if not 100.0 <= g <= 500.0:
            continue
        xc = np.correlate(out.traces[k].astype(np.float64), ww, mode="full")[ww.size - 1:]
        assert abs(int(np.argmax(xc)) * dt - 2.0 * time_depth(g)) <= 0.003, g
        checked += 1
    assert checked >= 70
