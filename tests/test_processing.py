import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.picks import PickSet, refraction_layers
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.core.velocity import VelocityPicks
from seisdesk.processing import REGISTRY, Context, create
from seisdesk.processing.decon import apply_filters, prediction_error_filters
from seisdesk.processing.filters import bandpass, fk_fan_filter
from seisdesk.processing.nmo import inverse_nmo, nmo_correct
from seisdesk.processing.statics import shift_traces
from seisdesk.processing.velan import semblance, supergather
from seisdesk.workflow.templates import templates

TRUE_V = [(0.0, 1500.0), (0.6, 1700.0), (1.1, 2100.0), (1.6, 2500.0), (2.2, 2900.0), (3.0, 3200.0)]


def true_picks(cdps=(1.0,)):
    vp = VelocityPicks()
    for c in cdps:
        for t, v in TRUE_V:
            vp.add(c, t, v)
    return vp


def middle_cmp(data):
    cdp, counts = np.unique(data.header("cdp"), return_counts=True)
    full = cdp[counts == counts.max()]
    return float(full[full.size // 2])


def event_times(traces, times, t_lo, t_hi):
    sel = (times >= t_lo) & (times <= t_hi)
    return times[sel][np.argmax(np.abs(traces[:, sel]), axis=1)]


# ---------------------------------------------------------------------------------------------- registry
def test_every_process_builds_with_defaults():
    assert len(REGISTRY) >= 40
    for name, cls in REGISTRY.items():
        proc = create(name)
        assert proc.label and proc.category
        for p in cls.params:
            assert p.name in proc.values


def test_templates_are_valid():
    for name, flow in templates().items():
        for step in flow.steps:
            step.build()


def test_param_coercion_and_range():
    proc = create("bandpass", corners="2, 4; 50,60")
    assert proc["corners"] == [2.0, 4.0, 50.0, 60.0]
    with pytest.raises(ValueError):
        create("agc", window=-1)
    with pytest.raises(ValueError):
        create("agc", no_such=1)
    assert create("sort", keys="fldr, offset")["keys"] == ["fldr", "offset"]


# ---------------------------------------------------------------------------------------------- gain and filters
def test_gain_steps(marine):
    out = create("agc", window=0.25)(marine)
    seg = out.traces[:, 100:600].astype(np.float64)
    rms = np.sqrt(np.mean(seg ** 2))
    assert 0.3 < rms < 3.0
    bal = create("trace_balance")(marine)
    assert np.allclose(np.sqrt(np.mean(bal.traces.astype(np.float64) ** 2, axis=1)), 1.0, rtol=1e-3)
    tp = create("tpow", power=1.0)(marine)
    assert np.allclose(tp.traces[:, 500], marine.traces[:, 500] * marine.times[500], rtol=1e-4, atol=1e-6)
    clipped = create("scale", factor=2.0, clip_percentile=90)(marine)
    assert np.abs(clipped.traces).max() <= np.percentile(np.abs(marine.traces * 2.0), 90) + 1e-6


def test_bandpass_and_notch():
    dt = 0.002
    t = np.arange(2000) * dt
    tr = (np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 40 * t) + np.sin(2 * np.pi * 50 * t))[None, :]
    out = bandpass(tr, dt, (20, 25, 45, 48))[0]
    spec = np.abs(np.fft.rfft(out[200:-200] * np.hanning(1600)))
    f = np.fft.rfftfreq(1600, dt)
    amp = lambda f0: spec[np.argmin(np.abs(f - f0))]
    assert amp(40) > 20 * amp(5) and amp(40) > 20 * amp(50)
    d = SeismicData(tr, dt)
    n = create("notch", frequency=50.0, quality=10.0)(d).traces[0]
    spec_n = np.abs(np.fft.rfft(n[200:-200] * np.hanning(1600)))
    assert spec_n[np.argmin(np.abs(f - 50))] < 0.1 * spec_n[np.argmin(np.abs(f - 40))]
    b = create("butterworth", low_cut=20.0, high_cut=0.0)(d).traces[0]
    spec_b = np.abs(np.fft.rfft(b[200:-200] * np.hanning(1600)))
    assert spec_b[np.argmin(np.abs(f - 5))] < 0.05 * spec_b[np.argmin(np.abs(f - 40))]


def test_fk_filter_rejects_slow_linear_event():
    dt, dx, ntr, nt = 0.004, 2.0, 64, 400                  # 2 m traces: the 400 m/s event is not aliased
    x = np.arange(ntr) * dx
    w, c = ricker(25.0, dt)
    slow = events_to_traces((0.1 + x / 400.0)[:, None], np.ones((ntr, 1)), nt, dt, w, c)
    flat = events_to_traces(np.full((ntr, 1), 0.8), np.ones((ntr, 1)), nt, dt, w, c)
    out = fk_fan_filter(slow + flat, dt, dx, 0.0, 1000.0, "reject", 0.1)
    inner = slice(8, ntr - 8)
    flat_win = slice(int(0.75 / dt), int(0.85 / dt))
    kept = np.sum(out[inner, flat_win] * flat[inner, flat_win]) / np.sum(flat[inner, flat_win] ** 2)
    assert kept > 0.8
    resid = out - flat
    assert np.sum(resid[inner] ** 2) < 0.2 * np.sum(slow[inner] ** 2)


# ---------------------------------------------------------------------------------------------- deconvolution
def test_spiking_decon_recovers_reflectivity():
    rng = np.random.default_rng(5)
    nt, dt = 3000, 0.004
    r = rng.standard_normal((4, nt))
    wavelet = np.array([1.0, 0.6, 0.2])                     # minimum phase
    tr = np.stack([np.convolve(x, wavelet)[:nt] for x in r])
    f = prediction_error_filters(tr, dt, 0.2, dt, 0.01, 0.0, nt * dt)
    out = apply_filters(tr, f)
    for i in range(4):
        cc = np.corrcoef(out[i, 100:-100], r[i, 100:-100])[0, 1]
        assert cc > 0.95
    d = SeismicData(tr, dt)
    out2 = create("decon", operator_length=0.2, gap=0.004, prewhitening=0.01, tmin=0.0, tmax=12.0)(d)
    assert np.corrcoef(out2.traces[0, 100:-100], r[0, 100:-100])[0, 1] > 0.95


# ---------------------------------------------------------------------------------------------- velocities, NMO
def test_semblance_peaks_at_true_velocity(marine_cmp):
    c = middle_cmp(marine_cmp)
    idx = supergather(marine_cmp, c, 1)
    vel = np.linspace(1400, 3500, 106)
    spec = semblance(marine_cmp.traces[idx], marine_cmp.header("offset")[idx], marine_cmp.times, marine_cmp.dt,
                     vel, 0.02)
    for t0, v_true in ((1.1, 2100.0), (1.6, 2500.0)):
        j = int(round(t0 / marine_cmp.dt))
        v_best = vel[np.argmax(spec[:, j - 2:j + 3].max(axis=1))]
        assert abs(v_best - v_true) / v_true < 0.04


def test_auto_velocity_close_to_truth(marine_cmp):
    c = middle_cmp(marine_cmp)
    sub = marine_cmp.subset(np.abs(marine_cmp.header("cdp") - c) <= 4)      # full-fold CMPs only
    ctx = Context()
    create("auto_velocity", cdp_step=1000, supergather=2, vmin=1400, vmax=3500, nv=106,
           pick_interval=0.05)(sub, ctx)                        # picks every 50 ms land on the events
    vp = ctx.velocity
    assert vp is not None and len(vp) > 3
    v = vp.function(vp.cdps[0], np.array([1.1, 1.6]))
    assert abs(v[0] - 2100) / 2100 < 0.07 and abs(v[1] - 2500) / 2500 < 0.07


def test_nmo_flattens_and_inverse_restores(marine_cmp):
    c = middle_cmp(marine_cmp)
    g = marine_cmp.gather("cdp", c)
    ctx = Context(velocity=true_picks())
    flat = create("nmo", velocity="project", stretch_limit=0.6)(g, ctx)
    t = event_times(flat.traces, flat.times, 1.05, 1.15)
    assert np.std(t) < 1.5 * g.dt
    raw_t = event_times(g.traces, g.times, 1.05, 1.35)
    assert np.ptp(raw_t) > 0.1                              # there was moveout to remove
    back = create("nmo", velocity="project", inverse=True)(flat, ctx)
    sel = (g.times > 1.0) & (g.times < 1.4)
    near = np.argsort(np.abs(g.header("offset")))[:4]
    cc = np.corrcoef(back.traces[near][:, sel].ravel(), g.traces[near][:, sel].ravel())[0, 1]
    assert cc > 0.9


def test_stretch_mute_zeroes_shallow_far_offsets():
    times = np.arange(500) * 0.004
    tr = np.ones((2, 500), dtype=np.float32)
    off = np.array([0.0, 1000.0])
    vf = np.full((2, 500), 1500.0)
    out = nmo_correct(tr, times, off, vf, 0.0, 0.3)
    assert np.all(out[0, 1:] == 1.0)
    assert np.all(out[1, :190] == 0.0)                        # at 1 km, t0 < 0.80 s is stretched more than 30 %
    assert out[1, 375] == 1.0                                 # t0 = 1.5 s: little stretch, inside the record
    back = inverse_nmo(out, times, off, vf)
    assert back.shape == out.shape


def test_eta_moveout_is_faster_at_far_offset():
    from seisdesk.processing.moveout import moveout_time
    t_h = moveout_time(np.array([1.0]), np.array([3000.0]), np.array([2000.0]), 0.0)
    t_a = moveout_time(np.array([1.0]), np.array([3000.0]), np.array([2000.0]), 0.1)
    assert t_a[0] < t_h[0]


def test_stack_fold_and_amplitude(marine_cmp):
    ctx = Context(velocity=true_picks())
    st = create("stack")(create("nmo")(marine_cmp, ctx), ctx)
    assert st.ntr == np.unique(marine_cmp.header("cdp")).size
    assert st.header("nhs").max() == 12
    j = np.argmin(np.abs(st.header("cdp") - middle_cmp(marine_cmp)))
    w = (st.times > 0.9) & (st.times < 1.3)
    assert abs(st.times[w][np.argmax(np.abs(st.traces[j, w]))] - 1.1) < 0.012


def test_rmo_reduces_residual_curvature(marine_cmp):
    c = middle_cmp(marine_cmp)
    g = marine_cmp.gather("cdp", c)
    wrong = VelocityPicks()
    for t, v in TRUE_V:
        wrong.add(1.0, t, v * 1.06)
    nm = create("nmo", velocity="project", stretch_limit=0.8)(g, Context(velocity=wrong))
    before = np.ptp(event_times(nm.traces, nm.times, 1.0, 1.2))
    fixed = create("residual_moveout", rmin=-0.06, rmax=0.06, nr=61, smooth=0.05)(nm, Context())
    after = np.ptp(event_times(fixed.traces, fixed.times, 1.0, 1.2))
    assert after < 0.6 * before


def test_radon_attenuates_multiples(marine_cmp):
    ctx = Context(velocity=true_picks())
    nm = create("nmo", stretch_limit=0.8)(marine_cmp, ctx)
    c = middle_cmp(marine_cmp)
    sel = nm.header("cdp") == c
    out = create("radon_demultiple", qmin=-0.05, qmax=0.5, nq=101, q_cut=0.04, fmax=60.0)(nm.subset(sel), ctx)
    g, o = nm.subset(sel), out
    far = np.abs(g.header("offset")) > 700
    mwin = (g.times > 1.22) & (g.times < 1.40)              # the first water-bottom multiple, under-corrected
    e_before = np.sum(g.traces[far][:, mwin] ** 2)
    e_after = np.sum(o.traces[far][:, mwin] ** 2)
    assert e_after < 0.5 * e_before
    pwin = (g.times > 1.06) & (g.times < 1.14)
    near = np.abs(g.header("offset")) < 500
    cc = np.corrcoef(g.traces[near][:, pwin].ravel(), o.traces[near][:, pwin].ravel())[0, 1]
    assert cc > 0.8


def test_dmo_keeps_flat_events(marine_cmp):
    ctx = Context(velocity=true_picks())
    nm = create("nmo")(marine_cmp, ctx)
    out = create("dmo", offset_bin=100.0, fmax=60.0)(nm, ctx)
    cdp = nm.header("cdp")
    mid = (cdp > np.percentile(cdp, 30)) & (cdp < np.percentile(cdp, 70))
    sel = (nm.times > 0.5) & (nm.times < 2.0)
    a = nm.traces[mid][:, sel].ravel()
    b = out.traces[mid][:, sel].ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.85


# ---------------------------------------------------------------------------------------------- statics
def test_shift_and_elevation_statics():
    dt = 0.002
    tr = np.zeros((1, 500), dtype=np.float32)
    tr[0, 250] = 1.0
    moved = shift_traces(tr, dt, np.array([0.010]))
    assert np.argmax(moved[0]) == 255
    d = SeismicData(tr, dt, {"selev": 100.0, "gelev": 100.0, "sdepth": 0.0})
    out = create("elevation_statics", datum=0.0, replacement_velocity=2000.0)(d)
    assert np.isclose(out.header("sstat")[0], -50.0) and np.isclose(out.header("gstat")[0], -50.0)
    assert np.argmax(out.traces[0]) == 200
    back = create("apply_statics", keys="sstat,gstat", sign=-1.0)(out)
    assert np.argmax(back.traces[0]) == 250


def test_residual_statics_recover_surface_consistent_shifts():
    from seisdesk.core.synthetic import land_refraction_line
    data = land_refraction_line(nshots=12, nrec=96, nt=601, v1=1e9, refractors=(),
                                reflectors=((0.45, 0.6, 2200.0), (0.7, -0.5, 2600.0)),
                                source_statics_ms=5.0, receiver_statics_ms=5.0, seed=7)
    ctx = Context(velocity=VelocityPicks.from_function([0.0, 0.45, 0.7, 1.2], [2200.0, 2200.0, 2600.0, 2600.0]))
    nm = create("nmo", stretch_limit=0.8)(data, ctx)
    create("residual_statics", tmin=0.3, tmax=0.95, max_shift=0.02, iterations=4)(nm, ctx)
    res = ctx.results["residual_statics"]
    src = np.array([res["source"][k] for k in sorted(res["source"])])
    rec = np.array([res["receiver"][k] for k in sorted(res["receiver"])])
    true_s = data.meta["synthetic"]["source_statics_s"] * 1000.0
    true_r = data.meta["synthetic"]["receiver_statics_s"] * 1000.0
    assert np.corrcoef(src, -true_s)[0, 1] > 0.7
    assert np.corrcoef(rec, -true_r)[0, 1] > 0.7


# ---------------------------------------------------------------------------------------------- migration
def diffraction_section(nx=101, nt=501, dt=0.004, dx=12.5, x0=625.0, z=600.0, v=2000.0):
    x = np.arange(nx) * dx
    t = 2.0 * np.hypot(z, x - x0) / v
    w, c = ricker(20.0, dt)
    tr = events_to_traces(t[:, None], np.ones((nx, 1)), nt, dt, w, c)
    heads = {"cdp": np.arange(1, nx + 1), "cdpx": x, "sx": x, "gx": x, "offset": 0.0}
    d = SeismicData(tr.astype(np.float32), dt, heads, name="diffraction")
    d.meta["geometry"] = {"type": "2D", "bin_size": dx}
    return d


def focus_ratio(traces, axis_values, x_index, y_value, dy, half_x=3, half_y=0.03):
    y = np.asarray(axis_values)
    sel_y = np.abs(y - y_value) <= half_y
    e = traces.astype(np.float64) ** 2
    inside = e[max(0, x_index - half_x):x_index + half_x + 1][:, sel_y].sum()
    return inside / e.sum()


@pytest.mark.parametrize("name,params,need", [("stolt", {"velocity": 2000.0}, 0.4),
                                              ("phase_shift", {"velocity": "2000", "fmax": 60.0}, 0.4),
                                              ("kirchhoff_pstm", {"velocity": "2000", "aperture": 1500.0}, 0.3)])
def test_time_migration_collapses_diffraction(name, params, need):
    d = diffraction_section()
    before = focus_ratio(d.traces, d.times, 50, 0.6, d.dt)
    out = create(name, **params)(d, Context())
    j = int(np.argmin(np.abs(out.header("cdpx") - 625.0)))
    after = focus_ratio(out.traces, out.times, j, 0.6, out.dt)
    assert after > need and after > 3 * before


@pytest.mark.parametrize("name,params", [("split_step", {"velocity": "2000", "dz": 10.0, "zmax": 1200.0, "fmax": 50.0}),
                                         ("kirchhoff_psdm", {"velocity": "2000", "dz": 10.0, "zmax": 1200.0,
                                                             "grid_dx": 12.5, "aperture": 1500.0})])
def test_depth_migration_collapses_diffraction(name, params):
    d = diffraction_section()
    out = create(name, **params)(d, Context())
    assert out.domain == "depth"
    j = int(np.argmin(np.abs(out.header("cdpx") - 625.0)))
    after = focus_ratio(out.traces, out.times, j, 600.0, out.dt, half_y=40.0)
    assert after > 0.35
    zpeak = out.times[np.argmax(np.abs(out.traces[j]))]
    assert abs(zpeak - 600.0) < 30.0


def test_pstm_image_gathers_shape(marine_cmp):
    ctx = Context(velocity=true_picks())
    small = marine_cmp.subset(marine_cmp.header("cdp") < np.percentile(marine_cmp.header("cdp"), 25))
    out = create("kirchhoff_pstm", velocity="project", aperture=500.0, output="image gathers",
                 offset_class=400.0)(small, ctx)
    ncdp = np.unique(small.header("cdp")).size
    assert out.ntr % ncdp == 0 and np.all(np.isfinite(out.traces))


# ---------------------------------------------------------------------------------------------- conversion
def test_time_depth_conversion_round_trip():
    dt = 0.004
    tr = np.zeros((3, 501), dtype=np.float32)
    tr[:, 125] = 1.0                                          # 0.5 s
    d = SeismicData(tr, dt, {"cdp": [1, 2, 3], "cdpx": [0.0, 25.0, 50.0]})
    ctx = Context(velocity=VelocityPicks.constant(2000.0))
    dz = create("time_to_depth", velocity="project", dz=5.0, zmax=1500.0)(d, ctx)
    assert dz.domain == "depth"
    assert abs(dz.times[np.argmax(dz.traces[1])] - 500.0) <= 5.0
    create("build_depth_model", source="constant", v0=2000.0, dz=5.0, zmax=1500.0)(d, ctx)
    assert ctx.depth_model is not None
    back = create("depth_to_time", dt=0.004, tmax=2.0)(dz, ctx)
    assert abs(back.times[np.argmax(back.traces[1])] - 0.5) <= 0.008
    via_model = create("time_to_depth", using="depth model", dz=5.0, zmax=1500.0)(d, ctx)
    assert abs(via_model.times[np.argmax(via_model.traces[1])] - 500.0) <= 5.0


def test_build_depth_model_from_picks(marine_cmp):
    ctx = Context(velocity=true_picks())
    create("build_depth_model", source="picks (Dix)", dz=20.0, zmax=2000.0, smooth=0.0)(marine_cmp, ctx)
    m = ctx.depth_model
    assert m.dx > 0 and m.values.shape[0] == 101
    assert np.isclose(m.values[0, 0], 1500.0, rtol=0.05)


# ---------------------------------------------------------------------------------------------- picking
def test_first_break_picker_consistent_with_truth(land):
    ctx = Context()
    out = create("auto_first_breaks", pickset="fb", gather_key="fldr", v_fast=6000.0, v_slow=500.0,
                 margin=0.04)(land, ctx)
    picks = out.header("laga") / 1000.0
    truth = land.meta["synthetic"]["first_break"]
    use = np.abs(land.header("offset")) > 60.0
    err = picks[use] - truth[use]
    assert abs(np.median(err)) < 0.04                     # onset vs zero-phase peak: a constant early bias
    assert np.percentile(np.abs(err - np.median(err)), 90) < 0.006
    assert ctx.picks["fb"].count() == land.ntr


def test_refraction_analysis_from_branch_picks(land):
    syn = land.meta["synthetic"]
    v1, refr = syn["v1"], syn["refractors"]
    ps = PickSet("fldr", "offset", "fb")
    names = [ps.new_phase(n) for n in ("direct", "refractor 1", "refractor 2")]
    shot = land.header("fldr") == 1
    x = land.header("offset")[shot]
    branches = np.stack([np.abs(x) / v1] + [np.abs(x) / v + ti for v, ti in refr])
    which = np.argmin(branches, axis=0)
    for xi, k in zip(x, which):
        if abs(xi) > 1.0:
            ps.add(1, names[k], xi, branches[k, list(x).index(xi)])
    ctx = Context()
    ctx.picks["fb"] = ps
    create("refraction_analysis", pickset="fb")(land, ctx)
    layers = ctx.results["refraction"]["layers"]
    expected = refraction_layers([{"velocity": v1, "intercept": 0.0, "slowness": 1 / v1}] +
                                 [{"velocity": v, "intercept": ti, "slowness": 1 / v} for v, ti in refr])
    for got, want in zip(layers, expected):
        assert np.isclose(got["thickness"], want["thickness"], rtol=0.02)


def test_picked_mute_uses_phase(marine):
    ps = PickSet("fldr", "offset", "mute")
    ph = ps.new_phase()
    for f in marine.gather_values("fldr")[::5]:
        ps.add(f, ph, -100.0, 0.2)
        ps.add(f, ph, -1300.0, 1.0)
    ctx = Context()
    ctx.picks["mute"] = ps
    out = create("picked_mute", picks="mute", phase=ph, kind="top", taper=0.0)(marine, ctx)
    far = np.argmax(np.abs(marine.header("offset")))
    assert np.all(out.traces[far, : int(0.9 / marine.dt)] == 0.0)
    assert np.any(out.traces[far, int(1.1 / marine.dt):] != 0.0)


# ---------------------------------------------------------------------------------------------- editing
def test_editing_steps(marine):
    killed = create("kill_traces", key="tracf", values="1,3-5", action="zero")(marine)
    hit = np.isin(marine.header("tracf"), [1, 3, 4, 5])
    assert np.all(killed.traces[hit] == 0) and np.all(killed.header("trid")[hit] == 2)
    removed = create("kill_traces", key="tracf", values="1,3-5", action="remove")(marine)
    assert removed.ntr == marine.ntr - hit.sum()
    sel = create("select_traces", key="fldr", min=1001, max=1005, every=2)(marine)
    assert set(np.unique(sel.header("fldr"))) == {1001, 1003, 1005}
    win = create("time_window", tmin=0.5, tmax=1.0)(marine)
    assert np.isclose(win.t0, 0.5) and np.isclose(win.tmax, 1.0)
    rs = create("resample", dt=0.008)(marine)
    assert np.isclose(rs.dt, 0.008) and abs(rs.nt - marine.nt / 2) <= 1
    rev = create("reverse_polarity")(marine)
    assert np.allclose(rev.traces, -marine.traces)


def test_process_reference_lists_every_step():
    from seisdesk.processing.reference import process_reference_html, process_reference_markdown
    md, html = process_reference_markdown(), process_reference_html()
    for name, cls in REGISTRY.items():
        assert f"`{name}`" in md and f"({name})" in html
