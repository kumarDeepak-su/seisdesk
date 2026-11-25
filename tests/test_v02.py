"""Tests of the version 0.2 steps: QC and editing, mutes, filters, refraction statics, surface-consistent
processing, velocity tools and stack methods."""

import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.picks import PickSet
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.core.velocity import VelocityPicks
from seisdesk.processing import REGISTRY, Context, create

NEW = ("auto_edit", "despike", "zone_mute", "cone_mute", "tv_bandpass", "spectral_whitening", "fx_decon",
       "median_filter", "taup_filter", "trace_mix", "refraction_statics", "sc_amplitude", "sc_decon", "exp_gain",
       "cvg_panels", "smooth_velocity")


def amp_at(seg, t, f):
    return 2.0 * abs(np.mean(seg * np.exp(-2j * np.pi * f * t)))


def test_new_steps_registered():
    for name in NEW:
        assert name in REGISTRY


# ---------------------------------------------------------------------------------------------- QC and editing
def test_auto_edit_finds_dead_and_noisy(marine):
    d = marine.gather("fldr", 1001.0, secondary=None)
    tr = d.traces.copy()
    tr[5] = 0.0
    tr[9] *= 20.0
    d = d.copy(tr)
    ctx = Context()
    out = create("auto_edit", gather_key="fldr", action="zero")(d, ctx)
    assert set(ctx.results["auto_edit"]["dead"]) == {5} and set(ctx.results["auto_edit"]["noisy"]) == {9}
    assert np.all(out.traces[9] == 0) and out.header("trid")[9] == 2 and out.has_header("qc_rms")
    assert create("auto_edit", action="remove")(d, Context()).ntr == d.ntr - 2


def test_despike_removes_a_burst(marine):
    d = marine.gather("fldr", 1001.0)
    tr = d.traces.copy()
    burst = slice(400, 420)
    tr[10, burst] += 50.0 * np.abs(d.traces).max()
    out = create("despike", gather_key="fldr", window=0.05, neighbours=7, threshold=3.0)(d.copy(tr))
    neighbours = np.abs(out.traces[[8, 9, 11, 12], burst]).max()
    assert np.abs(out.traces[10, burst]).max() < 3.0 * max(neighbours, 1e-6)
    assert np.allclose(out.traces[[0, 1, 30]], d.traces[[0, 1, 30]], atol=1e-6)


# ---------------------------------------------------------------------------------------------- mutes
def ones_dataset(ntr=10, nt=750, dt=0.004):
    off = np.linspace(0.0, 900.0, ntr)
    return SeismicData(np.ones((ntr, nt), np.float32), dt, {"fldr": 1.0, "offset": off, "tracf": np.arange(1, ntr + 1)})


def test_zone_and_cone_mutes():
    d = ones_dataset()
    ps = PickSet("fldr", "offset", "zone")
    for name, t0 in (("a", 0.1), ("b", 0.3)):
        ps.new_phase(name)
        for x in (0.0, 900.0):
            ps.add(1, name, x, t0 + x / 3000.0)
    ctx = Context()
    ctx.picks["zone"] = ps
    out = create("zone_mute", picks="zone", upper="a", lower="b", taper=0.0)(d, ctx)
    t, x = d.times, d.header("offset")[4]
    inside = (t > 0.1 + x / 3000 + 0.005) & (t < 0.3 + x / 3000 - 0.005)
    assert np.all(out.traces[4, inside] == 0) and np.all(out.traces[4, t > 0.35 + x / 3000] == 1)
    keep = create("zone_mute", picks="zone", upper="a", lower="b", mode="keep inside", taper=0.0)(d, ctx)
    assert np.all(keep.traces[4, inside] == 1) and np.all(keep.traces[4, t > 0.35 + x / 3000] == 0)
    cone = create("cone_mute", v_fast=1000.0, v_slow=250.0, t0=0.0, taper=0.0)(d, Context())
    j = int(np.argmin(np.abs(d.header("offset") - 500.0)))
    xo = d.header("offset")[j]
    assert np.all(cone.traces[j, (t > xo / 1000 + 0.01) & (t < xo / 250 - 0.01)] == 0)
    assert np.all(cone.traces[j, t < xo / 1000 - 0.01] == 1) and np.all(cone.traces[j, t > xo / 250 + 0.01] == 1)


# ---------------------------------------------------------------------------------------------- filters
def test_time_variant_bandpass():
    dt = 0.002
    t = np.arange(2001) * dt
    d = SeismicData((np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 40 * t))[None, :], dt)
    out = create("tv_bandpass", windows="0.0: 20,25,60,70; 4.0: 1,2,8,10")(d).traces[0]
    early, late = (t > 0.2) & (t < 0.7), (t > 3.3) & (t < 3.8)
    assert amp_at(out[early], t[early], 40) > 4 * amp_at(out[early], t[early], 5)
    assert amp_at(out[late], t[late], 5) > 4 * amp_at(out[late], t[late], 40)
    with pytest.raises(ValueError):
        create("tv_bandpass", windows="0.5 8,12,80")(d)


def test_spectral_whitening_flattens_band():
    from scipy.ndimage import uniform_filter1d
    rng = np.random.default_rng(2)
    dt = 0.002
    w, c = ricker(25.0, dt)
    refl = rng.standard_normal((6, 2000))
    tr = np.stack([np.convolve(r, w, mode="same") for r in refl])
    d = SeismicData(tr, dt)
    out = create("spectral_whitening", corners=[5, 10, 70, 90], smoothing=8.0)(d)

    def cv(x):
        f = np.fft.rfftfreq(4096, dt)
        a = uniform_filter1d(np.abs(np.fft.rfft(x, 4096, axis=1)).mean(axis=0), 30)
        band = (f > 15) & (f < 60)
        return np.std(a[band]) / np.mean(a[band])
    assert cv(out.traces) < 0.5 * cv(d.traces)
    assert np.allclose(np.sqrt(np.mean(out.traces.astype(float) ** 2, axis=1)),
                       np.sqrt(np.mean(tr ** 2, axis=1)), rtol=1e-3)


def linear_events(ntr=60, nt=500, dt=0.004, dx=10.0, noise=0.0, seed=0):
    x = np.arange(ntr) * dx
    w, c = ricker(25.0, dt)
    times = np.stack([0.4 + 0.0002 * x, 1.0 - 0.0001 * x, 1.5 + 0.00005 * x], axis=1)
    clean = events_to_traces(times, np.ones_like(times), nt, dt, w, c)
    noisy = clean + noise * np.random.default_rng(seed).standard_normal(clean.shape)
    return clean, noisy, x


def test_fx_decon_attenuates_random_noise():
    clean, noisy, x = linear_events(noise=0.4)
    d = SeismicData(noisy.astype(np.float32), 0.004, {"cdp": np.arange(1, 61), "cdpx": x})
    out = create("fx_decon", gather_key="", window=0.5, traces=30, filter_length=4, damping=1.0, fmax=70.0)(d)
    err_in = np.sqrt(np.mean((noisy - clean) ** 2))
    err_out = np.sqrt(np.mean((out.traces - clean) ** 2))
    assert err_out < 0.75 * err_in
    assert np.corrcoef(out.traces.ravel(), clean.ravel())[0, 1] > np.corrcoef(noisy.ravel(), clean.ravel())[0, 1]


def test_median_filter_removes_spikes():
    dt = 0.004
    tr = np.zeros((20, 300))
    tr[:, 150] = 1.0                                      # a flat event
    tr[7, 60] = 25.0                                      # spikes
    tr[12, 220] = -30.0
    out = create("median_filter", direction="across traces", length=5, output="filter")(SeismicData(tr, dt))
    assert abs(out.traces[7, 60]) < 1.0 and abs(out.traces[12, 220]) < 1.0
    assert np.allclose(out.traces[:, 150], 1.0)
    sub = create("median_filter", direction="across traces", length=5, output="subtract")(SeismicData(tr, dt))
    assert np.allclose(sub.traces[:, 150], 0.0) and abs(sub.traces[7, 60]) > 20.0


def test_taup_filter_removes_slow_linear_noise():
    dt, dx, ntr, nt = 0.004, 5.0, 64, 400
    x = np.arange(ntr) * dx
    w, c = ricker(25.0, dt)
    slow = events_to_traces((0.1 + x / 400.0)[:, None], np.ones((ntr, 1)), nt, dt, w, c)
    flat = events_to_traces(np.full((ntr, 1), 0.8), np.ones((ntr, 1)), nt, dt, w, c)
    d = SeismicData((slow + flat).astype(np.float32), dt, {"fldr": 1.0, "offset": x})
    out = create("taup_filter", gather_key="fldr", v_cut=1000.0, v_min=200.0, np=241, damping=0.01)(d).traces
    inner = slice(6, ntr - 6)
    assert np.sum((out - flat)[inner] ** 2) < 0.2 * np.sum(slow[inner] ** 2)
    fw = slice(int(0.75 / dt), int(0.85 / dt))
    assert np.corrcoef(out[inner, fw].ravel(), flat[inner, fw].ravel())[0, 1] > 0.9


def test_trace_mix_smooths_noise():
    rng = np.random.default_rng(4)
    tr = rng.standard_normal((50, 400))
    tr[:, 200] += 10.0
    out = create("trace_mix", weights=[1, 2, 1])(SeismicData(tr, 0.004)).traces
    assert np.std(out[5:-5, :150]) < 0.75 * np.std(tr[5:-5, :150])
    assert np.allclose(out[5:-5, 200].mean(), tr[5:-5, 200].mean(), atol=0.5)


# ---------------------------------------------------------------------------------------------- refraction statics
def refraction_case():
    v1, v2 = 600.0, 2200.0
    st = np.arange(0.0, 480.0, 10.0)
    z = 8.0 + 4.0 * np.sin(2 * np.pi * st / 300.0)
    elev = 100.0 + 3.0 * np.cos(2 * np.pi * st / 400.0)
    delay = z * np.sqrt(v2 ** 2 - v1 ** 2) / (v1 * v2)
    rows = [(k + 1, j + 1, s, g) for k, s in enumerate(st[::4]) for j, g in enumerate(st)]
    fldr, tracf, sx, gx = (np.array(c, float) for c in zip(*rows))
    off = gx - sx
    d = SeismicData(np.zeros((fldr.size, 50), np.float32), 0.002,
                    {"fldr": fldr, "tracf": tracf, "sx": sx, "gx": gx, "offset": off,
                     "selev": np.interp(sx, st, elev), "gelev": np.interp(gx, st, elev)})
    ps = PickSet("fldr", "offset", "fb")
    ps.new_phase("direct")
    ps.new_phase("refractor 1")
    for i in range(fldr.size):
        x = abs(off[i])
        if 0 < x <= 20:
            ps.add(fldr[i], "direct", off[i], x / v1)
        if x >= 40:
            ps.add(fldr[i], "refractor 1", off[i],
                   np.interp(sx[i], st, delay) + np.interp(gx[i], st, delay) + x / v2)
    return d, ps, st, z, elev, v1, v2


def test_refraction_statics_delay_time_inversion():
    d, ps, st, z, elev, v1, v2 = refraction_case()
    ctx = Context()
    ctx.picks["fb"] = ps
    out = create("refraction_statics", pickset="fb", phase="refractor 1", datum=80.0, smoothing=0.02)(d, ctx)
    res = ctx.results["refraction_statics"]
    assert abs(res["v2"] - v2) / v2 < 0.01 and abs(res["v1"] - v1) / v1 < 0.02
    th = np.interp(st, res["x"], res["thickness"])
    assert np.corrcoef(th, z)[0, 1] > 0.95 and np.mean(np.abs(th - z)) < 1.0
    want = -(z / v1 + (elev - z - 80.0) / v2) * 1000.0
    assert np.max(np.abs(np.interp(st, res["x"], res["static_ms"]) - want)) < 1.0
    assert np.allclose(out.header("gstat"), np.interp(d.header("gx"), res["x"], res["static_ms"]), atol=1e-6)
    assert res["rms_ms"] < 0.5


# ---------------------------------------------------------------------------------------------- surface consistent
def test_sc_amplitude_recovers_source_receiver_factors(marine):
    rng = np.random.default_rng(8)
    shots = np.unique(marine.header("fldr"))
    recs = np.unique(np.round(marine.header("gx"), 2))
    s_db = dict(zip(shots, rng.uniform(-6, 6, shots.size)))
    r_db = dict(zip(recs, rng.uniform(-6, 6, recs.size)))
    fac = np.array([10 ** ((s_db[f] + r_db[round(g, 2)]) / 20) for f, g in zip(marine.header("fldr"),
                                                                                 marine.header("gx"))])
    d = marine.copy((marine.traces * fac[:, None]).astype(np.float32))
    ctx = Context()
    out = create("sc_amplitude", tmin=0.0, offset_class=100.0)(d, ctx)
    got = ctx.results["sc_amplitude"]["source"]
    sx_of = {f: s for f, s in zip(d.header("fldr"), np.round(d.header("sx"), 2))}
    est = np.array([got[sx_of[f]] for f in shots])
    assert np.corrcoef(est, [s_db[f] for f in shots])[0, 1] > 0.8

    def shot_spread(x):
        from seisdesk.core.qc import mean_by, trace_rms
        _, m = mean_by(np.log(trace_rms(x)), x.header("fldr"))
        return np.std(m)
    assert shot_spread(out) < 0.4 * shot_spread(d)


def test_sc_decon_compresses_variable_wavelets():
    rng = np.random.default_rng(3)
    nt, dt = 1000, 0.004
    rows, refl, traces = [], [], []
    a = rng.uniform(0.3, 0.8, 12)
    b = rng.uniform(0.2, 0.7, 24)
    for k in range(12):
        for j in range(24):
            r = rng.standard_normal(nt)
            w = np.convolve([1.0, a[k]], [1.0, b[j]])
            traces.append(np.convolve(r, w)[:nt])
            refl.append(r)
            rows.append((k + 1, j + 1, 20.0 * k, 10.0 * j))
    fldr, tracf, sx, gx = (np.array(c, float) for c in zip(*rows))
    d = SeismicData(np.array(traces, np.float32), dt, {"fldr": fldr, "tracf": tracf, "sx": sx, "gx": gx,
                                                       "offset": gx - sx})
    out = create("sc_decon", tmin=0.0, tmax=3.9, offset_class=50.0, smoothing=2.0, prewhitening=0.1)(d)
    refl = np.array(refl)
    cc_in = np.mean([np.corrcoef(d.traces[i, 50:-50], refl[i, 50:-50])[0, 1] for i in range(d.ntr)])
    cc_out = np.mean([np.corrcoef(out.traces[i, 50:-50], refl[i, 50:-50])[0, 1] for i in range(d.ntr)])
    assert cc_out > 0.9 and cc_out > cc_in + 0.05


def test_exp_gain():
    d = SeismicData(np.ones((1, 501)), 0.002)
    out = create("exp_gain", db_per_s=6.0)(d)
    assert np.isclose(out.traces[0, 500], 10 ** (6.0 / 20.0), rtol=1e-4)


# ---------------------------------------------------------------------------------------------- velocity, stack
def test_cvg_panels_and_smooth_velocity(marine_cmp):
    out = create("cvg_panels", npanels=5, vmin=1600.0, vmax=2600.0)(marine_cmp, Context())
    c = float(np.unique(marine_cmp.header("cdp"))[np.unique(marine_cmp.header("cdp")).size // 2])
    n = int((marine_cmp.header("cdp") == c).sum())
    assert out.ntr == 5 * (n + 2) and set(np.unique(out.header("cvg_v"))) >= {1600.0, 2600.0}
    rng = np.random.default_rng(1)
    vp = VelocityPicks()
    cdps = np.unique(marine_cmp.header("cdp"))
    for cc in cdps[::5]:
        for t in (0.0, 1.0, 2.0, 3.0):
            vp.add(cc, t, 1500.0 + 450.0 * t + rng.uniform(-100, 100))
    ctx = Context(velocity=vp)

    def roughness(f):
        return np.std((f - np.convolve(f, np.ones(21) / 21, mode="same"))[15:-15])
    before = roughness(vp.field(cdps, [1.0])[:, 0])
    create("smooth_velocity", cdp_step=20, time_step=0.5, smooth_cdp=30.0, smooth_time=0.2)(marine_cmp, ctx)
    assert roughness(ctx.velocity.field(cdps, [1.0])[:, 0]) < 0.5 * before
    assert ctx.velocity.cdps.size == len(range(0, cdps.size, 20)) + (0 if (cdps.size - 1) % 20 == 0 else 1)


def test_median_and_diversity_stack_resist_bursts(marine_cmp):
    vp = VelocityPicks()
    for t, v in ((0.0, 1500.0), (0.6, 1700.0), (1.1, 2100.0), (1.6, 2500.0), (2.2, 2900.0), (3.0, 3200.0)):
        vp.add(1.0, t, v)
    ctx = Context(velocity=vp)
    nm = create("nmo", stretch_limit=0.6)(marine_cmp, ctx)
    c = float(np.unique(nm.header("cdp"))[np.unique(nm.header("cdp")).size // 2])
    idx = np.flatnonzero(nm.header("cdp") == c)
    tr = nm.traces.copy()
    tr[idx[3], 300:340] += 40.0
    noisy = nm.copy(tr)
    clean = create("stack")(nm, ctx)
    j = int(np.flatnonzero(clean.header("cdp") == c)[0])
    errs = {}
    for method in ("live", "median", "diversity"):
        st = create("stack", normalisation=method)(noisy, ctx)
        errs[method] = np.abs(st.traces[j, 300:340] - clean.traces[j, 300:340]).max()
    assert errs["median"] < 0.3 * errs["live"] and errs["diversity"] < 0.3 * errs["live"]
