import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.geometry import geometry_2d, geometry_3d, header_formula, streamer_geometry
from seisdesk.core.headers import choose_scalar, scale_factor, su_header_dtype
from seisdesk.core.picks import PickSet, refraction_layers
from seisdesk.core.velocity import DepthModel, VelocityPicks, dix, time_to_depth_axis, vint_to_vrms


def test_dataset_basics():
    d = SeismicData(np.random.default_rng(0).standard_normal((10, 50)), 0.002, {"fldr": np.repeat([1, 2], 5),
                                                                               "offset": np.arange(10)})
    assert d.ntr == 10 and d.nt == 50
    assert np.isclose(d.tmax, 49 * 0.002)
    assert list(d.gather_values("fldr")) == [1.0, 2.0]
    g = d.gather("fldr", 2)
    assert g.ntr == 5 and np.all(g.header("fldr") == 2)
    w = d.time_window(0.01, 0.05)
    assert np.isclose(w.t0, 0.01) and w.nt == 21
    s = d.sort(["fldr", "offset"])
    assert np.all(np.diff(s.header("fldr")) >= 0)
    with pytest.raises(ValueError):
        SeismicData(np.zeros((3, 4)), 0.0)


def test_iter_gathers_cover_all_traces(marine):
    seen = np.concatenate([idx for _, idx in marine.iter_gathers("cdp", None)])
    assert np.array_equal(np.sort(seen), np.arange(marine.ntr))


def test_scalars():
    assert np.allclose(scale_factor(np.array([0, 10, -100])), [1.0, 10.0, 0.01])
    assert choose_scalar(np.array([123456.78])) == -100
    assert su_header_dtype().itemsize == 240


def test_geometry_2d_marine(marine):
    off = marine.header("offset")
    assert np.all(off < 0)                                  # receivers behind the source
    mid = 0.5 * (marine.header("sx") + marine.header("gx"))
    cdp = marine.header("cdp")
    order = np.argsort(mid)
    assert np.all(np.diff(cdp[order]) >= 0)                 # CMP number grows with midpoint
    assert np.allclose(np.diff(np.unique(marine.header("cdpx"))), 12.5)


def test_geometry_3d_rotated_grid():
    rng = np.random.default_rng(3)
    a, b = rng.uniform(0, 1000, 400), rng.uniform(0, 500, 400)
    ang = np.deg2rad(30.0)
    x = a * np.cos(ang) - b * np.sin(ang)
    y = a * np.sin(ang) + b * np.cos(ang)
    d = SeismicData(np.zeros((400, 10)), 0.004, {"sx": x, "gx": x, "sy": y, "gy": y})
    out = geometry_3d(d, 25.0, 25.0, origin_x=0.0, origin_y=0.0, azimuth_deg=30.0)
    assert np.allclose(out.header("xline"), np.floor(a / 25.0 + 0.5) + 1)
    assert np.allclose(out.header("iline"), np.floor(b / 25.0 + 0.5) + 1)


def test_streamer_geometry_and_formula():
    d = SeismicData(np.zeros((6, 10)), 0.004, {"fldr": [5, 5, 5, 6, 6, 6], "tracf": [1, 2, 3, 1, 2, 3]})
    out = streamer_geometry(d, first_shot_x=1000, shot_interval=50, near_offset=100, group_interval=25)
    assert np.allclose(out.header("sx"), [1000, 1000, 1000, 1050, 1050, 1050])
    assert np.allclose(out.header("offset"), [-100, -125, -150] * 2)
    out = geometry_2d(out, 12.5)
    assert out.has_header("cdp")
    out = header_formula(out, "ep", 0.0, 1.0, "fldr", 0.0, "", 1.0)
    assert np.allclose(out.header("ep"), out.header("fldr"))


def test_velocity_picks_interpolation_and_files(tmp_path):
    vp = VelocityPicks()
    vp.add(100, 0.0, 1500)
    vp.add(100, 2.0, 2500)
    vp.add(200, 0.0, 1700)
    vp.add(200, 2.0, 2900, eta=0.1)
    assert np.isclose(vp.function(100, [1.0])[0], 2000.0)
    assert np.isclose(vp.function(150, [2.0])[0], 2700.0)
    assert np.isclose(vp.function(500, [0.0])[0], 1700.0)           # constant beyond the last CMP
    assert np.isclose(vp.eta_field([200], [2.0])[0, 0], 0.1)
    path = tmp_path / "v.txt"
    vp.save(str(path))
    back = VelocityPicks.load(str(path))
    assert len(back) == 4 and np.allclose(back.picks_at(200), vp.picks_at(200), atol=1e-3)
    assert VelocityPicks.from_json(vp.to_json()).picks_at(100).shape == (2, 3)


def test_dix_round_trip():
    t = np.linspace(0, 3, 301)
    vint = np.where(t < 1.0, 1800.0, np.where(t < 2.0, 2500.0, 3200.0))
    vrms = vint_to_vrms(t, vint)
    back = dix(t, vrms)
    inner = (np.abs(t - 1.0) > 0.03) & (np.abs(t - 2.0) > 0.03) & (t > 0.05)
    assert np.allclose(back[inner], vint[inner], rtol=0.02)
    z = time_to_depth_axis(t, np.full(t.size, 2000.0))
    assert np.isclose(z[-1], 3000.0, rtol=1e-6)


def test_depth_model_from_picks_reversed_positions():
    vp = VelocityPicks.constant(2000.0)
    m = DepthModel.from_picks(vp, {1.0: 500.0, 2.0: 400.0, 3.0: 300.0}, [1, 2, 3], 11, 10.0, 3, 100.0, 300.0)
    assert np.allclose(m.values, 2000.0)
    assert np.isclose(m.at(np.array([50.0]), np.array([350.0]))[0], 2000.0)


def test_pickset_phases_fit_and_layers(tmp_path):
    v1, v2, v3, h1, h2 = 800.0, 2000.0, 3500.0, 15.0, 40.0
    t1 = 2 * h1 * np.sqrt(v2 ** 2 - v1 ** 2) / (v1 * v2)
    t2 = 2 * h1 * np.sqrt(v3 ** 2 - v1 ** 2) / (v1 * v3) + 2 * h2 * np.sqrt(v3 ** 2 - v2 ** 2) / (v2 * v3)
    ps = PickSet("fldr", "offset", "fb")
    direct = ps.new_phase("direct")
    ps.new_phase()                                       # the break: next branch
    refr1 = ps.phase_order[-1]
    refr2 = ps.new_phase("refractor 2")
    for x in np.arange(10, 60, 10.0):
        ps.add(1, direct, x, x / v1)
        ps.add(1, direct, -x, x / v1)
    for x in np.arange(80, 300, 20.0):
        ps.add(1, refr1, x, t1 + x / v2)
    for x in np.arange(350, 900, 50.0):
        ps.add(1, refr2, x, t2 + x / v3)
    fits = [ps.fit_line(1, ph) for ph in ps.phase_order]
    assert np.allclose([f["velocity"] for f in fits], [v1, v2, v3], rtol=1e-6)
    layers = refraction_layers(fits)
    assert np.isclose(layers[0]["thickness"], h1, rtol=1e-6)
    assert np.isclose(layers[1]["thickness"], h2, rtol=1e-6)
    assert np.isclose(layers[1]["depth_to_base"], h1 + h2, rtol=1e-6)
    assert ps.remove_nearest(1, 10.0, 10.0 / v1, direct, max_distance=1.0)
    assert not ps.remove_nearest(1, 5000.0, 9.0, direct, max_distance=1.0)
    ps.rename_phase(refr1, "refractor 1")
    path = tmp_path / "p.json"
    ps.save(str(path))
    back = PickSet.load(str(path))
    assert back.phase_order == ["direct", "refractor 1", "refractor 2"]
    assert back.count() == ps.count()
    mute = back.interpolate(np.array([1.0, 1.0]), np.array([-20.0, 30.0]), "direct")
    assert np.allclose(mute, [20 / v1, 30 / v1])


def test_hyperbola_fit_and_dix_layers():
    from seisdesk.core.picks import dix_layers
    # two flat layers: 1800 m/s down to 450 m, 2600 m/s for 650 m more
    v1, h1, v2, h2 = 1800.0, 450.0, 2600.0, 650.0
    t1 = 2 * h1 / v1
    t2 = t1 + 2 * h2 / v2
    vrms2 = np.sqrt((v1 ** 2 * t1 + v2 ** 2 * (t2 - t1)) / t2)
    ps = PickSet("cdp", "offset", "reflections")
    for name, t0, v in (("R1", t1, v1), ("R2", t2, vrms2)):
        ps.new_phase(name)
        for x in np.arange(-1500.0, 1501.0, 100.0):
            ps.add(1, name, x, np.sqrt(t0 ** 2 + (x / v) ** 2))
    f1, f2 = ps.fit_hyperbola(1, "R1"), ps.fit_hyperbola(1, "R2")
    assert np.isclose(f1["velocity"], v1, rtol=1e-9) and np.isclose(f1["t0"], t1, rtol=1e-9)
    assert np.isclose(f2["velocity"], vrms2, rtol=1e-9) and f2["rms"] < 1e-9
    layers = dix_layers([f2, f1])                        # any order: sorted by t0
    assert np.isclose(layers[0]["velocity"], v1) and np.isclose(layers[0]["thickness"], h1)
    assert np.isclose(layers[1]["velocity"], v2) and np.isclose(layers[1]["depth_to_base"], h1 + h2)
    with pytest.raises(ValueError):
        ps.new_phase("line")
        for x in (100.0, 200.0, 300.0):
            ps.add(1, "line", x, 0.5 - x / 1000.0)        # time falling with offset: not a reflection
        ps.fit_hyperbola(1, "line")


def test_pick_table_with_positions(tmp_path):
    from seisdesk.core.picks import pick_table, write_pick_table
    d = SeismicData(np.zeros((3, 10)), 0.001, {"fldr": [5, 5, 5], "offset": [-10.0, 0.0, 10.0],
                                               "sx": 100.0, "gx": [90.0, 100.0, 110.0], "gelev": [1.0, 2.0, 3.0]})
    ps = PickSet("fldr", "offset", "fb")
    ph = ps.new_phase("direct")
    ps.add(5, ph, -10.0, 0.0125)
    ps.add(5, ph, 10.0, 0.0126)
    rows = pick_table(ps, d)
    assert len(rows) == 2 and rows[0]["gx"] == 90.0 and rows[1]["gelev"] == 3.0 and rows[0]["time_ms"] == 12.5
    path = tmp_path / "picks.csv"
    write_pick_table(rows, str(path))
    text = path.read_text().splitlines()
    assert text[0].startswith("fldr,phase,offset,time_ms") and len(text) == 3


def test_spectral_helpers():
    from seisdesk.core.spectra import (amplitude_spectrum, autocorrelation, first_zero_crossing, spectrum_band,
                                       suggest_bandpass)
    from seisdesk.core.synthetic import ricker
    dt = 0.002
    w, _ = ricker(30.0, dt)
    tr = np.zeros((4, 1000))
    tr[:, 400:400 + w.size] = w
    f, db, _ = amplitude_spectrum(tr, dt)
    peak = f[np.argmax(db)]
    assert 25.0 < peak < 35.0                            # a Ricker peaks at its central frequency
    lo, hi = spectrum_band(f, db, -20.0)
    assert lo < peak < hi and lo > 2.0 and hi < 100.0
    c = suggest_bandpass(f, db)
    assert c[0] <= c[1] < c[2] <= c[3]
    t = np.arange(2000) * dt
    lags, acf = autocorrelation(np.sin(2 * np.pi * 10.0 * t)[None, :], dt, max_lag=0.2)
    assert np.isclose(acf[0], 1.0) and abs(first_zero_crossing(lags, acf) - 0.025) < 0.003
