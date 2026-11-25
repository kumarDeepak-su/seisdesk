"""Tests of the version 0.3 modelling steps: model definitions, acoustic and elastic finite differences, Kirchhoff
modelling and ray-traced reflection synthetics."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.signal import hilbert

from seisdesk.core.dataset import SeismicData
from seisdesk.core.models import (GridModel, LayeredModel, elevation_surface, load_model_file, model_from_text,
                                  parse_layers, reflection_coefficient, reflectivity_grid)
from seisdesk.core.velocity import DepthModel
from seisdesk.processing import modelling
from seisdesk.processing import REGISTRY, Context, create

NEW = ("fd_acoustic", "fd_elastic", "kirchhoff_modelling", "ray_synthetics")


def geometry(sx, gx, nt=301, dt=0.002):
    """A dataset that holds only geometry: one zero trace per receiver x, sources at sx."""
    gx = np.asarray(gx, dtype=float)
    sx = np.broadcast_to(np.asarray(sx, dtype=float), gx.shape).copy()
    fldr = np.unique(sx, return_inverse=True)[1].reshape(-1) + 1.0
    heads = {"fldr": fldr, "tracf": np.arange(1, gx.size + 1, dtype=float), "sx": sx, "gx": gx, "offset": gx - sx}
    return SeismicData(np.zeros((gx.size, nt), np.float32), dt, heads, name="geometry")


def envelope_peak(trace, dt, t_expect, half_window):
    """Time of the envelope maximum within t_expect +/- half_window (insensitive to the wavelet phase)."""
    env = np.abs(hilbert(np.asarray(trace, dtype=float)))
    i0 = max(0, int(round((t_expect - half_window) / dt)))
    i1 = min(env.size, int(round((t_expect + half_window) / dt)) + 1)
    return (i0 + int(np.argmax(env[i0:i1]))) * dt


def two_layer_time(x, h1=300.0, v1=1500.0, h2=200.0, v2=2000.0):
    """Reflection time from the base of the second of two flat layers, solved for the angle in the first layer."""
    if x == 0:
        return 2.0 * (h1 / v1 + h2 / v2)

    def misfit(theta):
        s2 = v2 / v1 * np.sin(theta)
        return 2.0 * (h1 * np.tan(theta) + h2 * s2 / np.sqrt(1.0 - s2 ** 2)) - x

    theta = brentq(misfit, 0.0, np.arcsin(v1 / v2 * (1.0 - 1e-12)))
    s2 = v2 / v1 * np.sin(theta)
    return 2.0 * (h1 / (v1 * np.cos(theta)) + h2 / (v2 * np.sqrt(1.0 - s2 ** 2)))


def test_steps_registered():
    assert 0 < modelling.CFL < 0.6                          # importing the module registers the steps
    for name in NEW:
        assert name in REGISTRY and REGISTRY[name].category == "Modelling"


# ---------------------------------------------------------------------------------------------- models
def test_flat_layer_text_and_defaults():
    m = parse_layers("1500, 2000, 2600 / 300, 700")
    assert isinstance(m, LayeredModel) and len(m.layers) == 3 and m.is_flat
    assert [la.top_z[0] for la in m.layers] == [0.0, 300.0, 700.0]
    assert m.layers[1].vs == pytest.approx(2000.0 / 1.73)
    assert m.layers[1].rho == pytest.approx(310.0 * 2000.0 ** 0.25)
    full = parse_layers("1500, 2000 / 300 / 0, 1100 / 1000, 2100")
    assert full.layers[0].vs == 0.0 and full.layers[1].rho == 2100.0
    vp, vs, rho, air = full.sample(np.array([0.0, 50.0]), np.array([-10.0, 100.0, 299.0, 300.0, 500.0]))
    assert air[0].all() and not air[1:].any()
    assert np.allclose(vp[:, 0], [1500.0, 1500.0, 1500.0, 2000.0, 2000.0])
    assert np.allclose(rho[1:, 1], [1000.0, 1000.0, 2100.0, 2100.0])
    for bad in ("", "1500, 2000", "1500, 2000 / 300, 400", "1500, 2000 / -5"):
        with pytest.raises(ValueError):
            parse_layers(bad)


def test_model_file_polylines_and_pinch_out(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("# surface with topography\n"
                    "1800 - - : 0 10, 1000 -10\n"
                    "2400 1300 2200 : 0 300, 1000 100   # dipping\n"
                    "3000 - - : 200\n", encoding="utf-8")
    m = load_model_file(str(path))
    assert len(m.layers) == 3 and not m.is_flat
    assert m.surface_at(500.0) == pytest.approx(0.0)
    assert m.layers[1].vs == 1300.0 and m.layers[1].rho == 2200.0
    assert m.layers[0].vs == pytest.approx(1800.0 / 1.73)
    idx = m.layer_index(np.array([-5.0, 150.0, 250.0, 150.0]), np.array([0.0, 0.0, 0.0, 900.0]))
    assert idx.tolist() == [-1, 0, 2, 1]                       # layer 1 is pinched out at x = 0
    (tmp_path / "flat.txt").write_text("1600 : 0\n2200 : 150\n", encoding="utf-8")
    flat = model_from_text("flat.txt", work_dir=str(tmp_path))
    assert len(flat.layers) == 2 and flat.layers[1].top_at(0.0) == 150.0 and flat.is_flat


def test_reflectivity_and_coefficients():
    m = parse_layers("2000, 2500 / 300")
    x = np.arange(0.0, 101.0, 10.0)
    z = np.arange(0.0, 501.0, 5.0)
    refl, nrm_x, nrm_z = reflectivity_grid(m, x, z)
    assert np.flatnonzero(refl[:, 0]).tolist() == [60]
    z1, z2 = 2000.0 * m.layers[0].rho, 2500.0 * m.layers[1].rho
    assert refl[60, 0] == pytest.approx((z2 - z1) / (z2 + z1))
    assert np.allclose(nrm_z[60], 1.0) and np.allclose(nrm_x[60], 0.0)
    r0 = reflection_coefficient(2000.0, 1150.0, 2100.0, 2100.0, 1200.0, 2150.0)
    r_ar = reflection_coefficient(2000.0, 1150.0, 2100.0, 2100.0, 1200.0, 2150.0, p=0.0)
    assert r_ar == pytest.approx(r0, abs=2e-3)


def test_project_model_and_elevation_surface():
    dm = DepthModel.constant(2200.0, 50, 40, 10.0, 25.0)
    m = model_from_text("project model", depth_model=dm)
    assert isinstance(m, GridModel)
    vp, vs, rho, air = m.sample(np.array([0.0, 100.0]), np.array([0.0, 50.0]))
    assert np.allclose(vp, 2200.0) and np.allclose(vs, 2200.0 / 1.73) and not air.any()
    with pytest.raises(ValueError):
        model_from_text("project model", depth_model=None)
    x, zs, ref = elevation_surface([0.0, 100.0], [120.0, 110.0], [50.0, 100.0], [115.0, 110.0])
    assert ref == 120.0 and np.allclose(x, [0.0, 50.0, 100.0]) and np.allclose(zs, [0.0, 5.0, 10.0])
    topo = parse_layers("1500, 2000 / 300").with_surface(x, zs)
    assert topo.layer_index(np.array([2.0]), np.array([100.0]))[0] == -1


# ---------------------------------------------------------------------------------------------- ray tracing
def test_ray_synthetics_flat_layers_exact():
    gx = np.arange(0.0, 1001.0, 100.0)
    d = geometry(0.0, gx, nt=601, dt=0.002)
    ctx = Context()
    out = create("ray_synthetics", model="1500, 2000, 2600 / 300, 500")(d, ctx)
    res = ctx.results["ray_synthetics"]
    times = np.asarray(res["times"])
    assert res["method"] == "exact" and times.shape == (11, 2)
    t1 = 2.0 * np.sqrt(300.0 ** 2 + (gx / 2.0) ** 2) / 1500.0
    t2 = np.array([two_layer_time(x) for x in gx])
    assert np.allclose(times[:, 0], t1, atol=1e-3) and np.allclose(times[:, 1], t2, atol=1e-3)
    assert out.ntr == 11 and out.dt == 0.002 and np.allclose(out.header("gx"), gx)
    assert abs(envelope_peak(out.traces[0], 0.002, 0.4, 0.03) - 0.4) <= 0.004


def test_ray_synthetics_minimum_time_matches_exact():
    gx = np.arange(0.0, 601.0, 200.0)
    d = geometry(0.0, gx, nt=501, dt=0.002)
    ctx = Context()
    create("ray_synthetics", model="1500, 2000, 2600 / 300, 500", method="minimum time (grid)", dx=5.0)(d, ctx)
    grid_t = np.asarray(ctx.results["ray_synthetics"]["times"])
    exact_t = np.column_stack([2.0 * np.sqrt(300.0 ** 2 + (gx / 2.0) ** 2) / 1500.0,
                               [two_layer_time(x) for x in gx]])
    assert np.all(np.isfinite(grid_t)) and np.allclose(grid_t, exact_t, rtol=0.04)


# ---------------------------------------------------------------------------------------------- finite differences
def test_fd_acoustic_direct_wave_and_reflection():
    gx = np.array([100.0, 300.0, 500.0])
    d = geometry(0.0, gx, nt=401, dt=0.001)
    ctx = Context()
    out = create("fd_acoustic", model="2000, 3000 / 200", f0=25.0, dx=5.0, zmax=300.0, absorbing=20)(d, ctx)
    assert out.nt == 401 and np.all(np.isfinite(out.traces))
    assert np.abs(out.traces).max() == pytest.approx(1.0) and ctx.results["fd_acoustic"]["dx"] == 5.0
    for i in (1, 2):
        t = gx[i] / 2000.0
        assert abs(envelope_peak(out.traces[i], 0.001, t, 0.025) - t) <= 0.003
    t_refl = 2.0 * np.sqrt(200.0 ** 2 + 50.0 ** 2) / 2000.0
    assert abs(envelope_peak(out.traces[0], 0.001, t_refl, 0.025) - t_refl) <= 0.003


def test_fd_elastic_p_arrival_homogeneous():
    gx = np.array([200.0, 400.0])
    d = geometry(0.0, gx, nt=301, dt=0.001)
    out = create("fd_elastic", model="2000 / / 1150 / 2000", f0=20.0, dx=5.0, zmax=200.0, absorbing=30,
                 component="pressure", free_surface=False)(d, Context())
    assert np.all(np.isfinite(out.traces))
    for i in range(2):
        t = gx[i] / 2000.0
        assert abs(envelope_peak(out.traces[i], 0.001, t, 0.03) - t) <= 0.003


def test_free_surface_runs_stay_finite():
    d = geometry(0.0, np.array([60.0, 120.0]), nt=151, dt=0.001)
    for name, extra in (("fd_acoustic", {"boundary": "sponge"}), ("fd_elastic", {"component": "vertical"})):
        out = create(name, model="1800, 2400 / 60", f0=25.0, dx=4.0, zmax=100.0, absorbing=15, free_surface=True,
                     **extra)(d, Context())
        assert np.all(np.isfinite(out.traces)) and np.abs(out.traces).max() > 0


# ---------------------------------------------------------------------------------------------- Kirchhoff
def test_kirchhoff_modelling_flat_reflector():
    d = geometry(500.0, np.array([500.0, 900.0]), nt=301, dt=0.002)
    ctx = Context()
    out = create("kirchhoff_modelling", model="2000, 2500 / 300", f0=25.0, dx=5.0, aperture=1000.0)(d, ctx)
    assert np.all(np.isfinite(out.traces)) and ctx.results["kirchhoff_modelling"]["reflector_points"] > 100
    tp = envelope_peak(out.traces[0], 0.002, 0.3, 0.04)
    assert abs(tp - 0.3) <= 0.006 and out.traces[0, int(round(tp / 0.002))] > 0
    t1 = 2.0 * np.sqrt(300.0 ** 2 + 200.0 ** 2) / 2000.0
    assert abs(envelope_peak(out.traces[1], 0.002, t1, 0.04) - t1) <= max(0.006, 0.02 * t1)


def test_modelling_needs_geometry_and_a_layered_model_for_rays():
    d = SeismicData(np.zeros((3, 50), np.float32), 0.002, {"fldr": 1.0})
    with pytest.raises(ValueError):
        create("ray_synthetics")(d, Context())
    ctx = Context(depth_model=DepthModel.constant(2000.0, 20, 20, 10.0, 10.0))
    with pytest.raises(ValueError):
        create("ray_synthetics", model="project model")(geometry(0.0, [50.0, 100.0]), ctx)
