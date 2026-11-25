import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.velocity import DepthModel
from seisdesk.processing import Context, create
from seisdesk.processing.signal_tools import fractional_filter, structure_smooth
from seisdesk.processing.mva import scan_depth_moveout, straight_ray


def test_fractional_derivative_gaussian_and_inverse():
    dt = .001; t = np.arange(1001)*dt - .5
    g = np.exp(-(t/.05)**2)
    derivative = fractional_filter(g[None], dt, 1)[0]
    exact = -2*t/.05**2*g
    assert np.linalg.norm(derivative-exact)/np.linalg.norm(exact) < 1e-4
    recovered = fractional_filter(derivative[None], dt, -1)[0]
    # Removed DC leaves an arbitrary integration constant.
    assert np.std((recovered-g)[200:800]) < 2e-3


def test_panel_filter_rejects_selected_spatial_mode():
    x, t = np.arange(32), np.arange(128)
    wanted = np.cos(2*np.pi*(x[:, None]/32 + 8*t[None]/128))
    noise = np.cos(2*np.pi*(12*x[:, None]/32 + 8*t[None]/128))
    d = SeismicData(wanted+noise, .004, {"gx": x*10, "fldr": 1})
    out = create("kfilter", kmax=.01, fmax=100, taper=0)(d, Context())
    assert np.sqrt(np.mean((out.traces-wanted)**2)) < 1e-6


def test_radial_wavenumber_filter_and_units():
    x, z = np.arange(32), np.arange(128)
    low = np.cos(2*np.pi*(x[:, None]/32 + 4*z[None]/128))
    high = np.cos(2*np.pi*(12*x[:, None]/32 + 48*z[None]/128))
    d = SeismicData(low+high, 10, {"gx": x*10, "fldr": 1}, domain="depth")
    out = create("kfilter", mode="radial (depth)", kmax=.01, taper=0)(d, Context())
    assert np.sqrt(np.mean((out.traces-low)**2)) < 1e-6
    d.domain = "time"
    with pytest.raises(ValueError, match="depth section"):
        create("kfilter", mode="radial (depth)")(d, Context())


def test_structure_smoothing_preserves_dip_and_fills_dead_trace():
    t = np.arange(240); x = np.arange(15)
    truth = np.exp(-.5*((t[None]-90-2*x[:, None])/4)**2)
    noisy = truth + np.random.default_rng(9).normal(0, .2, truth.shape)
    result, _ = structure_smooth(noisy, width=7, max_dip=3, ndip=13)
    gate = slice(70, 140)
    assert np.mean((result[:, gate]-truth[:, gate])**2) < .65*np.mean((noisy[:, gate]-truth[:, gate])**2)
    missing = truth.copy(); missing[7] = 0
    filled, _ = structure_smooth(missing, width=7, max_dip=3, ndip=13)
    assert np.max(filled[7]) > .9
    assert abs(np.argmax(filled[7])-104) <= 1


def test_reciprocal_pairs_preserve_components_and_fold():
    d = SeismicData(np.ones((4, 100)), .004,
                    {"sx": [0, 10, 0, 10], "gx": [10, 0, 10, 0], "trid": [12, 12, 13, 13]})
    out = create("reciprocal_sum")(d, Context())
    assert out.ntr == 2 and np.all(out.traces == 2) and np.all(out.header("nhs") == 2)
    assert len(set(out.trace_ids)) == 2


def test_mva_coefficient_sign_and_straight_ray_length():
    z = np.arange(0, 1001, 2.); offsets = np.linspace(0, 800, 17)
    a = 1.05**2-1
    peaks = np.sqrt(500**2 + a*(offsets/2)**2)
    traces = np.exp(-.5*((z[None]-peaks[:, None])/4)**2)
    found = scan_depth_moveout(traces, z, offsets, [500], np.linspace(-.3, .3, 121), window=10)
    assert found[0]["a"] == pytest.approx(a, abs=.005)
    d = SeismicData(traces, 2, {"cdp": 1, "cdpx": 500, "offset": offsets}, domain="depth")
    ctx = Context(depth_model=DepthModel.constant(2100, 501, 101, 2, 10))
    create("mva_tomography", zmin=500, zmax=500, na=121, window=10)(d, ctx)
    assert np.mean(ctx.depth_model.values) == pytest.approx(2000, rel=.005)
    row = straight_ray(ctx.depth_model, 100, 500, 300)
    assert row.sum() == pytest.approx(500, rel=1e-12)


def test_straight_ray_mva_reduces_linear_residual_in_correct_direction():
    z = np.arange(0, 1001, 2.); offsets = np.linspace(0, 800, 17)
    peaks = np.sqrt(500**2 + (1.05**2-1)*(offsets/2)**2)
    traces = np.exp(-.5*((z[None]-peaks[:, None])/4)**2)
    d = SeismicData(traces, 2, {"cdp": 1, "cdpx": 500, "offset": offsets}, domain="depth")
    ctx = Context(depth_model=DepthModel.constant(2100, 51, 51, 20, 20))
    create("mva_tomography", method="straight-ray tomography", zmin=500, zmax=500,
           na=121, window=10, smoothing=.5, damping=.05)(d, ctx)
    r = ctx.results["mva_tomography"]
    assert np.linalg.norm(r["linear_residual_after"]) < .5*np.linalg.norm(r["linear_residual_before"])
    assert ctx.depth_model.values[:26].mean() < 2100


def test_statistics_and_panel_spectrum_dialogs(qtbot):
    from seisdesk.gui.stats_dialog import TraceStatisticsDialog
    from seisdesk.gui.k1k2_dialog import WavenumberSpectrumDialog
    n, nt = 32, 128
    samples = np.cos(2*np.pi*(3*np.arange(n)[:, None]/n + 8*np.arange(nt)[None]/nt))
    d = SeismicData(samples, .004, {"gx": np.arange(n)*10})
    stats = TraceStatisticsDialog(d); qtbot.addWidget(stats)
    assert stats.result["count"].sum() == n*nt
    assert np.allclose(np.diag(stats.result["similarity"]), 1)
    spec = WavenumberSpectrumDialog(d); qtbot.addWidget(spec)
    k1, k2, amp = spec.result
    i, j = np.unravel_index(np.argmax(amp), amp.shape)
    assert abs(k1[i]) == pytest.approx(3/(n*10))
    assert abs(k2[j]) == pytest.approx(8/(nt*.004))
