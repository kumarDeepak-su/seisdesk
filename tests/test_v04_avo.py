"""Tests of the version 0.4 AVO steps: incidence angles, angle mute and angle gathers, angle stacks, intercept and
gradient, reflection coefficients and two-layer modelling, the crossplot window."""

import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import ricker
from seisdesk.processing import Context, REGISTRY, create
from seisdesk.processing import avo

NEW = ("offset_to_angle", "angle_stack", "avo_attributes", "avo_modelling")
V = 2000.0
DT = 0.004
NT = 400
OFFSETS = np.arange(0.0, 1501.0, 50.0)

GAS = (2190.0, 820.0, 2.16, 1950.0, 1150.0, 1.95)          # shale over a low-impedance gas sand


def cmp_gather(traces, offsets=OFFSETS, cdp=1.0):
    n = len(offsets)
    heads = {"cdp": cdp, "offset": offsets, "fldr": 1.0, "tracf": np.arange(1, n + 1), "cdpx": 100.0}
    d = SeismicData(np.asarray(traces, dtype=np.float32), DT, heads)
    d.add_history("NMO correction: velocity=2000")
    return d


def test_avo_steps_registered():
    for name in NEW:
        assert name in REGISTRY


def test_incidence_angles_in_a_constant_velocity_medium():
    times = np.arange(NT) * DT
    th = avo.incidence_angles(times, OFFSETS, np.full(NT, V), np.full(NT, V))
    t0 = times[None, 25:]
    expected = np.degrees(np.arctan(OFFSETS[:, None] / (V * t0)))
    assert np.max(np.abs(th[:, 25:] - expected)) < 0.01
    assert np.all(th[0] == 0.0)


def test_angle_mute_keeps_only_the_angle_range():
    d = cmp_gather(np.ones((OFFSETS.size, NT)))
    out = create("offset_to_angle", velocity="2000", theta_min=0.0, theta_max=20.0, taper=0.0)(d, Context())
    th = avo.incidence_angles(d.times, OFFSETS, np.full(NT, V), np.full(NT, V))
    inside = th <= 20.0 - 1e-6
    outside = th > 20.0 + 1e-6
    assert np.allclose(out.traces[inside], 1.0) and np.allclose(out.traces[outside], 0.0)


def test_angle_gathers_have_one_trace_per_bin():
    d = cmp_gather(np.ones((OFFSETS.size, NT)))
    ctx = Context()
    out = create("offset_to_angle", velocity="2000", output="angle gathers", theta_min=0.0, theta_max=30.0,
                 angle_step=10.0)(d, ctx)
    assert out.ntr == 3 and np.allclose(out.header("angle"), [5.0, 15.0, 25.0])
    it = int(round(1.0 / DT))
    assert np.allclose(out.traces[:, it], 1.0)                      # averages of ones
    assert ctx.results["offset_to_angle"]["angles"].shape == d.traces.shape


def test_angle_stacks_are_the_means_of_the_right_traces():
    rng = np.random.default_rng(3)
    tr = rng.uniform(0.5, 1.5, (OFFSETS.size, NT)) * rng.choice([-1.0, 1.0], (OFFSETS.size, NT))
    d = cmp_gather(tr)
    out = create("angle_stack", velocity="2000", ranges="0-10; 10-20")(d, Context())
    assert out.ntr == 2 and np.allclose(out.header("angle"), [5.0, 15.0])
    th = avo.incidence_angles(d.times, OFFSETS, np.full(NT, V), np.full(NT, V))
    for it in (int(round(0.5 / DT)), int(round(1.0 / DT))):
        for r, (lo, hi) in enumerate(((0.0, 10.0), (10.0, 20.0))):
            m = (th[:, it] >= lo) & (th[:, it] < hi)
            assert m.any()
            assert abs(out.traces[r, it] - d.traces[m, it].mean()) < 1e-5


def shuey_gather(events, three=None):
    """Flat events (t0, A, B) whose amplitude at every sample follows A + B sin^2 of that sample's angle."""
    times = np.arange(NT) * DT
    th = np.deg2rad(np.nan_to_num(avo.incidence_angles(times, OFFSETS, np.full(NT, V), np.full(NT, V))))
    w, c = ricker(25.0, DT)
    tr = np.zeros((OFFSETS.size, NT))
    for t0, a, b in events:
        i0 = int(round(t0 / DT))
        pulse = np.zeros(NT)
        pulse[i0 - c:i0 - c + w.size] = w
        refl = a + b * np.sin(th) ** 2
        if three:
            refl += three * np.sin(th) ** 2 * np.tan(th) ** 2
        tr += pulse[None, :] * refl
    return cmp_gather(tr)


def test_intercept_and_gradient_are_recovered():
    events = ((0.6, 0.12, -0.30), (1.0, -0.10, -0.25), (1.4, 0.05, 0.20))
    d = shuey_gather(events)
    ctx = Context()
    a = create("avo_attributes", velocity="2000", output="intercept A", theta_max=35.0)(d, ctx)
    b = create("avo_attributes", velocity="2000", output="gradient B", theta_max=35.0)(d, Context())
    assert a.ntr == 1 and ctx.results["avo_attributes"]["A"].shape == (1, NT)
    for t0, a_true, b_true in events:
        it = int(round(t0 / DT))
        assert abs(a.traces[0, it] - a_true) <= 0.05 * abs(a_true) + 1e-4, t0
        assert abs(b.traces[0, it] - b_true) <= 0.05 * abs(b_true) + 1e-4, t0
    err = create("avo_attributes", velocity="2000", output="fit error")(d, Context())
    assert float(err.traces[0, int(round(0.6 / DT))]) < 0.02


def test_three_term_fit_recovers_the_curvature():
    d = shuey_gather(((0.8, 0.1, -0.2),), three=0.15)
    out = create("avo_attributes", velocity="2000", terms="three-term", output="curvature C (three-term)",
                 theta_max=40.0)(d, Context())
    assert abs(out.traces[0, int(round(0.8 / DT))] - 0.15) < 0.02


def test_zoeppritz_normal_incidence_and_aki_richards_for_small_contrasts():
    vp1, vs1, r1, vp2, vs2, r2 = 2500.0, 1200.0, 2.2, 3000.0, 1500.0, 2.4
    r0 = complex(avo.zoeppritz_rpp(vp1, vs1, r1, vp2, vs2, r2, 0.0))
    assert abs(r0.real - (r2 * vp2 - r1 * vp1) / (r2 * vp2 + r1 * vp1)) < 1e-12 and abs(r0.imag) < 1e-12
    small = (2000.0, 1000.0, 2.0, 2100.0, 1050.0, 2.1)
    th = np.linspace(0.0, 30.0, 16)
    exact = np.real(avo.zoeppritz_rpp(*small, th))
    approx = avo.aki_richards_rpp(*small, th)
    assert np.max(np.abs(exact - approx)) < 0.005
    a, b, _ = avo.shuey_terms(*small)
    assert abs(a - approx[0]) < 1e-9


def test_gas_sand_is_class_three_and_brightens_with_offset():
    a, b, _ = avo.shuey_terms(*GAS)
    assert a < 0 and b < 0 and avo.avo_class(a, b) == "III"
    keys = ("vp1", "vs1", "rho1", "vp2", "vs2", "rho2")
    d = SeismicData(np.zeros((1, 300), np.float32), 0.002, {"offset": 0.0})
    ctx = Context()
    g = create("avo_modelling", max_offset=1000.0, offset_step=100.0, t0=0.4, tmax=0.6,
               **dict(zip(keys, GAS)))(d, ctx)
    assert ctx.results["avo_modelling"]["class"] == "III" and g.ntr == 11
    peak = np.abs(g.traces).max(axis=1)
    assert peak[-1] > peak[0] > 0 and g.traces[0, int(round(0.4 / 0.002))] < 0
    assert avo.avo_class(0.15, -0.2) == "I" and avo.avo_class(-0.2, 0.1) == "IV" and avo.avo_class(0.0, -0.3) == "II"


def test_crossplot_dialog_fits_the_background_trend(qtbot):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from seisdesk.gui import common
    from seisdesk.gui.avo_dialog import AvoCrossplotDialog
    common.HEADLESS = True
    rng = np.random.default_rng(5)
    a = rng.normal(0.0, 0.1, (40, 200))
    b = -2.0 * a + rng.normal(0.0, 0.01, a.shape)
    dlg = AvoCrossplotDialog(a, b, None, dt=0.004)
    qtbot.addWidget(dlg)
    res = dlg.compute()
    assert res["n"] == a.size and abs(res["slope"] + 2.0) < 0.1 and res["r"] < -0.95
    dlg.tmax.setValue(0.4)
    dlg.refresh()
    assert dlg.result["n"] == 40 * 101
