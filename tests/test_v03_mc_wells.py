"""Tests of the version 0.3 multicomponent steps (horizontal and Alford rotation, polarisation analysis and filter)
and the well-log tools (LAS reading, synthetic seismogram, Backus/Thomsen, Gassmann, well tie, well-log window)."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from seisdesk.core import welllog as wl
from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.processing import REGISTRY, Context, create
from seisdesk.processing import multicomponent as mc

HAVE_QT = all(importlib.util.find_spec(m) is not None for m in ("PySide6", "pyqtgraph", "pytestqt"))

NEW = ("hrot", "alford", "polarisation", "polar_filter")
DT = 0.002
NT = 300


def pulse(t0, f0=25.0, amp=1.0):
    w, c = ricker(f0, DT)
    return events_to_traces(np.array([[t0]]), np.array([[amp]]), NT, DT, w, c)[0]


def test_new_steps_registered():
    for name in NEW:
        assert name in REGISTRY


# ---------------------------------------------------------------------------------------------- multicomponent
def test_hrot_puts_radial_motion_on_the_radial_component():
    sensor = 20.0
    angles = np.deg2rad([10.0, 75.0, 140.0, 200.0, 260.0, 330.0])
    radius = [120.0, 250.0, 180.0, 300.0, 90.0, 210.0]
    s = pulse(0.2)
    rows = []
    heads = {k: [] for k in ("fldr", "tracf", "trid", "sx", "sy", "gx", "gy")}
    for k, (phi, r) in enumerate(zip(angles, radius)):
        alpha = phi - np.deg2rad(sensor)
        for code, trace in ((12, 0.3 * s), (14, s * np.cos(alpha)), (13, s * np.sin(alpha))):
            rows.append(trace)
            for key, val in (("fldr", 1.0), ("tracf", k + 1.0), ("trid", code), ("sx", 0.0), ("sy", 0.0),
                             ("gx", r * np.cos(phi)), ("gy", r * np.sin(phi))):
                heads[key].append(val)
    d = SeismicData(np.array(rows, np.float32), DT, heads)
    out = create("hrot", sensor_azimuth=sensor)(d, Context())
    trid = out.header("trid")
    radial, transverse = out.traces[trid == 17], out.traces[trid == 16]
    assert radial.shape[0] == 6 and transverse.shape[0] == 6
    assert np.allclose(radial, s[None, :], atol=1e-4 * np.abs(s).max())
    assert np.sum(transverse.astype(np.float64) ** 2) < 1e-8 * np.sum(radial.astype(np.float64) ** 2)
    assert np.allclose(out.traces[trid == 12], d.traces[d.header("trid") == 12])
    fixed = create("hrot", mode="fixed angle", angle=90.0)(d, Context())
    ft = fixed.header("trid")
    assert np.allclose(fixed.traces[ft == 17], d.traces[d.header("trid") == 13], atol=1e-6)


def test_alford_recovers_the_fast_direction():
    theta0 = np.deg2rad(30.0)
    f = pulse(0.20)
    g = 0.8 * pulse(0.23)
    c, s = np.cos(theta0), np.sin(theta0)
    comps = ((1, c * c * f + s * s * g), (2, c * s * (f - g)), (3, c * s * (f - g)), (4, s * s * f + c * c * g))
    rows, heads = [], {"comp": [], "sx": [], "gx": []}
    for station in range(3):
        for code, tr in comps:
            rows.append(tr)
            heads["comp"].append(code)
            heads["sx"].append(0.0)
            heads["gx"].append(100.0 * (station + 1))
    d = SeismicData(np.array(rows, np.float32), DT, heads)
    ctx = Context()
    out = create("alford")(d, ctx)
    assert np.all(np.abs(out.header("alford_az") - 30.0) < 1.0)
    comp = out.header("comp")
    xx, xy, yx, yy = (out.traces[comp == k].astype(np.float64) for k in (1, 2, 3, 4))
    assert np.sum(xy ** 2) + np.sum(yx ** 2) < 1e-6 * (np.sum(xx ** 2) + np.sum(yy ** 2))
    assert np.allclose(xx[0], f, atol=1e-3) and np.allclose(yy[0], g, atol=1e-3)
    assert ctx.results["alford"]["offdiag_after"] < 1e-6 < ctx.results["alford"]["offdiag_before"]


def test_polarisation_of_linear_motion():
    az, inc = np.deg2rad(30.0), np.deg2rad(40.0)
    s = pulse(0.25)
    z, x, y = s * np.cos(inc), s * np.sin(inc) * np.cos(az), s * np.sin(inc) * np.sin(az)
    att = mc.polarisation_attributes(z, x, y, 21)
    k = int(np.argmax(np.abs(s)))
    assert att["rectilinearity"][k] > 0.99
    assert abs(att["azimuth"][k] - 30.0) < 1.0 and abs(att["incidence"][k] - 40.0) < 1.0
    d = SeismicData(np.array([z, x, y], np.float32), DT, {"fldr": 1.0, "tracf": 1.0, "trid": [12.0, 14.0, 13.0]})
    out = create("polarisation", attribute="incidence", window=0.04)(d, Context())
    assert out.ntr == 1 and abs(out.traces[0, k] - 40.0) < 1.0


def test_polar_filter_passes_linear_and_attenuates_elliptical():
    p = pulse(0.15)
    t = np.arange(NT) * DT
    env = np.exp(-0.5 * ((t - 0.45) / 0.05) ** 2)
    phase = 2.0 * np.pi * 15.0 * (t - 0.45)
    z = p + env * np.cos(phase)                  # P nearly vertical; ground roll circular in the z-inline plane
    x = 0.05 * p + env * np.sin(phase)
    y = np.zeros(NT)
    d = SeismicData(np.array([z, x, y], np.float32), DT, {"fldr": 1.0, "tracf": 1.0, "trid": [12.0, 14.0, 13.0]})
    out = create("polar_filter", window=0.066, power=4.0)(d, Context())
    zp = slice(int(0.10 / DT), int(0.20 / DT))
    ze = slice(int(0.38 / DT), int(0.52 / DT))
    assert np.abs(out.traces[0, zp]).max() > 0.7 * np.abs(z[zp]).max()
    assert np.abs(out.traces[0, ze]).max() < 0.3 * np.abs(z[ze]).max()


# ---------------------------------------------------------------------------------------------- well logs
LAS_TEXT = """~VERSION INFORMATION
 VERS.                  2.0 :   CWLS LOG ASCII STANDARD - VERSION 2.0
 WRAP.                   NO :   ONE LINE PER DEPTH STEP
~WELL INFORMATION
#MNEM.UNIT      DATA                 DESCRIPTION
 STRT.M        1000.0000            : START DEPTH
 STOP.M        1000.6000            : STOP DEPTH
 STEP.M           0.1500            : STEP
 NULL.         -999.2500            : NULL VALUE
 WELL.          TEST-1              : WELL
~CURVE INFORMATION
 DEPT.M                  : DEPTH
 DT  .US/F               : SONIC TRANSIT TIME
 RHOB.G/CC               : BULK DENSITY
~PARAMETER INFORMATION
 BHT .DEGC        35.0   : BOTTOM HOLE TEMPERATURE
~A  DEPTH     DT       RHOB
1000.00   100.0    2.30
1000.15   -999.25  2.31
1000.30   101.0    -999.2500
1000.45   102.0    2.33
1000.60   103.0    2.34
"""


def two_layer_las() -> str:
    """LAS text of a two-layer well: 2000 m/s, 2.0 g/cc above 1100 m; 3000 m/s, 2.4 g/cc below."""
    z = np.arange(1000.0, 1200.01, 0.5)
    lines = ["~Version", " VERS. 2.0 : version", " WRAP. NO : one line per depth", "~Well",
             " STRT.M 1000.0 : start", " STOP.M 1200.0 : stop", " STEP.M 0.5 : step", " NULL. -999.25 : null",
             " WELL. TWO-LAYER : well", "~Curve", " DEPT.M : depth", " DT.US/F : sonic", " RHOB.G/CC : density",
             "~A"]
    for depth in z:
        v, rho = (2000.0, 2.0) if depth < 1100.0 else (3000.0, 2.4)
        lines.append(f"{depth:.2f} {304800.0 / v:.4f} {rho:.3f}")
    return "\n".join(lines) + "\n"


def test_read_las(tmp_path):
    path = tmp_path / "w.las"
    path.write_text(LAS_TEXT, encoding="utf-8")
    log = wl.read_las(str(path))
    assert log.names == ["DEPT", "DT", "RHOB"] and log.depth_unit == "m" and log.name == "TEST-1"
    assert np.allclose(log.depth, [1000.0, 1000.15, 1000.3, 1000.45, 1000.6])
    assert np.isnan(log.curve("DT")[1]) and np.isnan(log.curve("RHOB")[2]) and log.curve("DT")[3] == 102.0
    assert log.units["DT"] == "US/F" and log.params["BHT"][0] == "35.0"
    assert wl.find_curve(log, "sonic") == "DT" and wl.find_curve(log, "density") == "RHOB"
    assert wl.find_curve(log, "shear sonic") is None
    assert np.isclose(wl.sonic_to_velocity(np.array([100.0]), "US/F")[0], 3048.0)
    assert np.isclose(wl.sonic_to_velocity(np.array([400.0]), "us/m")[0], 2500.0)
    assert np.isclose(wl.density_to_si(np.array([2.3]), "G/CC")[0], 2300.0)
    with pytest.raises(ValueError):
        wl.parse_las(LAS_TEXT.replace("NO :   ONE LINE", "YES :   ONE LINE"))


def test_two_layer_synthetic():
    z = np.arange(1000.0, 1200.01, 0.5)
    vp = np.where(z < 1100.0, 2000.0, 3000.0)
    rho = np.where(z < 1100.0, 2000.0, 2400.0)
    res = wl.synthetic_seismogram(z, vp, rho, 0.002, f0=30.0, v_above=2000.0)
    k = int(np.argmax(np.abs(res["synthetic"])))
    r_true = (3000.0 * 2400.0 - 2000.0 * 2000.0) / (3000.0 * 2400.0 + 2000.0 * 2000.0)
    assert abs(res["times"][k] - 1.1) <= 0.0021
    assert res["synthetic"][k] > 0 and abs(res["synthetic"][k] - r_true) < 0.05 * r_true
    assert abs(res["twt"][0] - 1.0) < 1e-9 and abs(np.sum(res["reflectivity"]) - r_true) < 1e-9


def test_checkshots_shift_the_times():
    z = np.arange(1000.0, 1100.01, 1.0)
    twt = wl.two_way_time(z, np.full(z.size, 2500.0), v_above=2000.0)
    fixed = wl.apply_checkshots(z, twt, [1000.0, 1100.0], [twt[0] + 0.010, twt[-1] + 0.020])
    assert abs(fixed[0] - twt[0] - 0.010) < 1e-9 and abs(fixed[-1] - twt[-1] - 0.020) < 1e-9


def test_backus_thomsen_parameters():
    z = np.arange(0.0, 100.0, 0.25)
    same = wl.backus_average(z, np.full(z.size, 3000.0), np.full(z.size, 1500.0), np.full(z.size, 2400.0), 10.0)
    for key in ("epsilon", "delta", "gamma"):
        assert np.nanmax(np.abs(same[key])) < 1e-9
    layer = (np.floor(z) % 2).astype(bool)
    vp = np.where(layer, 2200.0, 3000.0)
    vs = np.where(layer, 900.0, 1500.0)
    rho = np.where(layer, 2200.0, 2400.0)
    alt = wl.backus_average(z, vp, vs, rho, 10.0)
    mid = z.size // 2
    assert alt["epsilon"][mid] > 0.02 and alt["gamma"][mid] > 0.1
    assert wl.backus_length(900.0, 30.0) == pytest.approx(10.0)


def test_gassmann_brine_to_gas():
    vp2, vs2, rho2 = wl.gassmann(np.array([3000.0]), np.array([1600.0]), np.array([2300.0]), 0.25, 36.6e9, 2.8e9,
                                 1090.0, 0.04e9, 250.0)
    assert vp2[0] < 0.95 * 3000.0 and 1600.0 < vs2[0] < 1.08 * 1600.0 and rho2[0] == pytest.approx(2090.0)
    same = wl.gassmann(np.array([3000.0]), np.array([1600.0]), np.array([2300.0]), 0.0, 36.6e9, 2.8e9, 1090.0,
                       0.04e9, 250.0)
    assert same[0][0] == 3000.0 and same[1][0] == 1600.0


def test_well_tie_finds_the_shift():
    z = np.arange(1000.0, 1400.0, 0.5)
    vp = 2000.0 + 200.0 * (np.floor((z - 1000.0) / 25.0) % 4)
    rho = 2100.0 + 50.0 * (np.floor((z - 1000.0) / 40.0) % 3)
    res = wl.synthetic_seismogram(z, vp, rho, 0.002, f0=30.0)
    times = np.arange(1000) * 0.002
    seis = np.interp(times - 0.012, res["times"], res["synthetic"], left=0.0, right=0.0)
    tie = wl.well_tie(res["times"], res["synthetic"], times, seis, max_shift=0.05)
    assert tie["shift"] == pytest.approx(0.012, abs=0.0011) and tie["r"] > 0.95


@pytest.mark.skipif(not HAVE_QT, reason="needs PySide6, pyqtgraph and pytest-qt")
def test_well_dialog_synthetic_and_tie(qtbot, tmp_path):
    from seisdesk.gui import common
    from seisdesk.gui.well_dialog import WellLogDialog
    common.HEADLESS = True
    path = tmp_path / "two_layer.las"
    path.write_text(two_layer_las(), encoding="utf-8")
    z = np.arange(1000.0, 1200.01, 0.5)
    ref = wl.synthetic_seismogram(z, np.where(z < 1100.0, 2000.0, 3000.0), np.where(z < 1100.0, 2000.0, 2400.0),
                                  0.002, f0=30.0, v_above=2000.0)
    times = np.arange(800) * 0.002
    trace = np.interp(times - 0.008, ref["times"], ref["synthetic"], left=0.0, right=0.0)
    seis = SeismicData(np.tile(trace, (3, 1)).astype(np.float32), 0.002,
                       {"cdp": [1.0, 2.0, 3.0], "cdpx": [400.0, 500.0, 600.0]})
    dlg = WellLogDialog(SimpleNamespace(datasets={"line": seis}), None, str(path))
    qtbot.addWidget(dlg)
    assert dlg.sonic.currentText() == "DT" and dlg.density.currentText() == "RHOB"
    dlg.set_seismic("line", 500.0)
    assert dlg.compute()
    tie = dlg.result["tie"]
    assert abs(tie["shift"] - 0.008) < 0.0021 and tie["r"] > 0.9
    k = int(np.argmax(np.abs(dlg.result["synthetic"])))
    assert abs(dlg.result["times"][k] - 1.1) <= 0.0021
    assert dlg.synthetic_csv().startswith("time_s,") and dlg.time_depth_csv().startswith("depth_m,")
    dlg.export_synthetic(str(tmp_path / "syn.csv"))
    assert (tmp_path / "syn.csv").read_text(encoding="utf-8").count("\n") > 10
    b = dlg.compute_backus()
    assert np.nanmax(np.abs(b["epsilon"])) < 0.5
    g = dlg.compute_gassmann()
    ok = np.isfinite(g["vp"])
    assert np.all(np.isfinite(g["vp2"][ok])) and np.all(g["vp2"][ok] <= g["vp"][ok] + 1e-6)
