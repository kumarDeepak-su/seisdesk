from types import SimpleNamespace

import numpy as np
import pytest

from seisdesk.core import welltie as wt, welllog as wl, deviation as dev
from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import ricker


def test_deterministic_wavelet_and_phase_recovery():
    dt = .002; t = np.arange(1000)*dt
    rc = np.random.default_rng(5).normal(size=len(t))
    wavelet, centre = ricker(25, dt)
    wavelet = wt.rotate(wavelet, 35)
    seismic = wl.convolve_wavelet(rc, wavelet, centre)
    result = wt.deterministic_wavelet(t, rc, t, seismic, (len(wavelet)-1)*dt, prewhitening=.001)
    assert wt._corr(wavelet, result["wavelet"]) > .99
    original = wl.convolve_wavelet(rc, *ricker(25, dt))
    assert wt.constant_phase_scan(original, seismic)["angle"] == pytest.approx(35, abs=3)


def test_dynamic_warp_recovers_stretch_and_keeps_time_monotone():
    dt = .002; t = np.arange(1000)*dt
    f = wl.convolve_wavelet(np.random.default_rng(8).normal(size=t.size), *ricker(30, dt))
    shifts = .008*np.sin(np.pi*t/t[-1])
    g = wt.warp_trace(f, np.arange(t.size)+shifts/dt, t.size)
    result = wt.dynamic_warp(f, g, dt, .02, strain=.03)
    assert np.sqrt(np.mean((result["shifts"][100:-100]-shifts[100:-100])**2)) < 2*dt
    assert result["r_after"] > result["r_before"]
    assert result["max_strain"] <= .03 + 1e-6
    with pytest.raises(ValueError, match="increase"):
        wt.tie_point_shifts(t, [(1, 1.5), (2, 1.4)])


def test_minimum_curvature_matches_circle_and_vertical():
    radius = 1000
    theta = np.linspace(0, np.pi/3, 31)
    path = dev.minimum_curvature(radius*theta, np.rad2deg(theta), np.full(theta.size, 90))
    assert np.allclose(path["tvd"], radius*np.sin(theta), atol=1e-8)
    assert np.allclose(path["east"], radius*(1-np.cos(theta)), atol=1e-8)
    vertical = dev.minimum_curvature([0, 100, 300], [0, 0, 0], [20, 80, 100])
    assert np.allclose(vertical["tvd"], [0, 100, 300])


def test_zoeppritz_normal_incidence_and_small_contrast():
    args = (2500., 1400., 2200., 2525., 1410., 2210.)
    reflection = (args[3]*args[5]-args[0]*args[2])/(args[3]*args[5]+args[0]*args[2])
    assert wt.zoeppritz_pp(*args, 0) == pytest.approx(reflection, abs=1e-10)
    assert np.allclose(wt.zoeppritz_pp(*args, [0, 10, 20]), wt.aki_richards(*args, [0, 10, 20]), atol=2e-4)


def test_extended_well_dialog_exports_applied_tie(qtbot, tmp_path):
    from seisdesk.gui.well_dialog import WellLogDialog
    z = np.arange(100, 601, 1.)
    vp = 2000 + 150*np.sin(z/18); rho = 2200 + 100*np.sin(z/27)
    las = ["~Version", "VERS. 2.0 : version", "WRAP. NO", "~Well", "NULL. -999.25", "~Curve",
           "DEPT.M : depth", "VP.M/S : velocity", "RHOB.KG/M3 : density", "~A"]
    las += [f"{a} {b} {c}" for a, b, c in zip(z, vp, rho)]
    file = tmp_path/'well.las'; file.write_text('\n'.join(las))
    syn = wl.synthetic_seismogram(z, vp, rho, .002)
    times = np.arange(500)*.002
    trace = np.interp(times, syn["times"]+.02, syn["synthetic"], left=0, right=0)
    seismic = SeismicData(np.tile(trace, (11, 1)), .002, {"cdpx": np.arange(11)*10, "cdpy": 0})
    dlg = WellLogDialog(SimpleNamespace(datasets={"line": seismic}), las_path=str(file)); qtbot.addWidget(dlg)
    dlg.set_seismic("line", 50); dlg.compute()
    dlg.manual_points.setText("0.12:0.13,0.60:0.61"); dlg.manual_warp()
    assert np.allclose(dlg.tied_depth_times()-dlg.result["twt"], dlg.result["tie"]["shift"]+.01)
    exported = np.loadtxt(dlg.time_depth_csv().splitlines(), delimiter=',', skiprows=1)
    assert np.allclose(exported[:, 1], dlg.tied_depth_times(), atol=6e-6)
    assert dlg.tie_panel_data.ntr > seismic.ntr
    dlg.avo_depth.setValue(300)
    assert dlg.compute_avo()["gather"].shape[0] == 21
    deviation = tmp_path/'path.csv'; deviation.write_text('0,0,0\n700,0,0\n')
    dlg.load_deviation(str(deviation)); dlg.compute()
    assert np.allclose(dlg.result["depth"], z)
    dlg.f0.setValue(35)
    assert dlg.result is None
    dlg.compute()
    assert dlg.result is not None
