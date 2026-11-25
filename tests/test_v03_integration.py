"""Checks across the version 0.3 groups: Kirchhoff migrations keep a zero-phase wavelet zero phase (the half
derivative must be the anti-causal one), and the new menus list the new steps."""

import numpy as np
import pytest
from scipy.signal import hilbert

from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.core.velocity import VelocityPicks
from seisdesk.processing import Context, REGISTRY, create


def phase_at_peak(trace) -> float:
    """Instantaneous phase (degrees) at the envelope peak: 0 for a zero-phase wavelet of positive polarity."""
    a = hilbert(np.asarray(trace, dtype=np.float64))
    return float(np.degrees(np.angle(a[int(np.argmax(np.abs(a)))])))


def flat_section(nx=61, nt=400, dt=0.004, t0=0.8, dx=12.5):
    w, c = ricker(25.0, dt)
    tr = events_to_traces(np.full((nx, 1), t0), np.ones((nx, 1)), nt, dt, w, c)
    x = np.arange(nx) * dx
    heads = {"cdp": np.arange(1, nx + 1), "cdpx": x, "sx": x, "gx": x, "offset": 0.0}
    return SeismicData(tr.astype(np.float32), dt, heads)


def test_reference_wavelet_is_zero_phase():
    d = flat_section()
    assert abs(phase_at_peak(d.traces[30])) < 5.0


def test_kirchhoff_time_migration_keeps_zero_phase():
    d = flat_section()
    out = create("kirchhoff_pstm", velocity="2000", aperture=400.0)(d, Context())
    assert abs(phase_at_peak(out.traces[30])) < 25.0


def test_kirchhoff_depth_migration_keeps_zero_phase():
    d = flat_section()
    out = create("kirchhoff_psdm", velocity="2000", dz=5.0, zmax=1200.0, grid_dx=25.0, aperture=400.0,
                 smooth=0.0)(d, Context())
    # shortest-path traveltimes are slightly too long between the stencil directions (about 1 %, a few ms here), which
    # skews the summed wavelet; a wrong half-derivative would give about 135 degrees
    assert abs(phase_at_peak(out.traces[out.ntr // 2])) < 60.0


def test_3d_kirchhoff_migration_keeps_zero_phase():
    n, nt, dt, dx = 15, 120, 0.004, 25.0
    il, xl = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    il, xl = il.ravel().astype(float), xl.ravel().astype(float)
    w, c = ricker(25.0, dt)
    tr = events_to_traces(np.full((il.size, 1), 0.3), np.ones((il.size, 1)), nt, dt, w, c)
    heads = {"iline": il, "xline": xl, "cdp": (il - 1) * n + xl, "cdpx": (xl - 1) * dx, "cdpy": (il - 1) * dx}
    cube = SeismicData(tr.astype(np.float32), dt, heads)
    cube.meta["geometry"] = {"type": "3D", "inline_bin": dx, "xline_bin": dx}
    ctx = Context()
    ctx.velocity = VelocityPicks.constant(2000.0)
    out = create("kirchhoff_pstm_3d", velocity="project", aperture=200.0)(cube, ctx)
    centre = int(np.flatnonzero((il == 8) & (xl == 8))[0])
    assert abs(phase_at_peak(out.traces[centre])) < 30.0


def test_new_menus_list_the_new_steps(qtbot, monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from seisdesk.gui import main_window
    monkeypatch.setattr(main_window, "default_work_dir", lambda: str(tmp_path))
    monkeypatch.setattr(main_window.MainWindow, "_ask_save_changes", lambda self: True)
    w = main_window.MainWindow()
    qtbot.addWidget(w)
    texts = set()

    def walk(menu):
        for a in menu.actions():
            if a.menu() is not None:
                walk(a.menu())
            else:
                texts.add(a.text())

    for a in w.menuBar().actions():
        if a.menu() is not None:
            walk(a.menu())
    for name in ("phase_rotation", "vibro_correlation", "fx_interpolation", "datum_kirchhoff", "fd_migration",
                 "wave_psdm", "phase_shift_vti", "kirchhoff_pstm_3d", "attributes", "hrot", "fd_acoustic"):
        assert REGISTRY[name].label + "..." in texts, name
    assert "MASW dispersion analysis..." in texts and "Well logs and synthetic seismogram..." in texts
