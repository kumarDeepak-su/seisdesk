"""Manual screenshots of the multicomponent steps and the well-log window (m58-m61), on synthetic data."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from seisdesk.core import welllog as wl
from seisdesk.core.dataset import SeismicData

NT, DT = 601, 0.002
GRAY = "Gray (black = positive)"
SEISMIC_NAME = "stack near DEMO-1"
_CACHE: dict = {}


def three_component_shot(env, source_y: float = 150.0, noise: float = 0.01, seed: int = 4) -> SeismicData:
    """A 3C shot on 96 receivers along x (10 m apart), the source at x = 475 m, y = source_y: near-vertical P
    reflections, a P-to-S converted event (radial), an SH event (transverse) and elliptical ground roll in the
    vertical-radial plane. Inline sensors point along +x."""
    rng = np.random.default_rng(seed)
    syn = env.synthetic
    w, c = syn.ricker(25.0, DT)
    wg, cg = syn.ricker(12.0, DT)
    gx = np.arange(96) * 10.0
    sx, sy = 475.0, source_y
    dx, dy = gx - sx, -sy * np.ones(96)
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)

    def ev(times, wavelet=w, centre=c):
        return syn.events_to_traces(times[:, None], np.ones((96, 1)), NT, DT, wavelet, centre)

    zc, rc, tc = (np.zeros((96, NT)) for _ in range(3))
    for t0, v, amp in ((0.30, 1800.0, 1.0), (0.55, 2200.0, -0.7), (0.85, 2600.0, 0.6)):
        t = np.sqrt(t0 ** 2 + (r / v) ** 2)
        sin_i = np.clip(r / (v * t), 0.0, 0.95)
        e = amp * ev(t)
        zc += e * np.sqrt(1.0 - sin_i ** 2)[:, None]
        rc += e * sin_i[:, None]
    ps = 0.8 * ev(np.sqrt(0.70 ** 2 + (r / 1200.0) ** 2))
    rc += ps
    zc += 0.15 * ps
    tc += 0.6 * ev(np.sqrt(0.95 ** 2 + (r / 1100.0) ** 2))
    roll = 1.5 * ev(0.02 + r / 400.0, wg, cg) * np.exp(-r / 1500.0)[:, None]
    zc += roll
    rc += 0.7 * np.imag(hilbert(roll, axis=1))
    inline = rc * np.cos(phi)[:, None] - tc * np.sin(phi)[:, None]
    crossline = rc * np.sin(phi)[:, None] + tc * np.cos(phi)[:, None]
    scale = noise * float(np.abs(zc).max())
    traces = np.empty((3 * 96, NT), dtype=np.float32)
    for k, comp in enumerate((zc, inline, crossline)):
        traces[k::3] = comp + scale * rng.standard_normal(comp.shape)
    heads = {"fldr": 1.0, "tracf": np.repeat(np.arange(1.0, 97.0), 3), "trid": np.tile([12.0, 14.0, 13.0], 96),
             "sx": sx, "sy": sy, "gx": np.repeat(gx, 3), "gy": 0.0, "offset": np.repeat(np.sign(dx) * r, 3)}
    return SeismicData(traces, DT, heads, name="3C shot")


def _panels(env, items, name: str, size) -> None:
    """Side-by-side displays; items are (data, title, colour map, levels or None). Grey panels share one clip so their
    amplitudes compare; colour panels use the given levels."""
    host = env.QtWidgets.QWidget()
    lay = env.QtWidgets.QHBoxLayout(host)
    grey = [d for d, _t, cmap, _l in items if cmap == GRAY]
    clip = max(env.clip_value(d.traces, 99.0) for d in grey) if grey else 1.0
    for d, title, cmap, levels in items:
        p = env.SeismicPlot(host, y_label="Time", y_units="s")
        if levels is None:
            p.show_traces(d.traces, d.t0, d.dt, d.header("gx"), "Density", cmap, clip)
        else:
            p.show_traces(d.traces, d.t0, d.dt, d.header("gx"), "Density", cmap, 1.0, levels=levels)
        p.getPlotItem().setTitle(title)
        p.getPlotItem().setLabel("bottom", "receiver x (m)")
        p.reset_view()
        lay.addWidget(p)
    host.resize(*size)
    host.show()
    env.grab(host, name)
    host.close()


def _rotation(env) -> None:
    d = three_component_shot(env, source_y=150.0)
    out = env.create("hrot")(d, env.ctx)
    t_in, t_out = d.header("trid"), out.header("trid")
    _panels(env, [(d.subset(t_in == 14), "inline (x) component", GRAY, None),
                  (d.subset(t_in == 13), "crossline (y) component", GRAY, None),
                  (out.subset(t_out == 17), "radial after rotation", GRAY, None),
                  (out.subset(t_out == 16), "transverse after rotation", GRAY, None)],
            "m58_3c_rotation.png", (1600, 650))


def _polarisation(env) -> None:
    d = three_component_shot(env, source_y=0.0, seed=5)
    att = env.create("polarisation", attribute="rectilinearity", window=0.06)(d, env.ctx)
    filt = env.create("polar_filter", window=0.06, power=2.0, direction="near-vertical",
                      max_incidence=35.0)(d, env.ctx)
    _panels(env, [(d.subset(d.header("trid") == 12), "vertical component", GRAY, None),
                  (att, "rectilinearity (0 blue to 1 red)", "Rainbow", (0.0, 1.0)),
                  (filt.subset(filt.header("trid") == 12), "vertical after the polarisation filter", GRAY, None)],
            "m59_polarisation.png", (1500, 650))


def _demo_well(env):
    """A synthetic well written as a LAS file in the work folder, and a small seismic section that ties to it with a
    shift of about 12 ms; cached for both screenshots."""
    if "well" in _CACHE:
        return _CACHE["well"]
    rng = np.random.default_rng(11)
    z = np.arange(1000.0, 1500.01, 0.5)
    vp = 2300.0 + 0.6 * (z - 1000.0)
    for top, base, dv in ((1080.0, 1100.0, 400.0), (1200.0, 1240.0, 350.0), (1320.0, 1335.0, 1100.0),
                          (1420.0, 1426.0, -600.0)):
        vp = np.where((z >= top) & (z < base), vp + dv, vp)
    vp = vp + uniform_filter1d(rng.normal(0.0, 80.0, z.size), 7)
    rho = 0.31 * vp ** 0.25                                         # g/cc, Gardner
    vs = wl.mudrock_vs(vp)
    sand = (z >= 1200.0) & (z < 1240.0)
    phi = np.where(sand, 0.25, 0.08)
    gr = np.where(sand, 35.0, 95.0) + rng.normal(0.0, 5.0, z.size)
    lines = ["~Version information", " VERS.   2.0 : CWLS log ASCII standard 2.0",
             " WRAP.   NO  : one line per depth step", "~Well information", " STRT.M  1000.0 : start depth",
             " STOP.M  1500.0 : stop depth", " STEP.M  0.5 : step", " NULL.   -999.25 : null value",
             " WELL.   DEMO-1 : well name", "~Curve information", " DEPT.M : depth", " DT.US/F : P sonic",
             " DTS.US/F : S sonic", " RHOB.G/CC : bulk density", " PHIE.V/V : effective porosity",
             " GR.GAPI : gamma ray", "~A  DEPT DT DTS RHOB PHIE GR"]
    for k in range(z.size):
        lines.append(f"{z[k]:.2f} {304800.0 / vp[k]:.4f} {304800.0 / vs[k]:.4f} {rho[k]:.5f} {phi[k]:.3f} "
                     f"{gr[k]:.1f}")
    path = os.path.join(env.work, "DEMO-1.las")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    syn = wl.synthetic_seismogram(z, vp, rho * 1000.0, DT, f0=30.0, v_above=2000.0)
    times = np.arange(1001) * DT
    ntr = 41
    traces = np.empty((ntr, times.size))
    for k in range(ntr):
        lateral = 1.0 + 0.1 * np.sin(k / 6.0)
        delay = 0.012 + 0.002 * np.sin(k / 9.0)
        traces[k] = lateral * np.interp(times - delay, syn["times"], syn["synthetic"], left=0.0, right=0.0)
    noise = uniform_filter1d(rng.standard_normal(traces.shape), 5, axis=1)
    traces += 0.15 * float(np.abs(traces).max()) * noise / float(np.abs(noise).max())
    seis = SeismicData(traces.astype(np.float32), DT, {"cdp": np.arange(1.0, ntr + 1.0),
                                                        "cdpx": np.arange(ntr) * 25.0, "offset": 0.0},
                       name=SEISMIC_NAME)
    _CACHE["well"] = (path, SimpleNamespace(datasets={SEISMIC_NAME: seis}))
    return _CACHE["well"]


def _well_dialog(env):
    from seisdesk.gui.well_dialog import WellLogDialog
    path, project = _demo_well(env)
    dlg = WellLogDialog(project, env.win, path)
    dlg.set_seismic(SEISMIC_NAME, 500.0)
    return dlg


def _well_window(env) -> None:
    dlg = _well_dialog(env)
    dlg.compute()
    dlg.tabs.setCurrentIndex(0)
    env.dialog(dlg, "m60_well_log_window.png", (1350, 800))


def _gassmann(env) -> None:
    dlg = _well_dialog(env)
    dlg.zone_top.setValue(1195.0)
    dlg.zone_bottom.setValue(1245.0)
    dlg.compute_gassmann()
    dlg.tabs.setCurrentIndex(2)
    env.dialog(dlg, "m61_well_gassmann.png", (1350, 800))


def make(env) -> None:
    env.shot("m58", lambda: _rotation(env))
    env.shot("m59", lambda: _polarisation(env))
    env.shot("m60", lambda: _well_window(env))
    env.shot("m61", lambda: _gassmann(env))
