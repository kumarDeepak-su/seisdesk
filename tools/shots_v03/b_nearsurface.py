"""Manual screenshots of the version 0.3 near-surface tools: reduced time, refraction convolution section,
wave-equation datuming and the MASW window (m38-m41)."""

from __future__ import annotations

import numpy as np

from seisdesk.core.dataset import SeismicData
from seisdesk.core.masw import surface_wave_record
from seisdesk.core.synthetic import causal_wavelet, events_to_traces, ricker

F_C = np.array([4.0, 10.0, 20.0, 30.0, 50.0])
V_C = np.array([450.0, 330.0, 240.0, 210.0, 190.0])


def make(env):
    def reduced_time():
        shots = env.geo.gather_values("fldr")
        s = env.geo.gather("fldr", float(shots[len(shots) // 2]))
        red = env.create("lmo", velocity=2000.0)(s, env.ctx)
        env.panel([(s.time_window(0.0, 0.4), "land shot, true time", "offset"),
                   (red.time_window(0.0, 0.4), "reduced time t - |offset| / 2000 m/s", "offset")],
                  "m38_reduced_time.png", gain="AGC 0.25 s", size=(1300, 650))
    env.shot("m38", reduced_time)

    def convolution_section():
        rng = np.random.default_rng(11)
        v1, v2, dt, nt = 600.0, 2200.0, 0.001, 400
        rec = np.arange(0.0, 601.0, 5.0)
        shot_x = np.array([0.0, 300.0, 600.0])

        def time_depth(x):
            return 0.010 + 0.006 * np.sin(2 * np.pi * x / 450.0) + 0.003 * np.cos(2 * np.pi * x / 170.0)

        sx, gx = np.repeat(shot_x, rec.size), np.tile(rec, shot_x.size)
        x = np.abs(gx - sx)
        times = np.stack([x / v1, time_depth(sx) + time_depth(gx) + x / v2], axis=1)
        amps = np.stack([np.ones(x.size), np.full(x.size, 0.6)], axis=1)
        w, c = causal_wavelet(40.0, dt)
        tr = events_to_traces(times, amps, nt, dt, w, c) + 0.01 * rng.standard_normal((x.size, nt))
        heads = {"fldr": np.repeat(np.arange(1.0, shot_x.size + 1), rec.size), "sx": sx, "gx": gx, "offset": gx - sx,
                 "tracf": np.tile(np.arange(1, rec.size + 1), shot_x.size)}
        d = SeismicData(tr.astype(np.float32), dt, heads, name="refraction pairs")
        rcs = env.create("refraction_convolution", pairing="two shots", forward_ffid=1.0, reverse_ffid=3.0,
                         keep=0.04, tmax=0.08)(d, env.ctx)
        fwd = d.gather("fldr", 1.0, secondary="gx").time_window(0.0, 0.35)
        rev = d.gather("fldr", 3.0, secondary="gx").time_window(0.0, 0.35)
        env.panel([(fwd, "forward shot (x = 0 m)", "gx"), (rev, "reverse shot (x = 600 m)", "gx"),
                   (rcs, "convolution section (2 x time-depth)", "gx")],
                  "m39_refraction_convolution.png", gain="Trace normalise", size=(1500, 650))
    env.shot("m39", convolution_section)

    def datuming():
        v, dt, nt = 1500.0, 0.002, 300
        pos = np.arange(0.0, 601.0, 10.0)

        def elev(x):
            return 120.0 + 15.0 * np.sin(2 * np.pi * x / 400.0) + 8.0 * np.sin(2 * np.pi * x / 130.0 + 1.0)

        z_ref, datum = -50.0, 90.0
        sx, gx = np.repeat(pos, pos.size), np.tile(pos, pos.size)
        zs, zg = elev(sx), elev(gx)
        t = np.hypot(gx - sx, zg - (2.0 * z_ref - zs)) / v               # flat reflector, image-source travel time
        w, c = ricker(25.0, dt)
        tr = events_to_traces(t[:, None], np.ones((t.size, 1)), nt, dt, w, c)
        heads = {"fldr": np.repeat(np.arange(1.0, pos.size + 1), pos.size), "sx": sx, "gx": gx, "offset": gx - sx,
                 "selev": zs, "gelev": zg, "tracf": np.tile(np.arange(1, pos.size + 1), pos.size)}
        d = SeismicData(tr.astype(np.float32), dt, heads, name="topography line")
        stat = env.create("elevation_statics", datum=datum, replacement_velocity=v)(d, env.ctx)
        wed = env.create("datum_kirchhoff", datum=datum, velocity=v, aperture=200.0, max_angle=75.0,
                         fmax=60.0)(d, env.ctx)
        items = [(x.gather("fldr", 31.0, secondary="gx").time_window(0.1, 0.55), title, "gx")
                 for x, title in ((d, "shot at 300 m, on the topography"),
                                  (stat, "elevation statics to 90 m"),
                                  (wed, "wave-equation datuming to 90 m"))]
        env.panel(items, "m40_datuming.png", gain="None", size=(1500, 650))
    env.shot("m40", datuming)

    def masw_window():
        from seisdesk.gui.masw_dialog import MaswDialog
        rng = np.random.default_rng(12)
        off = np.arange(2.0, 97.0, 2.0)
        tr = surface_wave_record(off, 1000, 0.001, F_C, V_C, f0=18.0, t_delay=0.05)
        tr += 0.35 * surface_wave_record(off, 1000, 0.001, F_C, 1.7 * V_C, f0=30.0, t_delay=0.05)   # a higher mode
        tr += 0.02 * float(np.abs(tr).max()) * rng.standard_normal(tr.shape)
        d = SeismicData(tr.astype(np.float32), 0.001, {"fldr": 1.0, "offset": off, "tracf": np.arange(1, off.size + 1)},
                        name="surface-wave shot")
        dlg = MaswDialog(d, env.win)
        dlg.vmax.setValue(900.0)
        dlg.compute()
        dlg.auto_pick()
        env.dialog(dlg, "m41_masw.png", (1250, 680))
    env.shot("m41", masw_window)
