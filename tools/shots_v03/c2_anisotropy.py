"""Manual screenshots of the anisotropic, 3D and topographic migrations (m46-m49), on synthetic data."""

import numpy as np

from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.processing.moveout import moveout_time


def make(env):
    def colour_panels(items, name, size):
        """Side-by-side plots; items = (data, title, x values or None, axis label, colour map, levels or None)."""
        host = env.QtWidgets.QWidget()
        lay = env.QtWidgets.QHBoxLayout(host)
        for d, title, xvals, xlabel, cmap, levels in items:
            p = env.SeismicPlot(host, y_label="Time" if d.domain == "time" else "Depth",
                                y_units="s" if d.domain == "time" else "m")
            x = np.asarray(xvals, dtype=float) if xvals is not None else np.arange(1, d.ntr + 1, dtype=float)
            if levels is None:
                disp = env.display_gain(d.traces, d.times, d.dt, "AGC 0.5 s", d.domain)
                p.show_traces(disp, d.t0, d.dt, x, "Density", cmap, env.clip_value(disp, 98.5))
            else:
                p.show_traces(d.traces, d.t0, d.dt, x, "Density", cmap, 1.0, levels=levels)
            p.getPlotItem().setTitle(title)
            p.getPlotItem().setLabel("bottom", xlabel)
            p.reset_view()
            lay.addWidget(p)
        host.resize(*size)
        host.show()
        env.grab(host, name)
        host.close()

    # ------------------------------------------------------------------ m46: VTI phase-shift migration
    def vti():
        nx, nt, dt, dx, v, eta = 128, 300, 0.004, 12.5, 2200.0, 0.12
        x = np.arange(nx) * dx
        diffractors = ((400.0, 0.3), (800.0, 0.5), (1200.0, 0.7))
        times = np.stack([2.0 * moveout_time(0.5 * t0, np.abs(x - x0), v, eta) for x0, t0 in diffractors], axis=1)
        w, c = ricker(25.0, dt)
        tr = events_to_traces(times, np.ones_like(times), nt, dt, w, c)
        sec = SeismicData(tr.astype(np.float32), dt, {"cdp": np.arange(1, nx + 1), "cdpx": x}, name="VTI diffractions")
        iso = env.create("phase_shift_vti", velocity=f"{v:g}", eta_source="constant", eta=0.0)(sec)
        ani = env.create("phase_shift_vti", velocity=f"{v:g}", eta_source="constant", eta=eta)(sec)
        env.panel([(sec, "zero-offset section, VTI medium (eta 0.12)", "cdpx"),
                   (iso, "phase shift with eta = 0 (isotropic)", "cdpx"),
                   (ani, "phase shift for VTI, eta = 0.12", "cdpx")], "m46_vti_phase_shift.png", gain="None",
                  size=(1500, 650))
    env.shot("m46", vti)

    # ------------------------------------------------------------------ m47: residual-moveout scan of a marine CMP
    def rmo():
        nmo = env.create("nmo", velocity="project")(env.cmp_, env.ctx)
        cdps = np.unique(nmo.header("cdp"))
        c = float(cdps[cdps.size // 2])
        nr = 61
        scan = env.create("rmo_scan", cdps=f"{c:g}", mode="parabolic", rmin=-0.04, rmax=0.08, nr=nr)(nmo, env.ctx)
        panel = scan.subset(np.arange(nr))
        g = nmo.gather("cdp", c)
        colour_panels([(g, f"CMP {c:g} after NMO", g.header("offset"), "offset (m)", "Gray (black = positive)", None),
                       (panel, "residual moveout semblance", panel.header("rmo_r") * 1000.0,
                        "residual moveout at the far offset (ms)", "Rainbow", (0.0, 1.0))],
                      "m47_rmo_scan.png", (1300, 650))
    env.shot("m47", rmo)

    # ------------------------------------------------------------------ m48: 3D migration of a small cube
    def cube3d():
        n, nt, dt, d, v = 24, 150, 0.004, 25.0, 2500.0
        il, xl = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
        il, xl = il.ravel().astype(float), xl.ravel().astype(float)
        x, y = (xl - 1) * d, (il - 1) * d
        events = [np.sqrt(t0 ** 2 + 4.0 * ((x - xd) ** 2 + (y - 275.0) ** 2) / v ** 2)
                  for xd, t0 in ((150.0, 0.15), (300.0, 0.25), (450.0, 0.35))]
        events.append(0.45 + 0.0003 * (x - 300.0) + 0.0001 * (y - 275.0))           # a gently dipping plane
        times = np.stack(events, axis=1)
        amps = np.ones_like(times)
        amps[:, -1] = 0.5
        w, c = ricker(25.0, dt)
        tr = events_to_traces(times, amps, nt, dt, w, c)
        heads = {"iline": il, "xline": xl, "cdp": (il - 1) * n + xl, "cdpx": x, "cdpy": y}
        vol = SeismicData(tr.astype(np.float32), dt, heads, name="small 3D cube")
        vol.meta["geometry"] = {"type": "3D", "inline_bin": d, "xline_bin": d}
        ps = env.create("phase_shift_3d", velocity=f"{v:g}")(vol)
        kt = env.create("kirchhoff_pstm_3d", velocity=f"{v:g}", aperture=500.0)(vol)
        sel = [dd.gather("iline", 12.0, secondary="xline") for dd in (vol, ps, kt)]
        env.panel([(sel[0], "inline 12, unmigrated", "xline"), (sel[1], "after 3D phase-shift migration", "xline"),
                   (sel[2], "after 3D Kirchhoff time migration", "xline")], "m48_3d_migration.png", gain="None",
                  size=(1500, 600))
    env.shot("m48", cube3d)

    # ------------------------------------------------------------------ m49: depth migration from topography
    def topography():
        v, dt, nt = 2000.0, 0.004, 250

        def elev(xx):
            return 120.0 + 60.0 * np.sin(xx / 250.0)

        shots, recs = np.arange(0.0, 1001.0, 50.0), np.arange(0.0, 1001.0, 10.0)
        sx, gx = np.repeat(shots, recs.size), np.tile(recs, shots.size)
        times = np.stack([np.hypot(gx - sx, (elev(sx) - er) + (elev(gx) - er)) / v for er in (-150.0, -350.0)],
                         axis=1)
        w, c = ricker(25.0, dt)
        tr = events_to_traces(times, np.ones_like(times), nt, dt, w, c)
        cdpx = np.round(0.5 * (sx + gx) / 5.0) * 5.0
        heads = {"fldr": np.repeat(np.arange(1, shots.size + 1), recs.size), "sx": sx, "gx": gx, "offset": gx - sx,
                 "selev": elev(sx), "gelev": elev(gx), "cdp": cdpx / 5.0 + 1.0, "cdpx": cdpx}
        line = SeismicData(tr.astype(np.float32), dt, heads, name="land line over hills")
        common = dict(velocity=f"{v:g}", dz=5.0, zmax=700.0, grid_dx=10.0, aperture=600.0)
        flat = env.create("kirchhoff_psdm", surface="flat", **common)(line, env.ctx)
        topo = env.create("kirchhoff_psdm", surface="topography", **common)(line, env.ctx)
        ref = topo.meta.get("depth_reference_elevation", 0.0)
        env.panel([(flat, "flat surface assumed", "cdpx"),
                   (topo, f"from topography (depth below {ref:.0f} m)", "cdpx")],
                  "m49_psdm_topography.png", gain="None", size=(1300, 650))
    env.shot("m49", topography)
