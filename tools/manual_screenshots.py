"""Screenshots for the user manual: every image is a grab of the real SeisDesk windows on synthetic data.

    python tools/manual_screenshots.py figures/manual
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "manual_shots")
FAILED: list[str] = []


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    work = os.path.join(OUT, "_work")
    os.makedirs(work, exist_ok=True)
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="w", foreground="k")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    from seisdesk.core import synthetic
    from seisdesk.core.picks import PickSet
    from seisdesk.core.seg2 import read_seg2_files, write_seg2
    from seisdesk.gui import common, main_window
    from seisdesk.gui.common import clip_value, display_gain
    from seisdesk.gui.dialogs import (HeaderTableDialog, ImportDialog, SingleStepDialog, SpectrumDialog, SyntheticDialog,
                                      VelocityFieldDialog)
    from seisdesk.gui.plot import SeismicPlot
    from seisdesk.processing import create
    from seisdesk.processing.reference import process_reference_html
    from seisdesk.workflow.templates import templates

    common.HEADLESS = True
    main_window.MainWindow._ask_save_changes = lambda self: True
    main_window.default_work_dir = lambda: os.path.abspath(work)

    def settle(n: int = 6) -> None:
        for _ in range(n):
            app.processEvents()
            QtCore.QThread.msleep(30)

    def grab(widget, name: str) -> None:
        settle()
        widget.grab().save(os.path.join(OUT, name))
        print("saved", name, flush=True)

    def shot(name, fn):
        try:
            fn()
        except Exception:                                  # noqa: BLE001
            FAILED.append(name)
            traceback.print_exc()

    def panel(items, name, gain="AGC 0.5 s", size=(1500, 700)):
        host = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(host)
        for d, title, xkey in items:
            p = SeismicPlot(host, y_label="Time" if d.domain == "time" else "Depth",
                            y_units="s" if d.domain == "time" else "m")
            disp = display_gain(d.traces, d.times, d.dt, gain, d.domain)
            x = d.header(xkey).astype(float) if xkey else np.arange(1, d.ntr + 1, dtype=float)
            p.show_traces(disp, d.t0, d.dt, x, "Density", "Gray (black = positive)", clip_value(disp, 98.5))
            p.getPlotItem().setTitle(title)
            p.getPlotItem().setLabel("bottom", xkey or "trace")
            p.reset_view()
            lay.addWidget(p)
        host.resize(*size)
        host.show()
        grab(host, name)
        host.close()

    def dialog(dlg, name, size=None):
        if size:
            dlg.resize(*size)
        dlg.show()
        grab(dlg, name)
        dlg.close()

    win = main_window.MainWindow()
    win.resize(1600, 960)
    win.show()
    ctx = win.project.context
    v = win.viewer
    settle()

    # ---------------------------------------------------------------- land line as SEG-2 shot files + shot table
    land = synthetic.land_refraction_line(nshots=12, nrec=96, nt=601, noise=0.02)
    seg2_dir = os.path.join(work, "line1_seg2")
    os.makedirs(seg2_dir, exist_ok=True)
    paths = []
    for f in land.gather_values("fldr"):
        path = os.path.join(seg2_dir, f"{1000 + int(f)}.dat")
        write_seg2(land, path, record=f)
        paths.append(path)
    shot_x = np.unique(land.header("sx"))
    table = os.path.join(work, "geometry_line1.csv")
    with open(table, "w", encoding="utf-8") as fh:
        fh.write("file,ffid,shot_x,shot_z,first_rec_x,rec_dx,nchan\n")
        for k, (pth, sx) in enumerate(zip(paths, shot_x)):
            fh.write(f"line1_seg2/{os.path.basename(pth)},{k + 1},{sx:.1f},,0.0,10.0,96\n")
    topo = os.path.join(work, "topo_line1.txt")
    with open(topo, "w", encoding="utf-8") as fh:
        fh.write("x z\n")
        for x in np.arange(-50.0, 1001.0, 50.0):
            fh.write(f"{x:.1f} {120.0 + 6.0 * np.sin(x / 180.0):.2f}\n")
    raw = read_seg2_files(paths)
    lname = win.add_dataset(raw, "line1 (12 SEG-2 shots)")
    shot("m01", lambda: grab(win, "m01_main_window.png"))
    shot("m02", lambda: dialog(ImportDialog(os.path.join(work, "line1.sgy"), win), "m02_import_dialog.png", (520, 200)))
    shot("m03", lambda: dialog(SyntheticDialog(win), "m03_synthetic_dialog.png", (560, 220)))

    def shot_table():
        dlg = SingleStepDialog(win.project, "shot_table_geometry", lname, win)
        dlg.step.params.update({"table": os.path.basename(table), "topography": os.path.basename(topo)})
        dlg.form.set_step(dlg.step)
        dialog(dlg, "m04_shot_table_dialog.png", (760, 360))
    shot("m04", shot_table)
    geo = create("shot_table_geometry", table=table, topography=topo)(raw, ctx)
    win.project.datasets[lname] = geo
    geo.name = lname
    win.show_dataset(lname)
    shot("m05", lambda: dialog(HeaderTableDialog(geo, win), "m05_header_table.png", (1100, 520)))

    def viewer_shot():
        v.set_gather_key("fldr")
        v.show_gather(7.0)
        v.mode_combo.setCurrentText("Wiggle")
        v.gain_combo.setCurrentText("Trace normalise")
        grab(win, "m06_viewer_shot_gather.png")
    shot("m06", viewer_shot)

    # ---------------------------------------------------------------- refraction picking
    def picking():
        win.open_picking()
        ps = PickSet("fldr", "offset", "first breaks")
        ps.new_phase("direct")
        ctx.picks["first breaks"] = ps
        pnl = v.picking
        pnl.refresh_sets(select="first breaks")
        pnl.auto_pick(True)
        settle()
        grab(win, "m11_picking_automatic.png")
        for name_ in ("refractor 1", "refractor 2"):
            ps.new_phase(name_)
        for g in list(ps.data):
            pts = ps.points(g, "direct")
            ps.data[g]["direct"] = []
            for x, t in pts:
                ps.add(g, "direct" if abs(x) < 55 else ("refractor 1" if abs(x) < 230 else "refractor 2"), x, t)
        pnl.pooled_check.setChecked(True)
        pnl.refresh_phases(select="refractor 1")
        settle()
        bar = v.picking_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        grab(win, "m12_picking_phases_fits.png")
        bar.setValue(0)
    shot("m11-12", picking)

    # ---------------------------------------------------------------- marine line: display, spectra, filters, decon
    marine = synthetic.marine_line(nshots=40, ngroups=60, nt=1001, dt=0.004, noise=0.03,
                                   diffractor=(3000.0, 900.0, 2000.0, 0.8))
    mname = win.add_dataset(marine, "marine line")
    v.pick_btn.setChecked(False)

    def marine_views():
        v.show_gather(1020.0)
        v.mode_combo.setCurrentText("Density + wiggle")
        v.gain_combo.setCurrentText("t^2")
        grab(win, "m07_viewer_marine_shot.png")
        v.set_gather_key("cdp")
        v.show_gather(float(v.gather_values[v.gather_values.size // 2]))
        grab(win, "m08_viewer_cmp_gather.png")
        v.set_gather_key("fldr")
        v.show_gather(1020.0)
        v.plot.getPlotItem().vb.setRange(yRange=(0.4, 2.4), padding=0)
        dlg = SpectrumDialog(v.visible_data(), win)
        dlg.resize(1100, 600)
        dlg.show()
        grab(dlg, "m09_spectrum.png")
        dlg.tabs.setCurrentIndex(1)
        grab(dlg, "m10_autocorrelation.png")
        dlg.tabs.setCurrentIndex(2)
        grab(dlg, "m10b_fk_spectrum.png")
        dlg.close()
        v.plot.reset_view()
    shot("m07-10", marine_views)

    muted = create("velocity_mute", velocity=1500.0, t0=0.12, kind="top", taper=0.03)(marine, ctx)
    s1020 = muted.gather("fldr", 1020.0)

    def filters():
        dlg = SingleStepDialog(win.project, "bandpass", mname, win)
        dlg.step.params.update({"corners": [6.0, 10.0, 45.0, 60.0]})
        dlg.form.set_step(dlg.step)
        dialog(dlg, "m13_filter_dialog.png", (640, 260))
        bp = create("bandpass", corners=[6.0, 10.0, 45.0, 60.0])(s1020)
        fk = create("fk_filter", gather_key="fldr", position_key="offset", vmin=0.0, vmax=1600.0)(marine.gather(
            "fldr", 1020.0))
        panel([(marine.gather("fldr", 1020.0), "raw shot 1020", "offset"),
               (bp, "mute + band-pass 6-10-45-60 Hz", "offset"),
               (fk, "f-k reject below 1600 m/s", "offset")], "m14_filter_panels.png", gain="t^2")
        pre = create("tpow", power=2.0)(bp)
        sp = create("decon", operator_length=0.16, gap=0.004, prewhitening=0.1, tmin=0.3, tmax=3.0)(pre)
        pr = create("decon", operator_length=0.24, gap=0.024, prewhitening=0.1, tmin=0.3, tmax=3.0)(pre)
        panel([(pre, "before deconvolution", "offset"), (sp, "spiking (gap 4 ms, operator 160 ms)", "offset"),
               (pr, "predictive (gap 24 ms, operator 240 ms)", "offset")], "m15_decon_panels.png")
    shot("m13-15", filters)

    # ---------------------------------------------------------------- reflection phase picking on a shot
    def reflection_picking():
        win.show_dataset(mname)
        v.set_gather_key("fldr")
        v.show_gather(1020.0)
        v.mode_combo.setCurrentText("Density")
        v.gain_combo.setCurrentText("t^2")
        win.open_picking()
        pnl = v.picking
        ps = PickSet("fldr", "offset", "reflections")
        ps.new_phase("R1 0.6 s")
        ctx.picks["reflections"] = ps
        pnl.refresh_sets(select="reflections")
        pnl.fit_combo.setCurrentIndex(1)
        pnl.snap_combo.setCurrentText("peak")
        off = marine.header("offset")[v.idx]
        for k, (t0, vr, nm) in enumerate(((0.6, 1700.0, "R1 0.6 s"), (1.1, 2100.0, "R2 1.1 s"),
                                          (1.6, 2500.0, "R3 1.6 s"))):
            if k:
                ps.new_phase(nm)
                pnl.refresh_phases(select=nm)
            pnl.snap_combo.setCurrentText("peak" if k != 1 else "trough")
            for j in range(0, v.idx.size, 3):
                pnl.click(v, j, {"x": float(v.xpos[j]), "y": float(np.sqrt(t0 ** 2 + (off[j] / vr) ** 2)),
                                 "left": True, "right": False, "shift": False, "ctrl": False, "double": False})
        settle()
        bar = v.picking_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        grab(win, "m16_reflection_picking.png")
        bar.setValue(0)
        win.viewer.pick_btn.setChecked(False)
    shot("m16", reflection_picking)

    # ---------------------------------------------------------------- flows, velocity analysis, NMO ... migration
    def flow_editor():
        win.flow_editor.set_flow(templates()["Marine 2D: shots to migrated stack"])
        win.flow_editor.refresh_inputs(mname)
        win.flow_editor.steps.setCurrentRow(5)
        win.tabs.setCurrentWidget(win.flow_editor)
        grab(win, "m17_flow_editor.png")
    shot("m17", flow_editor)

    pre = create("decon", operator_length=0.16, gap=0.004, prewhitening=0.1, tmin=0.1, tmax=3.5)(
        create("tpow", power=2.0)(create("bandpass", corners=[4, 8, 60, 80])(muted, ctx), ctx), ctx)
    cmp_ = create("sort", keys="cdp,offset")(pre, ctx)
    create("auto_velocity", cdp_step=30, supergather=2, vmin=1400, vmax=3500, nv=106)(cmp_, ctx)
    cname = win.add_dataset(cmp_, "marine CMP-sorted")

    def velan():
        win.open_velan(cname)
        va = win.velan
        va.show_cdp(float(ctx.velocity.cdps[len(ctx.velocity.cdps) // 2]))
        va.compute()
        grab(win, "m18_velocity_analysis.png")
        va.mode_combo.setCurrentIndex(1)
        grab(win, "m19_eta_scan.png")
        va.mode_combo.setCurrentIndex(0)
    shot("m18-19", velan)

    nmo = create("nmo", velocity="project", stretch_limit=0.5)(cmp_, ctx)
    dem = create("radon_demultiple", qmin=-0.05, qmax=0.6, nq=101, q_cut=0.04, fmax=60.0)(nmo, ctx)
    c = float(np.unique(nmo.header("cdp"))[np.unique(nmo.header("cdp")).size // 2])
    shot("m20", lambda: panel([(cmp_.gather("cdp", c), f"CMP {c:g}", "offset"), (nmo.gather("cdp", c), "after NMO",
                                                                                     "offset"),
                               (dem.gather("cdp", c), "after Radon demultiple", "offset")], "m20_nmo_radon.png"))
    st = create("stack")(dem, ctx)
    pst = create("phase_shift", velocity="project", fmax=70.0)(st, ctx)
    kir = create("kirchhoff_pstm", velocity="project", aperture=1500.0)(cmp_, ctx)
    shot("m21", lambda: panel([(st, "CMP stack", "cdpx"), (pst, "phase-shift time migration", "cdpx"),
                               (kir, "Kirchhoff prestack time migration", "cdpx")], "m21_stack_time_migration.png",
                              gain="AGC 1 s", size=(1500, 760)))
    create("build_depth_model", source="picks (Dix)", dz=10.0, zmax=3000.0, smooth=200.0)(st, ctx)
    dep = create("split_step", velocity="project model", dz=10.0, zmax=3000.0, fmax=60.0)(st, ctx)
    shot("m22", lambda: panel([(st, "CMP stack (time)", "cdpx"), (dep, "split-step depth migration", "cdpx")],
                              "m22_depth_migration.png", gain="AGC 1 s", size=(1300, 760)))

    def vfield():
        dlg = VelocityFieldDialog(ctx.velocity, cmp_.header("cdp"), cmp_.times, win)
        dialog(dlg, "m24_velocity_field.png", (1100, 560))
    shot("m24", vfield)

    def horizons():
        sname = win.add_dataset(st, "marine stack")
        win.show_dataset(sname)
        win.open_picking()
        pnl = v.picking
        ps = PickSet("none", "cdp", "horizons")
        ps.new_phase("top 1.1 s")
        ctx.picks["horizons"] = ps
        pnl.refresh_sets(select="horizons")
        pnl.snap_combo.setCurrentText("trough")
        for j in range(0, v.idx.size, 8):
            pnl.click(v, j, {"x": float(v.xpos[j]), "y": 1.1, "left": True, "right": False, "shift": False,
                             "ctrl": False, "double": False})
        grab(win, "m25_horizon_picking.png")
        v.pick_btn.setChecked(False)
    shot("m25", horizons)

    # ---------------------------------------------------------------- residual statics
    def statics():
        kw = dict(nshots=24, nrec=96, nt=601, v1=1e9, refractors=(),
                  reflectors=((0.25, 0.6, 1800.0), (0.45, -0.5, 2200.0), (0.7, 0.5, 2600.0)), noise=0.02, seed=3)
        from seisdesk.core.velocity import VelocityPicks
        bad = synthetic.land_refraction_line(source_statics_ms=6.0, receiver_statics_ms=6.0, **kw)
        sctx = type(ctx)(velocity=VelocityPicks.from_function([0.0, 0.25, 0.45, 0.7, 1.2],
                                                              [1800.0, 1800.0, 2200.0, 2600.0, 2600.0]))
        nb = create("nmo", stretch_limit=0.6)(create("velocity_mute", velocity=1e6, t0=0.12, kind="top",
                                                      taper=0.02)(bad, sctx), sctx)
        fixed = create("residual_statics", tmin=0.15, tmax=0.95, max_shift=0.02, iterations=4)(nb, sctx)
        panel([(create("stack")(nb, sctx), "stack before residual statics", "cdpx"),
               (create("stack")(fixed, sctx), "after residual statics", "cdpx")], "m23_residual_statics.png",
              gain="None", size=(1100, 620))
    shot("m23", statics)

    # ---------------------------------------------------------------- version 0.2 tools
    from seisdesk.core.synthetic import events_to_traces, ricker
    from seisdesk.gui.dialogs import RefractionStaticsDialog, SurveyQCDialog

    def qc_tools():
        rng = np.random.default_rng(5)
        noisy = geo.copy()
        x = np.abs(noisy.header("offset"))
        w, c = ricker(12.0, noisy.dt)
        roll = events_to_traces((0.01 + x / 350.0)[:, None], np.full((noisy.ntr, 1), 2.5), noisy.nt, noisy.dt, w, c)
        with_roll = noisy.copy((noisy.traces + roll).astype(np.float32))
        tr = noisy.traces.copy()
        fl, ch = noisy.header("fldr"), noisy.header("tracf")
        tr[(ch == 30)] = 0.0                                     # a dead channel on every shot
        tr[(fl == 7) & (ch == 55)] *= 25.0                       # a noisy trace
        burst = np.flatnonzero((fl == 7) & (ch == 70))
        tr[burst, 250:275] += 1.5 * rng.standard_normal(25)      # a noise burst (too short for trace editing)
        noisy = noisy.copy(tr)
        dlg = SurveyQCDialog(noisy, win)
        dlg.resize(1150, 620)
        dlg.show()
        grab(dlg, "m27_survey_qc_geometry.png")
        dlg.tabs.setCurrentIndex(2)
        grab(dlg, "m28_survey_qc_amplitudes.png")
        dlg.close()
        edited = create("despike", gather_key="fldr", window=0.04, neighbours=7, threshold=3.0)(
            create("auto_edit", gather_key="fldr", action="zero")(noisy, ctx), ctx)
        s7 = lambda d: d.gather("fldr", 7.0)
        panel([(s7(noisy), "shot 7: dead channel, noisy trace, burst", "offset"),
               (s7(edited), "after automatic editing and despike", "offset")], "m29_auto_edit_despike.png",
              gain="None", size=(1200, 650))
        cone = create("cone_mute", v_fast=450.0, v_slow=280.0, t0=0.0, taper=0.03)(with_roll, ctx)
        taup = create("taup_filter", gather_key="fldr", v_cut=600.0, v_min=150.0, np=161, damping=0.1, fmin=2.0,
                      fmax=20.0)(with_roll.subset(with_roll.header("fldr") == 7), ctx)
        panel([(s7(with_roll), "ground roll (350 m/s)", "offset"), (s7(cone), "cone mute 280-450 m/s", "offset"),
               (taup, "tau-p, reject below 600 m/s, 2-20 Hz", "offset")], "m30_groundroll_cone_taup.png",
              gain="Trace normalise", size=(1500, 650))
    shot("m27-30", qc_tools)

    def fx_panel():
        rng = np.random.default_rng(6)
        noisy = st.copy((st.traces + 0.6 * float(np.std(st.traces)) * rng.standard_normal(st.traces.shape))
                        .astype(np.float32))
        fx = create("fx_decon", gather_key="", window=0.5, traces=30, filter_length=4, damping=1.0, fmax=70.0)(noisy)
        panel([(noisy, "stack with random noise", "cdpx"), (fx, "after f-x deconvolution", "cdpx")],
              "m31_fx_decon.png", gain="AGC 1 s", size=(1300, 700))
    shot("m31", fx_panel)

    def refraction_statics_window():
        v1, v2 = 600.0, 2200.0
        stn = np.arange(0.0, 480.0, 5.0)
        z = 8.0 + 4.0 * np.sin(2 * np.pi * stn / 300.0)
        elev = 120.0 + 6.0 * np.sin(stn / 180.0)
        delay = z * np.sqrt(v2 ** 2 - v1 ** 2) / (v1 * v2)
        rows = [(k + 1, j + 1, s_, g_) for k, s_ in enumerate(stn[::8]) for j, g_ in enumerate(stn)]
        fl, ch, sx_, gx_ = (np.array(col, float) for col in zip(*rows))
        off = gx_ - sx_
        d = type(geo)(np.zeros((fl.size, 20), np.float32), 0.002,
                      {"fldr": fl, "tracf": ch, "sx": sx_, "gx": gx_, "offset": off,
                       "selev": np.interp(sx_, stn, elev), "gelev": np.interp(gx_, stn, elev)})
        ps = PickSet("fldr", "offset", "fb demo")
        ps.new_phase("direct")
        ps.new_phase("refractor 1")
        for i in range(fl.size):
            xa = abs(off[i])
            if 0 < xa <= 20:
                ps.add(fl[i], "direct", off[i], xa / v1)
            if xa >= 40:
                ps.add(fl[i], "refractor 1", off[i], np.interp(sx_[i], stn, delay) + np.interp(gx_[i], stn, delay)
                       + xa / v2 + 0.0005 * rng2.standard_normal())
        ctx.picks["fb demo"] = ps
        create("refraction_statics", pickset="fb demo", phase="refractor 1", datum=100.0, smoothing=0.05)(d, ctx)
        dialog(RefractionStaticsDialog(ctx.results["refraction_statics"], win), "m32_refraction_statics.png",
               (1050, 680))
    rng2 = np.random.default_rng(9)
    shot("m32", refraction_statics_window)

    def cvg():
        out = create("cvg_panels", npanels=6, vmin=1700.0, vmax=2700.0, supergather=1)(cmp_, ctx)
        panel([(out, "constant-velocity gathers, 1700 to 2700 m/s (one panel per velocity)", None)],
              "m33_cvg_panels.png", gain="AGC 0.5 s", size=(1300, 650))
    shot("m33", cvg)

    # ---------------------------------------------------------------- version 0.3 tools: one module per group in
    # tools/shots_v03, each with make(env) that grabs its own screenshots through env.shot(...)
    import importlib.util
    from types import SimpleNamespace
    env = SimpleNamespace(win=win, ctx=ctx, create=create, panel=panel, dialog=dialog, grab=grab, shot=shot,
                          settle=settle, geo=geo, land=land, marine=marine, cmp_=cmp_, st=st, work=work,
                          synthetic=synthetic, PickSet=PickSet, SeismicPlot=SeismicPlot, QtWidgets=QtWidgets,
                          QtCore=QtCore, pg=pg, np=np, out_dir=OUT, display_gain=display_gain, clip_value=clip_value)
    shots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots_v03")
    for fname in sorted(os.listdir(shots_dir)) if os.path.isdir(shots_dir) else []:
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location("shots_v03_" + fname[:-3], os.path.join(shots_dir, fname))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.make(env)
        except Exception:                                  # noqa: BLE001
            FAILED.append(fname)
            traceback.print_exc()

    def reference():
        dlg = QtWidgets.QDialog(win)
        dlg.setWindowTitle("Process and parameter reference")
        br = QtWidgets.QTextBrowser(dlg)
        br.setHtml(process_reference_html())
        QtWidgets.QVBoxLayout(dlg).addWidget(br)
        dialog(dlg, "m26_process_reference.png", (900, 640))
    shot("m26", reference)

    win.dirty = False
    win.close()
    print("failed:", FAILED or "none")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
