"""Run SeisDesk on the synthetic lines and save screenshots of its windows plus the measured numbers.

    python tools/make_figures.py figures

Every image is a grab of the real SeisDesk widgets showing data processed by SeisDesk itself.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
LINES: list[str] = []
FAILED: list[str] = []


def note(text: str = "") -> None:
    print(text, flush=True)
    LINES.append(text)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="w", foreground="k")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    from seisdesk.core import synthetic
    from seisdesk.core.picks import PickSet
    from seisdesk.core.velocity import VelocityPicks
    from seisdesk.gui import common, main_window
    from seisdesk.gui.common import clip_value, display_gain
    from seisdesk.gui.dialogs import SpectrumDialog, VelocityFieldDialog
    from seisdesk.gui.plot import SeismicPlot
    from seisdesk.processing import create
    from seisdesk.workflow.templates import templates

    common.HEADLESS = True
    main_window.MainWindow._ask_save_changes = lambda self: True
    main_window.default_work_dir = lambda: os.path.abspath(OUT)

    def settle(n: int = 6) -> None:
        for _ in range(n):
            app.processEvents()
            QtCore.QThread.msleep(30)

    def grab(widget, name: str) -> None:
        settle()
        path = os.path.join(OUT, name)
        widget.grab().save(path)
        note(f"- saved `{name}`")

    def panel(items, name: str, gain: str = "AGC 0.5 s", size=(1800, 820), cmap="Gray (black = positive)") -> None:
        host = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(host)
        for d, title, xkey in items:
            p = SeismicPlot(host, y_label="Time" if d.domain == "time" else "Depth",
                            y_units="s" if d.domain == "time" else "m")
            disp = display_gain(d.traces, d.times, d.dt, gain, d.domain)
            x = d.header(xkey).astype(float) if xkey else np.arange(1, d.ntr + 1, dtype=float)
            p.show_traces(disp, d.t0, d.dt, x, "Density", cmap, clip_value(disp, 98.5))
            p.getPlotItem().setTitle(title)
            p.getPlotItem().setLabel("bottom", xkey or "trace", units="m" if xkey in ("cdpx", "offset", "gx") else None)
            p.reset_view()
            lay.addWidget(p)
        host.resize(*size)
        host.show()
        grab(host, name)
        host.close()

    def step(label: str, fn):
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception:                                  # noqa: BLE001 - keep going, report at the end
            FAILED.append(label)
            note(f"**FAILED: {label}**")
            note("```")
            note(traceback.format_exc())
            note("```")
            return None
        note(f"- {label}: {time.perf_counter() - t0:.1f} s")
        return out

    win = main_window.MainWindow()
    win.resize(1680, 1000)
    win.log_dock.hide()                                   # more room for the displays in the screenshots
    win.show()
    ctx = win.project.context
    settle()

    # ------------------------------------------------------------------ marine line
    note("## Marine line (synthetic: 4 reflectors, water-bottom multiples, a diffractor, noise)")
    marine = synthetic.marine_line(nshots=40, ngroups=60, nt=1001, dt=0.004, noise=0.03,
                                   diffractor=(3000.0, 900.0, 2000.0, 0.8))
    name = win.add_dataset(marine, "marine line")
    v = win.viewer

    def shot_view():
        v.mode_combo.setCurrentText("Density + wiggle")
        v.gain_combo.setCurrentText("t^2")
        v.show_gather(1020.0)
        grab(win, "01_viewer_shot_gather.png")
        v.set_gather_key("cdp")
        cdps = v.gather_values
        v.show_gather(float(cdps[cdps.size // 2]))
        grab(win, "02_viewer_cmp_gather.png")
    step("viewer: shot and CMP gathers", shot_view)

    pre = step("direct-wave mute + bandpass + t^2 gain + spiking deconvolution", lambda: create(
               "decon", operator_length=0.16, gap=0.004, prewhitening=0.1, tmin=0.1, tmax=3.5)(create(
               "tpow", power=2.0)(create("bandpass", corners=[4, 8, 60, 80])(create(
               "velocity_mute", velocity=1500.0, t0=0.12, kind="top", taper=0.03)(marine, ctx), ctx), ctx), ctx))
    cmp_ = create("sort", keys="cdp,offset")(pre, ctx)
    step("automatic velocity picking (every 30th CMP)", lambda: create(
        "auto_velocity", cdp_step=30, supergather=2, vmin=1400, vmax=3500, nv=106, pick_interval=0.1)(cmp_, ctx))
    vp = ctx.velocity
    if vp is not None and len(vp):
        truth = {0.6: 1700.0, 1.1: 2100.0, 1.6: 2500.0, 2.2: 2900.0}
        note("")
        note("Automatic RMS velocity picks against the true values (m/s):")
        note("")
        note("| CMP | " + " | ".join(f"t0 = {t:g} s (true {v_:.0f})" for t, v_ in truth.items()) + " |")
        note("|---|" + "---|" * len(truth))
        for c in vp.cdps:
            got = vp.function(c, np.array(list(truth)))
            note(f"| {c:g} | " + " | ".join(f"{g:.0f} ({(g / t_ - 1) * 100:+.1f} %)"
                                              for g, t_ in zip(got, truth.values())) + " |")
        note("")

    cname = win.add_dataset(cmp_, "marine CMP-sorted")

    def velan_view():
        win.open_velan(cname)
        va = win.velan
        va.vmin.setValue(1400)
        va.vmax.setValue(3500)
        if vp is not None and len(vp):
            va.show_cdp(float(vp.cdps[len(vp.cdps) // 2]))
        va.compute()
        grab(win, "03_velocity_analysis.png")
    step("velocity analysis window", velan_view)

    nmo = step("NMO with the picked velocities", lambda: create("nmo", velocity="project", stretch_limit=0.5)(cmp_, ctx))
    dem = step("parabolic Radon demultiple", lambda: create("radon_demultiple", qmin=-0.05, qmax=0.6, nq=101,
                                                             q_cut=0.04, fmax=60.0)(nmo, ctx))
    if nmo is not None and dem is not None:
        c = float(np.unique(nmo.header("cdp"))[np.unique(nmo.header("cdp")).size // 2])
        step("figure: Radon", lambda: panel([(cmp_.gather("cdp", c), f"CMP {c:g} raw", "offset"),
                                             (nmo.gather("cdp", c), "after NMO", "offset"),
                                             (dem.gather("cdp", c), "after Radon demultiple", "offset")],
                                            "04_nmo_and_radon_demultiple.png"))
    st = step("CMP stack", lambda: create("stack")(dem if dem is not None else nmo, ctx))
    pst = step("phase-shift time migration", lambda: create("phase_shift", velocity="project", fmax=70.0)(st, ctx))
    kir = step("Kirchhoff prestack time migration", lambda: create("kirchhoff_pstm", velocity="project",
                                                                     aperture=1500.0)(cmp_, ctx))
    if st is not None and pst is not None:
        step("figure: stack and time migrations", lambda: panel(
            [(st, "CMP stack (diffraction at x = 3000 m, t = 0.9 s)", "cdpx"),
             (pst, "phase-shift migration", "cdpx")] + ([(kir, "Kirchhoff prestack time migration", "cdpx")]
                                                        if kir is not None else []),
            "05_stack_and_time_migration.png", gain="AGC 1 s"))

        def focus(d, x_c=3000.0, t_c=0.9):
            j = int(np.argmin(np.abs(d.header("cdpx") - x_c)))
            sel_t = np.abs(d.times - t_c) <= 0.04
            win_e = (d.traces[max(0, j - 6):j + 7][:, sel_t].astype(float) ** 2).sum()
            ring = (d.traces[max(0, j - 40):j + 41][:, (d.times > 0.75) & (d.times < 1.4)].astype(float) ** 2).sum()
            return win_e / max(ring, 1e-30)
        note(f"- diffraction energy within 6 traces and 40 ms of the apex, as a fraction of its neighbourhood: "
             f"stack {focus(st):.2f}, phase shift {focus(pst):.2f}"
             + (f", Kirchhoff {focus(kir):.2f}" if kir is not None else ""))
        step("depth model (Dix) from the picks", lambda: create("build_depth_model", source="picks (Dix)", dz=10.0,
                                                                zmax=3000.0, smooth=200.0)(st, ctx))
        dep = step("split-step depth migration", lambda: create("split_step", velocity="project model", dz=10.0,
                                                                 zmax=3000.0, fmax=60.0)(st, ctx))
        if dep is not None:
            step("figure: depth migration", lambda: panel(
                [(st, "CMP stack (time)", "cdpx"), (dep, "split-step depth migration (depth)", "cdpx")],
                "06_depth_migration.png", gain="AGC 1 s", size=(1500, 820)))
            win.add_dataset(dep, "marine depth migration")
        win.add_dataset(st, "marine stack")
        mig_name = win.add_dataset(pst, "marine phase-shift migration")
        step("viewer: migrated section", lambda: (win.show_dataset(mig_name), grab(win, "07_viewer_migrated_section.png")))

        def spectra():
            dlg = SpectrumDialog(win.viewer.displayed_data(), win)
            dlg.resize(1200, 560)
            dlg.show()
            grab(dlg, "08_amplitude_spectrum.png")
            tabs = dlg.findChild(QtWidgets.QTabWidget)
            tabs.setCurrentIndex(1)
            grab(dlg, "09_fk_spectrum.png")
            dlg.close()
        step("spectra dialogs", spectra)

        def vfield():
            dlg = VelocityFieldDialog(ctx.velocity, cmp_.header("cdp"), cmp_.times, win)
            dlg.resize(1200, 620)
            dlg.show()
            grab(dlg, "10_velocity_field.png")
            dlg.close()
        step("velocity field dialog", vfield)

    def flow_view():
        win.flow_editor.set_flow(templates()["Marine 2D: shots to migrated stack"])
        win.flow_editor.refresh_inputs(name)
        win.flow_editor.steps.setCurrentRow(7)
        win.tabs.setCurrentWidget(win.flow_editor)
        grab(win, "11_flow_editor.png")
    step("flow editor window", flow_view)

    # ------------------------------------------------------------------ land refraction line
    note("")
    note("## Land refraction line (synthetic: direct wave 800 m/s, refractors 2000 and 3500 m/s)")
    land = synthetic.land_refraction_line(nshots=12, nrec=96, nt=601, noise=0.02)
    win.add_dataset(land, "land refraction line")

    def picking():
        v.set_gather_key("fldr")
        v.show_gather(7.0)
        v.mode_combo.setCurrentText("Wiggle")
        v.gain_combo.setCurrentText("Trace normalise")
        win.open_picking()
        ps = PickSet("fldr", "offset", "first breaks")
        ps.new_phase("direct")
        ctx.picks["first breaks"] = ps
        panel_ = v.picking
        panel_.refresh_sets(select="first breaks")
        panel_.auto_pick(True)                                     # energy-ratio first breaks, all shots
        # the user's breaks at the two crossovers seen on the gathers (about 55 m and 230 m)
        for name_ in ("refractor 1", "refractor 2"):
            ps.new_phase(name_)
        for g in list(ps.data):
            pts = ps.points(g, "direct")
            ps.data[g]["direct"] = []
            for x, t in pts:
                ph = "direct" if abs(x) < 55 else ("refractor 1" if abs(x) < 230 else "refractor 2")
                ps.add(g, ph, x, t)
        panel_.pooled_check.setChecked(True)
        panel_.refresh_phases(select="refractor 1")
        settle()
        bar = v.picking_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())                               # show the fit and layer tables
        grab(win, "12_refraction_picking.png")
        syn = land.meta["synthetic"]
        note("")
        note("Line fits pooled over all 12 shots (automatic picks, breaks at 55 m and 230 m):")
        note("")
        note("| phase | velocity (m/s) | intercept (ms) | rms (ms) | picks |")
        note("|---|---|---|---|---|")
        for f in panel_.fits:
            note(f"| {f['phase']} | {f['velocity']:.0f} | {f['intercept'] * 1000:.1f} | {f['rms'] * 1000:.2f} | {f['n']} |")
        from seisdesk.core.picks import refraction_layers
        true_layers = refraction_layers([{"velocity": syn["v1"], "intercept": 0.0, "slowness": 1 / syn["v1"]}] +
                                        [{"velocity": a, "intercept": b, "slowness": 1 / a} for a, b in syn["refractors"]])
        note("")
        note("| layer | thickness from picks (m) | true (m) | base depth from picks (m) | true (m) |")
        note("|---|---|---|---|---|")
        for got, want in zip(panel_.layers, true_layers):
            note(f"| {got['layer']} | {got['thickness']:.1f} | {want['thickness']:.1f} | {got['depth_to_base']:.1f} | "
                 f"{want['depth_to_base']:.1f} |")
        note("")
        note("True model: v1 = 800 m/s; refractor velocities 2000 and 3500 m/s; intercepts 40 and 90 ms; causal "
             "wavelet, so the true first break is the onset of energy.")
    step("refraction picking with breaks", picking)

    # ------------------------------------------------------------------ residual statics
    note("")
    note("## Residual statics (synthetic land reflections with random source and receiver statics up to 6 ms)")

    def statics():
        kw = dict(nshots=24, nrec=96, nt=601, v1=1e9, refractors=(),
                  reflectors=((0.25, 0.6, 1800.0), (0.45, -0.5, 2200.0), (0.7, 0.5, 2600.0)), noise=0.02, seed=3)
        bad = synthetic.land_refraction_line(source_statics_ms=6.0, receiver_statics_ms=6.0, **kw)
        good = synthetic.land_refraction_line(**kw)
        sctx = type(ctx)(velocity=VelocityPicks.from_function([0.0, 0.25, 0.45, 0.7, 1.2],
                                                              [1800.0, 1800.0, 2200.0, 2600.0, 2600.0]))

        def prep(d):
            d = create("velocity_mute", velocity=1e6, t0=0.12, kind="top", taper=0.02)(d, sctx)
            return create("nmo", stretch_limit=0.6)(d, sctx)
        nb, ng = prep(bad), prep(good)
        fixed = create("residual_statics", tmin=0.15, tmax=0.95, max_shift=0.02, iterations=4)(nb, sctx)
        res = sctx.results["residual_statics"]
        src = np.array([res["source"][k] for k in sorted(res["source"])])
        true_s = bad.meta["synthetic"]["source_statics_s"] * 1000.0
        rec = np.array([res["receiver"][k] for k in sorted(res["receiver"])])
        true_r = bad.meta["synthetic"]["receiver_statics_s"] * 1000.0
        note(f"- correlation of recovered and true statics: sources {np.corrcoef(src, -true_s)[0, 1]:.2f}, "
             f"receivers {np.corrcoef(rec, -true_r)[0, 1]:.2f}")
        err_s = np.std(src + (true_s - true_s.mean()))
        err_r = np.std(rec + (true_r - true_r.mean()))
        note(f"- residual error after correction (rms): sources {err_s:.2f} ms, receivers {err_r:.2f} ms "
             f"(statics before: {np.std(true_s):.2f} / {np.std(true_r):.2f} ms rms)")
        panel([(create("stack")(nb, sctx), "stack with statics", "cdpx"),
               (create("stack")(fixed, sctx), "after residual statics", "cdpx"),
               (create("stack")(ng, sctx), "reference: no statics", "cdpx")],
              "13_residual_statics.png", gain="None")
    step("residual statics", statics)

    with open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# SeisDesk on synthetic data\n\n" + "\n".join(LINES) + "\n")
    win.dirty = False
    win.close()
    print("failed steps:", FAILED or "none")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
