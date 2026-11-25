"""Manual screenshots of the version 0.3 attribute and noise tools: instantaneous attributes, the time-frequency
window, Harlan and eigenimage filters, deghosting."""


def make(env):
    np = env.np
    create = env.create

    def attributes_panel():
        st = env.st
        x = st.header("cdpx") if np.ptp(st.header("cdpx")) > 0 else np.arange(1, st.ntr + 1, dtype=float)
        host = env.QtWidgets.QWidget()
        lay = env.QtWidgets.QHBoxLayout(host)
        specs = (("envelope", "Viridis", None, "envelope"),
                 ("instantaneous phase", "Seismic", (-180.0, 180.0), "instantaneous phase (degrees)"),
                 ("instantaneous frequency", "Rainbow", (0.0, 60.0), "instantaneous frequency (0-60 Hz)"))
        for attr, cmap, levels, title in specs:
            a = create("attributes", attribute=attr, window=0.024)(st, env.ctx)
            if levels is None:
                levels = (0.0, float(np.percentile(a.traces, 99.0)) or 1.0)
            p = env.SeismicPlot(host)
            p.show_traces(a.traces, a.t0, a.dt, x, "Density", cmap, 1.0, levels=levels)
            p.getPlotItem().setTitle(title)
            p.getPlotItem().setLabel("bottom", "cdpx")
            p.reset_view()
            lay.addWidget(p)
        host.resize(1500, 700)
        host.show()
        env.grab(host, "m54_instantaneous_attributes.png")
        host.close()

    env.shot("m54", attributes_panel)

    def time_frequency():
        from seisdesk.gui.timefreq_dialog import TimeFrequencyDialog
        st = env.st
        dlg = TimeFrequencyDialog(st, env.win, trace=st.ntr // 2)
        dlg.method.setCurrentText("S-transform")
        dlg.fmax.setValue(80.0)
        dlg.compute()
        env.dialog(dlg, "m55_time_frequency.png", (1100, 680))

    env.shot("m55", time_frequency)

    def separation():
        rng = np.random.default_rng(11)
        st = env.st
        noisy = st.copy((st.traces + 0.8 * float(np.std(st.traces)) * rng.standard_normal(st.traces.shape))
                        .astype(np.float32))
        signal = create("harlan", traces=16, window=0.3, max_dip=6.0, threshold=2.0, fmax=70.0)(noisy, env.ctx)
        eigen = create("eigen_filter", rank=2, traces=9, window=0.4)(noisy, env.ctx)
        env.panel([(noisy, "stack with random noise", "cdpx"), (signal, "Harlan separation: signal", "cdpx"),
                   (eigen, "eigenimage filter: rank 2, 9-trace windows", "cdpx")], "m56_harlan_eigenimage.png",
                  gain="AGC 1 s", size=(1500, 700))

    env.shot("m56", separation)

    def deghosting():
        from seisdesk.processing.deghost import add_ghost
        shot = env.marine.gather("fldr", 1020.0)
        r, v = 0.95, 1500.0
        ghosted = add_ghost(add_ghost(shot.traces, shot.dt, 15.0, v, r), shot.dt, 10.0, v, r)
        g = shot.copy(ghosted.astype(np.float32))
        out = create("deghost", side="both", source_depth=10.0, receiver_depth=15.0, reflection=r, water_level=2.0,
                     velocity=v)(g, env.ctx)
        env.panel([(shot, "shot without ghosts", "offset"), (g, "source ghost 10 m, receiver ghost 15 m", "offset"),
                   (out, "after deghosting (vertical incidence)", "offset")], "m57_deghosting.png", gain="None",
                  size=(1500, 700))

    env.shot("m57", deghosting)
