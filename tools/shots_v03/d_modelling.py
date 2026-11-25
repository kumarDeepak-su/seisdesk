"""Manual screenshots of the version 0.3 modelling steps (m50-m53), on one shot of the land line."""

from __future__ import annotations

import os


def make(env):
    np = env.np
    from seisdesk.core.models import model_from_text

    shot6 = env.geo.subset(env.geo.header("fldr") == 6)
    layers = "1000, 2000, 3000 / 20, 250"
    cache = {}

    def model_and_gather(model, gather, title_model, title_gather, name, zmax, gain="AGC 0.25 s"):
        """The P-velocity model with the shot and receivers (left) and a modelled gather (right)."""
        xg = np.arange(-100.0, 1051.0, 5.0)
        zg = np.arange(0.0, zmax + 1.0, 2.5)
        vp = model.sample(xg, zg)[0]
        host = env.QtWidgets.QWidget()
        lay = env.QtWidgets.QHBoxLayout(host)
        p1 = env.SeismicPlot(host, y_label="Depth", y_units="m")
        p1.show_image(vp, float(xg[0]), 5.0, 0.0, 2.5, "Rainbow", (float(vp.min()), float(vp.max())))
        p1.addItem(env.pg.ScatterPlotItem(gather.header("gx"), np.zeros(gather.ntr), size=4,
                                          brush=env.pg.mkBrush(30, 60, 200), pen=None))
        p1.addItem(env.pg.ScatterPlotItem(gather.header("sx")[:1], [0.0], size=14, symbol="star",
                                          brush=env.pg.mkBrush(220, 30, 30), pen=None))
        p1.getPlotItem().setTitle(title_model)
        p1.getPlotItem().setLabel("bottom", "x (m)")
        p1.reset_view()
        p2 = env.SeismicPlot(host)
        disp = env.display_gain(gather.traces, gather.times, gather.dt, gain)
        p2.show_traces(disp, gather.t0, gather.dt, gather.header("offset"), "Density", "Gray (black = positive)",
                       env.clip_value(disp, 98.5))
        p2.getPlotItem().setTitle(title_gather)
        p2.getPlotItem().setLabel("bottom", "offset (m)")
        p2.reset_view()
        lay.addWidget(p1, 1)
        lay.addWidget(p2, 1)
        host.resize(1500, 650)
        host.show()
        env.grab(host, name)
        host.close()

    def acoustic_fd():
        if "fd" not in cache:
            cache["fd"] = env.create("fd_acoustic", model=layers, f0=20.0, dx=4.0, zmax=400.0, tmax=0.8)(shot6, env.ctx)
        return cache["fd"]

    def acoustic():
        model_and_gather(model_from_text(layers), acoustic_fd(), "P velocity (m/s), three layers",
                         "acoustic finite differences, shot 6", "m50_fd_acoustic.png", 400.0)
    env.shot("m50", acoustic)

    def elastic():
        fd = env.create("fd_elastic", model=layers, f0=12.0, dx=3.0, zmax=400.0, tmax=0.8,
                        component="vertical")(shot6, env.ctx)
        env.panel([(fd, "elastic finite differences, vertical component", "offset")], "m51_fd_elastic.png",
                  gain="AGC 0.25 s", size=(1000, 650))
    env.shot("m51", elastic)

    def rays():
        ray = env.create("ray_synthetics", model=layers, f0=20.0, tmax=0.8, first_arrivals=True)(shot6, env.ctx)
        env.panel([(ray, "ray-traced synthetics", "offset"), (acoustic_fd(), "acoustic finite differences", "offset")],
                  "m52_ray_vs_fd.png", gain="AGC 0.25 s", size=(1300, 650))
    env.shot("m52", rays)

    def kirchhoff():
        path = os.path.join(env.work, "curved_model.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# vp vs rho : top of the layer (x z, ...)\n"
                     "1800 - - : 0 0\n"
                     "2400 - - : 0 300, 300 380, 600 330, 1000 250\n"
                     "3000 - - : 0 520, 1000 420\n")
        km = env.create("kirchhoff_modelling", model=path, f0=25.0, tmax=0.8)(shot6, env.ctx)
        model_and_gather(model_from_text(path), km, "model file, P velocity (m/s)", "Kirchhoff modelling, shot 6",
                         "m53_kirchhoff_modelling.png", 700.0, gain="None")
    env.shot("m53", kirchhoff)
