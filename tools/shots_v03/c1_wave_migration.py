"""Manual screenshots of the wave-equation depth migrations (m42-m45)."""

import numpy as np


def make(env):
    create, panel, ctx = env.create, env.panel, env.ctx
    post = dict(velocity="project model", dz=10.0, zmax=3000.0, fmax=35.0)

    def poststack():
        fd = create("fd_migration", angle="45", **post)(env.st, ctx)
        ffd = create("ffd_migration", **post)(env.st, ctx)
        pspi = create("pspi_migration", **post)(env.st, ctx)
        panel([(fd, "finite difference, 45 degrees", "cdpx"), (ffd, "Fourier finite difference", "cdpx"),
               (pspi, "phase shift plus interpolation", "cdpx")], "m42_wave_migration_poststack.png",
              gain="AGC 0.5 s", size=(1500, 700))
    env.shot("m42", poststack)

    def prestack():
        fl = env.marine.header("fldr")
        chosen = np.unique(fl)[::4]
        shots = env.marine.subset(np.isin(fl, chosen))
        shots = create("velocity_mute", velocity=1500.0, t0=0.12, kind="top", taper=0.03)(shots, ctx)
        wem = create("wave_psdm", velocity="project model", dz=10.0, zmax=3000.0, fmin=3.0, fmax=30.0,
                     aperture=750.0)(shots, ctx)
        ss = create("split_step", **post)(env.st, ctx)
        panel([(ss, "stack, post-stack split-step", "cdpx"),
               (wem, f"{chosen.size} shots, prestack shot-profile migration", "cdpx")],
              "m43_wave_psdm.png", gain="AGC 0.5 s", size=(1400, 700))
    env.shot("m43", prestack)

    def beams():
        gb = create("gaussian_beam", velocity="project model", dz=10.0, zmax=3000.0, fmin=5.0,
                    fmax=35.0)(env.st, ctx)
        ss = create("split_step", **post)(env.st, ctx)
        panel([(ss, "split-step", "cdpx"), (gb, "Gaussian beams", "cdpx")], "m44_gaussian_beam.png",
              gain="AGC 0.5 s", size=(1400, 700))
    env.shot("m44", beams)

    def dip_limit():
        from seisdesk.core.dataset import SeismicData
        from seisdesk.core.synthetic import events_to_traces, ricker
        from seisdesk.processing import Context
        nx, dx, dt, nt, v = 201, 10.0, 0.004, 500, 2000.0
        x = np.arange(nx) * dx
        points = ((500.0, 300.0), (1000.0, 600.0), (1500.0, 900.0))
        times = np.stack([2.0 * np.hypot(x - xd, zd) / v for xd, zd in points], axis=1)
        w, c = ricker(25.0, dt)
        tr = events_to_traces(times, np.ones_like(times), nt, dt, w, c)
        heads = {"cdp": np.arange(1, nx + 1), "cdpx": x, "sx": x, "gx": x, "offset": np.zeros(nx)}
        d = SeismicData(tr.astype(np.float32), dt, heads, name="three diffractors")
        d.meta["geometry"] = {"bin_size": dx}
        local = Context()                       # constant velocity: keep the project's depth model untouched
        common = dict(velocity=str(v), dz=5.0, zmax=1200.0, fmax=50.0)
        fd45 = create("fd_migration", angle="45", **common)(d, local)
        fd65 = create("fd_migration", angle="65", **common)(d, local)
        ss = create("split_step", **common)(d, local)
        panel([(d, "three point diffractors", "cdpx"), (fd45, "finite difference, 45 degrees", "cdpx"),
               (fd65, "finite difference, 65 degrees", "cdpx"), (ss, "split-step (reference)", "cdpx")],
              "m45_fd_dip_limit.png", gain="AGC 0.25 s", size=(1600, 650))
    env.shot("m45", dip_limit)
