"""Manual screenshots m34-m37 (version 0.3): wavelet phase tools, shaping and signature deconvolution, vibroseis
correlation and trace interpolation."""

import os


def make(env):
    np = env.np
    create = env.create
    shot_1020 = env.marine.gather("fldr", 1020.0)

    def phase_tools():
        s = shot_1020.time_window(0.4, 1.8)
        rot = create("phase_rotation", angle=90.0)(s)
        mph = create("min_phase", mode="data wavelet", tmin=0.5, tmax=1.8)(s)
        env.panel([(s, "shot 1020, zero-phase wavelet", "offset"), (rot, "phase rotated by 90 degrees", "offset"),
                   (mph, "converted to minimum phase", "offset")], "m34_phase_tools.png", gain="AGC 0.5 s",
                  size=(1500, 650))
    env.shot("m34", phase_tools)

    def shaping_signature():
        from seisdesk.core.synthetic import causal_wavelet
        g = env.geo.gather("fldr", 6.0)
        w, _ = causal_wavelet(30.0, g.dt)                     # the source wavelet of the synthetic land line
        path = os.path.join(env.work, "land_source_wavelet.txt")
        np.savetxt(path, w)
        sig = create("signature_decon", source="wavelet file", wavelet_file=path, output="zero-phase band-pass",
                     corners="5,10,70,90", water_level=1.0)(g)
        shp = create("shaping_filter", source="wavelet file", wavelet_file=path, desired="zero-phase Ricker",
                     ricker_f0=30.0, operator_length=0.2)(g)
        env.panel([(g.time_window(0.0, 0.7), "land shot 6 (causal source wavelet)", "offset"),
                   (sig.time_window(0.0, 0.7), "signature decon, 5-80 Hz", "offset"),
                   (shp.time_window(0.0, 0.7), "shaping filter to a 30 Hz Ricker", "offset")],
                  "m35_shaping_signature.png", gain="AGC 0.25 s", size=(1500, 650))
    env.shot("m35", shaping_signature)

    def vibro():
        s = shot_1020.time_window(0.0, 2.0)
        sweep = dict(f1=8.0, f2=60.0, length=2.0, taper_start=0.2, taper_end=0.2)
        unc = create("vibro_convolve", **sweep)(s)
        cor = create("vibro_correlation", output_length=2.0, **sweep)(unc)
        kmp = create("klauder_min_phase", **sweep)(cor)
        env.panel([(unc, "uncorrelated record (2 s sweep, 8-60 Hz)", "offset"),
                   (cor, "correlated: zero-phase Klauder wavelet", "offset"),
                   (kmp, "Klauder wavelet to minimum phase", "offset")], "m36_vibroseis.png", gain="AGC 0.5 s",
                  size=(1500, 650))
    env.shot("m36", vibro)

    def interpolation():
        s = shot_1020.time_window(0.0, 2.5)
        dec = s.subset(np.arange(0, s.ntr, 2))
        spz = create("fx_interpolation", gather_key="fldr", order_key="offset", window=0.5, traces=24,
                     fmax=60.0)(dec)
        rng = np.random.default_rng(11)
        missing = rng.choice(np.arange(2, s.ntr - 2), 12, replace=False)
        gap = s.subset(np.setdiff1d(np.arange(s.ntr), missing))
        pocs = create("pocs_fill", gather_key="fldr", position_key="offset", iterations=60)(gap)
        env.panel([(dec, "50 m traces (aliased)", "offset"),
                   (spz, "Spitz interpolation, 25 m", "offset"),
                   (gap, "12 traces missing", "offset"),
                   (pocs, "gaps filled by POCS", "offset")], "m37_interpolation.png", gain="AGC 0.5 s",
                  size=(1700, 650))
    env.shot("m37", interpolation)
