"""Tests of the version 0.3 wavelet, vibroseis and trace interpolation steps."""

import numpy as np
from scipy.signal import hilbert

from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.processing.filters import ormsby_response
from seisdesk.processing import Context, REGISTRY, create
from seisdesk.processing import interpolation, vibroseis, wavelet  # noqa: F401
from seisdesk.processing.filters import bandpass
from seisdesk.processing.surface import minimum_phase

NEW = ("phase_rotation", "zero_phase", "min_phase", "shaping_filter", "signature_decon", "homomorphic_decon",
       "vibro_correlation", "vibro_convolve", "klauder_min_phase", "fx_interpolation", "pocs_fill")
DT = 0.002


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def min_phase_wavelet(f0=25.0, n=120, dt=DT, nfft=1024):
    """Minimum-phase wavelet with the amplitude spectrum of a Ricker wavelet (built the way the steps do)."""
    w, c = ricker(f0, dt)
    buf = np.zeros(nfft)
    buf[:w.size] = w
    amp = np.abs(np.fft.rfft(np.roll(buf, -c)))
    return np.fft.irfft(minimum_phase(np.log(amp + 1e-4 * amp.max())), nfft)[:n]


def best_corr(a, b, maxlag=25):
    """Correlation of two panels at the best common time shift (within maxlag samples)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    inner = slice(maxlag, a.shape[1] - maxlag)
    return max(corr(a[:, inner], np.roll(b, lag, axis=1)[:, inner]) for lag in range(-maxlag, maxlag + 1))


def spikes(ntr=16, nt=500, seed=0, nspk=8):
    rng = np.random.default_rng(seed)
    r = np.zeros((ntr, nt))
    for i in range(ntr):
        pos = rng.choice(np.arange(60, nt - 130), nspk, replace=False)
        r[i, pos] = rng.choice([-1.0, 1.0], nspk) * rng.uniform(0.5, 1.0, nspk)
    return r


def convolve_rows(r, w, centre=0):
    nt = r.shape[1]
    nfft = int(2 ** np.ceil(np.log2(nt + w.size)))
    buf = np.zeros(nfft)
    buf[:w.size] = w
    spec = np.fft.rfft(np.roll(buf, -centre))
    return np.fft.irfft(np.fft.rfft(r, nfft, axis=1) * spec, nfft, axis=1)[:, :nt]


def dataset(traces, dt=DT, **heads):
    ntr = traces.shape[0]
    h = {"fldr": 1.0, "tracf": np.arange(1, ntr + 1), "offset": np.arange(ntr) * 10.0, "trid": 1.0}
    h.update(heads)
    return SeismicData(np.asarray(traces, dtype=np.float32), dt, h)


def test_new_steps_registered():
    for name in NEW:
        assert name in REGISTRY
    assert "offset" in interpolation.POSITION_KEYS


# ---------------------------------------------------------------------------------------------- phase tools
def test_phase_rotation():
    t = np.arange(1000) * DT
    out = create("phase_rotation", angle=60.0)(dataset(np.cos(2 * np.pi * 20.0 * t)[None, :])).traces[0]
    expected = np.cos(2 * np.pi * 20.0 * t + np.deg2rad(60.0))
    assert np.max(np.abs(out[200:800] - expected[200:800])) < 0.05
    w, c = ricker(30.0, DT)
    tr = np.zeros((1, 300))
    tr[0, 150 - c:150 + c + 1] = w
    assert np.allclose(create("phase_rotation", angle=180.0)(dataset(tr)).traces[0], -tr[0], atol=1e-3)


def test_zero_phase_conversion(tmp_path):
    wm = min_phase_wavelet()
    r = spikes(seed=1)
    data = dataset(convolve_rows(r, wm))
    target = convolve_rows(r, *ricker(25.0, DT))
    path = tmp_path / "wavelet.txt"
    np.savetxt(path, wm)
    known = create("zero_phase", mode="wavelet file", wavelet_file=str(path), water_level=0.01)(data)
    assert corr(known.traces, target) > 0.95
    est = create("zero_phase", mode="data wavelet", tmin=0.0, tmax=1.0, smoothing=3.0)(data)
    assert corr(est.traces, target) > 0.85
    single = np.zeros((1, 400))
    single[0, 100:100 + wm.size] = wm
    out = create("zero_phase", mode="per trace")(dataset(single)).traces[0].astype(np.float64)
    p = int(np.argmax(np.abs(out)))
    seg = out[p - 40:p + 41]
    assert corr(seg, seg[::-1]) > 0.98
    assert corr(np.abs(np.fft.rfft(out, 1024)), np.abs(np.fft.rfft(single[0], 1024))) > 0.99


def test_min_phase_conversion():
    wm = min_phase_wavelet()
    w25, c25 = ricker(25.0, DT)
    r = spikes(seed=2)
    data = dataset(convolve_rows(r, w25, c25))
    est = create("min_phase", mode="data wavelet", tmin=0.0, tmax=1.0, smoothing=3.0)(data)
    assert corr(est.traces, convolve_rows(r, wm)) > 0.85
    single = np.zeros((1, 400))
    single[0, 200 - c25:200 + c25 + 1] = w25
    out = create("min_phase", mode="per trace", water_level=0.01)(dataset(single)).traces[0]
    env = np.abs(hilbert(single[0], N=1024)[:400])                 # as the step computes it
    onset = int(np.argmax(env >= 0.1 * env.max()))
    expected = np.zeros(400)
    expected[onset:onset + wm.size] = wm[:400 - onset]
    assert corr(out, expected) > 0.95


# ---------------------------------------------------------------------------------------------- shaping and decon
def test_shaping_filter(tmp_path):
    wm = min_phase_wavelet()
    r = spikes(seed=3)
    data = dataset(convolve_rows(r, wm))
    target = convolve_rows(r, *ricker(25.0, DT))
    path = tmp_path / "input_wavelet.txt"
    np.savetxt(path, wm)
    ctx = Context()
    out = create("shaping_filter", source="wavelet file", wavelet_file=str(path), desired="zero-phase Ricker",
                 ricker_f0=25.0, operator_length=0.3, prewhitening=0.01)(data, ctx)
    assert corr(out.traces, target) > 0.9
    assert len(ctx.results["shaping_filter"]["filter"]) == 150
    est = create("shaping_filter", source="estimated from the data", tmin=0.0, tmax=1.0, smoothing=3.0,
                 desired="zero-phase Ricker", ricker_f0=25.0, operator_length=0.3)(data)
    assert corr(est.traces, target) > 0.8


def test_signature_decon_from_file_and_from_traces(tmp_path):
    wm = min_phase_wavelet()
    r = spikes(seed=4)
    data = dataset(convolve_rows(r, wm))
    target = bandpass(r, DT, (5.0, 10.0, 40.0, 55.0))
    path = tmp_path / "signature.txt"
    np.savetxt(path, np.column_stack([np.arange(wm.size) * DT * 1000.0, wm]))       # two columns, time in ms
    out = create("signature_decon", source="wavelet file", wavelet_file=str(path), output="zero-phase band-pass",
                 corners="5,10,40,55", water_level=0.1)(data)
    assert corr(out.traces, target) > 0.95
    # two shots, each with its own near-field signature trace (trid 9) at time zero
    w40 = min_phase_wavelet(40.0)
    r2 = spikes(seed=5)
    rows, fl, trid = [], [], []
    for shot, (refl, w) in enumerate(((r[:8], wm), (r2[:8], w40))):
        sig = np.zeros(500)
        sig[:w.size] = w
        rows += [sig] + list(convolve_rows(refl, w))
        fl += [shot + 1.0] * 9
        trid += [9.0] + [1.0] * 8
    d2 = dataset(np.array(rows), fldr=np.array(fl), trid=np.array(trid))
    out2 = create("signature_decon", source="trace in the data", trace_key="trid", trace_value=9.0,
                  gather_key="fldr", output="zero-phase band-pass", corners="5,10,40,55", water_level=0.1)(d2)
    assert corr(out2.traces[1:9], bandpass(r[:8], DT, (5.0, 10.0, 40.0, 55.0))) > 0.9
    assert corr(out2.traces[10:18], bandpass(r2[:8], DT, (5.0, 10.0, 40.0, 55.0))) > 0.9


def test_homomorphic_decon_estimates_the_wavelet_spectrum():
    w, c = ricker(30.0, DT)
    w = wavelet.rotate_phase(w[None, :], 45.0)[0]                  # a mixed-phase wavelet
    r = spikes(ntr=60, nt=400, seed=7, nspk=6)
    d = dataset(convolve_rows(r, w, c))
    ctx = Context()
    out = create("homomorphic_decon", tmin=0.0, tmax=0.8, cutoff=0.03, output="spike")(d, ctx)
    est = np.asarray(ctx.results["homomorphic_decon"]["wavelet"])
    a_true = np.abs(np.fft.rfft(w, 512))
    a_est = np.abs(np.fft.rfft(est, 512))
    band = a_true > 0.2 * a_true.max()
    assert corr(a_true[band], a_est[band]) > 0.9
    assert np.all(np.isfinite(out.traces))
    # with a zero-phase band-pass as desired output the result should resemble the band-limited reflectivity; the
    # phase estimate of the cepstral method is approximate, so this checks the shape only loosely
    corners = [5.0, 10.0, 50.0, 70.0]
    zp = create("homomorphic_decon", tmin=0.0, tmax=0.8, cutoff=0.03, output="zero-phase band-pass",
                corners=corners)(d, Context()).traces
    nfft = 1024
    band_pass = ormsby_response(np.fft.rfftfreq(nfft, DT), *corners)
    target = np.fft.irfft(np.fft.rfft(r, nfft, axis=1) * band_pass, nfft, axis=1)[:, :r.shape[1]]
    assert best_corr(zp, target) > 0.75


# ---------------------------------------------------------------------------------------------- vibroseis
SWEEP = dict(f1=10.0, f2=60.0, length=0.6, taper_start=0.05, taper_end=0.05)


def test_vibroseis_convolution_and_correlation_round_trip():
    nt = 600
    r = np.zeros((3, nt))
    r[0, 100], r[1, 250], r[2, 400] = 1.0, -0.5, 0.8
    unc = create("vibro_convolve", **SWEEP)(dataset(r))
    assert unc.nt == nt + 300 - 1
    cor = create("vibro_correlation", output_length=nt * DT, **SWEEP)(unc)
    assert cor.nt == nt
    k, c = vibroseis.klauder(vibroseis.sweep(10.0, 60.0, 0.6, DT, "linear", 0.05, 0.05))
    assert corr(cor.traces, convolve_rows(r, k, c)) > 0.99
    assert int(np.argmax(np.abs(cor.traces[1]))) == 250 and cor.traces[1, 250] < 0
    assert abs(float(cor.traces[0, 100]) - 1.0) < 0.05


def test_vibroseis_pilot_trace_correlation():
    nt = 600
    s = vibroseis.sweep(10.0, 60.0, 0.6, DT, "linear", 0.05, 0.05)
    r = np.zeros((2, nt))
    r[0, 100], r[1, 300] = 1.0, 0.5
    unc = create("vibro_convolve", **SWEEP)(dataset(r))
    pilot = np.zeros((1, unc.nt))
    pilot[0, :s.size] = s
    d = dataset(np.vstack([pilot, unc.traces]), trid=np.array([6.0, 1.0, 1.0]))
    out = create("vibro_correlation", sweep="pilot trace", pilot_key="trid", pilot_value=6.0,
                 output_length=nt * DT)(d)
    ref = create("vibro_correlation", output_length=nt * DT, **SWEEP)(unc)
    assert out.ntr == 2
    assert corr(out.traces, ref.traces) > 0.999


def test_sweep_spectrum_slope():
    f = np.fft.rfftfreq(8192, DT)

    def band_db(x, lo, hi):
        p = np.abs(np.fft.rfft(x, 8192)) ** 2
        m = (f >= lo) & (f <= hi)
        return 10.0 * np.log10(p[m].mean())

    lin = vibroseis.sweep(10.0, 100.0, 4.0, DT, "linear", 0.02, 0.02)
    boost = vibroseis.sweep(10.0, 100.0, 4.0, DT, "dB per octave", 0.02, 0.02, db_per_octave=6.0)
    assert abs(band_db(lin, 80.0, 96.0) - band_db(lin, 20.0, 24.0)) < 2.0
    assert 9.0 < band_db(boost, 80.0, 96.0) - band_db(boost, 20.0, 24.0) < 15.0     # two octaves at 6 dB/octave


def test_klauder_to_minimum_phase():
    s = vibroseis.sweep(10.0, 60.0, 1.0, DT, "linear", 0.1, 0.1)
    k, c = vibroseis.klauder(s)
    r = np.zeros((1, 800))
    r[0, 300] = 1.0
    d = dataset(convolve_rows(r, k, c))
    out = create("klauder_min_phase", f1=10.0, f2=60.0, length=1.0, taper_start=0.1, taper_end=0.1,
                 water_level=0.1)(d).traces[0].astype(np.float64)
    assert np.sum(out[:295] ** 2) < 0.05 * np.sum(out ** 2)
    assert np.sum(d.traces[0, :300].astype(np.float64) ** 2) > 0.3 * np.sum(d.traces[0].astype(np.float64) ** 2)
    assert corr(np.abs(np.fft.rfft(out, 2048)), np.abs(np.fft.rfft(d.traces[0], 2048))) > 0.98


# ---------------------------------------------------------------------------------------------- interpolation
def linear_events(ntr, dx, dips, t0s, nt=256, dt=0.004, f0=25.0):
    x = np.arange(ntr) * dx
    times = np.stack([t0 + p * x for t0, p in zip(t0s, dips)], axis=1)
    w, c = ricker(f0, dt)
    return events_to_traces(times, np.ones_like(times), nt, dt, w, c), x


def test_fx_interpolation_restores_aliased_dips():
    full, x = linear_events(47, 10.0, dips=(0.0, 0.0012, -0.0005), t0s=(0.15, 0.2, 0.7))
    coarse = dataset(full[0::2], dt=0.004, offset=x[0::2], tracf=np.arange(1, 25))
    out = create("fx_interpolation", gather_key="fldr", order_key="offset", window=1.1, traces=24,
                 fmax=100.0)(coarse)
    assert out.ntr == 47
    assert np.allclose(out.traces[0::2], coarse.traces, atol=1e-6)
    assert np.allclose(out.header("offset"), x)
    assert np.all(out.header("interp")[1::2] == 1) and np.all(out.header("interp")[0::2] == 0)
    truth = full[1::2]
    spitz = corr(out.traces[1::2], truth)
    linear = corr(0.5 * (full[0:-1:2] + full[2::2]), truth)
    assert spitz > 0.85 and spitz > linear + 0.05
    assert create("fx_interpolation", factor="4", window=1.1, traces=24)(coarse).ntr == 4 * 23 + 1


def test_pocs_fills_missing_and_dead_traces():
    full, x = linear_events(48, 10.0, dips=(0.0, 0.0003, -0.0002), t0s=(0.25, 0.4, 0.65))
    missing = np.array([5, 9, 13, 17, 26, 30, 33, 38, 41, 44])
    dead = 22
    keep = np.setdiff1d(np.arange(48), missing)
    tr = full[keep].copy()
    tr[np.flatnonzero(keep == dead)[0]] = 0.0
    d = dataset(tr, dt=0.004, offset=x[keep], tracf=keep + 1.0)
    out = create("pocs_fill", gather_key="fldr", position_key="offset", iterations=60)(d)
    assert out.ntr == 48
    assert np.allclose(out.header("offset"), x)
    rebuilt = np.flatnonzero(out.header("interp") == 1)
    assert set(rebuilt.tolist()) == set(missing.tolist()) | {dead}
    assert corr(out.traces[rebuilt], full[rebuilt]) > 0.85
    live = np.setdiff1d(keep, [dead])
    assert np.allclose(out.traces[live], full[live], atol=1e-5)
