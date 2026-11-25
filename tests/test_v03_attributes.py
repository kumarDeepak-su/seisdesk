"""Tests of the version 0.3 attribute and noise steps: Hilbert transform, instantaneous attributes, spectral
decomposition and time-frequency maps, eigenimage and Harlan filters, phase-weighted stack, deghosting."""

import numpy as np
import pytest

from seisdesk.core import timefreq
from seisdesk.core.dataset import SeismicData
from seisdesk.core.synthetic import events_to_traces, ricker
from seisdesk.processing import attributes, deghost, separation, stack
from seisdesk.processing import REGISTRY, create

NEW = ("hilbert", "attributes", "spectral_decomp", "eigen_filter", "harlan", "deghost")


def corr(a, b):
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def test_group_e_steps_registered():
    for name in NEW:
        assert name in REGISTRY
    choices = {p.name: p for p in REGISTRY["stack"].params}["normalisation"].choices
    assert "phase-weighted" in choices and choices[0] == "live"
    assert callable(separation.eigenimage) and callable(stack.cmp_stack)


# ---------------------------------------------------------------------------------------------- attributes
def test_hilbert_of_a_cosine_is_a_sine():
    dt, nt, f = 0.002, 500, 10.0
    t = np.arange(nt) * dt
    d = SeismicData(np.cos(2 * np.pi * f * t)[None, :].astype(np.float32), dt)
    q = create("hilbert")(d).traces[0]
    mid = slice(125, 375)
    assert np.max(np.abs(q[mid] - np.sin(2 * np.pi * f * t[mid]))) < 0.05
    q2 = create("hilbert", sign="-H{x} (+90 degrees)")(d).traces[0]
    assert np.allclose(q2, -q, atol=1e-6)


def test_envelope_of_a_modulated_gaussian():
    dt, nt = 0.002, 600
    t = np.arange(nt) * dt
    g = np.exp(-0.5 * ((t - 0.6) / 0.08) ** 2)
    d = SeismicData((g * np.cos(2 * np.pi * 40.0 * t))[None, :].astype(np.float32), dt)
    env = create("attributes", attribute="envelope")(d).traces[0]
    region = g > 0.05
    assert np.max(np.abs(env[region] - g[region])) < 0.03


def test_instantaneous_frequency_of_a_chirp():
    dt, nt = 0.002, 500
    t = np.arange(nt) * dt
    f0, k = 10.0, 40.0
    d = SeismicData(np.cos(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))[None, :].astype(np.float32), dt)
    fi = create("attributes", attribute="instantaneous frequency", window=0.02)(d).traces[0]
    mid = (t > 0.2) & (t < 0.8)
    assert np.max(np.abs(fi[mid] - (f0 + k * t[mid]))) < 2.0


def test_all_attributes_are_finite_and_polarity_follows_the_event():
    dt, nt = 0.002, 500
    w, c = ricker(30.0, dt)
    tr = events_to_traces(np.array([[0.3, 0.7]]), np.array([[1.0, -1.0]]), nt, dt, w, c)
    d = SeismicData(np.vstack([tr, 0.5 * tr]).astype(np.float32), dt)
    for name in attributes.ATTRIBUTES:
        out = create("attributes", attribute=name)(d)
        assert out.traces.shape == d.traces.shape and np.all(np.isfinite(out.traces)), name
    pol = create("attributes", attribute="apparent polarity")(d).traces[0]
    assert pol[int(round(0.3 / dt))] > 0.5 and pol[int(round(0.7 / dt))] < -0.5


# ---------------------------------------------------------------------------------------------- time-frequency
@pytest.mark.parametrize("method", ["gabor", "stransform", "cwt_morlet"])
def test_time_frequency_maps_localise_a_burst(method):
    dt, nt = 0.002, 600
    t = np.arange(nt) * dt
    burst = np.exp(-0.5 * ((t - 0.6) / 0.05) ** 2) * np.cos(2 * np.pi * 30.0 * (t - 0.6))
    times, freqs, m = getattr(timefreq, method)(burst, dt, fmax=80.0, nf=160)
    amp = np.abs(m)
    assert amp.shape == (nt, freqs.size)
    it, jf = np.unravel_index(int(np.argmax(amp)), amp.shape)
    assert abs(times[it] - 0.6) < 0.03 and abs(freqs[jf] - 30.0) < 3.0


@pytest.mark.parametrize("method", ["S-transform", "STFT (Gabor)", "CWT (Morlet)"])
def test_spectral_decomposition_selects_the_frequency(method):
    dt, nt = 0.002, 500
    t = np.arange(nt) * dt
    d = SeismicData(np.vstack([np.sin(2 * np.pi * 20.0 * t), np.sin(2 * np.pi * 50.0 * t)]).astype(np.float32), dt)
    out = create("spectral_decomp", frequency=20.0, method=method, window=0.1)(d)
    mid = slice(150, 350)
    assert abs(float(np.median(out.traces[0, mid])) - 1.0) < 0.1
    assert float(np.max(out.traces[1, mid])) < 0.2
    ph = create("spectral_decomp", frequency=20.0, method=method, output="phase (degrees)")(d)
    assert np.all(np.abs(ph.traces) <= 180.0 + 1e-3)


# ---------------------------------------------------------------------------------------------- eigenimage, Harlan
def gather(ntr=20, nt=300, dt=0.004, t0=0.4, slowness=0.0, noise=0.0, seed=0):
    w, c = ricker(25.0, dt)
    off = np.arange(ntr) * 50.0
    clean = events_to_traces((t0 + slowness * off)[:, None], np.ones((ntr, 1)), nt, dt, w, c)
    noisy = clean + noise * np.random.default_rng(seed).standard_normal(clean.shape)
    heads = {"cdp": 1.0, "fldr": 1.0, "offset": off, "tracf": np.arange(1, ntr + 1)}
    return clean, SeismicData(noisy.astype(np.float32), dt, heads)


def test_eigen_filter_keeps_a_flat_event_and_removes_noise():
    clean, d = gather(noise=0.25)
    out = create("eigen_filter", rank=1, mode="pass")(d).traces
    assert np.sum((out - clean) ** 2) < 0.2 * np.sum((d.traces - clean) ** 2)
    flat = SeismicData(clean.astype(np.float32), d.dt, d.headers)
    rejected = create("eigen_filter", rank=1, mode="reject")(flat).traces
    assert np.max(np.abs(rejected)) < 1e-3 * np.max(np.abs(clean))


def test_eigen_filter_flattens_by_linear_moveout():
    clean, d = gather(t0=0.2, slowness=1.0 / 2000.0, noise=0.1)
    out = create("eigen_filter", rank=1, mode="pass", lmo_velocity=2000.0)(d).traces
    assert corr(out, clean) > 0.95


def test_harlan_separation_improves_signal_to_noise():
    dt, nt, ntr = 0.004, 256, 24
    w, c = ricker(25.0, dt)
    clean = events_to_traces((0.3 + 0.002 * np.arange(ntr))[:, None], np.ones((ntr, 1)), nt, dt, w, c)
    noisy = clean + 0.5 * np.random.default_rng(3).standard_normal(clean.shape)
    d = SeismicData(noisy.astype(np.float32), dt, {"tracf": np.arange(1, ntr + 1)})
    opts = dict(traces=24, window=0.5, max_dip=6.0, threshold=2.0, fmax=80.0)
    sig = create("harlan", **opts)(d).traces
    assert np.sum((sig - clean) ** 2) < 0.5 * np.sum((d.traces - clean) ** 2)
    noise = create("harlan", output="noise", **opts)(d).traces
    assert np.allclose(sig + noise, d.traces, atol=1e-4)


# ---------------------------------------------------------------------------------------------- phase-weighted stack
def test_phase_weighted_stack_suppresses_incoherent_noise():
    dt, nt, ntr = 0.004, 300, 24
    w, c = ricker(25.0, dt)
    clean = events_to_traces(np.full((ntr, 1), 0.4), np.ones((ntr, 1)), nt, dt, w, c)
    noisy = clean + 0.2 * np.random.default_rng(4).standard_normal(clean.shape)
    d = SeismicData(noisy.astype(np.float32), dt, {"cdp": 1.0, "offset": np.arange(ntr) * 25.0})
    lin = create("stack")(d).traces[0]
    pws = create("stack", normalisation="phase-weighted", pws_power=2.0)(d).traces[0]
    quiet = slice(150, 300)
    assert np.std(pws[quiet]) < 0.3 * np.std(lin[quiet])
    peak = int(round(0.4 / dt))
    assert pws[peak] > 0.7 * lin[peak] > 0


# ---------------------------------------------------------------------------------------------- deghosting
def test_deghost_recovers_a_ghosted_wavelet_at_vertical_incidence():
    dt, nt = 0.001, 512
    w, c = ricker(30.0, dt)
    clean = events_to_traces(np.full((3, 1), 0.2), np.ones((3, 1)), nt, dt, w, c)
    ghosted = deghost.add_ghost(clean, dt, 8.0, 1500.0, 0.95)
    d = SeismicData(ghosted.astype(np.float32), dt, {"gelev": -8.0, "fldr": 1.0, "offset": [0.0, 10.0, 20.0]})
    out = create("deghost", side="receiver", reflection=0.95, water_level=1.0)(d).traces
    for k in range(3):
        assert corr(out[k], clean[k]) > 0.9
    assert corr(ghosted[0], clean[0]) < corr(out[0], clean[0])
    with pytest.raises(ValueError):
        create("deghost", side="source")(d)                      # no sdepth header and no source depth given


def test_deghost_in_tau_p_handles_a_flat_event_like_vertical_incidence():
    dt, nt, ntr = 0.002, 400, 12
    w, c = ricker(25.0, dt)
    clean = events_to_traces(np.full((ntr, 1), 0.3), np.ones((ntr, 1)), nt, dt, w, c)
    ghosted = deghost.add_ghost(clean, dt, 10.0, 1500.0, 0.95)
    d = SeismicData(ghosted.astype(np.float32), dt, {"fldr": 1.0, "offset": np.arange(ntr) * 25.0, "gelev": -10.0})
    out = create("deghost", side="receiver", mode="angle-dependent (tau-p)", reflection=0.95, water_level=1.0,
                 fmax=60.0)(d).traces
    assert corr(out, clean) > 0.9 and corr(out, clean) > corr(ghosted, clean)


# ---------------------------------------------------------------------------------------------- dialog
def test_time_frequency_dialog_finds_the_burst(qtbot, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from seisdesk.gui import common
    from seisdesk.gui.timefreq_dialog import TimeFrequencyDialog
    monkeypatch.setattr(common, "HEADLESS", True)
    dt, nt = 0.004, 300
    t = np.arange(nt) * dt
    burst = np.exp(-0.5 * ((t - 0.5) / 0.04) ** 2) * np.cos(2 * np.pi * 25.0 * (t - 0.5))
    d = SeismicData(np.vstack([np.zeros(nt), burst]).astype(np.float32), dt, {"fldr": 1.0, "tracf": [1.0, 2.0]})
    dlg = TimeFrequencyDialog(d, None, trace=1)
    qtbot.addWidget(dlg)
    for k in range(dlg.method.count()):
        dlg.method.setCurrentIndex(k)
        res = dlg.compute()
        t_peak, f_peak = res["peak"]
        assert abs(t_peak - 0.5) < 0.03 and abs(f_peak - 25.0) < 4.0
        assert res["amplitude"].shape == (nt, dlg.NF)
