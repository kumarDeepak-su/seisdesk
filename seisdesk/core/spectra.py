"""Spectral analysis helpers: amplitude spectrum, usable band, band-pass suggestion, autocorrelation."""

from __future__ import annotations

import numpy as np


def amplitude_spectrum(traces: np.ndarray, dt: float):
    """Average amplitude spectrum of the traces (Hann taper over the window). Returns (frequencies, dB re peak,
    linear amplitude)."""
    tr = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    n = tr.shape[1]
    nfft = int(2 ** np.ceil(np.log2(max(n, 2)) + 1))
    window = np.hanning(n) if n > 8 else np.ones(n)
    amp = np.abs(np.fft.rfft((tr - tr.mean(axis=1, keepdims=True)) * window, nfft, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(nfft, dt)
    db = 20.0 * np.log10(np.maximum(amp, 1e-30) / max(float(amp.max()), 1e-30))
    return freqs, db, amp


def spectrum_band(freqs: np.ndarray, db: np.ndarray, level_db: float) -> tuple[float, float]:
    """The contiguous band around the spectral peak where the amplitude stays above ``level_db`` (e.g. -20)."""
    k = int(np.argmax(db))
    lo = k
    while lo > 0 and db[lo - 1] >= level_db:
        lo -= 1
    hi = k
    while hi < db.size - 1 and db[hi + 1] >= level_db:
        hi += 1
    return float(freqs[lo]), float(freqs[hi])


def suggest_bandpass(freqs: np.ndarray, db: np.ndarray) -> tuple[float, float, float, float]:
    """Ormsby corners from the -20 dB band: the pass band is the -20 dB band, tapers of 30 % below and 20 % above."""
    lo, hi = spectrum_band(freqs, db, -20.0)
    nyq = float(freqs[-1])
    f2, f3 = max(lo, 1.0), min(hi, 0.9 * nyq)
    f1 = max(0.0, 0.7 * f2)
    f4 = min(1.2 * f3, nyq)
    return round(f1, 1), round(f2, 1), round(f3, 1), round(f4, 1)


def autocorrelation(traces: np.ndarray, dt: float, max_lag: float = 0.3):
    """Average normalised autocorrelation for lags 0..max_lag (s). Returns (lags, values)."""
    tr = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    n = tr.shape[1]
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(tr - tr.mean(axis=1, keepdims=True), nfft, axis=1)
    acf = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)
    acf = acf / np.maximum(acf[:, :1], 1e-30)
    m = min(n - 1, int(round(max_lag / dt)))
    return np.arange(m + 1) * dt, acf[:, : m + 1].mean(axis=0)


def first_zero_crossing(lags: np.ndarray, acf: np.ndarray) -> float | None:
    s = np.flatnonzero((acf[:-1] > 0) & (acf[1:] <= 0))
    if s.size == 0:
        return None
    i = int(s[0])
    return float(lags[i] + (lags[i + 1] - lags[i]) * acf[i] / (acf[i] - acf[i + 1]))
