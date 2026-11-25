"""Time-frequency analysis of seismic traces: Gabor short-time Fourier transform, S-transform (Stockwell, Mansinha &
Lowe 1996) and Morlet continuous wavelet transform (Grossmann & Morlet 1984), all computed as Gaussian band-pass
filters in the frequency domain.

At frequency f every method keeps the analytic (positive-frequency) part of the trace around f with a Gaussian
frequency window of standard deviation sigma_f:

- Gabor (STFT): constant, 1 / (2 pi sigma_t) with sigma_t a quarter of the window length (the window is +-2 sigma_t);
- S-transform: f / (2 pi), i.e. a time window one period wide;
- Morlet CWT: f / w0 (the S-transform is the Morlet transform with w0 = 2 pi, apart from the phase reference).

Amplitudes are normalised so that a unit-amplitude sinusoid at f has amplitude 1 at f.
"""

from __future__ import annotations

import numpy as np

METHODS = ("S-transform", "STFT (Gabor)", "CWT (Morlet)")


def sigma_f(method: str, f: float, window: float = 0.1, w0: float = 6.0) -> float:
    """Standard deviation (Hz) of the Gaussian frequency window of a method at frequency f."""
    if method.startswith("S-"):
        return max(abs(f), 1e-6) / (2.0 * np.pi)
    if method.startswith("CWT") or method.lower().startswith("morlet"):
        return max(abs(f), 1e-6) / max(w0, 1e-6)
    sigma_t = max(window, 1e-6) / 4.0
    return 1.0 / (2.0 * np.pi * sigma_t)


def _fft_length(nt: int, dt: float, sf: float) -> int:
    """FFT length with room for the filter's impulse response: 4 time standard deviations on each side, at most two
    trace lengths (beyond that the wrap-around of very low frequencies is accepted)."""
    sigma_t = 1.0 / (2.0 * np.pi * max(sf, 1e-6))
    pad = min(int(np.ceil(4.0 * sigma_t / dt)), 2 * nt)
    return int(2 ** np.ceil(np.log2(max(nt + 2 * pad, 2))))


def _positive_weights(freqs: np.ndarray) -> np.ndarray:
    """2 for positive frequencies, 1 at zero, 0 for negative ones: the analytic part, scaled to keep amplitudes."""
    return np.where(freqs > 0, 2.0, np.where(freqs == 0, 1.0, 0.0))


def band_component(traces: np.ndarray, dt: float, f: float, sf: float) -> np.ndarray:
    """Complex analytic component of every trace (ntr, nt) around frequency f, Gaussian frequency window of standard
    deviation sf (Hz). Its modulus is the time-frequency amplitude at f, its angle the phase of that component."""
    x = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    nt = x.shape[1]
    n = _fft_length(nt, dt, sf)
    freqs = np.fft.fftfreq(n, dt)
    w = _positive_weights(freqs) * np.exp(-0.5 * ((freqs - f) / sf) ** 2)
    return np.fft.ifft(np.fft.fft(x, n, axis=1) * w[None, :], axis=1)[:, :nt]


def tf_map(trace: np.ndarray, dt: float, freqs: np.ndarray, method: str = "S-transform", window: float = 0.1,
           w0: float = 6.0) -> np.ndarray:
    """Complex time-frequency map (nt, nf) of one trace at the given frequencies (Hz)."""
    x = np.asarray(trace, dtype=np.float64).ravel()
    nt = x.size
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    out = np.zeros((nt, freqs.size), dtype=np.complex128)
    if nt == 0 or freqs.size == 0:
        return out
    sfs = np.array([sigma_f(method, f, window, w0) for f in freqs])
    n = _fft_length(nt, dt, float(sfs.min()))                  # room for the narrowest frequency window
    fax = np.fft.fftfreq(n, dt)
    spec = np.fft.fft(x, n)
    pos = _positive_weights(fax)
    for j, (f, sf) in enumerate(zip(freqs, sfs)):
        out[:, j] = np.fft.ifft(spec * pos * np.exp(-0.5 * ((fax - f) / sf) ** 2))[:nt]
    return out


def frequency_axis(dt: float, fmin: float = 0.0, fmax: float = 0.0, nf: int = 128, allow_zero: bool = False):
    """nf frequencies from fmin (or one step above zero) to fmax (0 = Nyquist), kept below the Nyquist frequency."""
    nyq = 0.5 / dt
    hi = min(fmax, nyq) if fmax > 0 else nyq
    hi = min(hi, 0.999 * nyq)
    nf = max(int(nf), 2)
    lo = fmin if fmin > 0 else (0.0 if allow_zero else hi / nf)
    lo = min(lo, 0.5 * hi)
    return np.linspace(lo, hi, nf)


def gabor(trace: np.ndarray, dt: float, window: float = 0.1, fmin: float = 0.0, fmax: float = 0.0, nf: int = 128,
          t0: float = 0.0):
    """Gabor transform (short-time Fourier transform with a Gaussian window of length +-2 standard deviations):
    (times, freqs, complex map (nt, nf))."""
    freqs = frequency_axis(dt, fmin, fmax, nf, allow_zero=True)
    m = tf_map(trace, dt, freqs, "STFT (Gabor)", window=window)
    return t0 + np.arange(m.shape[0]) * dt, freqs, m


def stransform(trace: np.ndarray, dt: float, fmin: float = 0.0, fmax: float = 0.0, nf: int = 128, t0: float = 0.0):
    """S-transform (Stockwell, Mansinha & Lowe 1996): (times, freqs, complex map (nt, nf))."""
    freqs = frequency_axis(dt, fmin, fmax, nf)
    m = tf_map(trace, dt, freqs, "S-transform")
    return t0 + np.arange(m.shape[0]) * dt, freqs, m


def cwt_morlet(trace: np.ndarray, dt: float, w0: float = 6.0, fmin: float = 0.0, fmax: float = 0.0, nf: int = 128,
               t0: float = 0.0):
    """Continuous wavelet transform with the Morlet wavelet of centre frequency w0 (radians), scales converted to
    frequencies f = w0 / (2 pi scale): (times, freqs, complex map (nt, nf))."""
    freqs = frequency_axis(dt, fmin, fmax, nf)
    m = tf_map(trace, dt, freqs, "CWT (Morlet)", w0=w0)
    return t0 + np.arange(m.shape[0]) * dt, freqs, m
