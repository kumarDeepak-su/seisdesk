"""Vibroseis: sweep generation, correlation of uncorrelated records with the sweep (parametric, pilot trace or file),
convolution with a sweep, and conversion of the zero-phase Klauder wavelet to minimum phase."""

from __future__ import annotations

import numpy as np

from .base import Param, Process, register
from .wavelet import _nfft, apply_spectrum, min_phase_of, read_wavelet


def sweep(f1: float, f2: float, length: float, dt: float, kind: str = "linear", taper_start: float = 0.0,
          taper_end: float = 0.0, db_per_octave: float = 3.0) -> np.ndarray:
    """Vibroseis sweep sin(phase(t)), 0 <= t < length, with cosine tapers at both ends.

    linear: instantaneous frequency f1 + (f2 - f1) t / T, a flat power spectrum. 'dB per octave' (Goupillaud 1976):
    the sweep rate falls as f^-p with p = dB / (10 log10 2), so the power spectrum rises by that many dB per octave;
    f(t) = (f1^q + (f2^q - f1^q) t / T)^(1/q) with q = p + 1."""
    n = max(2, int(round(length / dt)))
    t = np.arange(n) * dt
    total = n * dt
    if kind == "linear" or f1 <= 0 or f2 <= 0:
        finst = f1 + (f2 - f1) * t / total
    else:
        q = db_per_octave / (10.0 * np.log10(2.0)) + 1.0
        if abs(q) < 1e-6:
            finst = f1 * (f2 / f1) ** (t / total)
        else:
            finst = (f1 ** q + (f2 ** q - f1 ** q) * t / total) ** (1.0 / q)
    phase = 2.0 * np.pi * np.concatenate(([0.0], np.cumsum(0.5 * (finst[1:] + finst[:-1]) * dt)))
    s = np.sin(phase)
    for m, end in ((int(round(taper_start / dt)), False), (int(round(taper_end / dt)), True)):
        m = min(m, n)
        if m > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
            if end:
                s[n - m:] *= ramp[::-1]
            else:
                s[:m] *= ramp
    return s


def klauder(s: np.ndarray) -> tuple[np.ndarray, int]:
    """Klauder wavelet: the autocorrelation of the sweep, peak normalised to 1; returns it and its centre index."""
    s = np.asarray(s, dtype=np.float64)
    n = s.size
    nfft = _nfft(2 * n)
    acf = np.fft.irfft(np.abs(np.fft.rfft(s, nfft)) ** 2, nfft)
    k = np.concatenate([acf[nfft - (n - 1):], acf[:n]]) if n > 1 else acf[:1]
    peak = float(acf[0])
    return (k / peak if peak > 0 else k), n - 1


SWEEP_PARAMS = (Param("sweep", "choice", "parametric", "Sweep", choices=("parametric", "pilot trace", "sweep file")),
                Param("f1", "float", 10.0, "Start frequency", unit="Hz", minimum=0.0),
                Param("f2", "float", 80.0, "End frequency", unit="Hz", minimum=0.0),
                Param("length", "float", 8.0, "Sweep length", unit="s", minimum=0.01),
                Param("kind", "choice", "linear", "Sweep type", choices=("linear", "dB per octave")),
                Param("db_per_octave", "float", 3.0, "Boost (dB per octave)", unit="dB"),
                Param("taper_start", "float", 0.3, "Start taper", unit="s", minimum=0.0),
                Param("taper_end", "float", 0.3, "End taper", unit="s", minimum=0.0),
                Param("pilot_key", "key", "trid", "Pilot trace header"),
                Param("pilot_value", "float", 6.0, "Pilot trace header value"),
                Param("gather_key", "key", "fldr", "One pilot per gather of (empty = first only)"),
                Param("sweep_file", "infile", "", "Sweep file (text)"),
                Param("file_dt", "float", 0.0, "Sample interval of a one-column file (0 = data)", unit="ms",
                      minimum=0.0))


def resolve_sweeps(values: dict, data) -> tuple[list, np.ndarray, np.ndarray]:
    """The sweeps and, for every trace, the index of the sweep it uses; also the indices of pilot traces."""
    mode = values["sweep"]
    none = np.zeros(0, dtype=np.int64)
    if mode == "parametric":
        s = sweep(values["f1"], values["f2"], values["length"], data.dt, values["kind"], values["taper_start"],
                  values["taper_end"], values["db_per_octave"])
        return [s], np.zeros(data.ntr, dtype=np.int64), none
    if mode == "sweep file":
        s, _t0 = read_wavelet(values["sweep_file"], data.dt, values["file_dt"] / 1000.0)
        return [s], np.zeros(data.ntr, dtype=np.int64), none
    key, value = values["pilot_key"], values["pilot_value"]
    pilots = np.flatnonzero(np.abs(data.header(key) - value) <= 1e-6)
    if pilots.size == 0:
        raise ValueError(f"no pilot trace with {key} = {value:g}")
    use = np.zeros(data.ntr, dtype=np.int64)
    gk = values["gather_key"]
    if gk and pilots.size > 1:
        g = data.header(gk)
        first = {}
        for n, k in enumerate(pilots):
            first.setdefault(float(g[k]), n)
        use = np.array([first.get(float(v), 0) for v in g], dtype=np.int64)
    return [data.traces[k].astype(np.float64) for k in pilots], use, pilots


@register
class VibroCorrelation(Process):
    name = "vibro_correlation"
    label = "Vibroseis correlation"
    category = "Vibroseis"
    description = ("Cross-correlates every uncorrelated vibroseis record with the sweep, which compresses each "
                   "sweep-long reflection into a short zero-phase Klauder wavelet at the reflection time. The sweep is "
                   "built from its parameters (linear or dB-per-octave, with tapers), taken from a pilot trace in the "
                   "data (e.g. trid 6, one per gather) or read from a file. The output keeps the listen time (record "
                   "length minus sweep length unless given) and is scaled by the sweep energy, so a reflection keeps "
                   "its amplitude.")
    params = SWEEP_PARAMS + (Param("output_length", "float", 0.0, "Output length (0 = record minus sweep)", unit="s",
                                   minimum=0.0),
                             Param("remove_pilot", "bool", True, "Remove the pilot traces"))

    def run(self, data, context):
        sweeps, use, pilots = resolve_sweeps(self.values, data)
        ns = max(s.size for s in sweeps)
        if self["output_length"] > 0:
            n_out = int(round(self["output_length"] / data.dt))
        else:
            n_out = data.nt - ns
        n_out = int(min(max(n_out, 1), data.nt))
        nfft = _nfft(data.nt + ns)
        spec = np.fft.rfft(data.traces.astype(np.float64), nfft, axis=1)
        out = np.zeros((data.ntr, n_out))
        for k, s in enumerate(sweeps):
            rows = np.flatnonzero(use == k)
            if rows.size == 0:
                continue
            energy = float(np.sum(s * s))
            if energy <= 0:
                raise ValueError("the sweep is zero")
            c = np.fft.irfft(spec[rows] * np.conj(np.fft.rfft(s, nfft))[None, :], nfft, axis=1)
            out[rows] = c[:, :n_out] / energy
        result = data.copy(out.astype(np.float32))
        if pilots.size and self["remove_pilot"]:
            keep = np.ones(data.ntr, dtype=bool)
            keep[pilots] = False
            result = result.subset(keep)
        context.log(f"    correlated with {len(sweeps)} sweep(s) of {ns * data.dt:g} s; output {n_out * data.dt:g} s")
        return result


@register
class VibroConvolve(Process):
    name = "vibro_convolve"
    label = "Convolve with a sweep (uncorrelated records)"
    category = "Vibroseis"
    description = ("Convolves every trace with the sweep, turning reflectivity or correlated data into uncorrelated "
                   "vibroseis records (the record grows by the sweep length). For tests, teaching and checking the "
                   "correlation parameters.")
    params = SWEEP_PARAMS

    def run(self, data, context):
        sweeps, use, _pilots = resolve_sweeps(self.values, data)
        ns = max(s.size for s in sweeps)
        n_out = data.nt + ns - 1
        nfft = _nfft(n_out)
        spec = np.fft.rfft(data.traces.astype(np.float64), nfft, axis=1)
        out = np.zeros((data.ntr, n_out))
        for k, s in enumerate(sweeps):
            rows = np.flatnonzero(use == k)
            if rows.size:
                out[rows] = np.fft.irfft(spec[rows] * np.fft.rfft(s, nfft)[None, :], nfft, axis=1)[:, :n_out]
        return data.copy(out.astype(np.float32))


@register
class KlauderMinPhase(Process):
    name = "klauder_min_phase"
    label = "Klauder wavelet to minimum phase"
    category = "Vibroseis"
    description = ("Correlated vibroseis data carry the zero-phase Klauder wavelet, but predictive deconvolution "
                   "assumes a minimum-phase wavelet. This all-pass filter W_min / K turns the Klauder wavelet K (the "
                   "autocorrelation of the sweep) into its minimum-phase equivalent W_min (Gibson & Larner 1984); run "
                   "it after correlation and before deconvolution, with the same sweep.")
    params = SWEEP_PARAMS + (Param("water_level", "float", 0.1, "Stabilisation (water level)", unit="%",
                                   minimum=0.0),)

    def run(self, data, context):
        sweeps, use, _pilots = resolve_sweeps(self.values, data)
        nfft = _nfft(2 * data.nt + max(s.size for s in sweeps))
        wl = self["water_level"] / 100.0
        out = data.traces.astype(np.float64).copy()
        for k, s in enumerate(sweeps):
            rows = np.flatnonzero(use == k)
            if rows.size == 0:
                continue
            power = np.abs(np.fft.rfft(s, nfft)) ** 2
            power /= max(float(power.max()), 1e-30)
            wmin = min_phase_of(power, max(wl, 1e-6))
            h = wmin / (power + max(wl, 1e-6))
            out[rows] = apply_spectrum(out[rows], h, nfft)
        return data.copy(out.astype(np.float32))
