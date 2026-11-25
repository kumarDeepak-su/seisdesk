"""Small processing utilities with explicit sampling, geometry and amplitude conventions."""
import os

import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import uniform_filter1d

from ..core.dataset import SeismicData
from ..core.segy_io import read_any
from .base import Param, Process, register


def fractional_filter(traces, dt, order, phase=True):
    n = next_fast_len(2 * traces.shape[1])
    w = 2 * np.pi * np.fft.rfftfreq(n, dt)
    multiplier = np.zeros(w.size, dtype=complex)
    multiplier[1:] = w[1:] ** order * (np.exp(0.5j * np.pi * order) if phase else 1)
    multiplier[0] = 1 if order == 0 else 0
    return np.fft.irfft(np.fft.rfft(traces, n, axis=1) * multiplier, n, axis=1)[:, :traces.shape[1]]


@register
class Taper(Process):
    name, label, category = "taper", "Taper (trace ends and panel edges)", "Editing"
    description = "Cosine, linear or Gaussian edge tapers; fractions refer to each end of the selected axis."
    params = (Param("time", "float", 0.05, "Fraction at each trace end", minimum=0, maximum=0.5),
              Param("space", "float", 0.0, "Fraction at each panel edge", minimum=0, maximum=0.5),
              Param("shape", "choice", "cosine", "Shape", choices=("cosine", "linear", "gaussian")),
              Param("gather_key", "key", "fldr", "Panel gather header"))

    def run(self, data, context):
        def window(n, fraction):
            m = min(n // 2, int(round(n * fraction)))
            w = np.ones(n)
            if m:
                q = np.linspace(0, 1, m) if m > 1 else np.zeros(1)
                ramp = q if self["shape"] == "linear" else (np.sin(q * np.pi / 2) ** 2 if self["shape"] == "cosine"
                                                            else np.exp(-4.5 * (1 - q) ** 2))
                w[:m], w[-m:] = ramp, ramp[::-1]
            return w
        out = data.traces * window(data.nt, self["time"])
        for _, idx in data.iter_gathers(self["gather_key"]):
            out[idx] *= window(len(idx), self["space"])[:, None]
        return data.copy(out.astype(np.float32))


@register
class FractionalDerivative(Process):
    name, label, category = "frac_derivative", "Fractional derivative / integral", "Filtering"
    description = "Zero-padded Fourier multiplier (i omega)^order. Negative orders integrate with the DC term removed."
    params = (Param("order", "float", 0.5, "Derivative order", minimum=-3, maximum=3),
              Param("phase", "bool", True, "Include derivative phase rotation"))
    def run(self, data, context):
        return data.copy(fractional_filter(data.traces, data.dt, self["order"], self["phase"]).astype(np.float32))


def ordered_panels(data, key, xkey):
    if xkey not in data.headers:
        raise ValueError(f"panel spacing requires the {xkey} header")
    for _, idx in data.iter_gathers(key, xkey):
        spacing = np.diff(data.header(xkey)[idx])
        if len(idx) < 2 or np.any(spacing <= 0) or not np.allclose(spacing, np.median(spacing), rtol=0.02, atol=1e-6):
            raise ValueError("spatial filtering requires distinct, regularly spaced traces within each panel")
        yield idx, float(np.median(spacing))


PANEL = (Param("gather_key", "key", "fldr", "Panel gather header"),
         Param("x_key", "key", "gx", "Position header (metres)"))


@register
class WavenumberFilter(Process):
    name, label, category = "kfilter", "2D Fourier panel filter", "Filtering"
    description = ("Rectangular pass band in spatial wavenumber and sample-axis frequency, or radial "
                   "wavenumber band on depth sections, with cosine edges. Time data use Hz on the second axis; "
                   "depth data use cycles/metre on both axes. Radial mode uses kmin/kmax. Regular 2D panels only.")
    params = PANEL + (Param("mode", "choice", "box", "Pass band", choices=("box", "radial (depth)")),
                      Param("kmin", "float", 0, "Minimum |spatial wavenumber|", unit="cycles/m", minimum=0),
                      Param("kmax", "float", 0.02, "Maximum |spatial wavenumber|", unit="cycles/m", minimum=1e-9),
                      Param("fmin", "float", 0, "Minimum |sample-axis frequency|", minimum=0),
                      Param("fmax", "float", 100, "Maximum |sample-axis frequency|", minimum=1e-9),
                      Param("taper", "float", 0.1, "Transition fraction", minimum=0, maximum=0.5))

    def run(self, data, context):
        if self["mode"] == "radial (depth)" and data.domain != "depth":
            raise ValueError("radial wavenumber filtering requires a depth section (both axes in metres)")
        def band(axis, lo, hi):
            if hi <= lo:
                raise ValueError("pass-band maxima must exceed minima")
            a = np.abs(axis); w = ((a >= lo) & (a <= hi)).astype(float)
            edge = (hi - lo) * self["taper"]
            if edge:
                w *= np.sin(np.clip((hi - a) / edge, 0, 1) * np.pi / 2) ** 2
                if lo > 0:
                    w *= np.sin(np.clip((a - lo) / edge, 0, 1) * np.pi / 2) ** 2
            return w
        out = data.traces.copy()
        for idx, spacing in ordered_panels(data, self["gather_key"], self["x_key"]):
            spec = np.fft.fft2(data.traces[idx])
            kx, kt = np.fft.fftfreq(len(idx), spacing), np.fft.fftfreq(data.nt, data.dt)
            if self["mode"] == "radial (depth)":
                weight = band(np.hypot(kx[:, None], kt[None, :]), self["kmin"], self["kmax"])
            else:
                weight = band(kx, self["kmin"], self["kmax"])[:, None] * band(kt, self["fmin"], self["fmax"])[None, :]
            out[idx] = np.fft.ifft2(spec * weight).real
        return data.copy(out)


@register
class FractionalWavenumber(Process):
    name, label, category = "kfrac", "Fractional spatial-wavenumber power", "Filtering"
    description = "Multiply every panel's spatial spectrum by |2 pi k|^order; zero wavenumber is removed for nonzero order."
    params = PANEL + (Param("order", "float", 1.0, "Spatial derivative order", minimum=-3, maximum=3),)
    def run(self, data, context):
        out = data.traces.copy()
        for idx, spacing in ordered_panels(data, self["gather_key"], self["x_key"]):
            k = np.abs(2 * np.pi * np.fft.fftfreq(len(idx), spacing))
            w = np.zeros_like(k); w[k > 0] = k[k > 0] ** self["order"]
            w[0] = 1 if self["order"] == 0 else 0
            out[idx] = np.fft.ifft(np.fft.fft(data.traces[idx], axis=0) * w[:, None], axis=0).real
        return data.copy(out)


def fx_mix(traces, dt, width, method, trim=0.2, threshold=0.5, cutoff=10.0):
    width = max(1, int(width) | 1)
    spec = np.fft.rfft(traces, axis=1)
    pad = np.pad(spec, ((width // 2, width // 2), (0, 0)), mode="edge")
    result = np.empty_like(spec)
    for i in range(spec.shape[0]):
        neighbours = pad[i:i + width]
        if method == "alpha trim":
            order = np.argsort(np.abs(neighbours), axis=0)
            trim_count = min((width - 1) // 2, int(width * trim))
            ranked = np.take_along_axis(neighbours, order, axis=0)
            result[i] = ranked[trim_count:width - trim_count].mean(axis=0)
        elif method == "correlation":
            cross = uniform_filter1d(neighbours * np.conj(spec[i]), 5, axis=1)
            power = uniform_filter1d(np.abs(neighbours) ** 2, 5, axis=1)
            central = uniform_filter1d(np.abs(spec[i]) ** 2, 5)
            coherence = np.abs(cross) / np.sqrt(np.maximum(power * central, 1e-30))
            weights = np.where(coherence >= threshold, coherence, 0)
            result[i] = (neighbours * weights).sum(axis=0) / np.maximum(weights.sum(axis=0), 1e-30)
        else:
            weights = np.exp(-0.5 * ((np.arange(width) - width // 2) / max(width / 4, 1)) ** 2)
            result[i] = (neighbours * weights[:, None]).sum(axis=0) / weights.sum()
    if method == "low frequency":
        f = np.fft.rfftfreq(traces.shape[1], dt)
        blend = 0.5 * (1 + np.cos(np.clip((f - 0.8 * cutoff) / (0.2 * cutoff), 0, 1) * np.pi))
        result = blend * result + (1 - blend) * spec
    return np.fft.irfft(result, n=traces.shape[1], axis=1)


class _FXMix(Process):
    category = "Filtering"
    params = (Param("gather_key", "key", "fldr", "Panel gather header"),
              Param("width", "int", 5, "Trace window (odd)", minimum=1, maximum=101),
              Param("trim", "float", 0.2, "Fraction trimmed from each end (alpha trim)", minimum=0, maximum=0.45),
              Param("threshold", "float", 0.5, "Minimum correlation", minimum=0, maximum=1),
              Param("cutoff", "float", 10.0, "Low-frequency cutoff", unit="Hz", minimum=0.01))
    def run(self, data, context):
        out = data.traces.copy()
        for idx, _ in ordered_panels(data, self["gather_key"], "gx"):
            out[idx] = fx_mix(data.traces[idx], data.dt, self["width"], self.method, self["trim"], self["threshold"], self["cutoff"])
        return data.copy(out)


@register
class FXCorrelation(_FXMix):
    name, label, method = "fx_cc_filter", "f-x correlation-weighted filter", "correlation"
    description = "Average neighbouring complex spectra only where their local frequency correlation exceeds the threshold."


@register
class FXAlphaTrim(_FXMix):
    name, label, method = "fx_alpha_trim", "f-x alpha-trimmed mean", "alpha trim"
    description = "Trim low and high spectral amplitudes in the trace window, then average the remaining complex spectra."


@register
class FXWeightedMix(_FXMix):
    name, label, method = "fx_weighted_mix", "f-x weighted trace mix", "weighted"
    description = "Gaussian spatial weights on neighbouring complex spectra. Coherent steep dips may be attenuated."


@register
class LowFrequencyArray(_FXMix):
    name, label, method = "lf_array_forming", "Low-frequency array forming", "low frequency"
    description = "Spatial trace mixing confined to the low-frequency band with a cosine transition at the cutoff."


def structure_smooth(traces, width=5, max_dip=3.0, ndip=13, time_window=7):
    g = np.asarray(traces, dtype=float); ntr, nt = g.shape
    width = max(1, int(width) | 1)
    live_trace = np.any(g != 0, axis=1)
    t = np.arange(nt)
    best, result, dips = np.full_like(g, -1), g.copy(), np.zeros_like(g)
    for dip in np.linspace(-max_dip, max_dip, ndip):
        total, energy, fold = np.zeros_like(g), np.zeros_like(g), np.zeros_like(g)
        for shift in range(-width // 2 + 1, width // 2 + 1):
            for i in range(ntr):
                j = i + shift
                if not 0 <= j < ntr or not live_trace[j]:
                    continue
                q = t + dip * shift
                live = (q >= 0) & (q <= nt - 1)
                samples = np.interp(q, t, g[j], left=0, right=0)
                total[i] += samples; energy[i] += samples ** 2; fold[i] += live
        semblance = uniform_filter1d(total ** 2, time_window, axis=1) / np.maximum(
            uniform_filter1d(fold * energy, time_window, axis=1), 1e-30)
        keep = semblance > best
        result[keep] = (total / np.maximum(fold, 1))[keep]
        dips[keep], best[keep] = dip, semblance[keep]
    return result, dips


@register
class StructureSmooth(Process):
    name, label, category = "structure_smooth", "Structure-oriented smoothing (local semblance)", "Filtering"
    description = "Scan local slopes in samples/trace and average along the most coherent slope; regular ordered 2D panels."
    params = (Param("gather_key", "key", "fldr", "Panel gather header"),
              Param("width", "int", 5, "Neighbour trace window", minimum=3, maximum=21),
              Param("max_dip", "float", 3, "Maximum slope", unit="samples/trace", minimum=0, maximum=20),
              Param("ndip", "int", 13, "Slope trials", minimum=3, maximum=81),
              Param("time_window", "int", 7, "Semblance sample window", minimum=1, maximum=101),
              Param("dead_only", "bool", False, "Interpolate dead traces only"))
    def run(self, data, context):
        out = data.traces.copy()
        groups = list(ordered_panels(data, self["gather_key"], "gx"))
        for k, (idx, _) in enumerate(groups):
            context.report(k / max(len(groups), 1))
            smoothed, _ = structure_smooth(data.traces[idx], self["width"], self["max_dip"], self["ndip"], self["time_window"])
            if self["dead_only"]:
                dead = np.max(np.abs(data.traces[idx]), axis=1) == 0
                out[idx[dead]] = smoothed[dead]
            else:
                out[idx] = smoothed
        return data.copy(out)


@register
class TraceWeight(Process):
    name, label, category = "trace_weight", "Weight traces by a header", "Gain"
    description = "Multiply every trace by scale times its selected numeric header."
    params = (Param("key", "key", "nhs", "Weight header"), Param("scale", "float", 1.0, "Scale"))
    def run(self, data, context):
        return data.copy(data.traces * (data.header(self["key"], None) * self["scale"])[:, None])


@register
class HeaderMath(Process):
    name, label, category = "header_math", "Trace arithmetic with header values", "Utility"
    description = "Add, subtract, multiply or divide samples by a trace-header value; zero divisors are rejected."
    params = (Param("key", "key", "nhs", "Header"),
              Param("operation", "choice", "multiply", "Operation", choices=("add", "subtract", "multiply", "divide")))
    def run(self, data, context):
        h = data.header(self["key"], None)[:, None]
        if self["operation"] == "divide" and np.any(h == 0):
            raise ValueError("header contains zero divisors")
        op = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply, "divide": np.divide}[self["operation"]]
        return data.copy(op(data.traces, h).astype(np.float32))


@register
class DatasetMath(Process):
    name, label, category = "dataset_math", "Combine with another dataset", "Utility"
    description = "Trace-aligned arithmetic with a SEG-Y/SU dataset; sample axes and source/receiver geometry must agree."
    params = (Param("file", "infile", "", "Other dataset"),
              Param("operation", "choice", "difference", "Operation", choices=("difference", "sum", "product", "ratio")))
    def run(self, data, context):
        other = read_any(os.path.join(context.work_dir, self["file"]))
        if (other.traces.shape != data.traces.shape or other.domain != data.domain
                or not np.isclose(other.dt, data.dt, rtol=1e-9, atol=0)
                or not np.isclose(other.t0, data.t0, rtol=0, atol=data.dt * 1e-6)):
            raise ValueError("datasets must have identical trace counts and sample axes")
        for key in ("fldr", "cdp", "offset", "sx", "sy", "gx", "gy"):
            if (key in data.headers) != (key in other.headers):
                raise ValueError(f"both datasets must supply the {key} header for alignment")
            if key in data.headers and key in other.headers and not np.allclose(data.header(key), other.header(key), atol=0.01, rtol=0):
                raise ValueError(f"trace alignment differs in header {key}; sort and align datasets first")
        op = {"difference": np.subtract, "sum": np.add, "product": np.multiply, "ratio": np.divide}[self["operation"]]
        if self["operation"] == "ratio" and np.any(other.traces == 0):
            raise ValueError("other dataset contains zero divisors")
        return data.copy(op(data.traces, other.traces).astype(np.float32))


@register
class ReciprocalSum(Process):
    name, label, category = "reciprocal_sum", "Sum reciprocal source-receiver pairs", "Stack"
    description = "Sum traces whose source/receiver coordinates are interchanged; retain components separately and write fold."
    params = (Param("normalise", "bool", False, "Average instead of sum"),)
    def run(self, data, context):
        if not all(k in data.headers for k in ("sx", "gx")):
            raise ValueError("reciprocity requires source and receiver coordinates")
        coordinates = {k: data.header(k) for k in ("sx", "sy", "gx", "gy", "trid")}
        groups = {}
        for i in range(data.ntr):
            a = tuple(round(float(coordinates[k][i]), 6) for k in ("sx", "sy"))
            b = tuple(round(float(coordinates[k][i]), 6) for k in ("gx", "gy"))
            groups.setdefault((min(a, b), max(a, b), coordinates["trid"][i]), []).append(i)
        idx = np.array([g[0] for g in groups.values()])
        heads = {k: v[idx].copy() for k, v in data.headers.items()}
        heads["nhs"] = np.array([len(g) for g in groups.values()])
        traces = np.stack([data.traces[g].mean(axis=0) if self["normalise"] else data.traces[g].sum(axis=0)
                           for g in groups.values()])
        return SeismicData(traces, data.dt, heads, data.t0, data.name + " reciprocal", data.domain, list(data.history))


@register
class FractionalAutocorrelation(Process):
    name, label, category = "frac_autocorr", "Fractional autocorrelation", "Utility"
    description = "Inverse Fourier transform of power spectrum raised to an exponent; output axis is positive lag from zero."
    params = (Param("power", "float", 1.0, "Power-spectrum exponent", minimum=0.1, maximum=3),
              Param("normalise", "bool", True, "Normalise zero lag"))
    def run(self, data, context):
        n = next_fast_len(2 * data.nt)
        corr = np.fft.irfft(np.abs(np.fft.rfft(data.traces, n, axis=1)) ** (2 * self["power"]), n, axis=1)[:, :data.nt]
        if self["normalise"]:
            corr /= np.maximum(corr[:, :1], 1e-30)
        return data.copy(corr.astype(np.float32), t0=0)


@register
class DipDivergence(Process):
    name, label, category = "dip_divcor", "Approximate local-dip divergence weighting", "Gain"
    description = "Time-squared gain divided by local dip cosine, estimated from image slopes; not true-amplitude migration."
    params = PANEL + (Param("velocity", "float", 2000, "Constant velocity", unit="m/s", minimum=100),
                      Param("max_gain", "float", 10, "Maximum dip multiplier", minimum=1, maximum=100))
    def run(self, data, context):
        if data.domain != "time":
            raise ValueError("dip divergence weighting requires a time section")
        out = data.traces.copy()
        for idx, dx in ordered_panels(data, self["gather_key"], self["x_key"]):
            _, dip = structure_smooth(data.traces[idx])
            sine = np.clip(self["velocity"] * dip * data.dt / (2 * dx), -0.999, 0.999)
            factor = np.minimum(1 / np.sqrt(1 - sine ** 2), self["max_gain"])
            out[idx] *= factor * np.maximum(data.times, 0)[None, :] ** 2
        return data.copy(out)
