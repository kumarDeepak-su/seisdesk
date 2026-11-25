"""Automatic trace editing (dead and noisy traces) and despiking of amplitude bursts."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from ..core.qc import trace_rms
from .base import Param, Process, register


@register
class AutoEdit(Process):
    name = "auto_edit"
    label = "Automatic trace editing (dead / noisy traces)"
    category = "Editing"
    description = ("Compares the RMS amplitude of every trace in a window with the median of its gather: traces below "
                   "'low' x median are dead or weak, above 'high' x median noisy. They are zeroed (trid = 2), removed "
                   "or only flagged. The RMS goes to header 'qc_rms' and the edited traces are listed in the log.")
    params = (Param("gather_key", "key", "fldr", "Gather key (median per gather)"),
              Param("tmin", "float", 0.0, "Window start", unit="s"),
              Param("tmax", "float", 0.0, "Window end (0 = trace end)", unit="s"),
              Param("low", "float", 0.05, "Dead / weak below (x gather median)", minimum=0.0),
              Param("high", "float", 4.0, "Noisy above (x gather median)", minimum=1.0),
              Param("action", "choice", "zero", "Action", choices=("zero", "remove", "flag only")))

    def run(self, data, context):
        rms = trace_rms(data, self["tmin"], self["tmax"])
        ratio = np.ones(data.ntr)
        for _, idx in data.iter_gathers(self["gather_key"], None):
            med = float(np.median(rms[idx]))
            ratio[idx] = rms[idx] / med if med > 0 else 0.0
        dead = ratio < self["low"]
        noisy = ratio > self["high"]
        bad = dead | noisy
        out = data.copy()
        out.set_header("qc_rms", rms)
        flag = np.zeros(data.ntr)
        flag[dead] = 1.0
        flag[noisy] = 2.0
        out.set_header("qc_flag", flag)
        g, c = data.header(self["gather_key"]), data.header("tracf")
        listed = ", ".join(f"{g[i]:g}/{c[i]:g}" for i in np.flatnonzero(bad)[:30])
        context.log(f"    {dead.sum()} dead/weak and {noisy.sum()} noisy traces of {data.ntr}"
                    + (f" ({self['gather_key']}/channel: {listed}{' ...' if bad.sum() > 30 else ''})" if bad.any() else ""))
        context.results["auto_edit"] = {"dead": np.flatnonzero(dead).tolist(), "noisy": np.flatnonzero(noisy).tolist()}
        if self["action"] == "remove":
            return out.subset(~bad)
        if self["action"] == "zero":
            out.traces[bad] = 0.0
            trid = out.header("trid", 1.0).copy()
            trid[bad] = 2.0
            out.set_header("trid", trid)
        return out


@register
class Despike(Process):
    name = "despike"
    label = "Despike / noise-burst suppression"
    category = "Editing"
    description = ("Compares each trace's short-window RMS envelope with the running median of the envelopes of its "
                   "neighbours (same gather, same time); where it exceeds threshold x median, the samples are scaled "
                   "down to the median level. Removes spikes, amplitude bursts and swell noise, keeps coherent events.")
    params = (Param("gather_key", "key", "fldr", "Gather key"),
              Param("window", "float", 0.05, "Envelope window", unit="s", minimum=0.002),
              Param("neighbours", "int", 7, "Traces in the running median (odd)", minimum=3),
              Param("threshold", "float", 3.0, "Threshold (x median envelope)", minimum=1.0))

    def run(self, data, context):
        out = data.traces.astype(np.float64).copy()
        nwin = max(1, int(round(self["window"] / data.dt)))
        scaled = 0
        secondary = "offset" if data.has_header("offset") else None
        for _, idx in data.iter_gathers(self["gather_key"], secondary):
            if idx.size < 3:
                continue
            g = out[idx]
            env = np.sqrt(np.maximum(uniform_filter1d(g * g, nwin, axis=1, mode="nearest"), 0.0))
            k = min(self["neighbours"], idx.size)
            k = k if k % 2 else k - 1
            med = median_filter(env, size=(max(k, 1), 1), mode="nearest")
            floor = 1e-9 * max(float(env.max()), 1e-30)
            ratio = env / np.maximum(med, floor)
            scale = np.where(ratio > self["threshold"], np.maximum(med, floor) / np.maximum(env, floor), 1.0)
            scaled += int((scale < 1.0).sum())
            out[idx] = g * scale
        context.log(f"    {scaled} samples scaled down ({100.0 * scaled / max(out.size, 1):.3f} % of the data)")
        return data.copy(out.astype(np.float32))
