"""CMP stacking with live-sample fold normalisation, median, diversity and phase-weighted stacks."""

from __future__ import annotations

import numpy as np

from ..core.dataset import SeismicData
from scipy.ndimage import uniform_filter1d

from .attributes import analytic_signal
from .base import Param, Process, cdp_values, register


def cmp_stack(data: SeismicData, normalisation: str = "live", diversity_window: float = 0.2, pws_power: float = 2.0,
              pws_window: float = 0.0) -> SeismicData:
    """One trace per CMP: sum divided by the number of live (non-zero) samples, by sqrt of it or by the fold; or the
    median of the live samples; or a diversity stack weighting every sample by 1 / (local power of its trace); or a
    phase-weighted stack (Schimmel & Paulssen 1997): the live stack times |mean of exp(i instantaneous phase)| to the
    power pws_power, the phase coherence optionally smoothed over pws_window seconds."""
    cdp_values(data)
    nwin = max(1, int(round(diversity_window / data.dt)))
    npws = int(round(pws_window / data.dt))
    groups = list(data.iter_gathers("cdp", secondary=None))
    out = np.zeros((len(groups), data.nt), dtype=np.float64)
    heads = {k: np.zeros(len(groups)) for k in data.headers}
    fold = np.zeros(len(groups))
    for j, (value, idx) in enumerate(groups):
        g = data.traces[idx].astype(np.float64)
        s = g.sum(axis=0)
        if normalisation == "live":
            n = np.maximum((g != 0).sum(axis=0), 1)
            out[j] = s / n
        elif normalisation == "median":
            masked = np.where(g != 0, g, np.nan)
            live = np.any(g != 0, axis=0)
            out[j, live] = np.nanmedian(masked[:, live], axis=0)
        elif normalisation == "diversity":
            power = uniform_filter1d(g * g, nwin, axis=1, mode="nearest")
            w = np.where((g != 0) & (power > 0), 1.0 / np.maximum(power, 1e-30), 0.0)
            wsum = w.sum(axis=0)
            out[j] = np.where(wsum > 0, (w * g).sum(axis=0) / np.maximum(wsum, 1e-30), 0.0)
        elif normalisation == "phase-weighted":
            live = g != 0
            n = np.maximum(live.sum(axis=0), 1)
            z = analytic_signal(g)
            mag = np.abs(z)
            unit = np.where(live & (mag > 0), z / np.maximum(mag, 1e-30), 0.0)
            coherence = np.abs(unit.sum(axis=0)) / n                  # 1 where all live phases agree
            if npws > 1:
                coherence = uniform_filter1d(coherence, npws, mode="nearest")
            out[j] = (s / n) * coherence ** pws_power
        elif normalisation == "sqrt":
            out[j] = s / np.sqrt(np.maximum((g != 0).sum(axis=0), 1))
        else:
            out[j] = s / idx.size
        for k in heads:
            heads[k][j] = data.headers[k][idx[0]]
        fold[j] = idx.size
    heads["cdp"] = np.array([v for v, _ in groups])
    heads["offset"] = np.zeros(len(groups))
    heads["nhs"] = fold
    if "cdpx" in heads and np.any(heads["cdpx"]):
        heads["sx"] = heads["cdpx"].copy()
        heads["gx"] = heads["cdpx"].copy()
    heads["tracl"] = np.arange(1, len(groups) + 1, dtype=np.float64)
    st = SeismicData(out.astype(np.float32), data.dt, heads, data.t0, data.name + " stack", data.domain,
                     list(data.history), dict(data.meta))
    st.meta["stacked"] = True
    return st


@register
class Stack(Process):
    name = "stack"
    label = "CMP stack"
    category = "Stack"
    description = ("Stack every CMP gather into one trace; fold goes to header nhs. 'live' divides the sum by the "
                   "number of non-zero samples, 'median' takes the median (robust to noise bursts), 'diversity' "
                   "weights each sample by the inverse power of its trace in a sliding window (noisy traces count "
                   "less), 'phase-weighted' multiplies the live stack by the coherence of the instantaneous phases "
                   "to a power (incoherent noise is suppressed strongly, aligned events are kept).")
    params = (Param("normalisation", "choice", "live", "Method", choices=("live", "sqrt", "fold", "median",
                                                                         "diversity", "phase-weighted")),
              Param("diversity_window", "float", 0.2, "Diversity power window", unit="s", minimum=0.004),
              Param("pws_power", "float", 2.0, "Phase-weighted stack power", minimum=0.0),
              Param("pws_window", "float", 0.0, "Phase coherence smoothing window (0 = none)", unit="s",
                    minimum=0.0))

    def run(self, data, context):
        return cmp_stack(data, self["normalisation"], self["diversity_window"], self["pws_power"],
                         self["pws_window"])
