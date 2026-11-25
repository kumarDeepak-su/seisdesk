"""Dip moveout (Hale 1984, f-k) on NMO-corrected common-offset sections."""

from __future__ import annotations

import numpy as np

from .base import Param, Process, cdp_values, register


def hale_dmo_section(section: np.ndarray, dt: float, dx: float, half_offset: float, fmax: float | None = None,
                     t0: float = 0.0) -> np.ndarray:
    """DMO of one common-offset section (ncmp, nt), NMO-corrected, regular CMP spacing dx.

    P0(w, k) = sum_tn p(tn, k) exp(-i sqrt((w tn)^2 + (k h)^2))  — the phase-only Hale (1984) operator. At k = 0 it
    reduces to the Fourier transform (flat events are unchanged). The operator depends on k^2, so it is built once
    for |k| and applied to +k and -k."""
    ncmp, nt = section.shape
    nk = int(2 ** np.ceil(np.log2(2 * ncmp)))
    ntf = int(2 ** np.ceil(np.log2(nt + max(8, nt // 8))))
    P = np.fft.fft(section.astype(np.float64), nk, axis=0).astype(np.complex64)      # (nk, nt), FFT over CMP
    k = 2 * np.pi * np.fft.fftfreq(nk, dx)
    w = 2 * np.pi * np.fft.rfftfreq(ntf, dt)
    nw = w.size if fmax is None else int(np.searchsorted(w, 2 * np.pi * fmax, side="right"))
    tn = (t0 + np.arange(nt) * dt).astype(np.float64)
    wt2 = (np.outer(w[:nw], tn) ** 2).astype(np.float32)                             # (nw, nt); float32 halves the cost
    out_spec = np.zeros((nk, ntf // 2 + 1), dtype=np.complex64)
    for ik in range(nk // 2 + 1):
        kh2 = (k[ik] * half_offset) ** 2
        op = np.exp(-1j * np.sqrt(wt2 + np.float32(kh2)))                              # (nw, nt), complex64
        out_spec[ik, :nw] = op @ P[ik]
        if 0 < ik < nk // 2:
            out_spec[nk - ik, :nw] = op @ P[nk - ik]
    # out_spec holds S0(k, w >= 0) of a real section: invert over k first (giving s0(x, w >= 0), the half spectrum
    # of each real output trace), then over w with irfft — the reverse order would treat complex k-rows as real
    if t0 != 0.0:                       # the sums used absolute times tn; put the output back on the t0 + n dt axis
        out_spec[:, :nw] *= np.exp(1j * w[:nw] * t0).astype(np.complex64)[None, :]
    s_xw = np.fft.ifft(out_spec, axis=0)
    return np.fft.irfft(s_xw, ntf, axis=1)[:ncmp, :nt].astype(np.float32)


@register
class DMO(Process):
    name = "dmo"
    label = "DMO (Hale f-k)"
    category = "Moveout"
    description = ("Dip moveout on NMO-corrected data, offset bin by offset bin (common-offset sections on the regular "
                   "CMP grid). Follow with inverse NMO / velocity analysis / NMO, then stack.")
    params = (Param("offset_bin", "float", 0.0, "Offset bin width (0 = offset step of the fullest CMP)", unit="m",
                    minimum=0.0),
              Param("fmax", "float", 90.0, "Highest frequency", unit="Hz", minimum=1.0))

    def run(self, data, context):
        cdp = cdp_values(data)
        geom = data.meta.get("geometry", {})
        bin_size = float(geom.get("bin_size", 0.0))
        if bin_size <= 0:
            cx = np.unique(data.header("cdpx"))
            bin_size = float(np.median(np.diff(cx))) if cx.size > 1 else 0.0
        if bin_size <= 0:
            raise ValueError("CMP spacing unknown: run 2D geometry first")
        off = np.abs(data.header("offset"))
        width = self["offset_bin"]
        if width <= 0:                      # bins as wide as the offset step inside a CMP fill every CMP of a section
            fullest = max(data.iter_gathers("cdp", None), key=lambda g: g[1].size)[1]
            steps = np.diff(np.sort(off[fullest]))
            steps = steps[steps > 1e-6]
            width = float(np.median(steps)) if steps.size else 50.0
        obin = np.floor(off / width).astype(np.int64)
        cmin, cmax = int(cdp.min()), int(cdp.max())
        ncmp = cmax - cmin + 1
        out = data.traces.copy()
        bins = np.unique(obin)
        for n, b in enumerate(bins):
            idx = np.flatnonzero(obin == b)
            col = (cdp[idx] - cmin).astype(np.int64)
            section = np.zeros((ncmp, data.nt), dtype=np.float32)
            count = np.zeros(ncmp)
            np.add.at(section, col, data.traces[idx])
            np.add.at(count, col, 1.0)
            filled = count > 0
            section[filled] /= count[filled][:, None]
            h = 0.5 * float(np.mean(off[idx]))
            dmo = hale_dmo_section(section, data.dt, bin_size, h, self["fmax"], data.t0)
            out[idx] = dmo[col]
            context.report((n + 1) / bins.size)
        return data.copy(out)
