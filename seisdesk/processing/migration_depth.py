"""Depth migration: split-step Fourier (post-stack, v(x, z)) and Kirchhoff prestack depth migration (grid traveltimes,
from a flat surface or from topography)."""

from __future__ import annotations

import numpy as np

from ..core.dataset import SeismicData
from ..core.velocity import DepthModel
from .base import Param, Process, model_grid, pad_to_time_zero, register, resolve_velocity, trace_positions
from .migration_time import _section_spacing, rho_filter
from .traveltime import traveltime_tables, traveltime_tables_xz


def split_step(section: np.ndarray, dt: float, dx: float, vel: np.ndarray, dz: float, fmax: float | None = None,
               progress=None) -> np.ndarray:
    """Exploding-reflector split-step Fourier depth migration; vel (nz, nx) on the section's trace positions."""
    nx, nt = section.shape
    nz = vel.shape[0]
    nxp = int(2 ** np.ceil(np.log2(2 * nx)))
    ntp = int(2 ** np.ceil(np.log2(2 * nt)))
    w = 2 * np.pi * np.fft.rfftfreq(ntp, dt)
    nw = w.size if fmax is None else int(np.searchsorted(w, 2 * np.pi * fmax, side="right"))
    w = w[:nw]
    D = np.zeros((nxp, nw), dtype=np.complex128)
    D[:nx] = np.fft.rfft(section.astype(np.float64), ntp, axis=1)[:, :nw]
    kx = 2 * np.pi * np.fft.fftfreq(nxp, dx)[:, None]
    weights = np.ones(nw)
    weights[0] = 0.5
    image = np.zeros((nz, nx))
    u_full = np.ones((nxp,)) * 0.5 * vel[0].mean()
    for iz in range(nz):
        image[iz] = 2.0 * np.real(D[:nx] @ weights) / ntp
        u = 0.5 * vel[iz]
        u_full[:nx] = u
        u_full[nx:] = u[-1]
        u_ref = float(np.mean(u))
        Dk = np.fft.fft(D, axis=0)
        kz2 = (w[None, :] / u_ref) ** 2 - kx ** 2
        prop = kz2 > 0
        Dk = np.where(prop, Dk * np.exp(1j * np.sqrt(np.where(prop, kz2, 0.0)) * dz), 0.0)
        D = np.fft.ifft(Dk, axis=0)
        D *= np.exp(1j * w[None, :] * dz * (1.0 / u_full[:, None] - 1.0 / u_ref))
        if progress is not None and iz % 20 == 0:
            progress(iz / nz)
    return image


def _model_for(data: SeismicData, context, nz: int, dz: float, source: str, smooth: float) -> tuple[DepthModel, np.ndarray]:
    """The depth model (from the context, or built from the velocity picks by Dix) and the trace x positions."""
    cdp = data.header("cdp")
    ucdp, first = np.unique(cdp, return_index=True)
    x = trace_positions(data)[first]
    if source == "project model" and context.depth_model is not None:
        return context.depth_model, x
    picks = resolve_velocity(source if source != "project model" else "project", context)
    x0, dx, nx = model_grid(x)
    model = DepthModel.from_picks(picks, dict(zip(ucdp.tolist(), x.tolist())), ucdp, nz, dz, nx, dx, x0,
                                  tmax=max(data.tmax, 1.0), smooth_m=smooth)
    if context.depth_model is None:          # never replace a model the user built or loaded
        context.depth_model = model
    return model, x


@register
class SplitStepMigration(Process):
    name = "split_step"
    label = "Split-step depth migration (post-stack)"
    category = "Depth migration"
    description = ("Depth migration of a stacked section in an interval-velocity model v(x, z) — the project depth "
                   "model, or one built by Dix conversion of the velocity picks. Output is a depth section.")
    params = (Param("velocity", "velocity", "project model", "Velocity model", "'project model', picks or a file"),
              Param("dz", "float", 10.0, "Depth step", unit="m", minimum=0.5),
              Param("zmax", "float", 4000.0, "Maximum depth", unit="m", minimum=10.0),
              Param("smooth", "float", 200.0, "Smoothing of a Dix model", unit="m", minimum=0.0),
              Param("fmax", "float", 60.0, "Highest frequency", unit="Hz", minimum=1.0))

    def run(self, data, context):
        data = pad_to_time_zero(data)
        nz = int(self["zmax"] / self["dz"]) + 1
        model, x = _model_for(data, context, nz, self["dz"], self["velocity"], self["smooth"])
        vel = np.stack([model.at(np.full(x.size, z), x) for z in np.arange(nz) * self["dz"]])
        img = split_step(data.traces, data.dt, _section_spacing(data), vel, self["dz"], self["fmax"], context.report)
        out = SeismicData(img.T.astype(np.float32), self["dz"], {k: v.copy() for k, v in data.headers.items()}, 0.0,
                          data.name + " depth", "depth", list(data.history), dict(data.meta))
        out.meta["migrated"] = "split_step"
        return out


def surface_depth(xg: np.ndarray, sx: np.ndarray, gx: np.ndarray, selev: np.ndarray, gelev: np.ndarray):
    """The reference (highest ground) elevation and the depth of the ground below it at the grid columns xg, from the
    source and receiver positions and elevations (averaged where several share a position)."""
    xs = np.concatenate([sx, gx]).astype(np.float64)
    es = np.concatenate([selev, gelev]).astype(np.float64)
    ux, inv = np.unique(xs, return_inverse=True)
    ue = np.bincount(inv.ravel(), weights=es, minlength=ux.size) / np.bincount(inv.ravel(), minlength=ux.size)
    z_ref = float(es.max())
    return z_ref, z_ref - np.interp(xg, ux, ue)


@register
class KirchhoffPSDM(Process):
    name = "kirchhoff_psdm"
    label = "Kirchhoff depth migration (prestack)"
    category = "Depth migration"
    description = ("Prestack Kirchhoff depth migration with first-arrival traveltimes computed on the velocity grid "
                   "(shortest paths); output a depth image, or image gathers by offset class. Recording surface "
                   "'topography' migrates from the real source and receiver elevations (selev - sdepth, gelev) "
                   "instead of a flat datum: depth is then measured below the highest ground elevation, the model "
                   "starts there, cells above the ground take the velocity just below it, and the image above the "
                   "ground is blank.")
    params = (Param("velocity", "velocity", "project model", "Velocity model", "'project model', picks or a file"),
              Param("dz", "float", 10.0, "Image depth step", unit="m", minimum=0.5),
              Param("zmax", "float", 3000.0, "Maximum depth", unit="m", minimum=10.0),
              Param("grid_dx", "float", 25.0, "Traveltime grid spacing", unit="m", minimum=1.0),
              Param("aperture", "float", 2000.0, "Aperture (half width)", unit="m", minimum=10.0),
              Param("smooth", "float", 200.0, "Smoothing of a Dix model", unit="m", minimum=0.0),
              Param("output", "choice", "image", "Output", choices=("image", "image gathers")),
              Param("offset_class", "float", 250.0, "Offset class width (gathers)", unit="m", minimum=1.0),
              Param("surface", "choice", "flat", "Recording surface",
                    "flat: sources and receivers at z = 0 (data corrected to a flat datum); topography: at their "
                    "elevations, depth below the highest elevation", choices=("flat", "topography")))

    def run(self, data, context):
        nz = int(self["zmax"] / self["dz"]) + 1
        model, x_out = _model_for(data, context, nz, self["dz"], self["velocity"], self["smooth"])
        sx, gx = data.header("sx"), data.header("gx")
        gdx = self["grid_dx"]
        xmin = min(float(sx.min()), float(gx.min()), float(x_out.min())) - gdx
        xmax = max(float(sx.max()), float(gx.max()), float(x_out.max())) + gdx
        xg = np.arange(xmin, xmax + gdx, gdx)
        zg = np.arange(0.0, self["zmax"] + gdx, gdx)
        vg = np.stack([model.at(np.full(xg.size, z), xg) for z in zg])
        z_ref, zsurf = None, None
        if self["surface"] == "topography":
            selev, gelev = data.header("selev"), data.header("gelev")
            if np.any(selev) or np.any(gelev):
                z_ref, zsurf = surface_depth(xg, sx, gx, selev, gelev)
                for j in range(xg.size):                 # cells above the ground: velocity of the first cell below
                    k = int(min(np.ceil(zsurf[j] / gdx - 1e-9), zg.size - 1))
                    if k > 0:
                        vg[:k, j] = vg[k, j]
            else:
                context.log("    no elevations (selev, gelev) in the headers: flat surface used")
        context.log(f"    traveltime grid {zg.size} x {xg.size}, {np.unique(np.concatenate([sx, gx])).size} surface positions")
        if z_ref is None:
            zs = zr = np.zeros(data.ntr)
            tables = traveltime_tables(vg, gdx, gdx, xmin, np.concatenate([sx, gx]))
        else:
            zs = z_ref - (data.header("selev") - data.header("sdepth"))
            zr = z_ref - data.header("gelev")
            context.log(f"    topography: depth below elevation {z_ref:g} m, ground {zsurf.min():.1f}-{zsurf.max():.1f} m down")
            tables = traveltime_tables_xz(vg, gdx, gdx, xmin, np.concatenate([sx, gx]), np.concatenate([zs, zr]))

        def node(xpos, zpos):
            col = int(np.clip(np.rint((xpos - xmin) / gdx), 0, xg.size - 1))
            if z_ref is None:
                return col
            return int(np.clip(np.rint(zpos / gdx), 0, zg.size - 1)), col
        # image grid: depth samples dz, x = output CMP positions; tables interpolated in z (linear) at those columns
        z_img = np.arange(nz) * self["dz"]
        col_of_x = np.clip(np.rint((x_out - xmin) / gdx).astype(np.int64), 0, xg.size - 1)
        zpos = z_img / gdx
        z0 = np.clip(np.floor(zpos).astype(np.int64), 0, zg.size - 2)
        zf = zpos - z0

        def table_at(key):
            t = tables[key]
            tt = t[:, col_of_x]
            return (1 - zf)[:, None] * tt[z0] + zf[:, None] * tt[z0 + 1]            # (nz, nx_out)

        filt = rho_filter(data.traces.astype(np.float64), data.dt)
        edges = None
        if self["output"] == "image gathers":
            omax = float(np.max(np.abs(gx - sx))) + self["offset_class"]
            edges = np.arange(0.0, omax + self["offset_class"], self["offset_class"])
        image = np.zeros((nz, x_out.size))
        cig = None if edges is None else np.zeros((edges.size - 1, nz, x_out.size))
        cache = {}
        max_cache = max(16, int(5e7 // max(1, nz * x_out.size)))         # about 400 MB of float64 tables
        for i in range(data.ntr):
            mid = 0.5 * (sx[i] + gx[i])
            cols = np.flatnonzero(np.abs(x_out - mid) <= self["aperture"])
            if cols.size == 0:
                continue
            pair = []
            for px, pz in ((sx[i], zs[i]), (gx[i], zr[i])):
                k = node(px, pz)
                tab = cache.get(k)
                if tab is None:
                    tab = table_at(k)
                    if len(cache) >= max_cache:
                        cache.pop(next(iter(cache)))
                    cache[k] = tab
                pair.append(tab)
            t = pair[0][:, cols] + pair[1][:, cols]
            pos = (t - data.t0) / data.dt
            ok = np.isfinite(pos) & (pos >= 0) & (pos < data.nt - 1)
            i0 = np.clip(np.floor(np.where(ok, pos, 0)).astype(np.int64), 0, data.nt - 2)
            fr = np.where(ok, pos - i0, 0.0)
            val = (1 - fr) * filt[i, i0] + fr * filt[i, i0 + 1]
            contrib = np.where(ok, val, 0.0)
            image[:, cols] += contrib
            if cig is not None:
                b = int(np.searchsorted(edges, abs(gx[i] - sx[i]), side="right") - 1)
                if 0 <= b < cig.shape[0]:
                    cig[b][:, cols] += contrib
            if i % 200 == 0:
                context.report(i / data.ntr)
        if zsurf is not None:                            # nothing is imaged above the ground
            above = z_img[:, None] < np.interp(x_out, xg, zsurf)[None, :]
            image[above] = 0.0
            if cig is not None:
                cig[:, above] = 0.0
        ucdp = np.unique(data.header("cdp"))
        if cig is None:
            heads = {"cdp": ucdp, "cdpx": x_out, "sx": x_out, "gx": x_out, "offset": np.zeros(x_out.size)}
            out = SeismicData(image.T.astype(np.float32), self["dz"], heads, 0.0, data.name + " PSDM", "depth",
                              list(data.history), dict(data.meta))
        else:
            nb = cig.shape[0]
            traces = np.transpose(cig, (2, 0, 1)).reshape(-1, nz)
            centres = 0.5 * (edges[:-1] + edges[1:])
            heads = {"cdp": np.repeat(ucdp, nb), "cdpx": np.repeat(x_out, nb), "offset": np.tile(centres, ucdp.size)}
            out = SeismicData(traces.astype(np.float32), self["dz"], heads, 0.0, data.name + " PSDM gathers", "depth",
                              list(data.history), dict(data.meta))
        out.meta["migrated"] = "kirchhoff_psdm"
        if z_ref is not None:
            out.meta["depth_reference_elevation"] = z_ref
        return out
