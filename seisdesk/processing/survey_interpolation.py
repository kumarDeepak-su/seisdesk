"""Geometry-aware whole-survey 3D / 5D Fourier reconstruction."""
from math import prod
import uuid

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..core.reconstruction import reconstruct_survey, reconstruction_budget
from .base import Param, Process, register


@register
class SurveyPOCS(Process):
    name, label, category = "survey_pocs", "Survey interpolation (3D / 5D POCS)", "Interpolation"
    description = ("Reconstruct missing midpoint bins across a whole 3D stack or 5D prestack survey "
                   "(midpoint x,y; full signed offset x,y; time). Presence or an explicit live header defines "
                   "observations; zero-valued live traces stay unchanged. Input coordinates must lie within the "
                   "configured bin tolerance. Recorded samples are preserved; duplicate bins require an explicit "
                   "mean policy. New traces have interp=1. Geometry is binned and original geometry retained in "
                   "diagnostics. New station IDs and varying elevations/statics are unknown (zero); assign these "
                   "before shot-domain processing. CPU Fourier POCS assumes coherent, sparsely represented events "
                   "and periodic spatial support; inspect held-out traces and acquisition gaps before interpretation.")
    params = (
        Param("mode", "choice", "3D stack", "Survey domain", choices=("3D stack", "5D prestack")),
        Param("dx", "float", 25, "Midpoint x bin", unit="m", minimum=.001),
        Param("dy", "float", 25, "Midpoint y bin", unit="m", minimum=.001),
        Param("dhx", "float", 50, "Full offset x bin (5D)", unit="m", minimum=.001),
        Param("dhy", "float", 50, "Full offset y bin (5D)", unit="m", minimum=.001),
        Param("x0", "float", 0, "Midpoint x grid reference", unit="m"),
        Param("y0", "float", 0, "Midpoint y grid reference", unit="m"),
        Param("hx0", "float", 0, "Full offset x grid reference", unit="m"),
        Param("hy0", "float", 0, "Full offset y grid reference", unit="m"),
        Param("tolerance", "float", .1, "Maximum bin mismatch (fraction of each spacing)", minimum=0, maximum=.49),
        Param("live_key", "key", "", "Live header (>0 observed; empty = all present)"),
        Param("duplicates", "choice", "error", "Multiple traces in one bin", choices=("error", "mean")),
        Param("max_gap", "float", 3, "Maximum output distance to an observed bin", unit="bins", minimum=.1,
              help="Euclidean distance in spacing-normalised midpoint/offset coordinates. No output beyond the observed axis bounds."),
        Param("iterations", "int", 80, "Iterations", minimum=2, maximum=1000),
        Param("final_threshold", "float", .01, "Final spectral threshold / initial maximum", minimum=.000001, maximum=.999),
        Param("frequency_block", "int", 8, "Frequencies in memory at once", minimum=1, maximum=256),
        Param("memory_gb", "float", 2, "Working memory budget", unit="GiB", minimum=.01),
    )

    def run(self, data, context):
        if data.domain != "time" or data.ntr < 2 or not np.isfinite(data.traces).all():
            raise ValueError("survey interpolation requires finite time-domain traces")
        if not np.isfinite(self["memory_gb"]) or self["memory_gb"] <= 0:
            raise ValueError("give a finite positive working memory budget")
        five = self["mode"] == "5D prestack"
        if five:
            sx, sy, gx, gy = [data.header(key, None) for key in ("sx", "sy", "gx", "gy")]
            coords = np.column_stack(((sy+gy)/2, (sx+gx)/2, gy-sy, gx-sx))
            spacing = np.array([self["dy"], self["dx"], self["dhy"], self["dhx"]])
            origin = np.array([self["y0"], self["x0"], self["hy0"], self["hx0"]])
        else:
            coords = np.column_stack((data.header("cdpy", None), data.header("cdpx", None)))
            spacing, origin = np.array([self["dy"], self["dx"]]), np.array([self["y0"], self["x0"]])
        if not np.isfinite(np.r_[coords.ravel(), spacing, origin]).all():
            raise ValueError("bin spacings, reference coordinates and trace geometry must be finite")
        live = data.header(self["live_key"], None) > 0 if self["live_key"] else np.ones(data.ntr, bool)
        selected = np.flatnonzero(live)
        if len(selected) < 2:
            raise ValueError("at least two explicitly live input traces are required")
        relative = (coords[selected]-origin)/spacing
        rounded = np.rint(relative)
        if np.any(np.abs(relative-rounded) > self["tolerance"]+1e-9):
            raise ValueError("trace coordinates exceed bin tolerance; correct grid spacing/reference or geometry")
        if np.any(np.abs(rounded) > 2**50):
            raise ValueError("coordinate/bin ratio is too large; check coordinate units")
        index = rounded.astype(np.int64)
        lower, upper = index.min(axis=0), index.max(axis=0)
        shape = tuple(int(v) for v in upper-lower+1); count = prod(shape)
        budget = reconstruction_budget(shape, data.nt, len(selected), self["frequency_block"])
        if budget > self["memory_gb"]*1024**3:
            raise ValueError(f"survey grid {shape} needs approximately {budget/1024**3:.3f} GiB; crop or coarsen it")
        flat = np.ravel_multi_index((index-lower).T, shape)
        occupied, first, inverse, counts = np.unique(flat, return_index=True, return_inverse=True, return_counts=True)
        if np.any(counts > 1) and self["duplicates"] == "error":
            raise ValueError("multiple traces occupy the same bin; select mean explicitly or refine geometry")
        observations = np.zeros((len(occupied), data.nt), np.float32)
        np.add.at(observations, inverse, data.traces[selected])
        observations /= counts[:, None]
        known = np.zeros(count, bool); known[occupied] = True
        distance = distance_transform_edt(~known.reshape(shape)).ravel()
        keep = np.flatnonzero(distance <= self["max_gap"])
        context.log(f"Survey POCS: grid {shape}, {len(occupied)}/{count} occupied bins; "
                    f"{len(keep)-len(occupied)} output gaps; working estimate {budget/1024**3:.3f} GiB")
        output = reconstruct_survey(observations, occupied, shape, iterations=self["iterations"],
                                     final_threshold=self["final_threshold"], frequency_block=self["frequency_block"],
                                     memory_gb=self["memory_gb"], work_dir=context.work_dir, report=context.report)
        positions = (np.column_stack(np.unravel_index(keep, shape))+lower)*spacing + origin
        nout = len(keep); observed_rows = np.searchsorted(keep, occupied)
        headers = {}
        for key, values in data.headers.items():
            # Preserve measured headers at occupied bins, but do not invent varying
            # source IDs, elevations or statics for a predicted trace.
            headers[key] = np.full(nout, values[selected[0]] if np.all(values[selected] == values[selected[0]]) else 0.)
            headers[key][observed_rows] = values[selected[first]]
        my, mx = positions[:, 0], positions[:, 1]
        hy, hx = (positions[:, 2], positions[:, 3]) if five else (np.zeros(nout), np.zeros(nout))
        headers.update(cdpx=mx, cdpy=my, sx=mx-hx/2, sy=my-hy/2, gx=mx+hx/2, gy=my+hy/2,
                       offset=np.hypot(hx, hy), offset_x=hx, offset_y=hy,
                       tracl=np.arange(nout)+1, interp=(~known[keep]).astype(float),
                       interp_distance=distance[keep], bin_fold=np.zeros(nout),
                       geometry_inferred=(~known[keep]).astype(float))
        headers["bin_fold"][observed_rows] = counts
        prefix = uuid.uuid4().hex
        identities = [f"{prefix}:{i}" for i in range(nout)]
        single = counts == 1
        for out_index, source_index in zip(observed_rows[single], selected[first[single]]):
            identities[out_index] = str(data.trace_ids[source_index])
        out = data.copy(output[keep], headers=headers, trace_ids=identities, name=data.name+" survey POCS")
        original = dict(trace_ids=data.trace_ids[selected], coordinates=coords[selected], bin_indices=flat)
        result = dict(shape=shape, spacing=spacing, origin=origin+lower*spacing, occupied=occupied,
                      output_bins=keep, original_geometry=original, bin_counts=counts,
                      observed_fold=known.reshape(shape).sum(axis=(2, 3)) if five else known.reshape(shape).astype(int),
                      generated_fold=(~known & (distance <= self["max_gap"])).reshape(shape).sum(axis=(2, 3))
                          if five else (~known & (distance <= self["max_gap"])).reshape(shape).astype(int),
                      working_bytes_estimate=budget, max_bin_shift_m=np.max(np.abs(coords[selected]-(index*spacing+origin)), axis=0))
        context.results[self.name] = result
        out.meta["survey_reconstruction"] = dict(mode=self["mode"], shape=list(shape),
            spacing=spacing.tolist(), origin=result["origin"].tolist(), observed_bins=len(occupied),
            generated_traces=nout-len(occupied), duplicate_policy=self["duplicates"], max_gap_bins=self["max_gap"],
            original_geometry_result=self.name,
            assumptions="Binned Fourier POCS; periodic spatial support; predicted station IDs and varying elevation/statics unknown")
        return out
