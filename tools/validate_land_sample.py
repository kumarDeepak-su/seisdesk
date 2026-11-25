"""Small public land-data checks. Run on a compute node, not the login node.

Sirente DOI 10.17632/9rrjbnvmkm.1, RFL 1 (CC BY 4.0): waveform I/O and
operator agreement. Koenigsee: independent published first-arrival picks,
including a deterministic holdout. Neither is subsurface ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from seisdesk.core.seg2 import read_seg2, read_seg2_files
from seisdesk.core.segy_io import write_segy
from seisdesk.core.tomography import NodeGrid, first_arrivals, invert
from seisdesk.core.velocity import DepthModel
from seisdesk.workflow.project import Project
from validate_field_sample import main as compare_su


def reflection(args, root):
    import obspy
    files = sorted(p for p in Path(args.reflection).glob("*.DAT") if not p.name.startswith("."))
    if len(files) != 26:
        raise ValueError("expected the 26 files from Sirente RFL 1")
    result = dict(source="https://doi.org/10.17632/9rrjbnvmkm.1", licence="CC BY 4.0",
                  credit="Torrese, Rossi, Ori, Rainone and Ormo (2019)", files=[],
                  interpretation_accuracy_validated=False)
    for path in files:
        native = read_seg2(str(path))
        reference = obspy.read(str(path), format="SEG2")
        samples = np.stack([tr.data.astype(float)*tr.stats.calib for tr in reference]).astype(np.float32)
        np.testing.assert_array_equal(native.traces, samples)
        assert native.ntr == 12
        for trace in reference:
            assert abs(native.dt-trace.stats.delta) < 1e-12
            assert float(trace.stats.seg2.DELAY) == 0
        result["files"].append(dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   ntr=native.ntr, nt=native.nt, dt=native.dt))
    data = read_seg2_files(files, name="Sirente RFL 1: original header geometry")
    sx = np.unique(data.header("sx"))
    receiver_spacing = []
    for shot in np.unique(data.header("fldr")):
        receiver_spacing.extend(np.diff(np.sort(data.header("gx")[data.header("fldr")==shot])).tolist())
    result["original_header_geometry"] = dict(source_increment_m=np.unique(np.diff(sx)).tolist(),
                                              receiver_increment_m=np.unique(receiver_spacing).tolist(),
                                              absolute_offset_m=[float(abs(data.header("offset")).min()),
                                                                 float(abs(data.header("offset")).max())])
    result["geometry_note"] = ("The repository description gives 2 m source/receiver spacing and 6 m near offset. "
                               "The original SEG-2 coordinates differ. They are retained for this software "
                               "operator comparison, not treated as surveyed truth or silently overwritten.")
    result["obspy_scaled_samples_exact"] = True
    result["ntr"], result["shots"] = data.ntr, len(files)
    midpoint = .5*(data.header("sx")+data.header("gx"))
    _, bins = np.unique(midpoint, return_inverse=True)
    data.set_header("cdp", bins+1)
    result["cdp_assignment"] = "Equal original-header midpoints grouped for operator comparison only"
    path = root/"sirente-original-headers.sgy"
    write_segy(data, str(path))
    compare_su(SimpleNamespace(input=str(path), output=str(root/"su-comparison"), su_bin=args.su_bin))
    result["su_comparison"] = json.loads((root/"su-comparison"/"report.json").read_text())
    return result


def refraction(args, root):
    source_file = Path(args.refraction)
    lines = [s.split("#", 1)[0].strip() for s in source_file.read_text().splitlines()]
    lines = [s for s in lines if s]
    count = int(lines[0])
    sensors = np.array([[float(v) for v in row.split()] for row in lines[1:1+count]])
    measurements = int(lines[1+count])
    records = np.array([[float(v) for v in row.split()] for row in lines[2+count:]])
    if sensors.shape != (63, 2) or records.shape != (714, 3) or measurements != len(records):
        raise ValueError("expected the published 63-sensor, 714-pick Koenigsee dataset")
    source_index, receiver_index = records[:, :2].astype(int).T-1
    observed = records[:, 2]
    dx, dz, zmax = .5, .25, 15.
    x0, xmax = np.floor(sensors[:, 0].min()/dx)*dx, np.ceil(sensors[:, 0].max()/dx)*dx
    x = np.arange(x0, xmax+dx*.5, dx)
    elevation = float(sensors[:, 1].max())
    surface = elevation-np.interp(x, sensors[:, 0], sensors[:, 1])
    grid = NodeGrid(x0, dx, len(x), dz, int(zmax/dz)+1, surface, radius=3)
    nodes = np.array([grid.surface_node(p) for p in sensors[:, 0]])
    source, receiver = nodes[source_index], nodes[receiver_index]
    if np.any(source == receiver):
        raise ValueError("grid merged a recorded source/receiver pair")
    initial = 500+200*np.maximum(grid.z[:, None]-surface[None, :], 0)
    # Reserve one in five published picks before any fitting. Thresholds below
    # are fixed in this script before the first field inversion is run.
    held = np.arange(len(records)) % 5 == 0
    initial_times = first_arrivals(grid, initial, source, receiver, rays=False)[0]
    fit = invert(grid, initial, source[~held], receiver[~held], observed[~held],
                 iterations=20, smoothing=1, damping=.1, vmin=300, vmax=4500, log=print)
    predicted, _, hits = first_arrivals(grid, fit["velocity"], source, receiver)
    rms = lambda values: float(np.sqrt(np.mean(values**2)))
    initial_hold, final_hold = rms(initial_times[held]-observed[held]), rms(predicted[held]-observed[held])
    result = dict(source="https://github.com/gimli-org/example-data/blob/master/traveltime/koenigsee.sgt",
                  documentation="https://www.pygimli.org/_examples_auto/2_seismics/plot_04_koenigsee/",
                  credit="pyGIMLi development team, Koenigsee field example", sensor_count=count,
                  sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(), observations=len(records),
                  training_picks=int((~held).sum()), held_out_picks=int(held.sum()),
                  training_rms_seconds=fit["rms"], initial_holdout_rms_seconds=initial_hold,
                  final_holdout_rms_seconds=final_hold, all_predictions_finite=bool(np.isfinite(predicted).all()),
                  thresholds=dict(max_holdout_rms_seconds=.0015, max_holdout_fraction_of_initial=.7),
                  physical_velocity_ground_truth_available=False, raw_waveform_picking_validated=False)
    np.savez(root/"koenigsee-results.npz", x=grid.x, z=grid.z, surface=surface, initial=initial,
             velocity=fit["velocity"], hits=hits, observed=observed, predicted=predicted,
             source_x=sensors[source_index, 0], receiver_x=sensors[receiver_index, 0], held_out=held)
    project = Project()
    project.context.depth_model = DepthModel(grid.fill_air(fit["velocity"]), dz, dx, x0=x0)
    project.context.results["refraction_tomography"] = dict(fit, x=grid.x, z=grid.z, surface=surface,
                                                           observed=observed, t_pred=predicted, hits=hits,
                                                           starting_model=initial, held_out=held,
                                                           used=np.ones(len(observed), bool))
    project.save(str(root/"koenigsee-project"))
    result["status"] = ("passed" if result["all_predictions_finite"] and final_hold <= .0015 and
                        final_hold <= .7*initial_hold and np.all(np.diff(fit["rms"]) < 0) else "failed")
    return result


def main(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    report = dict(scope="public land waveform/operator checks and held-out refraction-pick prediction",
                  full_land_processing_accuracy_validated=False)
    for name, function in (("reflection", reflection), ("refraction", refraction)):
        try:
            report[name] = function(args, root)
            report[name].setdefault("status", "passed")
        except Exception as exc:
            import traceback
            report[name] = dict(status="failed", error=str(exc), traceback=traceback.format_exc())
        (root/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if any(report[name]["status"] != "passed" for name in ("reflection", "refraction")):
        raise SystemExit("one or more land checks failed; see the retained report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reflection"); parser.add_argument("refraction"); parser.add_argument("output")
    parser.add_argument("--su-bin", required=True)
    main(parser.parse_args())
