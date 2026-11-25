"""Small labelled synthetic examples, run only on Windows or a compute allocation."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from seisdesk.core.tomography3d import NodeGrid3D, arrivals3d, invert3d
from seisdesk.core.dataset import SeismicData
from seisdesk.processing import Context, create


def examples(work_dir):
    if os.name != "nt" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run this scientific example on Windows CI or inside a Slurm compute allocation")
    grid = NodeGrid3D((9, 9, 9), (10, 10, 10), radius=2)
    z, y, x = np.meshgrid(*grid.axes, indexing="ij")
    truth = 2000+250*np.exp(-((x-40)**2+(y-40)**2+(z-30)**2)/25**2)
    definitions = {"P": dict(kind="first", uncertainty_ms=.2),
                   "R1": dict(kind="reflection", depth=30, uncertainty_ms=.2),
                   "R2": dict(kind="reflection", depth=60, uncertainty_ms=.2)}
    sources, receivers, phases = [], [], []
    for sy, sx in ((0, 0), (0, 80), (80, 0), (80, 80)):
        for gy, gx in ((0, 20), (0, 60), (80, 20), (80, 60), (20, 0), (60, 0), (20, 80), (60, 80)):
            for phase in definitions:
                sources.append([0, sy, sx]); receivers.append([0, gy, gx]); phases.append(phase)
    source, source_snap = grid.nodes_for(sources); receiver, receiver_snap = grid.nodes_for(receivers)
    observed = arrivals3d(grid, truth, source, receiver, phases, definitions)[0]
    result = invert3d(grid, np.full(grid.shape, 1850.), source, receiver, observed, phases, definitions,
                      iterations=5, bootstrap=3, seed=17, smoothing=2, damping=.1)
    result.update(reference_elevation=0., source_snap_m=source_snap, receiver_snap_m=receiver_snap,
                  trace_ids=np.array([f"synthetic-trace-{i//3}" for i in range(len(source))]))
    shape, nt = (6, 6, 5, 5), 64
    yy, xx, hy, hx = np.indices(shape, dtype=float)
    a = yy/6+xx/6+hy/5+hx/5
    b = yy/6-xx/6+hy/5-hx/5
    t = np.arange(nt)/nt
    clean = (np.cos(2*np.pi*(a[..., None]+4*t))+.4*np.sin(2*np.pi*(b[..., None]+9*t))).reshape(-1, nt).astype(np.float32)
    known = np.random.default_rng(61).random(len(clean)) > .35
    known[0] = known[-1] = True
    my, mx, oy, ox = yy.ravel()*25, xx.ravel()*25, (hy.ravel()-2)*50, (hx.ravel()-2)*50
    data = SeismicData(clean[known], .004, dict(sx=(mx-ox/2)[known], gx=(mx+ox/2)[known],
                                               sy=(my-oy/2)[known], gy=(my+oy/2)[known]), name="Synthetic 5D plane waves")
    context = Context(work_dir=str(work_dir))
    output = create("survey_pocs", mode="5D prestack", dx=25, dy=25, dhx=50, dhy=50,
                    iterations=90, max_gap=10)(data, context)
    error = np.linalg.norm(output.traces[~known]-clean[~known])/np.linalg.norm(clean[~known])
    return dict(tomography=result, truth_velocity=truth, reconstruction=context.results["survey_pocs"],
                clean=clean, reconstructed=output.traces, known=known, reconstruction_shape=shape,
                held_out_relative_l2=float(error))
