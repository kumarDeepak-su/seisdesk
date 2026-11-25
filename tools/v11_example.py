"""Independent plane-reflection example for compute-host validation and GUI previews."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from seisdesk.core.tomography3d import NodeGrid3D
from seisdesk.core.interfaces3d import invert_interfaces3d


def example(bootstrap=3):
    if os.name != "nt" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run on Windows CI or in an Athena compute allocation")
    grid = NodeGrid3D((15, 9, 9), (5, 10, 10), radius=3)
    sources, phases, clean = [], [], []
    for y in (20, 40, 60):
        for x in (20, 40, 60):
            for phase in ("R1", "R2"):
                sources.append([0, y, x]); phases.append(phase)
                # Exact distance to an infinite plane for zero-offset reflection.
                clean.append(2*(25+.125*x+.125*y)/(2000*np.sqrt(1+2*.125**2)) if phase == "R1" else 2*55/2000)
    nodes, snap = grid.nodes_for(sources)
    branches = {"R1":dict(kind="reflection", depth=35, control_shape=[2,2], interface="dipping",
                         depth_bounds=[15,50], uncertainty_ms=2),
                "R2":dict(kind="reflection", depth=60, interface="deep", depth_bounds=[45,65], uncertainty_ms=2)}
    observed = np.asarray(clean)+np.random.default_rng(19).normal(0,.0004,len(clean))
    result = invert_interfaces3d(grid, np.full(grid.shape, 2000.), nodes, nodes, observed, phases, branches,
                                iterations=4, update_velocity=False, warp_velocity=False, max_shift=15, max_step=10,
                                bootstrap=bootstrap, seed=23)
    result.update(reference_elevation=0., trace_ids=np.array([f"zero-offset-{i//2}" for i in range(len(nodes))]),
                  source_snap_m=snap, receiver_snap_m=snap)
    yy, xx = np.meshgrid(grid.axes[1], grid.axes[2], indexing="ij")
    return result, {"dipping":25+.125*xx+.125*yy, "deep":np.full(grid.shape[1:],55.)}
