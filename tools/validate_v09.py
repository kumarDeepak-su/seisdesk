"""Persist v0.9 controlled example metrics and scientific figures on a compute node."""
import json
import os
from pathlib import Path
import platform
import sys

from v09_example import examples
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seisdesk.workflow.results import save_results


def main(destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    data = examples(destination)
    save_results(data, destination/"synthetic-results.npz")
    r = data["tomography"]
    report = dict(source_revision=os.environ.get("SEISDESK_SOURCE_REVISION", "working-tree snapshot"),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), host=platform.node(), python=platform.python_version(),
                  tomography=dict(arrivals=len(r["observed"]), phases=list(r["branches"]),
                      rms_ms=[v*1000 for v in r["rms"]], covered_nodes=int(np.count_nonzero(r["hits"])),
                      total_nodes=int(r["velocity"].size), uncertainty=r["uncertainty_ensemble"]),
                  reconstruction=dict(shape=data["reconstruction_shape"],
                      recorded_traces=int(data["known"].sum()), withheld_traces=int((~data["known"]).sum()),
                      held_out_relative_l2=data["held_out_relative_l2"],
                      max_recorded_sample_change=float(np.max(np.abs(data["clean"][data["known"]]-data["reconstructed"][data["known"]])))),
                  scope="Controlled synthetic examples only; not independent heterogeneous field or survey-scale validation")
    assert data["held_out_relative_l2"] < .05
    assert report["reconstruction"]["max_recorded_sample_change"] == 0
    assert r["rms"][-1] < r["rms"][0]
    (destination/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    y_index = r["velocity"].shape[1]//2
    images = (("Synthetic true velocity", data["truth_velocity"], "m/s", "viridis", (1800, 2250)),
              ("Starting velocity", r["starting_model"], "m/s", "viridis", (1800, 2250)),
              ("Recovered velocity (masked where uncovered)", np.where(r["hits"] > 0, r["velocity"], np.nan), "m/s", "viridis", (1800, 2250)),
              ("Arrival coverage", r["hits"], "rays/node", "magma", None),
              ("Conditional timing-noise standard deviation", r["velocity_std"], "m/s", "viridis", None))
    z, y, x = r["axes"]
    for ax, (title, values, units, cmap, limits) in zip(axes.flat, images):
        image = ax.imshow(np.ma.masked_invalid(values[:, y_index]), origin="upper", extent=(x[0], x[-1], z[-1], z[0]),
                          cmap=cmap, vmin=limits[0] if limits else None, vmax=limits[1] if limits else None)
        ax.set(title=title, xlabel="Easting (m)", ylabel="Depth (m)")
        fig.colorbar(image, ax=ax, label=units, shrink=.8)
    axes[1, 2].plot(np.arange(len(r["rms"])), np.asarray(r["rms"])*1000, "o-")
    axes[1, 2].set(title="All named arrivals retained", xlabel="Accepted update", ylabel="Travel-time RMS (ms)")
    fig.suptitle("3D fixed-interface tomography: controlled synthetic example; y = 40 m\n"
                 "Timing-only uncertainty from three diagnostic fits; not total geological uncertainty")
    fig.savefig(destination/"v09_synthetic_tomography.png", dpi=150); plt.close(fig)
    shape = data["reconstruction_shape"]
    clean = data["clean"].reshape(*shape, -1)[:, :, 2, 2].reshape(-1, 64)
    recovered = data["reconstructed"].reshape(*shape, -1)[:, :, 2, 2].reshape(-1, 64)
    known = data["known"].reshape(shape)[:, :, 2, 2].ravel()
    displayed = np.where(known[:, None], clean, np.nan)
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), constrained_layout=True)
    for ax, title, values in zip(axes, ("Analytic complete slice", "Recorded subset", "Reconstructed slice", "Reconstruction − truth"),
                                 (clean, displayed, recovered, recovered-clean)):
        image = ax.imshow(np.ma.masked_invalid(values.T), cmap="RdBu_r", aspect="auto", extent=(0, 36, 252, 0), vmin=-1.4, vmax=1.4)
        ax.set(title=title, xlabel="Midpoint trace index", ylabel="Time (ms)")
    fig.colorbar(image, ax=axes, label="Amplitude", shrink=.8)
    fig.suptitle(f"5D analytic plane waves: full signed offset (0,0) slice\n"
                 f"All withheld traces relative L2 error {100*data['held_out_relative_l2']:.4g}%; recorded samples unchanged")
    fig.savefig(destination/"v09_synthetic_reconstruction.png", dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
