"""Withheld analytic Ricker events: curved moveout and varying amplitudes.

An additional benchmark deliberately avoids exact periodic Fourier-grid events.
Run on a Slurm compute node or Windows CI, never on the Mac/login node.
"""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seisdesk.core.reconstruction import reconstruct_survey


def main(destination):
    if os.name != "nt" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run on a compute node or Windows")
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    cases = []
    for shape in ((16, 16), (9, 9, 5, 5)):
        axes = np.indices(shape, dtype=float)
        y, x = axes[0]*25, axes[1]*25
        oy, ox = (axes[2]*100-200, axes[3]*100-200) if len(shape) == 4 else (np.zeros(shape), np.zeros(shape))
        offset2 = ox**2+oy**2
        t1 = np.sqrt((.3+.00011*x+.00007*y+.0000002*(x-y)**2)**2 + offset2/2000**2)
        t2 = np.sqrt((.58-.00008*x+.00003*y)**2 + offset2/2700**2)
        amp1 = (1-.22*offset2/80000)*(1+.15*np.sin(x/150))
        amp2 = .45*(1+.15*np.cos(y/170))
        times = np.arange(256)*.004
        def ricker(arrival, frequency):
            phase = (np.pi*frequency*(times-arrival[..., None]))**2
            return (1-2*phase)*np.exp(-phase)
        first, second = ricker(t1, 22), ricker(t2, 17)
        truth = (amp1[..., None]*first+amp2[..., None]*second).reshape(-1, 256).astype(np.float32)
        known = np.random.default_rng(724).random(len(truth)) > .3
        recovered = reconstruct_survey(truth[known], np.flatnonzero(known), shape, iterations=120,
                                        final_threshold=.008, work_dir=destination)
        relative_l2 = float(np.linalg.norm(recovered[~known]-truth[~known])/np.linalg.norm(truth[~known]))
        # Independently evaluate event amplitudes by projection onto analytic wavelets.
        first_flat = first.reshape(-1, 256)
        fitted_amplitude = np.sum(recovered*first_flat, axis=1)/np.sum(first_flat**2, axis=1)
        amplitude_relative_l2 = float(np.linalg.norm((fitted_amplitude-amp1.ravel())[~known])/np.linalg.norm(amp1.ravel()[~known]))
        record = dict(dimensions=len(shape)+1, spatial_shape=shape, recorded=int(known.sum()), withheld=int((~known).sum()),
                      held_out_relative_l2=relative_l2, first_event_amplitude_relative_l2=amplitude_relative_l2,
                      zero_fill_relative_l2=1., max_recorded_change=float(np.max(np.abs(recovered[known]-truth[known]))))
        cases.append(record)
        if relative_l2 >= .5 or record["max_recorded_change"] != 0:
            raise AssertionError(f"withheld curved-event benchmark failed: {record}")
        selection = (slice(None), slice(None)) if len(shape) == 2 else (slice(None), slice(None), 2, 2)
        a = truth.reshape(*shape, -1)[selection].reshape(-1, 256)
        b = recovered.reshape(*shape, -1)[selection].reshape(-1, 256)
        present = known.reshape(shape)[selection].ravel()
        fig, panels = plt.subplots(1, 4, figsize=(15, 6), constrained_layout=True)
        for panel, title, values in zip(panels, ("Analytic reflections", "Recorded subset", "POCS reconstruction", "Reconstruction − truth"),
                                        (a, np.where(present[:, None], a, np.nan), b, b-a)):
            image = panel.imshow(np.ma.masked_invalid(values.T), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                                 extent=(0, len(a), 1.024, 0))
            panel.set(title=title, xlabel="Midpoint trace index", ylabel="Time (s)")
        fig.colorbar(image, ax=panels, label="Amplitude", shrink=.8)
        fig.suptitle(f"{len(shape)+1}D curved Ricker-event benchmark: held-out L2 error {100*relative_l2:.2f}%\n"
                     f"First-event amplitude L2 error {100*amplitude_relative_l2:.2f}%; recorded samples unchanged")
        fig.savefig(destination/f"v09_curved_{len(shape)+1}d.png", dpi=150); plt.close(fig)
    report = dict(implementation_revision=os.environ.get("SEISDESK_SOURCE_REVISION", "unknown"),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), cases=cases,
                  scope="Analytic curved reflections, offset-dependent amplitudes and randomly withheld traces; not field validation")
    (destination/"curved-events-report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
