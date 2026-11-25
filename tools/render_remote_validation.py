"""Render remote CPU/CUDA predictions on a compute node, from a lifecycle report."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(report_file, destination):
    report = json.loads(Path(report_file).read_text(encoding="utf-8"))
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    cases = report["cases"]
    cpu_root = Path(cases["cpu"]["remote_directory"])/"bundle/output"
    gpu_root = Path(cases["cuda"]["remote_directory"])/"bundle/output"
    cpu = json.loads((cpu_root/"job.json").read_text(encoding="utf-8"))
    gpu = json.loads((gpu_root/"job.json").read_text(encoding="utf-8"))
    metrics = []
    for left, right in zip(cpu["shot_results"], gpu["shot_results"]):
        a, b = np.load(left["files"]["predicted"]), np.load(right["files"]["predicted"])
        relative = float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-30))
        if not np.isfinite(b).all() or relative > 1e-5:
            raise AssertionError(f"CPU/CUDA mismatch: relative L2={relative:g}")
        metrics.append(dict(relative_l2=relative, max_absolute=float(np.max(np.abs(a-b))), shape=list(a.shape)))
    with np.load(cpu["shots"][0]["path"], allow_pickle=False) as shot:
        dt = float(shot["dt"])
        offsets = shot["h_offset"]
    a = np.load(cpu["shot_results"][0]["files"]["predicted"])
    b = np.load(gpu["shot_results"][0]["files"]["predicted"])
    time_ms = np.arange(len(a))*dt*1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    amplitude = max(float(np.max(np.abs(a))), 1e-20)
    for index in range(a.shape[1]):
        axes[0].plot(time_ms, a[:, index]/amplitude+index*1.5, color="#1565a6", linewidth=1.6,
                     label="CPU" if index == 0 else None)
        axes[0].plot(time_ms, b[:, index]/amplitude+index*1.5, color="#d46b22", linestyle="--", linewidth=1,
                     label="CUDA" if index == 0 else None)
        axes[1].plot(time_ms, (b[:, index]-a[:, index])/amplitude,
                     label=f"Offset {offsets[index]:g} m")
    axes[0].set_yticks(np.arange(a.shape[1])*1.5, [f"{v:g} m" for v in offsets])
    axes[0].set(xlabel="Recording time (ms)", ylabel="Source–receiver offset; normalised traces",
                title="First shot: CPU and CUDA predictions")
    axes[1].set(xlabel="Recording time (ms)", ylabel="(CUDA − CPU) / peak CPU amplitude",
                title="Numerical difference")
    for axis in axes:
        axis.grid(alpha=.18); axis.legend(loc="upper right")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("SeisDesk remote execution validation", fontsize=17)
    fig.text(.5, -.035, "Two shots; constant 1900 m/s acoustic model. Small coarse-grid fixture for execution checks, "
             "not a field image or a dispersion benchmark.\n"
             f"Slurm CPU {cases['cpu']['job_id']} / CUDA {cases['cuda']['job_id']} · "
             f"maximum relative waveform L2 difference {max(m['relative_l2'] for m in metrics):.3g}",
             ha="center", va="top", fontsize=9)
    fig.savefig(root/"athena-cpu-cuda-predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    result = dict(cpu_job=cases["cpu"]["job_id"], cuda_job=cases["cuda"]["job_id"], shots=metrics,
                  fixture="coarse-grid job-control fixture; not field data or an independent physics benchmark")
    (root/"cpu-cuda-comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
