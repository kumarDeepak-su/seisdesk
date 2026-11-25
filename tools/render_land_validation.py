"""Render saved land benchmark diagnostics on the compute host (no model fitting)."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    root = Path(root)
    report = json.loads((root/"report.json").read_text())["refraction"]
    with np.load(root/"koenigsee-results.npz", allow_pickle=False) as data:
        x, z, surface = data["x"], data["z"], data["surface"]
        observed, predicted, held = data["observed"], data["predicted"], data["held_out"]
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        extent = [x[0], x[-1], z[-1], z[0]]
        air = z[:, None] < surface[None, :]
        for ax, key, title in zip(axes[0], ("initial", "velocity", "hits"),
                                  ("Starting velocity", "Fitted velocity", "Ray coverage")):
            values = np.ma.array(data[key], mask=air)
            if key == "velocity":
                values = np.ma.masked_where(data["hits"] == 0, values)
            image = ax.imshow(values, extent=extent, aspect="auto", cmap="viridis",
                              **({"vmin":300, "vmax":4500} if key != "hits" else {}))
            ax.plot(x, surface, "k-", lw=1)
            ax.set(title=title, xlabel="Line position (m)", ylabel="Depth below datum (m)")
            figure.colorbar(image, ax=ax, label="m/s" if key != "hits" else "Arrivals")
        for flag, label, colour in ((False, "571 fitted picks", "#4b718c"), (True, "143 held-out picks", "#c64b28")):
            mask = held == flag
            axes[1, 0].scatter(observed[mask]*1000, predicted[mask]*1000, s=9, color=colour, label=label, alpha=.7)
            axes[1, 1].scatter(np.abs(data["receiver_x"][mask]-data["source_x"][mask]),
                               (predicted[mask]-observed[mask])*1000, s=9, color=colour, alpha=.7)
        limit = max(observed.max(), predicted.max())*1000
        axes[1, 0].plot([0, limit], [0, limit], "k--", lw=1)
        axes[1, 0].set(title="Predicted arrivals", xlabel="Observed time (ms)", ylabel="Predicted time (ms)")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].axhline(0, color="k", lw=1)
        axes[1, 1].set(title="Arrival residuals", xlabel="Absolute offset (m)", ylabel="Predicted − observed (ms)")
        rms = np.array(report["training_rms_seconds"])*1000
        axes[1, 2].plot(np.arange(len(rms)), rms, "o-", ms=3, label="Fitted-pick RMS")
        initial, final = report["initial_holdout_rms_seconds"]*1000, report["final_holdout_rms_seconds"]*1000
        axes[1, 2].scatter([0, len(rms)-1], [initial, final], marker="s", color="#c64b28", label="Held-out RMS")
        axes[1, 2].set(title=f"Held-out RMS: {initial:.2f} → {final:.2f} ms", xlabel="Accepted update", ylabel="RMS (ms)")
        axes[1, 2].legend(fontsize=8)
        figure.suptitle("Koenigsee public refraction picks · no velocity ground truth\n"
                        "Fitted velocity hidden where no rays sample the model; datum is highest sensor elevation", fontsize=12)
        figure.savefig(root/"koenigsee-validation.png", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main(sys.argv[1])
