"""Small job-control fixture; run only on a validation compute node, not a desktop."""
import json
from pathlib import Path
import sys

import numpy as np

from seisdesk.core.dataset import SeismicData
from seisdesk.core.segy_io import write_su
from seisdesk.core.volume import VelocityVolume


def main(destination):
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    VelocityVolume(np.full((9, 8, 10), 1900., dtype=np.float32), (10, 10, 10)).save(root/"model.npz")
    survey = SeismicData(np.zeros((6, 85), dtype=np.float32), .001,
                         dict(sx=[20]*3+[60]*3, sy=30, selev=-20,
                              gx=[10, 50, 70]*2, gy=[20, 35, 50]*2, gelev=-20,
                              fldr=[1]*3+[2]*3, tracf=[1, 2, 3]*2, offset=[-10, 30, 50, -50, -10, 10]))
    write_su(survey, str(root/"survey.su"))
    params = dict(input_file="survey.su", model="model.npz", mode="model", device="cpu", workers=1,
                  memory_gb=.2, frequency=45, source_delay=.025, source_amplitude=1e6, padding=3, block=11,
                  vmax=2400, vmin=1000, allow_coarse_grid=True)
    (root/"parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
