import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from seisdesk.core import synthetic


@pytest.fixture(scope="session")
def marine():
    """Small streamer line: 4 reflectors, water-bottom multiples, direct wave, no noise."""
    return synthetic.marine_line(nshots=24, ngroups=48, nt=751, dt=0.004)


@pytest.fixture(scope="session")
def marine_cmp(marine):
    return marine.sort(["cdp", "offset"])


@pytest.fixture(scope="session")
def land():
    return synthetic.land_refraction_line(nshots=6, nrec=96, nt=601)


def peak_time(trace, times):
    return float(times[int(np.argmax(np.abs(trace)))])
