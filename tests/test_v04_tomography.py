import numpy as np
import pytest

from seisdesk.core.tomography import NodeGrid, first_arrivals, invert
from seisdesk.processing.grm import grm_functions
from seisdesk.processing.tomography import column_statics
from seisdesk.core.velocity import DepthModel


def test_graph_ray_matrix_and_homogeneous_traveltimes():
    grid = NodeGrid(0, 10, 31, 10, 21, radius=3)
    src = np.array([grid.node_at(0, 0), grid.node_at(0, 0), grid.node_at(0, 0)])
    rec = np.array([grid.node_at(300, 0), grid.node_at(300, 100), grid.node_at(300, 200)])
    velocity = np.full((21, 31), 2000.)
    t, matrix, _ = first_arrivals(grid, velocity, src, rec)
    exact = np.hypot(300, [0, 100, 200]) / 2000
    assert np.allclose(t, exact, rtol=0.01)
    assert np.allclose(t, matrix @ (1 / velocity.ravel()), atol=1e-12)


def test_refraction_tomography_reduces_misfit():
    grid = NodeGrid(0, 10, 31, 10, 16, radius=3)
    sources, receivers = [], []
    for xs in [0, 100, 200, 300]:
        for xr in np.arange(0, 301, 20):
            if xs != xr:
                sources.append(grid.node_at(xs, 0)); receivers.append(grid.node_at(xr, 0))
    truth = np.broadcast_to(700 + 18 * grid.z[:, None], (grid.nz, grid.nx)).copy()
    observed = first_arrivals(grid, truth, sources, receivers)[0]
    result = invert(grid, truth * 0.9, sources, receivers, observed, iterations=5,
                    smoothing=0.5, damping=0.05, vmin=300, vmax=5000)
    assert result["rms"][-1] < result["rms"][0] * 0.3
    assert np.all(np.diff(result["rms"]) < 0)


def test_grm_horizontal_layer_exact_depth():
    x = np.arange(0, 1001, 10.)
    v1, v2, h = 500., 2000., 20.
    delay = h * np.sqrt(v2 ** 2 - v1 ** 2) / (v1 * v2)
    a, b = 2 * delay + x / v2, 2 * delay + (1000 - x) / v2
    result = grm_functions(x, a, x, b, 2 * delay + 1000 / v2, [0, 20, 40], v1)
    for candidate in result["candidates"]:
        assert candidate["velocity"] == pytest.approx(v2, rel=1e-8)
        assert np.allclose(candidate["depth"], h, atol=1e-8)


def test_vertical_tomostatic_replacement_sign():
    model = DepthModel.constant(1000, 21, 3, 5, 10)
    shifts = column_statics(model, np.array([10.]), np.array([0.]), 20, 2000, bottom=100)
    # 100 ms observed down to base, 40 ms from datum to base: advance by 60 ms.
    assert shifts[0] == pytest.approx(-0.06)


def test_topographic_graph_does_not_jump_through_air():
    grid = NodeGrid(0, 1, 7, 1, 6, surface=[0, 0, 3, 3, 3, 0, 0], radius=3)
    times, _, _ = first_arrivals(grid, np.ones((6, 7)), [grid.node_at(1, 0)], [grid.node_at(5, 0)])
    assert times[0] > 6  # direct 4-unit edge chain would incorrectly cut through air
