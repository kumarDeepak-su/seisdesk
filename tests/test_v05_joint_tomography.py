import numpy as np
import pytest

from seisdesk.core.joint_tomography import invert_joint, joint_arrivals
from seisdesk.core.tomography import NodeGrid


def test_reflections_retain_later_events_and_obey_constant_velocity_geometry():
    grid = NodeGrid(0, 5, 41, 5, 31, radius=3)
    velocity = np.full((31, 41), 2000.)
    source = [grid.surface_node(20)]*3
    receiver = [grid.surface_node(100)]*3
    phases = ["direct", "R1", "R2"]
    branches = {"direct": {"kind": "first"}, "R1": {"kind": "reflection", "depth": 50},
                "R2": {"kind": "reflection", "depth": 100}}
    predicted, matrix, _ = joint_arrivals(grid, velocity, source, receiver, phases, branches)
    expected = np.array([80, np.hypot(80, 100), np.hypot(80, 200)])/2000
    np.testing.assert_allclose(predicted, expected, rtol=.015)
    np.testing.assert_allclose(matrix @ (1/velocity.ravel()), predicted, rtol=1e-12)
    assert np.all(np.diff(predicted) > 0)


def test_head_wave_critical_branch_and_slowness_derivative():
    grid = NodeGrid(0, 2, 101, 2, 31, radius=3)
    velocity = np.full((31, 101), 1000.)
    velocity[11:] = 2500
    branches = {"refractor": {"kind": "head", "depth": 20}}
    source, receiver = [grid.surface_node(10)], [grid.surface_node(190)]
    predicted, matrix, _ = joint_arrivals(grid, velocity, source, receiver, ["refractor"], branches)
    analytic = 180/2500 + 2*20*np.sqrt(1/1000**2-1/2500**2)
    np.testing.assert_allclose(predicted, analytic, rtol=.02)
    np.testing.assert_allclose(matrix @ (1/velocity.ravel()), predicted, rtol=1e-12)
    with pytest.raises(ValueError, match="head-wave leg"):
        joint_arrivals(grid, velocity, source, [grid.surface_node(12)], ["refractor"], branches)


def test_joint_tomography_reduces_all_branch_misfit():
    grid = NodeGrid(0, 10, 21, 10, 16, radius=3)
    source, receiver, phases = [], [], []
    for shot in (0, 100, 200):
        for rec in (20, 60, 140, 180):
            for phase in ("direct", "R1", "R2"):
                source.append(grid.surface_node(shot)); receiver.append(grid.surface_node(rec)); phases.append(phase)
    branches = {"direct": {"kind": "first"}, "R1": {"kind": "reflection", "depth": 50},
                "R2": {"kind": "reflection", "depth": 100}}
    observed, _, _ = joint_arrivals(grid, np.full((16, 21), 2200.), source, receiver, phases, branches)
    result = invert_joint(grid, np.full((16, 21), 1900.), source, receiver, observed, phases, branches,
                          iterations=4, smoothing=.5)
    assert result["rms"][-1] < .15*result["rms"][0]
    assert len(result["t_pred"]) == 36 and result["used"].all()
    assert set(result["phase_rms"]) == set(branches)
    assert np.all(np.diff(result["weighted_misfit"]) < 0)
