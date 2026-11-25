from copy import deepcopy

import numpy as np
import pytest

from seisdesk.core.interface_inversion import invert_interfaces
from seisdesk.core.joint_tomography import joint_arrivals
from seisdesk.core.tomography import NodeGrid


def geometry(grid, names):
    sources, receivers, phases = [], [], []
    for sx in (0, 50, 100):
        for gx in (10, 30, 70, 90):
            for name in names:
                sources.append(grid.surface_node(sx)); receivers.append(grid.surface_node(gx)); phases.append(name)
    return sources, receivers, phases


def test_recovers_two_interface_depths_with_known_velocity():
    grid = NodeGrid(0, 5, 21, 2, 46)
    velocity = np.full((grid.nz, grid.nx), 2000.)
    truth = {"R1": dict(kind="reflection", depth=30), "R2": dict(kind="reflection", depth=60)}
    source, receiver, phases = geometry(grid, truth)
    observed = joint_arrivals(grid, velocity, source, receiver, phases, truth)[0]
    start = {"R1": dict(kind="reflection", depth=20), "R2": dict(kind="reflection", depth=70)}
    untouched = deepcopy(start)
    fit = invert_interfaces(grid, velocity, source, receiver, observed, phases, start, iterations=5,
                            update_velocity=False, warp_velocity=False, max_step=8, max_shift=20)
    assert start == untouched
    assert fit["branches"]["R1"]["depth"] == 30 and fit["branches"]["R2"]["depth"] == 60
    assert fit["rms"][-1] < 1e-10 and np.all(np.diff(fit["weighted_misfit"]) < 0)
    np.testing.assert_array_equal(fit["velocity"], velocity)


def test_interface_depths_against_independent_analytic_reflection_times():
    grid = NodeGrid(0, 2, 61, 1, 81, radius=5)
    source, receiver, phases, observed = [], [], [], []
    truth = {"R1": 30.4, "R2": 59.6}
    for sx in (0, 60, 120):
        for gx in (12, 40, 80, 108):
            for phase, depth in truth.items():
                source.append(grid.surface_node(sx)); receiver.append(grid.surface_node(gx)); phases.append(phase)
                observed.append(np.hypot(gx-sx, 2*depth)/2000.)
    start = {"R1": dict(kind="reflection", depth=20), "R2": dict(kind="reflection", depth=70)}
    fit = invert_interfaces(grid, np.full((grid.nz, grid.nx), 2000.), source, receiver, observed, phases,
                            start, update_velocity=False, warp_velocity=False, max_shift=20, max_step=8)
    assert fit["rms"][-1] < .0006
    for phase, depth in truth.items():
        assert abs(fit["branches"][phase]["depth"]-depth) <= grid.dz
    coarse = NodeGrid(0, 2, 61, 1, 81, radius=3)
    coarse_fit = invert_interfaces(coarse, np.full((grid.nz, grid.nx), 2000.), source, receiver, observed,
                                   phases, start, update_velocity=False, warp_velocity=False, max_shift=20, max_step=8)
    assert fit["rms"][-1] < .6*coarse_fit["rms"][-1]


def test_interface_controls_are_bounded_and_can_be_fixed():
    grid = NodeGrid(0, 5, 21, 2, 46)
    velocity = np.full((grid.nz, grid.nx), 2000.)
    truth = {"R1": dict(kind="reflection", points=[[0, 30], [100, 40]]),
             "R2": dict(kind="reflection", depth=65)}
    source, receiver, phases = geometry(grid, truth)
    observed = joint_arrivals(grid, velocity, source, receiver, phases, truth)[0]
    start = {"R1": dict(kind="reflection", points=[[0, 20], [100, 20]], depth_bounds=[18, 28]),
             "R2": dict(kind="reflection", depth=60, invert_depth=False)}
    fit = invert_interfaces(grid, velocity, source, receiver, observed, phases, start, iterations=4,
                            update_velocity=False, warp_velocity=False, max_step=8, max_shift=20)
    controls = np.asarray(fit["branches"]["R1"]["points"])[:, 1]
    assert np.all((controls >= 18) & (controls <= 28))
    assert fit["branches"]["R2"]["depth"] == 60
    assert fit["rms"][-1] < fit["rms"][0]


def test_shared_interface_and_velocity_contrast_move_together():
    grid = NodeGrid(0, 5, 41, 2, 36)
    source = [grid.surface_node(0)]*2
    receiver = [grid.surface_node(200)]*2
    phases = ["R", "H"]
    velocity = np.full((grid.nz, grid.nx), 1200.)
    velocity[16:] = 3000
    truth = {p: dict(kind=k, depth=30, interface="base") for p, k in zip(phases, ("reflection", "head"))}
    observed = joint_arrivals(grid, velocity, source, receiver, phases, truth)[0]
    initial = np.full_like(velocity, 1200.)
    initial[11:] = 3000
    start = {p: dict(kind=k, depth=20, interface="base") for p, k in zip(phases, ("reflection", "head"))}
    fit = invert_interfaces(grid, initial, source, receiver, observed, phases, start, iterations=5,
                            update_velocity=False, max_step=4, max_shift=16)
    assert fit["branches"]["R"]["depth"] == fit["branches"]["H"]["depth"]
    assert fit["rms"][-1] < .5*fit["rms"][0]
    assert not np.array_equal(fit["velocity"], initial)
    bad = deepcopy(start); bad["H"]["depth"] = 22
    with pytest.raises(ValueError, match="identical geometry"):
        invert_interfaces(grid, initial, source, receiver, observed, phases, bad)


def test_alternating_velocity_and_depth_reduces_joint_misfit():
    grid = NodeGrid(0, 5, 21, 2, 36)
    truth = {"D": dict(kind="first"), "R": dict(kind="reflection", depth=40)}
    source, receiver, phases = geometry(grid, truth)
    observed = joint_arrivals(grid, np.full((grid.nz, grid.nx), 2000.), source, receiver, phases, truth)[0]
    start = deepcopy(truth); start["R"]["depth"] = 32
    fit = invert_interfaces(grid, np.full((grid.nz, grid.nx), 1900.), source, receiver, observed, phases,
                            start, iterations=5, max_step=4, max_shift=16)
    assert fit["rms"][-1] < .25*fit["rms"][0]
    assert np.all(np.diff(fit["weighted_misfit"]) < 0)
    assert any(h["update"] == "velocity" for h in fit["interface_history"])
    assert any(h["update"].startswith("interface") for h in fit["interface_history"])


def test_picked_phases_flow_to_depth_inversion_and_survive_project_save(tmp_path):
    import json
    from seisdesk.core.dataset import SeismicData
    from seisdesk.core.picks import PickSet
    from seisdesk.processing import create
    from seisdesk.workflow.project import Project
    sx = np.repeat([0, 50, 100], 4)
    gx = np.tile([10, 30, 70, 90], 3)
    data = SeismicData(np.zeros((12, 64)), .001, dict(sx=sx, gx=gx, offset=gx-sx, fldr=sx+1))
    picks = PickSet(name="arrivals")
    branches = {"R1": dict(kind="reflection", depth=20), "R2": dict(kind="reflection", depth=70)}
    for phase, depth in (("R1", 30), ("R2", 60)):
        picks.new_phase(phase, "reflection")
        for i, (shot, receiver) in enumerate(zip(sx, gx)):
            picks.add(shot+1, phase, receiver-shot, np.hypot(receiver-shot, 2*depth)/2000, data.trace_ids[i])
    project = Project(); project.add_dataset(data)
    project.context.picks["arrivals"] = picks
    process = create("joint_tomography", pickset="arrivals", branches=json.dumps(branches), model="gradient",
                     v0=2000, gradient=0, dx=5, dz=2, zmax=90, invert_interfaces=True,
                     update_velocity=False, warp_velocity=False, interface_max_step=8, interface_max_shift=20)
    result_data = process(data, project.context)
    result = project.context.results["joint_tomography"]
    assert result_data.domain == "depth" and len(result["observed"]) == 24
    assert result["rms"][-1] < .001
    assert len(set(result["trace_ids"])) == 12  # Two separate arrivals on every trace.
    project.save(str(tmp_path/"saved"))
    restored = Project.load(str(tmp_path/"saved")).context.results["joint_tomography"]
    assert restored["branches"] == result["branches"]
    assert restored["interface_history"] == result["interface_history"]
    np.testing.assert_array_equal(restored["t_pred"], result["t_pred"])
