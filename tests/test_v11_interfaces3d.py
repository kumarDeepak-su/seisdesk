from copy import deepcopy
import json

import numpy as np
import pytest

from seisdesk.core.interfaces3d import invert_interfaces3d, save_interfaces3d
from seisdesk.core.tomography3d import NodeGrid3D, arrivals3d


def flat_example():
    grid = NodeGrid3D((12, 9, 9), (10, 10, 10), radius=3)
    sources, receivers, phases, observed = [], [], [], []
    for sy, sx, gy, gx in ((10, 10, 70, 70), (70, 10, 10, 70), (10, 70, 70, 10), (70, 70, 10, 10)):
        for phase, depth in (("P", 0), ("R1", 30), ("R2", 60)):
            sources.append([0, sy, sx]); receivers.append([0, gy, gx]); phases.append(phase)
            observed.append(np.sqrt((sy-gy)**2+(sx-gx)**2+4*depth**2)/2000)
    branches = {"P": dict(kind="first"),
                "R1": dict(kind="reflection", depth=20, interface="shallow", depth_bounds=[10, 50]),
                "R2": dict(kind="reflection", depth=80, interface="deep", depth_bounds=[40, 100])}
    return grid, grid.nodes_for(sources)[0], grid.nodes_for(receivers)[0], phases, branches, np.array(observed)


def test_two_independent_mirror_reflections_recover_depths_and_preserve_inputs():
    grid, source, receiver, phases, branches, observed = flat_example()
    original = deepcopy(branches)
    velocity = np.full(grid.shape, 2000.)
    progress = []
    result = invert_interfaces3d(grid, velocity, source, receiver, observed, phases, branches,
                                iterations=4, update_velocity=False, warp_velocity=False, max_shift=30, max_step=20,
                                report=progress.append)
    np.testing.assert_allclose(result["sampled_interfaces"]["shallow"], 30)
    np.testing.assert_allclose(result["sampled_interfaces"]["deep"], 60)
    np.testing.assert_allclose(result["t_pred"], observed, atol=1e-12)
    assert result["rms"][-1] < 1e-12 and np.all(np.diff(result["weighted_misfit"]) < 0)
    assert branches == original and np.all(velocity == 2000)
    assert result["used"].all() and len(result["observed"]) == 12
    assert progress[-1] == 1 and np.all(np.diff(progress) >= 0)
    assert sum(a.sum() for a in result["interface_hits"].values()) == 8


def test_shared_reflection_head_boundary_and_velocity_layer_remapping():
    grid = NodeGrid3D((8, 3, 23), (1, 1, 1), radius=3)
    velocity = np.full(grid.shape, 1500.); velocity[3:] = 3000
    source, receiver, phases, observed = [], [], [], []
    for y in (0, 1, 2):
        for phase in ("R", "H"):
            source.append([0, y, 1]); receiver.append([0, y, 21]); phases.append(phase)
            observed.append(np.sqrt(20**2+6**2)/1500 if phase == "R" else 20/3000+6*np.sqrt(1/1500**2-1/3000**2))
    branches = {p: dict(kind=k, depth=2, interface="shared", depth_bounds=[1, 5])
                for p, k in (("R", "reflection"), ("H", "head"))}
    result = invert_interfaces3d(grid, velocity, grid.nodes_for(source)[0], grid.nodes_for(receiver)[0], observed,
                                phases, branches, iterations=3, update_velocity=False, max_shift=2, max_step=2)
    np.testing.assert_allclose(result["sampled_interfaces"]["shared"], 3)
    np.testing.assert_array_equal(result["branches"]["R"]["surface"], result["branches"]["H"]["surface"])
    assert np.all(result["velocity"][3] == 1500) and np.all(result["velocity"][4] == 3000)
    assert result["rms"][-1] < result["rms"][0]*.15
    assert result["interface_hits"]["shared"].sum() > len(observed)


def test_bilinear_boundary_deformation_improves_laterally_varying_example():
    grid = NodeGrid3D((12, 5, 5), (5, 10, 10), radius=2)
    yy, xx = np.meshgrid(grid.axes[1], grid.axes[2], indexing="ij")
    truth = 20+.25*xx+.25*yy
    stations = [[0, y, x] for y in grid.axes[1] for x in grid.axes[2]]
    source = receiver = grid.nodes_for(stations)[0]
    phases = ["R"]*len(source); velocity = np.full(grid.shape, 2000.)
    observed = arrivals3d(grid, velocity, source, receiver, phases, {"R":dict(kind="reflection", surface=truth)})[0]
    branches = {"R":dict(kind="reflection", depth=30, control_shape=[2, 2], depth_bounds=[10, 45])}
    result = invert_interfaces3d(grid, velocity, source, receiver, observed, phases, branches,
                                iterations=4, update_velocity=False, warp_velocity=False, max_shift=15, max_step=10)
    assert result["rms"][-1] < .25*result["rms"][0]
    assert np.mean(np.abs(result["sampled_interfaces"]["R"]-truth)) <= grid.spacing[0]
    assert np.ptp(result["interface_controls"]["R"]) >= 10


def test_joint_velocity_steps_and_fixed_boundary():
    grid, source, receiver, phases, branches, observed = flat_example()
    branches["R2"].update(depth=60, invert_interface=False)
    result = invert_interfaces3d(grid, np.full(grid.shape, 1900.), source, receiver, observed, phases, branches,
                                iterations=3, max_shift=20, max_step=10, warp_velocity=False, smoothing=1, damping=.05)
    assert result["rms"][-1] < .4*result["rms"][0]
    assert any(row["update"] == "velocity" for row in result["interface_history"])
    assert np.all(result["sampled_interfaces"]["deep"] == 60)
    assert np.all(np.diff(result["weighted_misfit"]) < 0)


def test_conditional_ensemble_original_bounds_masks_and_portable_export(tmp_path):
    grid, source, receiver, phases, branches, observed = flat_example()
    result = invert_interfaces3d(grid, np.full(grid.shape, 2000.), source, receiver, observed, phases, branches,
                                uncertainty=np.full(len(source), .004), iterations=3, update_velocity=False,
                                warp_velocity=False, max_shift=20, max_step=10, bootstrap=3, seed=22)
    for name in result["sampled_interfaces"]:
        covered = result["interface_hits"][name] > 0
        assert np.isfinite(result["interface_std"][name][covered]).all()
        assert np.isnan(result["interface_std"][name][~covered]).all()
        assert np.all(np.abs(result["interface_control_ensemble"][name]) <= 20)
        assert result["interface_control_ensemble"][name].shape == (3, 1, 1)
        assert np.all(result["interface_p95"][name][covered] >= result["interface_p05"][name][covered])
    path = tmp_path/"interfaces.npz"; save_interfaces3d(result, path)
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        assert metadata["interface_0"]["name"] == "shallow"
        np.testing.assert_allclose(saved["interface_0_sampled_interfaces"], result["sampled_interfaces"]["shallow"])
        assert metadata["uncertainty"]["replicates"] == 3


def test_invalid_shared_bounds_crossing_uncertainty_and_cancellation():
    grid, source, receiver, phases, branches, observed = flat_example()
    velocity = np.full(grid.shape, 2000.)
    def run(definitions=branches, **kwargs):
        return invert_interfaces3d(grid, velocity, source, receiver, observed, phases, definitions,
                                    update_velocity=False, **kwargs)
    conflicting = deepcopy(branches); conflicting["R2"]["interface"] = "shallow"
    with pytest.raises(ValueError, match="shared interface"):
        run(conflicting)
    crossing = deepcopy(branches); crossing["R2"].update(depth=20, depth_bounds=[10, 50])
    with pytest.raises(ValueError, match="ordered"):
        run(crossing)
    outside = deepcopy(branches); outside["R1"]["depth_bounds"] = [30, 50]
    with pytest.raises(ValueError, match="bounds"):
        run(outside)
    with pytest.raises(ValueError, match="uncertainty"):
        run(uncertainty=np.zeros(len(source)))
    def stop(value):
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError, match="cancelled"):
        run(report=stop)
    with pytest.raises(ValueError, match="depth cell"):
        run(max_step=1)


def test_interface_processing_flow_and_project_persistence(tmp_path):
    from seisdesk.core.dataset import SeismicData
    from seisdesk.core.picks import PickSet
    from seisdesk.core.volume import VelocityVolume
    from seisdesk.processing import create
    from seisdesk.workflow.project import Project
    from seisdesk.workflow.flow import FlowStep
    grid, source, receiver, phases, branches, observed = flat_example()
    # Three distinct events on each of four trace identities.
    s, r = grid.coordinates(source[::3]), grid.coordinates(receiver[::3])
    data = SeismicData(np.zeros((4, 32)), .002, dict(sx=s[:, 2], sy=s[:, 1], selev=-s[:, 0],
            gx=r[:, 2], gy=r[:, 1], gelev=-r[:, 0], fldr=np.arange(4)+1,
            offset=np.linalg.norm(s[:, 1:]-r[:, 1:], axis=1)))
    picks = PickSet(name="multi")
    for phase in ("P", "R1", "R2"):
        picks.new_phase(phase, "direct" if phase == "P" else "reflection")
    for index, (phase, time) in enumerate(zip(phases, observed)):
        trace = index//3
        picks.add(data.header("fldr")[trace], phase, data.header("offset")[trace], time, data.trace_ids[trace])
    project = Project(str(tmp_path)); project.context.picks["multi"] = picks
    VelocityVolume(np.full(grid.shape, 2000.), grid.spacing).save(tmp_path/"start.npz")
    step = FlowStep("joint_tomography_3d", dict(model="start.npz", pickset="multi", branches=json.dumps(branches),
        invert_interfaces=True, update_velocity=False, interface_warp=False, interface_max_shift=30,
        interface_step=20, iterations=3, radius=3))
    output = step.build()(data, project.context)
    result = project.context.results["joint_tomography_3d"]
    assert len(set(zip(result["trace_ids"], result["phases"]))) == 12
    with np.load(result["interface_bundle"], allow_pickle=False) as saved:
        assert "interface_0_sampled_interfaces" in saved
    project.add_dataset(output); project.save()
    restored = Project.load(str(tmp_path)).context.results["joint_tomography_3d"]
    np.testing.assert_allclose(restored["sampled_interfaces"]["deep"], 60)
    assert restored["interface_parameters"]["shallow"]["shape"] == [1, 1]
