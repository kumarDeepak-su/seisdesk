import numpy as np
import pytest

from seisdesk.core.tomography3d import NodeGrid3D, arrivals3d, invert3d


def nodes(grid, xyz):
    return grid.nodes_for(np.asarray(xyz, float))[0]


def test_3d_homogeneous_first_reflection_and_reciprocity():
    grid = NodeGrid3D((9, 11, 11), (10, 10, 10), radius=3)
    source = nodes(grid, [[0, 20, 20], [0, 20, 20], [0, 20, 20]])
    receiver = nodes(grid, [[0, 80, 80]]*3)
    phases = ['first', 'R1', 'R2']
    branches = {'first': {'kind': 'first'}, 'R1': {'kind': 'reflection', 'depth': 30},
                'R2': {'kind': 'reflection', 'depth': 60}}
    velocity = np.full(grid.shape, 2000.)
    predicted, matrix, hits = arrivals3d(grid, velocity, source, receiver, phases, branches)
    # Independent constant-velocity direct / mirror-source travel times.
    expected = np.sqrt(60**2+60**2+np.array([0, 60, 120])**2)/2000
    np.testing.assert_allclose(predicted, expected, rtol=1e-12)
    np.testing.assert_allclose(matrix @ (1/velocity.ravel()), predicted, rtol=1e-12)
    reverse = arrivals3d(grid, velocity, receiver, source, phases, branches)[0]
    np.testing.assert_allclose(reverse, predicted, rtol=1e-12)
    assert np.count_nonzero(hits) > 0 and len(predicted) == 3


def test_3d_head_wave_matches_two_layer_analytical_time():
    grid = NodeGrid3D((7, 3, 23), (1, 1, 1), radius=3)
    velocity = np.full(grid.shape, 1500.)
    velocity[4:] = 3000
    source = nodes(grid, [[0, 1, 1]])
    receiver = nodes(grid, [[0, 1, 21]])
    branches = {'head': dict(kind='head', depth=3)}
    predicted, matrix, _ = arrivals3d(grid, velocity, source, receiver, ['head'], branches)
    expected = 20/3000 + 2*3*np.sqrt(1/1500**2-1/3000**2)
    assert abs(predicted[0]-expected)/expected < .015
    np.testing.assert_allclose(matrix @ (1/velocity.ravel()), predicted, rtol=1e-12)
    np.testing.assert_allclose(arrivals3d(grid, velocity, receiver, source, ['head'], branches)[0], predicted)
    with pytest.raises(ValueError, match='faster'):
        arrivals3d(grid, np.full(grid.shape, 1500.), source, receiver, ['head'], branches)


def test_3d_slowness_derivative_topography_and_memory_validation():
    surface = np.zeros((7, 7)); surface[3, :] = 20
    grid = NodeGrid3D((7, 7, 7), (10, 10, 10), surface=surface, radius=2)
    rng = np.random.default_rng(40)
    slow = (1/2000)*(1+.05*rng.normal(size=grid.shape))
    source = nodes(grid, [[0, 0, 0], [0, 60, 60]])
    receiver = nodes(grid, [[0, 60, 60], [30, 0, 40]])
    time, matrix, _ = arrivals3d(grid, 1/slow, source, receiver, ['P', 'P'], {'P': dict(kind='first')})
    assert time[0] > np.sqrt(60**2+60**2)/2200
    direction = rng.normal(size=grid.size)*1e-6
    step = 1e-4
    forward = arrivals3d(grid, 1/(slow+step*direction.reshape(grid.shape)), source, receiver,
                         ['P', 'P'], {'P': dict(kind='first')}, rays=False)[0]
    backward = arrivals3d(grid, 1/(slow-step*direction.reshape(grid.shape)), source, receiver,
                          ['P', 'P'], {'P': dict(kind='first')}, rays=False)[0]
    np.testing.assert_allclose((forward-backward)/(2*step), matrix @ direction, rtol=1e-7, atol=1e-11)
    with pytest.raises(ValueError, match='above'):
        grid.nodes_for([[0, 30, 30]])
    with pytest.raises(ValueError, match='outside'):
        grid.nodes_for([[0, -1, 0]])
    with pytest.raises(ValueError, match='memory|GiB'):
        NodeGrid3D((1000, 1000, 1000), (1, 1, 1), memory_gb=.1)


def example():
    grid = NodeGrid3D((5, 5, 5), (10, 10, 10), radius=2)
    a = [[0, 0, 0], [0, 40, 0], [0, 0, 40], [0, 40, 40]]
    b = [[40, y, x] for y in (0, 20, 40) for x in (0, 20, 40)]
    source = nodes(grid, [s for s in a for _ in b])
    receiver = nodes(grid, b*len(a))
    phases = ['P']*len(source)
    branches = {'P': dict(kind='first', uncertainty_ms=.2)}
    observed = arrivals3d(grid, np.full(grid.shape, 2000.), source, receiver, phases, branches)[0]
    return grid, source, receiver, phases, branches, observed


def test_3d_inversion_uncertainty_and_cancellation():
    grid, source, receiver, phases, branches, observed = example()
    progress = []
    result = invert3d(grid, np.full(grid.shape, 1800.), source, receiver, observed, phases, branches,
                      iterations=4, smoothing=.5, damping=.03, bootstrap=3, seed=7, report=progress.append)
    assert result['rms'][-1] < .2*result['rms'][0]
    assert np.all(np.diff(result['weighted_misfit']) < 0)
    assert result['used'].all() and len(result['t_pred']) == len(observed)
    covered = result['hits'] > 0
    assert np.isfinite(result['velocity_std'][covered]).all()
    assert np.any(result['velocity_std'][covered] > 0)
    assert np.isnan(result['velocity_std'][~covered]).all()
    assert np.all(result['velocity_p95'][covered] >= result['velocity_p05'][covered])
    assert progress[-1] == 1 and np.all(np.diff(progress) >= -1e-12)
    def cancelled(fraction):
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        invert3d(grid, np.full(grid.shape, 1800.), source, receiver, observed, phases, branches, report=cancelled)
    with pytest.raises(ValueError, match='uncertainty'):
        invert3d(grid, np.full(grid.shape, 1800.), source, receiver, observed, phases, branches,
                 uncertainty=np.zeros(len(observed)))


def test_3d_processing_preserves_arrival_identities_uncertainty_and_project(tmp_path):
    import csv
    import json
    from seisdesk.core.dataset import SeismicData
    from seisdesk.core.picks import PickSet
    from seisdesk.core.volume import VelocityVolume
    from seisdesk.processing import create
    from seisdesk.workflow.project import Project
    grid, source, receiver, phases, branches, observed = example()
    s, r = grid.coordinates(source), grid.coordinates(receiver)
    data = SeismicData(np.zeros((len(source), 32)), .002,
        dict(sx=s[:, 2], sy=s[:, 1], selev=-s[:, 0], gx=r[:, 2], gy=r[:, 1], gelev=-r[:, 0],
             fldr=source+1, offset=np.linalg.norm(s[:, 1:]-r[:, 1:], axis=1)))
    picks = PickSet(name="arrivals")
    picks.new_phase("P", "direct"); picks.new_phase("Q", "unclassified")
    # Two selected labels per identified trace must survive the entire pipeline.
    for i, time in enumerate(observed):
        for phase in ("P", "Q"):
            picks.add(data.header("fldr")[i], phase, data.header("offset")[i], time, data.trace_ids[i])
    project = Project(str(tmp_path)); project.context.picks["arrivals"] = picks
    volume = VelocityVolume(np.full(grid.shape, 1800.), grid.spacing)
    volume.save(tmp_path/"start.npz")
    with (tmp_path/"sigma.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["trace_id", "phase", "uncertainty_ms"])
        writer.writerow([data.trace_ids[0], "Q", .7])
    branches["Q"] = dict(kind="first", uncertainty_ms=.2)
    process = create("joint_tomography_3d", model="start.npz", pickset="arrivals",
                     branches=json.dumps(branches), uncertainties="sigma.csv", iterations=3)
    output = process(data, project.context)
    result = project.context.results[process.name]
    assert len(result["observed"]) == 2*data.ntr
    assert set(zip(result["trace_ids"], result["phases"])) == set(zip(np.tile(data.trace_ids, 2), ["P"]*data.ntr+["Q"]*data.ntr))
    special = [i for i, key in enumerate(zip(result["trace_ids"], result["phases"])) if key == (data.trace_ids[0], "Q")][0]
    assert result["uncertainty"][special] == pytest.approx(.0007)
    assert output.domain == "depth" and output.meta["volume"]["shape"] == list(grid.shape)
    np.testing.assert_allclose(VelocityVolume.load(result["final_model"]).values, result["velocity"])
    with open(result["arrival_qc"], newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 2*data.ntr and {row["phase"] for row in exported} == {"P", "Q"}
    project.add_dataset(output); project.save()
    restored = Project.load(str(tmp_path)).context.results[process.name]
    np.testing.assert_array_equal(restored["trace_ids"], result["trace_ids"])
    np.testing.assert_allclose(restored["velocity"], result["velocity"])
    (tmp_path/"sigma.csv").write_text("trace_id,phase,uncertainty_ms\nunknown,P,1\n")
    with pytest.raises(ValueError, match="unknown or repeated"):
        process(data, project.context)
