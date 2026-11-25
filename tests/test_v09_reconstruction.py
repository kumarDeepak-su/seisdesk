import numpy as np
import pytest

from seisdesk.core.reconstruction import reconstruct_survey
from seisdesk.core.dataset import SeismicData
from seisdesk.processing import Context, create


def plane_waves(shape, nt=64):
    axes = np.indices(shape, dtype=float)
    a = sum((i+1)*axis/n for i, (axis, n) in enumerate(zip(axes, shape)))
    b = sum((-1)**i*axis/n for i, (axis, n) in enumerate(zip(axes, shape)))
    t = np.arange(nt)/nt
    # Independent analytic multidimensional events, with different dips/frequencies.
    return (np.cos(2*np.pi*(a[..., None]+4*t)) + .4*np.sin(2*np.pi*(b[..., None]+9*t))).reshape(-1, nt).astype(np.float32)


@pytest.mark.parametrize("shape", [(16, 16), (5, 5, 5, 5)])
def test_3d_and_5d_reconstruct_held_out_plane_waves(shape, tmp_path):
    truth = plane_waves(shape)
    known = np.random.default_rng(61).random(len(truth)) > .35
    progress = []
    result = reconstruct_survey(truth[known], np.flatnonzero(known), shape, iterations=90,
                                 final_threshold=.01, work_dir=tmp_path, report=progress.append)
    relative = np.linalg.norm(result[~known]-truth[~known])/np.linalg.norm(truth[~known])
    assert relative < .05
    np.testing.assert_array_equal(result[known], truth[known])
    assert progress[-1] == 1 and np.all(np.diff(progress) >= 0)
    assert not list(tmp_path.iterdir())


def test_live_zero_traces_memory_validation_and_cancel_cleanup(tmp_path):
    truth = plane_waves((8, 8)); truth[0] = 0
    known = np.arange(0, 64, 2)
    result = reconstruct_survey(truth[known], known, (8, 8), iterations=10, work_dir=tmp_path)
    np.testing.assert_array_equal(result[0], 0)
    def cancel(fraction):
        if fraction > 0:
            raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        reconstruct_survey(truth[known], known, (8, 8), report=cancel, work_dir=tmp_path)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="memory"):
        reconstruct_survey(truth[:2], np.array([0, 1]), (1000000, 1000000), memory_gb=.01)
    with pytest.raises(ValueError, match="unique"):
        reconstruct_survey(truth[:2], np.array([0, 0]), (8, 8))


def test_survey_geometry_identities_mask_and_explicit_duplicate_policy(tmp_path):
    ny, nx = 8, 8
    truth = plane_waves((ny, nx)); yy, xx = np.indices((ny, nx))
    live = np.ones(ny*nx); live[[11, 20, 29, 31]] = 0
    truth[0] = 0
    original = SeismicData(truth, .004, dict(cdpx=xx.ravel()*10+1000, cdpy=yy.ravel()*20+2000, live=live),
                           trace_ids=["long-imported-trace-identifier-"+"x"*60+str(i) for i in range(len(truth))])
    process = create("survey_pocs", dx=10, dy=20, live_key="live", iterations=30)
    context = Context(work_dir=str(tmp_path))
    out = process(original, context)
    assert out.ntr == ny*nx and out.header("interp").sum() == 4
    np.testing.assert_array_equal(out.traces[live > 0], original.traces[live > 0])
    np.testing.assert_array_equal(out.trace_ids[live > 0], original.trace_ids[live > 0])
    assert len(set(out.trace_ids)) == out.ntr
    np.testing.assert_array_equal(out.traces[0], 0)
    np.testing.assert_allclose(out.header("gx"), out.header("cdpx"))
    assert context.results["survey_pocs"]["observed_fold"].sum() == 60
    shifted = original.copy(); shifted.headers["cdpx"][1] += 3
    with pytest.raises(ValueError, match="tolerance"):
        process(shifted, context)
    duplicate = original.copy(); duplicate.headers["cdpx"][1] = duplicate.headers["cdpx"][0]
    with pytest.raises(ValueError, match="multiple traces"):
        process(duplicate, context)
    mean = create("survey_pocs", dx=10, dy=20, live_key="live", duplicates="mean", iterations=5)(duplicate, context)
    np.testing.assert_allclose(mean.traces[0], duplicate.traces[:2].mean(axis=0), atol=1e-7)
    assert mean.header("bin_fold")[0] == 2


def test_5d_signed_offset_geometry_and_gap_limit(tmp_path):
    shape = (3, 3, 3, 3)
    y, x, hy, hx = np.indices(shape).reshape(4, -1)*20
    hy, hx = hy-20, hx-20
    truth = plane_waves(shape)
    known = np.ones(len(truth), bool); known[40] = False
    data = SeismicData(truth[known], .004,
        dict(sx=(x-hx/2)[known], gx=(x+hx/2)[known], sy=(y-hy/2)[known], gy=(y+hy/2)[known]))
    context = Context(work_dir=str(tmp_path))
    out = create("survey_pocs", mode="5D prestack", dx=20, dy=20, dhx=20, dhy=20,
                 max_gap=1, iterations=20)(data, context)
    assert out.ntr == len(truth) and out.header("interp").sum() == 1
    np.testing.assert_allclose(out.header("gx")-out.header("sx"), hx)
    np.testing.assert_allclose(out.header("gy")-out.header("sy"), hy)
    assert context.results["survey_pocs"]["shape"] == shape
    assert out.meta["survey_reconstruction"]["generated_traces"] == 1
