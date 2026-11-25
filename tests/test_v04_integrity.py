"""Regression cases for trace identity and nonuniform recording metadata."""
import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.headers import su_header_dtype
from seisdesk.core.picks import PickSet, pick_table
from seisdesk.core.segy_io import read_segy, read_su, write_segy, write_su
from seisdesk.processing import Context, create
from seisdesk.workflow.project import Project


def duplicate_offsets():
    return SeismicData(np.ones((2, 101)), 0.002,
                       {"fldr": [1, 1], "offset": [100, 100], "sx": [0, 0], "sy": [0, 0],
                        "gx": [100, 0], "gy": [0, 100], "tracf": [1, 2]})


def test_equal_offsets_keep_picks_and_geometry_after_sort_and_save(tmp_path):
    d = duplicate_offsets()
    ps = PickSet(name="mixed")
    ps.new_phase("R1", "reflection")
    ps.add(1, "R1", 100, 0.12, d.trace_ids[0])
    ps.add(1, "R1", 100, 0.16, d.trace_ids[1])
    ps.add(1, "R1", 100, 0.11, d.trace_ids[0])
    assert ps.count() == 2
    swapped = d.subset([1, 0]).copy()
    assert np.allclose(ps.times_for(swapped, "R1"), [0.16, 0.11])
    proj = Project()
    proj.add_dataset(swapped, "shot")
    proj.context.picks[ps.name] = ps
    proj.save(str(tmp_path / "project"))
    loaded = Project.load(str(tmp_path / "project"))
    back = loaded.context.picks["mixed"]
    assert back.phase_type("R1") == "reflection"
    rows = {r["trace_id"]: r for r in pick_table(back, loaded.datasets["shot"])}
    assert rows[d.trace_ids[0]]["gx"] == 100 and rows[d.trace_ids[0]]["gy"] == 0
    assert rows[d.trace_ids[1]]["gx"] == 0 and rows[d.trace_ids[1]]["gy"] == 100
    assert back.remove_nearest(1, 100, 0.11, "R1", trace_id=d.trace_ids[0])
    assert back.count() == 1 and back.records(1, "R1")[0][2] == d.trace_ids[1]


def test_legacy_ambiguous_geometry_requires_repick():
    ps = PickSet.from_dict({"phase_order": ["P"], "data": {"1": {"P": [[100, 0.1]]}}})
    assert ps.count() == 1 and ps.phase_type("P") == "unclassified"
    with pytest.raises(ValueError, match="multiple traces"):
        pick_table(ps, duplicate_offsets())


def test_picked_mute_uses_trace_identity():
    d = duplicate_offsets()
    ps = PickSet(name="mute")
    for i, t in enumerate([0.04, 0.12]):
        ps.add(1, "top", 100, t, d.trace_ids[i])
    result = create("picked_mute", picks="mute", phase="top", kind="top", taper=0.0)(
        d, Context(picks={"mute": ps}))
    assert result.traces[0, 40] == 1 and result.traces[1, 40] == 0


def test_mixed_event_types_excluded_from_refraction_layers():
    d = duplicate_offsets()
    ps = PickSet()
    ps.new_phase("direct", "direct")
    ps.new_phase("head", "refraction")
    ps.new_phase("reflector", "reflection")
    for x in [10, 20, 30]:
        ps.add(1, "direct", x, x / 500)
        ps.add(1, "head", x, 0.02 + x / 1500)
        ps.add(1, "reflector", x, np.hypot(0.1, x / 2000))
    ctx = Context(picks={"picks": ps})
    create("refraction_analysis", pickset="picks")(d, ctx)
    assert [f["phase"] for f in ctx.results["refraction"]["fits"]] == ["direct", "head"]
    with pytest.raises(ValueError, match="direct and refracted"):
        create("refraction_analysis", pickset="picks", phases="direct,reflector")(d, ctx)


@pytest.mark.parametrize("field,value,message", [("delrt", 50, "recording delays"),
                                                ("dt", 4000, "sample intervals"),
                                                ("ns", 100, "sample counts")])
def test_su_rejects_inconsistent_headers(tmp_path, field, value, message):
    path = tmp_path / "test.su"
    write_su(duplicate_offsets(), str(path))
    raw = bytearray(path.read_bytes())
    h = np.frombuffer(raw, dtype=su_header_dtype().newbyteorder("<"), count=1, offset=240 + 101 * 4)
    h[field][0] = value
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        read_su(str(path))


def test_su_rejects_truncated_trace(tmp_path):
    path = tmp_path / "test.su"
    write_su(duplicate_offsets(), str(path))
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="truncated"):
        read_su(str(path))


def test_segy_rejects_variable_delay_without_changing_source(tmp_path):
    segyio = pytest.importorskip("segyio")
    path = tmp_path / "test.sgy"
    write_segy(duplicate_offsets(), str(path))
    with segyio.open(str(path), "r+", ignore_geometry=True) as f:
        f.header[1][segyio.TraceField.DelayRecordingTime] = 100
    before = path.read_bytes()
    with pytest.raises(ValueError, match="recording delays"):
        read_segy(str(path))
    assert path.read_bytes() == before


@pytest.mark.parametrize("delay", [-0.02, 0.12])
def test_uniform_delay_roundtrip(tmp_path, delay):
    d = duplicate_offsets().copy(t0=delay)
    p = tmp_path / "test.su"
    write_su(d, str(p))
    back = read_su(str(p))
    assert back.t0 == pytest.approx(delay)
    assert np.array_equal(d.traces, back.traces)
