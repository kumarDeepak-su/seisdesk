import json

import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.segy_io import read_any, read_su, write_su
from seisdesk.core.velocity import DepthModel, VelocityPicks
from seisdesk.processing import Context
from seisdesk.workflow.flow import Flow
from seisdesk.workflow.project import Project, load_dataset, save_dataset


def test_su_round_trip(tmp_path, marine):
    path = tmp_path / "line.su"
    write_su(marine, str(path))
    back = read_su(str(path))
    assert back.ntr == marine.ntr and back.nt == marine.nt and np.isclose(back.dt, marine.dt)
    assert np.allclose(back.traces, marine.traces)
    for key in ("fldr", "tracf", "cdp"):
        assert np.allclose(back.header(key), marine.header(key))
    assert np.allclose(back.header("sx"), marine.header("sx"), atol=0.01)
    assert np.allclose(back.header("offset"), marine.header("offset"))
    assert read_any(str(path)).ntr == marine.ntr


def test_segy_round_trip_with_scaled_coordinates(tmp_path):
    pytest.importorskip("segyio")
    from seisdesk.core.segy_io import read_segy, write_segy
    rng = np.random.default_rng(1)
    heads = {"fldr": np.repeat([10, 11], 5), "tracf": np.tile(np.arange(1, 6), 2),
             "sx": 512345.67 + np.zeros(10), "gx": 512345.67 + np.arange(10) * 12.5, "gelev": 101.25,
             "offset": np.arange(10) * 12.5}
    d = SeismicData(rng.standard_normal((10, 300)).astype(np.float32), 0.002, heads, t0=0.1, name="x")
    path = tmp_path / "x.sgy"
    write_segy(d, str(path))
    back = read_segy(str(path))
    assert back.ntr == 10 and back.nt == 300 and np.isclose(back.dt, 0.002)
    assert np.isclose(back.t0, 0.1)
    assert np.allclose(back.traces, d.traces, atol=1e-6)
    assert np.allclose(back.header("gx"), heads["gx"], atol=0.01)
    assert np.allclose(back.header("gelev"), 101.25, atol=0.01)
    assert np.allclose(back.header("fldr"), heads["fldr"])
    assert back.meta["textual_header"].startswith("C 1")


def test_dataset_npz_round_trip(tmp_path, land):
    path = tmp_path / "d.npz"
    save_dataset(land, str(path))
    back = load_dataset(str(path))
    assert np.array_equal(back.traces, land.traces) and back.dt == land.dt
    assert set(back.headers) == set(land.headers)
    assert back.meta["synthetic"]["v1"] == land.meta["synthetic"]["v1"]


def test_flow_json_and_run(marine_cmp):
    flow = Flow(name="test")
    flow.add("bandpass", corners=[4, 8, 50, 70]).add("nmo", velocity="0:1500,1.1:2100,2.2:2900").add("stack")
    flow.steps[0].enabled = False
    text = flow.to_json()
    again = Flow.from_json(text)
    assert [s.process for s in again.steps] == ["bandpass", "nmo", "stack"] and not again.steps[0].enabled
    progress = []
    ctx = Context(progress=progress.append)
    out, inter = again.run(marine_cmp, ctx, keep_intermediate=True)
    assert out.ntr == np.unique(marine_cmp.header("cdp")).size
    assert len(inter) == 2 and progress[-1] == 1.0
    assert all(0.0 <= p <= 1.0 for p in progress)
    assert out.history[-1].startswith("CMP stack")


def test_flow_cancel(marine_cmp):
    flow = Flow().add("agc").add("stack")
    ctx = Context(cancelled=lambda: True)
    with pytest.raises(InterruptedError):
        flow.run(marine_cmp, ctx)


def test_project_save_load(tmp_path, marine, land):
    proj = Project()
    proj.add_dataset(marine.copy())
    proj.add_dataset(land.copy(), "land line")
    proj.context.velocity = VelocityPicks.constant(2000.0)
    proj.context.depth_model = DepthModel.constant(2000.0, 10, 5, 10.0, 25.0)
    ps = proj.pick_set("fb")
    ps.add(1, ps.new_phase(), 10.0, 0.05)
    proj.flows["f"] = Flow().add("agc", window=0.3)
    folder = tmp_path / "proj"
    proj.save(str(folder))
    back = Project.load(str(folder))
    assert list(back.datasets) == list(proj.datasets)
    assert np.allclose(back.datasets["land line"].traces, land.traces)
    assert len(back.context.velocity) == 1 and back.context.depth_model.shape == (10, 5)
    assert back.context.picks["fb"].count() == 1
    assert back.flows["f"].steps[0].params["window"] == 0.3
    proj.remove_dataset("land line")
    proj.save(str(folder))
    assert len(list((folder / "datasets").glob("*.npz"))) == 1
    doc = json.loads((folder / "project.json").read_text())
    assert doc["format"] == 1


def test_unique_dataset_names(marine):
    proj = Project()
    a = proj.add_dataset(marine.copy(), "x")
    b = proj.add_dataset(marine.copy(), "x")
    assert a == "x" and b == "x (2)"
    assert proj.rename_dataset("x (2)", "y") == "y"


def test_cli(tmp_path, capsys):
    from seisdesk.cli import main
    assert main(["processes"]) == 0
    assert "nmo" in capsys.readouterr().out
    assert main(["processes", "kirchhoff_psdm"]) == 0
    out = tmp_path / "m.su"
    assert main(["synthetic", "marine", str(out)]) == 0
    assert main(["info", str(out)]) == 0
    flow = tmp_path / "f.json"
    Flow(name="f").add("agc", window=0.3).add("sort", keys="cdp,offset").save(str(flow))
    res = tmp_path / "r.su"
    assert main(["run", str(flow), str(out), str(res)]) == 0
    assert read_su(str(res)).ntr == read_su(str(out)).ntr
    assert main(["info", str(tmp_path / "missing.su")]) == 1


def test_seg2_round_trip_delay_and_multi_file(tmp_path, land):
    from seisdesk.core.seg2 import is_seg2, read_seg2, read_seg2_files, write_seg2
    shots = land.gather_values("fldr")[:2]
    paths = []
    for f in shots:
        p = tmp_path / f"{int(f):04d}.dat"
        write_seg2(land, str(p), record=f)
        paths.append(str(p))
    assert is_seg2(paths[0]) and not is_seg2(__file__)
    g = land.gather("fldr", shots[0], secondary=None)
    one = read_seg2(paths[0])
    assert one.ntr == g.ntr and one.nt == g.nt and np.isclose(one.dt, land.dt)
    assert np.allclose(one.traces, g.traces, atol=1e-6)
    assert np.allclose(one.header("gx"), g.header("gx"), atol=1e-3)
    assert np.allclose(one.header("tracf"), g.header("tracf"))
    assert np.all(one.header("fldr") == shots[0])
    both = read_seg2_files(paths)
    assert both.ntr == 2 * g.ntr and set(np.unique(both.header("fldr"))) == set(shots)
    assert read_any(paths[1]).ntr == g.ntr
    pre = g.copy(t0=-0.01)                                   # 10 ms of pre-trigger samples
    path = tmp_path / "pre.dat"
    write_seg2(pre, str(path))
    back = read_seg2(str(path))
    assert back.t0 == 0.0 and back.nt == g.nt - 5
    assert np.allclose(back.traces, g.traces[:, 5:], atol=1e-6)


def test_shot_table_geometry_and_table_statics(tmp_path):
    from seisdesk.core.geometry import shot_table_geometry
    from seisdesk.processing import create
    tr = np.zeros((6, 200), dtype=np.float32)
    tr[:, 100] = 1.0
    d = SeismicData(tr, 0.001, {"fldr": [1, 1, 1, 2, 2, 2], "tracf": [1, 2, 3, 1, 2, 3]})
    table = tmp_path / "geom.csv"
    table.write_text("file,ffid,shot_x,shot_z,first_rec_x,rec_dx,nchan\n"
                     "raw/1.dat,1,-1.0,,0.0,2.0,3\nraw/2.dat,2,7.0,,0.0,2.0,3\n")
    topo = tmp_path / "topo.txt"
    topo.write_text("x z\n-10 100\n20 130\n")
    out = shot_table_geometry(d, str(table), str(topo))
    gx = np.array([0, 2, 4, 0, 2, 4.0])
    assert np.allclose(out.header("gx"), gx) and np.allclose(out.header("sx"), [-1, -1, -1, 7, 7, 7])
    assert np.allclose(out.header("offset"), [1, 3, 5, -7, -5, -3])
    assert np.allclose(out.header("gelev"), gx + 110.0) and np.isclose(out.header("selev")[0], 109.0)
    assert out.has_header("cdp") and out.meta["geometry"]["bin_size"] == 1.0
    ctx = Context(work_dir=str(tmp_path))
    via_step = create("shot_table_geometry", table="geom.csv", topography="topo.txt")(d, ctx)
    assert np.allclose(via_step.header("gx"), gx)
    (tmp_path / "st.txt").write_text("-5 4\n10 4\n")
    shifted = create("table_statics", file="st.txt", mode="remove")(out, ctx)
    assert np.all(np.argmax(shifted.traces, axis=1) == 92)
    assert np.allclose(shifted.header("tstat"), -8.0)
    with pytest.raises(ValueError):
        shot_table_geometry(SeismicData(tr, 0.001, {"fldr": [9] * 6, "tracf": [1] * 6}), str(table))
