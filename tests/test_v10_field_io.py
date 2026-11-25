"""Timing/amplitude checks, binary fixtures, geometry identities and project round trips."""
import struct
import numpy as np
import pytest

from seisdesk.core.dataset import SeismicData
from seisdesk.core.sampling import common_samples
from seisdesk.core.segy_io import read_segy, read_su, write_segy, write_su
from seisdesk.core.segd import decode_segd, read_segd, is_segd
from seisdesk.core.sps import apply_sps
from seisdesk.workflow.project import Project


def test_mixed_rates_fractional_delays_and_antialiasing():
    signal = lambda t: np.sin(2*np.pi*23*t)+.3*np.cos(2*np.pi*61*t)
    starts, dt = [-.01325, .00275], [.001, .002]
    rows = [signal(start+np.arange(2001)*d) for start, d in zip(starts, dt)]
    out, newdt, t0, meta = common_samples(rows, dt, starts, .004, "intersection")
    times = t0+np.arange(out.shape[1])*newdt
    np.testing.assert_allclose(out[:, 50:-50], np.broadcast_to(signal(times)[None, 50:-50], out[:, 50:-50].shape), atol=3e-5)
    high = np.sin(2*np.pi*180*np.arange(4001)*.001)
    low, *_ = common_samples([high], [.001], [0], .004, "intersection")
    assert np.sqrt(np.mean(low[0, 50:-50]**2)) < 1e-4
    assert meta["original_start_s"] == starts


def test_union_padding_exact_samples_intersection_and_bounds():
    out, dt, t0, _ = common_samples([np.arange(4), 10+np.arange(3)], [.002]*2, [-.002, .002], .002, "union")
    np.testing.assert_array_equal(out, [[0,1,2,3,0], [0,0,10,11,12]])
    assert t0 == -.002
    for options, match in [({}, "differ"), ({"window":"union"}, "explicit target"),
                           ({"target_dt":.002, "window":"union", "max_mb":1e-6}, "limit")]:
        with pytest.raises(ValueError, match=match):
            common_samples([np.ones(10), np.ones(12)], [.002, .004], [0, .002], **options)
    with pytest.raises(InterruptedError):
        common_samples([np.ones(12)], [.001], [0], cancelled=lambda: True)


def variable_fixture(path, endian, su, code=5, ext=False):
    chunks = []
    if not su:
        b = bytearray(400)
        for offset, value, fmt in [(16,1000,"H"), (20,9,"H"), (24,code,"H"), (54,1,"H"), (300,256,"H"), (304,int(ext),"h")]:
            struct.pack_into(endian+fmt, b, offset, value)
        chunks = [b"C 1 TIMING FIXTURE".ljust(3200), bytes(b)]
        if ext:
            chunks.append(b"((SEG: EndText))".ljust(3200))
    for k, (ns, dt, start) in enumerate([(9,1000,-2), (5,2000,0)]):
        h = bytearray(240)
        struct.pack_into(endian+"ii", h, 8, 4, k+1)
        struct.pack_into(endian+"h", h, 108, start)
        struct.pack_into(endian+"HH", h, 114, ns, dt)
        chunks.extend([bytes(h), np.arange(ns, dtype=endian+{2:"i4",3:"i2",5:"f4",8:"i1"}[code]).tobytes()])
    path.write_bytes(b"".join(chunks))
    return str(path)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("code", [2,3,5,8])
def test_variable_segy_lengths_rates_and_text_extensions(tmp_path, endian, code):
    path = variable_fixture(tmp_path/"test.sgy", endian, False, code, ext=True)
    out = read_segy(path, endian="little" if endian == "<" else "big", target_dt=.001, window="union")
    assert out.dt == .001 and out.t0 == -.002 and out.nt == 11
    np.testing.assert_array_equal(out.traces[0, :9], np.arange(9))
    np.testing.assert_allclose(out.traces[1, 2::2], np.arange(5), atol=1e-6)
    np.testing.assert_array_equal(out.header("input_ns"), [9,5])
    assert "TIMING FIXTURE" in out.meta["textual_header"]


@pytest.mark.parametrize("endian", ["<", ">"])
def test_su_nonuniform_and_corrupt_tail(tmp_path, endian):
    p = tmp_path/"test.su"
    variable_fixture(p, endian, True)
    out = read_su(str(p), byteorder=endian, target_dt=.001, window="intersection")
    assert out.t0 == 0 and out.nt == 7
    np.testing.assert_array_equal(out.traces[0], np.arange(2,9))
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        read_su(str(p), byteorder=endian, target_dt=.001, window="union")


def test_seg2_mixed_rate_record_and_multiple_files(tmp_path):
    from test_v06_seg2 import fixture
    from seisdesk.core.seg2 import read_seg2, read_seg2_files
    a = fixture(tmp_path/"a.dat", [np.arange(9), np.arange(5)],
                headers=[{}, {"SAMPLE_INTERVAL":.002, "DELAY":.002}])
    b = fixture(tmp_path/"b.dat", [np.arange(5)], headers=[{"SAMPLE_INTERVAL":.002, "DELAY":.004}])
    out = read_seg2(a, target_dt=.001, window="union")
    assert out.nt == 11
    both = read_seg2_files([a,b], target_dt=.001, window="union")
    assert both.ntr == 3 and both.nt == 13
    np.testing.assert_array_equal(both.header("input_dt_s"), [.001,.002,.002])
    np.testing.assert_allclose(both.traces[2, 4::2], np.arange(5), atol=1e-6)


@pytest.mark.parametrize("writer", [write_su, write_segy])
@pytest.mark.parametrize("dt,t0", [(.0010005,0), (.001,.0005), (.001,40)])
def test_export_never_rounds_unrepresentable_timing(tmp_path, writer, dt, t0):
    data = SeismicData(np.ones((1,10)),dt,t0=t0)
    path = str(tmp_path/"out")
    with pytest.raises(ValueError, match="exact timing"):
        writer(data,path)
    assert not (tmp_path/"out").exists()


def test_segy_feet_convert_consistently_and_round_trip(tmp_path):
    import segyio
    data=SeismicData(np.arange(20,dtype=np.float32)[None,:],.001,{"sx":[1000],"gx":[1100],"selev":[50],"offset":[100]})
    path=str(tmp_path/"feet.sgy"); write_segy(data,path)
    with segyio.open(path,"r+",ignore_geometry=True) as f:f.bin[segyio.BinField.MeasurementSystem]=2
    uniform=read_segy(path); aligned=read_segy(path,target_dt=.001,window="union")
    for key in ('sx','gx','selev','offset'):
        np.testing.assert_allclose(uniform.header(key),data.header(key)*.3048)
        np.testing.assert_allclose(aligned.header(key),uniform.header(key))
    back=str(tmp_path/"metres.sgy"); write_segy(aligned,back)
    np.testing.assert_allclose(read_segy(back).header('sx'),[304.8],atol=.001)


def segd_fixture(path, code=8058, payload=None, ns=8, mp=(0, 0), skew=0):
    # Independent byte construction from SEG-D 2.1 table offsets, one channel set.
    g1, g2, cs, th, ext = (bytearray(n) for n in (32,32,32,20,32))
    g1[:4] = bytes.fromhex("0001"+str(code)); g1[11]=0x10; g1[22]=16
    g1[27]=1; g1[28]=1
    g2[10:12]=b"\x02\x01"; g2[18]=2; g2[21]=7
    cs[:2]=b"\x01\x01"; cs[4:6]=(4).to_bytes(2,"big"); cs[6:8]=bytes(mp)
    cs[8:10]=b"\x00\x01"; cs[10]=0x10; cs[28]=1; cs[29]=1
    th[:6]=bytes.fromhex("000101010001"); th[9]=1; th[10]=skew
    ext[:3]=(10).to_bytes(3,"big"); ext[3:6]=(20).to_bytes(3,"big"); ext[6]=1
    ext[7:10]=ns.to_bytes(3,"big"); ext[20]=2
    if payload is None:
        payload = np.arange(ns,dtype=">f4").tobytes()
    path.write_bytes(g1+g2+cs+th+ext+payload)
    return path


@pytest.mark.parametrize("code,raw,values", [
    (8015, bytes.fromhex("0123 4000 bfff 0000 ffff"), [.5,-1,0,0]),
    (8022, bytes.fromhex("08 87 18 97"), [.5,-.5,2,-2]),
    (8024, bytes.fromhex("0800 87ff 1800 97ff"), [.5,-.5,2,-2]),
    (8036, bytes.fromhex("000001 ffffff 7fffff 800000"), [1,-1,8388607,-8388608]),
    (8038, bytes.fromhex("00000001 ffffffff 00000000 80000000"), [1,-1,0,-2147483648]),
    (8042, bytes.fromhex("10 90 30 b0"), [.5,-.5,8,-8]),
    (8044, bytes.fromhex("1000 9000 3000 b000"), [.5,-.5,8,-8]),
    (8048, bytes.fromhex("41100000 c1100000 00000000 40800000"), [1,-1,0,.5]),
    (8058, bytes.fromhex("3f800000 bf800000 00000000 3f000000"), [1,-1,0,.5]),
])
def test_all_segd_encodings_and_mp_descaling(tmp_path, code, raw, values):
    np.testing.assert_array_equal(decode_segd(raw,code),values)
    path = segd_fixture(tmp_path/"trace.segd",code,raw,4,mp=(0,4))
    assert is_segd(path)
    data = read_segd(path)
    np.testing.assert_array_equal(data.traces[0], np.asarray(values,np.float32)*2)
    assert data.header("segd_line_sequence")[0] == 7
    assert "sx" not in data.headers


def test_segd_skew_concatenation_and_rejection(tmp_path):
    p=segd_fixture(tmp_path/"a.segd",skew=128)
    original=p.read_bytes()
    out=read_segd(p)
    assert out.t0 == .0005
    p.write_bytes(original+original)
    assert read_segd(p).ntr == 2
    p.write_bytes(original[:-1])
    with pytest.raises(ValueError,match="truncated"):
        read_segd(p)
    raw=bytearray(original); raw[32+10]=3; p.write_bytes(raw)
    with pytest.raises(ValueError,match="revision"):
        read_segd(p)


def point_line(kind, line, point, index, x, y, z, code="G1", static=0):
    return f"{kind}{line:10.2f}{point:10.2f}  {index:1d}{code:2s}{static:4d}{0:4.1f}{0:4d}{0:2d}{0:6.1f}{x:9.1f}{y:10.1f}{z:6.1f}{1:3d}000000"


def relation_line(first=1,last=4,components=1,receiver_first=40,receiver_last=10,tape="A"):
    return f"X{tape:<6}{7:8d}11{10:10.2f}{1:10.2f}1{first:5d}{last:5d}{components:1d}{20:10.2f}{receiver_first:10.2f}{receiver_last:10.2f}2"


def sps_fixture(tmp_path, relation=None):
    head="H00 SPS format version num".ljust(32)+"SPS 2.1;\n"+"H201Factor to metre".ljust(32)+"    0.30480000\n"
    s,r,x=[tmp_path/name for name in ("s.sps","r.rps","x.xps")]
    s.write_text(head+point_line("S",10,1,1,1000,2000,100,"V1",4)+"\n")
    r.write_text(head+"\n".join(point_line("R",20,p,2,1000+p,2000,90+p/10,static=-1) for p in (10,20,30,40))+"\n")
    x.write_text(head+(relation or relation_line())+"\n")
    return s,r,x


def test_sps_reverse_receiver_order_units_statics_and_persistence(tmp_path):
    paths=sps_fixture(tmp_path)
    data=SeismicData(np.ones((4,10)),.001,{"fldr":[7]*4,"tracf":[1,2,3,4]})
    out=apply_sps(data,*paths,apply_statics=True)
    np.testing.assert_allclose(out.header("gx"),np.array([1040,1030,1020,1010])*.3048)
    np.testing.assert_array_equal(out.header("selev"),[100]*4)  # elevation is metres, not grid feet
    np.testing.assert_array_equal(out.header("sps_receiver_index"),[2]*4)
    np.testing.assert_array_equal(out.header("sstat"),[4]*4)
    np.testing.assert_array_equal(out.traces,data.traces)
    np.testing.assert_array_equal(out.trace_ids,data.trace_ids)
    project=Project(); project.add_dataset(out); project.save(str(tmp_path/"project"))
    back=Project.load(str(tmp_path/"project")).datasets[out.name]
    assert back.meta["sps_geometry"] == out.meta["sps_geometry"]
    np.testing.assert_array_equal(back.header("gx"),out.header("gx"))


def test_sps_multicomponent_missing_and_ambiguous_matches(tmp_path):
    paths=sps_fixture(tmp_path,relation_line(last=6,components=3,receiver_first=10,receiver_last=20))
    data=SeismicData(np.ones((6,8)),.001,{"fldr":[7]*6,"tracf":np.arange(1,7)})
    out=apply_sps(data,*paths)
    np.testing.assert_array_equal(out.header("sps_receiver_point"),[10,10,10,20,20,20])
    np.testing.assert_array_equal(out.header("sps_component"),[1,2,3,1,2,3])
    x=paths[2]
    x.write_text(x.read_text()+relation_line(last=6,components=3,receiver_first=10,receiver_last=20,tape="B")+"\n")
    with pytest.raises(ValueError,match="2 SPS relations"):
        apply_sps(data,*paths)
    apply_sps(data,*paths,tape="A")
    data.set_header("tracf",[1,2,3,4,5,99])
    with pytest.raises(ValueError,match="0 SPS relations"):
        apply_sps(data,*paths,tape="A")


def test_batch_import_preserves_original_timing_in_project(tmp_path):
    from seisdesk.cli import main
    p=tmp_path/'input.su';variable_fixture(p,'<',True)
    assert main(['import',str(p),str(tmp_path/'project'),'--dt-ms','1','--window','union']) == 0
    project=Project.load(str(tmp_path/'project')); data=next(iter(project.datasets.values()))
    assert data.t0 == -.002 and data.nt == 11
    np.testing.assert_array_equal(data.header('input_ns'),[9,5])
