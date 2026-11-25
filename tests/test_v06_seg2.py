"""Byte-level SEG-2 fixtures, timing controls, and optional independent ObsPy comparison."""
import struct

import numpy as np
import pytest

from seisdesk.core.seg2 import read_seg2, read_seg2_files


def fixture(path, rows, *, endian="<", code=4, headers=None, file_headers=None, payload=None, ns=None):
    def strings(values):
        result = b""
        for key, value in values.items():
            raw = f"{key} {value}".encode()+b"\0"
            result += struct.pack(endian+"H", len(raw)+2)+raw
        return result+b"\0\0"
    fstrings = strings(file_headers or {})
    ntr = len(rows)
    position = 32+4*ntr+len(fstrings)
    pointers, blocks = [], []
    for i, row in enumerate(rows):
        info = dict(SAMPLE_INTERVAL=.001, DELAY=0, CHANNEL_NUMBER=i+1)
        info.update((headers or [{}]*ntr)[i])
        text = strings(info)
        size = 32+len(text)
        size += -size % 4
        raw = payload if payload is not None else np.asarray(row, dtype=endian+{1:"i2", 2:"i4", 4:"f4", 5:"f8"}[code]).tobytes()
        samples = len(row) if ns is None else ns
        tdb = struct.pack(endian+"HHIIB", 0x4422, size, len(raw), samples, code).ljust(32, b"\0")
        blocks.append((tdb+text).ljust(size, b"\0")+raw)
        pointers.append(position)
        position += size+len(raw)
    header = struct.pack(endian+"HHHHB2sB2s", 0x3a55, 1, 4*ntr, ntr, 1, b"\0\0", 1, b"\n\0").ljust(32, b"\0")
    path.write_bytes(header+struct.pack(endian+f"{ntr}I", *pointers)+fstrings+b"".join(blocks))
    return path


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("code", [1, 2, 4, 5])
def test_standard_encodings_and_file_header_descaling(tmp_path, endian, code):
    values = [1, -3, 0, 32767]
    path = fixture(tmp_path/"record.dat", [values], endian=endian, code=code,
                   file_headers={"DESCALING_FACTOR": .5})
    out = read_seg2(path)
    np.testing.assert_array_equal(out.traces[0], np.array(values)*.5)
    np.testing.assert_array_equal(read_seg2(path, descale=False).traces[0], values)
    assert out.meta["seg2_trace_info"][0]["DESCALING_FACTOR"] == "0.5"


def packed_file(tmp_path, endian):
    words = [0x3210, 1, 0xfffe, 0xffff, 0x8000, 0xf001, 0x7fff, 0, 0xffff, 0x8001]
    return fixture(tmp_path/"packed.dat", [[0]*8], endian=endian, code=3,
                   payload=struct.pack(endian+"10H", *words))


@pytest.mark.parametrize("endian", ["<", ">"])
def test_packed20_sign_exponent_negative_zero_and_large_values(tmp_path, endian):
    out = read_seg2(packed_file(tmp_path, endian))
    expected = np.array([1, -2, 0, -262136, 65534, 0, 0, -1073676288], np.float32)
    np.testing.assert_array_equal(out.traces[0], expected)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_packed20_agrees_with_independent_obspy(tmp_path, endian):
    obspy = pytest.importorskip("obspy")
    path = packed_file(tmp_path, endian)
    reference = obspy.read(str(path), format="SEG2")
    np.testing.assert_array_equal(read_seg2(path).traces[0], reference[0].data.astype(np.float32))


def test_each_trace_delay_is_aligned_without_changing_integer_samples(tmp_path):
    rows = [np.arange(8), 10+np.arange(8), 20+np.arange(8)]
    path = fixture(tmp_path/"delays.dat", rows, headers=[{}, {"DELAY":.002}, {"DELAY":-.002}])
    out = read_seg2(path)
    np.testing.assert_array_equal(out.traces, [np.arange(6), [0, 0, 10, 11, 12, 13], np.arange(22, 28)])
    np.testing.assert_array_equal(out.header("seg2_delay_s"), [0, .002, -.002])
    assert out.t0 == 0
    with pytest.raises(ValueError, match="different start times"):
        read_seg2(path, remove_delay=False)


def test_fractional_delay_matches_analytic_bandlimited_signal(tmp_path):
    dt, start = .001, .00035
    signal = lambda time: np.sin(2*np.pi*20*time)+.3*np.sin(2*np.pi*70*time)
    row = signal(start+np.arange(800)*dt)
    path = fixture(tmp_path/"fractional.dat", [row], headers=[{"DELAY": start}])
    out = read_seg2(path)
    np.testing.assert_allclose(out.traces[0, 32:-32], signal(out.times[32:-32]), atol=5e-4)
    raw = read_seg2(path, remove_delay=False)
    assert raw.t0 == start
    np.testing.assert_array_equal(raw.traces[0], row.astype(np.float32))


def test_skew_is_preserved_and_only_applied_explicitly(tmp_path):
    path = fixture(tmp_path/"skew.dat", [np.arange(8)], headers=[{"SKEW": .002}])
    with pytest.warns(UserWarning, match="SKEW is retained"):
        raw = read_seg2(path)
    np.testing.assert_array_equal(raw.traces[0], np.arange(8))
    aligned = read_seg2(path, apply_skew=True)
    np.testing.assert_array_equal(aligned.traces[0], [0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    assert aligned.meta["seg2_timing"]["skew_applied"]


@pytest.mark.parametrize("header", [{"SAMPLE_INTERVAL": .002}, {"SAMPLE_INTERVAL": "nan"},
                                   {"SAMPLE_INTERVAL": 0}, {"DELAY": "bad"}, {"SKEW": "inf"}])
def test_invalid_or_nonuniform_trace_timing_is_rejected(tmp_path, header):
    path = fixture(tmp_path/"invalid.dat", [np.arange(8)]*2, headers=[{}, header])
    with pytest.raises(ValueError, match="SEG-2"):
        read_seg2(path)


def test_truncated_blocks_and_bad_packed_count_are_rejected(tmp_path):
    path = packed_file(tmp_path, "<")
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        read_seg2(path)
    path = fixture(tmp_path/"bad-count.dat", [[0]*7], code=3, payload=b"\0"*20)
    with pytest.raises(ValueError, match="divisible by four"):
        read_seg2(path)


def test_multiple_files_must_share_time_origin_when_not_aligned(tmp_path):
    a = fixture(tmp_path/"1.dat", [np.arange(8)])
    b = fixture(tmp_path/"2.dat", [np.arange(8)], headers=[{"DELAY":.002}])
    with pytest.raises(ValueError, match="different start times"):
        read_seg2_files([a, b], remove_delay=False)
    out = read_seg2_files([a, b])
    assert out.ntr == 2 and out.nt == 8 and len(out.meta["seg2_records"]) == 2
    np.testing.assert_array_equal(out.traces[1], [0, 0, 0, 1, 2, 3, 4, 5])
