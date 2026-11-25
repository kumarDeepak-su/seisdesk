"""SEG-D rev 2.0/2.1 demultiplexed disk records, all nine standard encodings.

No tape labels/wrappers or proprietary header interpretation. Sample counts must
be present in trace extension 1. Source/receiver line and point numbers are survey
identifiers, not coordinates: use SPS for positions. MP descaling gives millivolts,
not instrument-response correction. References: SEG-D 2.1 §§5–8 (January 2006).
"""
from __future__ import annotations

import hashlib
import os
import numpy as np

from .dataset import SeismicData
from .field_io import decode_samples
from .sampling import common_samples, timing_headers


def _bcd(raw):
    n = 0
    for byte in raw:
        a, b = byte >> 4, byte & 15
        if a > 9 or b > 9:
            raise ValueError("invalid SEG-D BCD field (wrong revision, tape wrapper or corrupt header)")
        n = n*100+a*10+b
    return n


def _uint(raw):
    return int.from_bytes(raw, "big")


def _fixed(raw):
    return int.from_bytes(raw, "big", signed=True)/65536


def _size(code, ns):
    sizes = {8022: 1, 8024: 2, 8036: 3, 8038: 4, 8042: 1, 8044: 2, 8048: 4, 8058: 4}
    if code == 8015:
        if ns % 4:
            raise ValueError("SEG-D 8015 sample count must be divisible by four")
        return ns//4*10
    if code not in sizes:
        raise ValueError(f"SEG-D code {code} is unsupported; demultiplexed rev 2 encodings are required")
    return sizes[code]*ns


def decode_segd(raw, code):
    """Decode unscaled big-endian SEG-D samples (including one's-complement fractions)."""
    if code in (8038, 8048, 8058):
        return decode_samples(raw, {8038: 2, 8048: 1, 8058: 5}[code], ">")
    if code == 8036:
        b = np.frombuffer(raw, dtype="u1").astype(np.int64).reshape(-1, 3)
        x = (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
        return np.where(x & 0x800000, x-0x1000000, x).astype(float)
    if code == 8015:
        words = np.frombuffer(raw, dtype=">u2").astype(np.int64).reshape(-1, 5)
        exponent = (words[:, :1] >> np.array([12, 8, 4, 0])) & 15
        mantissa = np.where(words[:, 1:] & 0x8000, -(words[:, 1:] ^ 0xffff), words[:, 1:])
        return (mantissa*np.power(2., exponent-15)).ravel()
    if code in (8022, 8024, 8042, 8044):
        bits = 8 if code in (8022, 8042) else 16
        x = np.frombuffer(raw, dtype="u1" if bits == 8 else ">u2").astype(np.int64)
        negative = (x >> (bits-1)).astype(bool)
        quaternary = code in (8022, 8024)
        fraction_bits = bits-(4 if quaternary else 3)
        mask = (1 << fraction_bits)-1
        exponent = (x >> fraction_bits) & (7 if quaternary else 3)
        fraction = x & mask
        if quaternary:
            fraction = np.where(negative, fraction ^ mask, fraction)
        return np.where(negative, -1., 1.)*fraction/(1 << fraction_bits)*np.power(4. if quaternary else 16., exponent)
    raise ValueError(f"unsupported SEG-D encoding {code}")


def is_segd(path):
    try:
        with open(path, "rb") as f:
            header = f.read(64)
        return len(header) == 64 and _bcd(header[2:4]) in (8015,8022,8024,8036,8038,8042,8044,8048,8058) and header[11] >> 4 >= 1
    except (OSError, ValueError):
        return False


def read_segd(path, name=None, *, target_dt=None, window="strict", descale=True,
              include_aux=False, channel_set=0, max_mb=1024):
    """Read one or more contiguous disk records; reject incomplete/ambiguous layouts."""
    if not np.isfinite(max_mb) or max_mb <= 0:
        raise ValueError("import memory limit must be positive and finite")
    rows, heads, intervals, starts, records = [], [], [], [], []
    allocated = 0
    with open(path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        def take(n):
            if n < 0 or f.tell()+n > file_size or n > max_mb*1024**2:
                raise ValueError("truncated SEG-D block or block exceeds import limit")
            raw = f.read(n)
            if len(raw) != n:
                raise ValueError("truncated SEG-D block")
            return raw
        while f.tell() < file_size:
            record_offset = f.tell()
            g1 = take(32)
            additional = g1[11] >> 4
            if additional < 1:
                raise ValueError("SEG-D needs general header 2; rev 0/1 and tape wrappers are unsupported")
            extra = take(32*additional); g2 = extra[:32]
            if (g2[10], g2[11]) not in ((2, 0), (2, 1)) or g2[18] != 2:
                raise ValueError("SEG-D reader supports revision 2.0/2.1 disk records only")
            code = _bcd(g1[2:4]); _size(code, 4)
            ffid = _uint(g2[:3]) if g1[:2] == b"\xff\xff" else _bcd(g1[:2])
            base_dt = g1[22]/16000.
            if base_dt <= 0:
                raise ValueError("SEG-D base scan interval is zero")
            scan_count = _bcd(g1[27:28])
            set_count = _uint(g2[3:5]) if g1[28] == 255 else _bcd(g1[28:29])
            skew_count = _bcd(g1[29:30])
            ext_count = _uint(g2[5:7]) if g1[30] == 255 else _bcd(g1[30:31])
            external_count = _uint(g2[7:9]) if g1[31] == 255 else _bcd(g1[31:32])
            if scan_count < 1 or set_count < 1:
                raise ValueError("SEG-D has no scan/channel sets")
            sets, set_raw = {}, []
            for _ in range(scan_count):
                for _ in range(set_count):
                    c = take(32); set_raw.append(c.hex())
                    scan = _bcd(c[:1]); number = _uint(c[26:28]) if c[1] == 255 else _bcd(c[1:2])
                    count = _bcd(c[8:10])
                    if not count:
                        continue
                    if c[28] >> 4:
                        raise ValueError("SEG-D channel set uses a proprietary extended descriptor; convert with the vendor reader")
                    if (scan, number) in sets or c[11] >> 4 > 9:
                        raise ValueError("duplicate channel set or invalid SEG-D subscan exponent")
                    mp = ((c[7] & 127)*256+c[6])/1024 * (-1 if c[7] & 128 else 1)
                    sets[scan, number] = dict(count=count, seen=0, dt=base_dt/2**(c[11] >> 4),
                                               kind=c[10] >> 4, mp=mp, extensions=c[28] & 15,
                                               start=_uint(c[2:4])*.002, end=_uint(c[4:6])*.002)
                take(32*skew_count)
            opaque = take(32*(ext_count+external_count))
            record_info = dict(offset=record_offset, ffid=ffid, revision=f"{g2[10]}.{g2[11]}", code=code,
                               general_header_hex=(g1+extra).hex(), channel_set_headers_hex=set_raw,
                               external_header_bytes=len(opaque), external_header_sha256=hashlib.sha256(opaque).hexdigest())
            source = dict(segd_source_line=0., segd_source_point=0., segd_source_index=0.)
            if additional >= 2:
                g3 = extra[32:64]
                source = dict(segd_source_line=_fixed(g3[3:8]), segd_source_point=_fixed(g3[8:13]), segd_source_index=g3[13])
            total = sum(s["count"] for s in sets.values())
            for _ in range(total):
                h = take(20)
                trace_ffid = _uint(h[17:20]) if h[:2] == b"\xff\xff" else _bcd(h[:2])
                number = _uint(h[15:17]) if h[3] == 255 else _bcd(h[3:4])
                if h[2] == 255:
                    scans = [scan for scan, num in sets if num == number]
                    if len(scans) != 1:
                        raise ValueError("ambiguous expanded SEG-D channel set / scan type")
                    scan = scans[0]
                else:
                    scan = _bcd(h[2:3])
                if trace_ffid != ffid or (scan, number) not in sets:
                    raise ValueError("SEG-D trace identifiers disagree with general/channel headers")
                c = sets[scan, number]
                if not 1 <= h[9] <= 15 or h[9] != c["extensions"]:
                    raise ValueError("SEG-D requires consistent trace extensions with explicit sample counts")
                extension = take(32*h[9]); e = extension[:32]
                ns = _uint(e[7:10])
                if ns < 1 or ("ns" in c and ns != c["ns"]):
                    raise ValueError("SEG-D sample count is zero or varies within a channel set")
                c["ns"] = ns; c["seen"] += 1
                raw = take(_size(code, ns))
                if (channel_set and number != channel_set) or (not include_aux and c["kind"] != 1):
                    continue
                allocated += ns*8
                if allocated > max_mb*1024**2:
                    raise ValueError(f"SEG-D samples exceed the {max_mb:g} MiB import limit")
                samples = decode_segd(raw, code)
                if descale:
                    samples *= 2**c["mp"]
                # Timing-word wrap is unambiguous only when anchored to the channel-set start.
                timing = _uint(h[6:9])/256000.
                timing += max(0, int(np.floor((c["start"]-timing+base_dt)/65.536)))*65.536
                if abs(timing-c["start"]) > max(.002, base_dt):
                    raise ValueError("SEG-D first timing word disagrees with the channel-set start")
                start = timing+h[10]/256*base_dt
                row = dict(fldr=ffid, tracf=_bcd(h[4:6]), trid=2 if h[11] in (1, 2) else (1 if c["kind"] == 1 else 3),
                           segd_channel_set=number, segd_scan_type=scan, segd_channel_type=c["kind"],
                           segd_line_sequence=_uint(g2[20:22]), segd_edit=h[11], segd_sensor=e[20], segd_mp=c["mp"],
                           segd_receiver_line=_fixed(e[10:15]) if e[:3] == b"\xff\xff\xff" else int.from_bytes(e[:3], "big", signed=True),
                           segd_receiver_point=_fixed(e[15:20]) if e[3:6] == b"\xff\xff\xff" else int.from_bytes(e[3:6], "big", signed=True),
                           segd_receiver_index=int.from_bytes(e[6:7], "big", signed=True), **source)
                heads.append(row); rows.append(samples); starts.append(start); intervals.append(c["dt"])
            if any(c["seen"] != c["count"] for c in sets.values()):
                raise ValueError("SEG-D trace count disagrees with channel-set descriptors")
            trailer = take(_uint(g2[12:14])*32)
            record_info["trailer_sha256"] = hashlib.sha256(trailer).hexdigest()
            record_info["total_traces"] = total
            records.append(record_info)
    if not rows:
        raise ValueError("SEG-D contains no traces matching the channel selection")
    traces, dt, t0, timing = common_samples(rows, intervals, starts, target_dt, window, max_mb)
    headers = timing_headers({k: np.array([h[k] for h in heads]) for k in heads[0]}, timing, traces.shape[1], dt, t0)
    out = SeismicData(traces, dt, headers, t0, name or os.path.splitext(os.path.basename(path))[0])
    out.meta.update(source_file=os.path.abspath(path), import_timing=timing, segd_records=records,
                    segd_descale_applied=bool(descale), sample_units="mV" if descale else "recorded units",
                    geometry_status="SPS line/point identifiers only; surveyed coordinates not assigned",
                    segd_selected_channel_set=channel_set, segd_aux_included=bool(include_aux))
    out.add_history(f"read SEG-D: {len(records)} records, {out.ntr} selected traces; {window} time window")
    return out
