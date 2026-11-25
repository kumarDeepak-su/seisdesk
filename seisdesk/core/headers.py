"""Trace-header definitions shared by the SEG-Y and Seismic Unix readers and writers.

Names follow Seismic Unix (``fldr`` = FFID / field record, ``ep`` = shot point, ``cdp`` = CMP number, ``gx`` = receiver
x, ...), byte positions follow SEG-Y rev 1 (1-based). Inside SeisDesk every header is held as a float64 array in real
units: coordinates (sx, sy, gx, gy, cdpx, cdpy) and offsets in metres after ``scalco``, elevations and depths in metres
after ``scalel``, time headers (delrt, muts, mute, sstat, gstat, tstat, laga, lagb) in milliseconds as in SEG-Y, ``dt``
in microseconds.
"""

from __future__ import annotations

import numpy as np

# (name, 1-based byte, dtype, description) — bytes 1-180 are common to SEG-Y and SU
STANDARD = [
    ("tracl", 1, "i4", "trace sequence number within line"),
    ("tracr", 5, "i4", "trace sequence number within file"),
    ("fldr", 9, "i4", "field record number (FFID)"),
    ("tracf", 13, "i4", "trace number within field record (channel)"),
    ("ep", 17, "i4", "energy source point (shot) number"),
    ("cdp", 21, "i4", "CMP ensemble number"),
    ("cdpt", 25, "i4", "trace number within CMP ensemble"),
    ("trid", 29, "i2", "trace identification code (1 = seismic, 2 = dead)"),
    ("nvs", 31, "i2", "number of vertically summed traces"),
    ("nhs", 33, "i2", "number of horizontally summed traces"),
    ("duse", 35, "i2", "data use (1 production, 2 test)"),
    ("offset", 37, "i4", "source-receiver offset"),
    ("gelev", 41, "i4", "receiver group elevation"),
    ("selev", 45, "i4", "source surface elevation"),
    ("sdepth", 49, "i4", "source depth below surface"),
    ("gdel", 53, "i4", "datum elevation at receiver"),
    ("sdel", 57, "i4", "datum elevation at source"),
    ("swdep", 61, "i4", "water depth at source"),
    ("gwdep", 65, "i4", "water depth at receiver"),
    ("scalel", 69, "i2", "scalar for elevations and depths"),
    ("scalco", 71, "i2", "scalar for coordinates"),
    ("sx", 73, "i4", "source x"),
    ("sy", 77, "i4", "source y"),
    ("gx", 81, "i4", "receiver x"),
    ("gy", 85, "i4", "receiver y"),
    ("counit", 89, "i2", "coordinate units (1 length, 2 arc seconds)"),
    ("wevel", 91, "i2", "weathering velocity"),
    ("swevel", 93, "i2", "subweathering velocity"),
    ("sut", 95, "i2", "uphole time at source (ms)"),
    ("gut", 97, "i2", "uphole time at receiver (ms)"),
    ("sstat", 99, "i2", "source static (ms)"),
    ("gstat", 101, "i2", "receiver static (ms)"),
    ("tstat", 103, "i2", "total static applied (ms)"),
    ("laga", 105, "i2", "lag time A (ms)"),
    ("lagb", 107, "i2", "lag time B (ms)"),
    ("delrt", 109, "i2", "delay recording time (ms)"),
    ("muts", 111, "i2", "mute start time (ms)"),
    ("mute", 113, "i2", "mute end time (ms)"),
    ("ns", 115, "u2", "number of samples"),
    ("dt", 117, "u2", "sample interval (microseconds)"),
    ("gain", 119, "i2", "gain type"),
    ("igc", 121, "i2", "instrument gain constant"),
    ("igi", 123, "i2", "instrument early gain"),
    ("corr", 125, "i2", "correlated (1 no, 2 yes)"),
    ("sfs", 127, "i2", "sweep start frequency"),
    ("sfe", 129, "i2", "sweep end frequency"),
    ("slen", 131, "i2", "sweep length (ms)"),
    ("styp", 133, "i2", "sweep type"),
    ("stas", 135, "i2", "sweep taper at start (ms)"),
    ("stae", 137, "i2", "sweep taper at end (ms)"),
    ("tatyp", 139, "i2", "taper type"),
    ("afilf", 141, "i2", "alias filter frequency"),
    ("afils", 143, "i2", "alias filter slope"),
    ("nofilf", 145, "i2", "notch filter frequency"),
    ("nofils", 147, "i2", "notch filter slope"),
    ("lcf", 149, "i2", "low-cut frequency"),
    ("hcf", 151, "i2", "high-cut frequency"),
    ("lcs", 153, "i2", "low-cut slope"),
    ("hcs", 155, "i2", "high-cut slope"),
    ("year", 157, "i2", "year recorded"),
    ("day", 159, "i2", "day of year"),
    ("hour", 161, "i2", "hour"),
    ("minute", 163, "i2", "minute"),
    ("sec", 165, "i2", "second"),
    ("timbas", 167, "i2", "time basis code"),
    ("trwf", 169, "i2", "trace weighting factor"),
    ("grnors", 171, "i2", "geophone group number of roll switch position one"),
    ("grnofr", 173, "i2", "geophone group number of first trace of original field record"),
    ("grnlof", 175, "i2", "geophone group number of last trace of original field record"),
    ("gaps", 177, "i2", "gap size"),
    ("otrav", 179, "i2", "overtravel taper code"),
]

# SEG-Y rev 1 extensions read and written through SEG-Y files
SEGY_EXTRA = [
    ("cdpx", 181, "i4", "CMP x coordinate"),
    ("cdpy", 185, "i4", "CMP y coordinate"),
    ("iline", 189, "i4", "3D inline number"),
    ("xline", 193, "i4", "3D crossline number"),
    ("sp", 197, "i4", "shotpoint number"),
    ("scalsp", 201, "i2", "shotpoint scalar"),
]

# Seismic Unix extensions (bytes 181-240 of an SU trace header)
SU_EXTRA = [
    ("d1", 181, "f4", "sample spacing for non-seismic data"),
    ("f1", 185, "f4", "first sample location for non-seismic data"),
    ("d2", 189, "f4", "trace spacing"),
    ("f2", 193, "f4", "first trace location"),
    ("ungpow", 197, "f4", "negative of power used for range compression"),
    ("unscale", 201, "f4", "reciprocal of scaling factor"),
    ("ntr", 205, "i4", "number of traces"),
    ("mark", 209, "i2", "mark selected traces"),
    ("shortpad", 211, "i2", "alignment padding"),
]

COORDINATE_KEYS = ("sx", "sy", "gx", "gy", "cdpx", "cdpy")
ELEVATION_KEYS = ("gelev", "selev", "sdepth", "gdel", "sdel", "swdep", "gwdep")
DESCRIPTIONS = {name: text for name, _, _, text in STANDARD + SEGY_EXTRA + SU_EXTRA}
SEGY_FIELDS = STANDARD + SEGY_EXTRA
BYTE_OF = {name: byte for name, byte, _, _ in SEGY_FIELDS}

# keys offered as gather keys in the GUI, with their usual names
GATHER_KEYS = {
    "fldr": "FFID / field record",
    "ep": "shot point",
    "sx": "source x",
    "gx": "receiver x",
    "tracf": "channel",
    "cdp": "CMP",
    "offset": "offset",
    "iline": "inline",
    "xline": "crossline",
}


def scale_factor(scalar: np.ndarray) -> np.ndarray:
    """Multiplier implied by a SEG-Y scalar: s > 0 multiplies, s < 0 divides by |s|, 0 means 1."""
    s = np.asarray(scalar, dtype=np.float64)
    out = np.ones_like(s)
    out[s > 0] = s[s > 0]
    out[s < 0] = 1.0 / np.abs(s[s < 0])
    return out


def choose_scalar(values: np.ndarray, max_int: int = 2_000_000_000) -> int:
    """A SEG-Y scalar that keeps 2 decimals where the range allows (-100, -10, 1 or a positive power of ten)."""
    v = np.abs(np.asarray(values, dtype=np.float64))
    peak = float(v.max()) if v.size else 0.0
    for s in (-100, -10, 1):
        if peak * (1.0 / scale_factor(np.array([s]))[0]) < max_int:
            return s
    s = 10
    while peak / s >= max_int:
        s *= 10
    return s


def su_header_dtype() -> np.dtype:
    """The 240-byte Seismic Unix trace header as a NumPy structured dtype (native byte order)."""
    names, formats, offsets = [], [], []
    for name, byte, fmt, _ in STANDARD + SU_EXTRA:
        names.append(name)
        formats.append(fmt)
        offsets.append(byte - 1)
    names.append("unass")
    formats.append(("i2", 14))
    offsets.append(212)
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": 240})
