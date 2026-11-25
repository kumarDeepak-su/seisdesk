"""Survey geometry and datum utilities. Positions are metres unless explicitly geographic."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

from ..core.dataset import SeismicData
from .base import Param, Process, register
from .statics import shift_traces


def utm_wgs84(longitude, latitude, zone=0):
    """WGS84 UTM, Snyder transverse-Mercator series through A^6; one zone/hemisphere per line."""
    lon, lat = np.broadcast_arrays(np.asarray(longitude, float), np.asarray(latitude, float))
    if not np.all(np.isfinite(lon+lat)) or np.any((lat < -80) | (lat > 84) | (lon < -180) | (lon > 180)):
        raise ValueError("UTM requires finite WGS84 longitude [-180,180] and latitude [-80,84] in decimal degrees")
    if np.any(lat < 0) and np.any(lat >= 0):
        raise ValueError("split the line at the equator before UTM conversion")
    if zone == 0:
        zones = np.minimum(60, np.floor((lon+180)/6).astype(int)+1)
        zones = np.where((lat >= 56) & (lat < 64) & (lon >= 3) & (lon < 12), 32, zones)
        for left, right, special in ((0, 9, 31), (9, 21, 33), (21, 33, 35), (33, 42, 37)):
            zones = np.where((lat >= 72) & (lat < 84) & (lon >= left) & (lon < right), special, zones)
        if np.unique(zones).size != 1:
            raise ValueError("survey spans UTM zones; choose one explicit zone for the complete line")
        zone = int(zones.flat[0])
    if not 1 <= zone <= 60:
        raise ValueError("UTM zone must lie between 1 and 60")
    central = 6*zone-183
    if np.any(np.abs(lon-central) > 6):
        raise ValueError("chosen UTM zone is too far from the survey")
    a, e2, k0 = 6378137., 6.6943799901413165e-3, .9996
    phi = np.deg2rad(lat); ep2 = e2/(1-e2)
    n = a/np.sqrt(1-e2*np.sin(phi)**2); t = np.tan(phi)**2; c = ep2*np.cos(phi)**2
    A = np.cos(phi)*np.deg2rad(lon-central)
    m = a*((1-e2/4-3*e2**2/64-5*e2**3/256)*phi
           -(3*e2/8+3*e2**2/32+45*e2**3/1024)*np.sin(2*phi)
           +(15*e2**2/256+45*e2**3/1024)*np.sin(4*phi)-35*e2**3/3072*np.sin(6*phi))
    east = 500000+k0*n*(A+(1-t+c)*A**3/6+(5-18*t+t*t+72*c-58*ep2)*A**5/120)
    north = k0*(m+n*np.tan(phi)*(A*A/2+(5-t+9*c+4*c*c)*A**4/24+(61-58*t+t*t+600*c-330*ep2)*A**6/720))
    south = bool(np.any(lat < 0)); north += 10000000 if south else 0
    return east, north, zone, south


@register
class UTM(Process):
    name, label, category = "utm", "WGS84 longitude / latitude to UTM", "Geometry"
    description = "Convert source and receiver decimal-degree headers into one WGS84 UTM zone; keep original geographic coordinates."
    params = (Param("zone", "int", 0, "UTM zone (0 = automatic)", minimum=0, maximum=60),)
    def run(self, data, context):
        coords = [data.header(k, None) for k in ("sx", "sy", "gx", "gy")]
        lon, lat = np.r_[coords[0], coords[2]], np.r_[coords[1], coords[3]]
        e, n, zone, south = utm_wgs84(lon, lat, self["zone"])
        out = data.copy()
        for k in ("sx", "sy", "gx", "gy"):
            out.set_header(k+"_degrees", data.header(k))
        for k, values in (("sx", e[:data.ntr]), ("gx", e[data.ntr:]), ("sy", n[:data.ntr]), ("gy", n[data.ntr:])):
            out.set_header(k, values)
        out.set_header("counit", 1)
        out.meta["crs"] = f"EPSG:{32700+zone if south else 32600+zone}"
        return out


@register
class Azimuth(Process):
    name, label, category = "azimuth", "Azimuth, radial offset and midpoint headers", "Geometry"
    description = "Azimuth clockwise from north, unsigned radial offset, and geometric midpoint; source/receiver coordinates in metres."
    params = ()
    def run(self, data, context):
        sx, sy, gx, gy = (data.header(k, None) for k in ("sx", "sy", "gx", "gy"))
        out = data.copy(); distance = np.hypot(gx-sx, gy-sy)
        out.set_header("azimuth", np.where(distance > 0, np.rad2deg(np.arctan2(gx-sx, gy-sy)) % 360, 0))
        out.set_header("offset", distance); out.set_header("cdpx", (sx+gx)/2); out.set_header("cdpy", (sy+gy)/2)
        return out


@register
class FlexBinning(Process):
    name, label, category = "flex_binning", "Flex binning (borrow neighbouring CMP traces)", "Geometry"
    description = "Fill low-fold CMPs with nearest traces from nearby bins. Duplicated traces receive new IDs; original geometry and origin CMP are retained."
    params = (Param("target_fold", "int", 12, "Target fold", minimum=1, maximum=1000),
              Param("radius", "int", 2, "Maximum neighbour bin-number distance", minimum=1, maximum=100),)
    def run(self, data, context):
        cdp = data.header("cdp", None); index, bins = [], []
        for c in np.unique(cdp):
            own = np.flatnonzero(cdp == c)
            candidates = np.flatnonzero((cdp != c) & (np.abs(cdp-c) <= self["radius"]))
            candidates = candidates[np.argsort(np.abs(cdp[candidates]-c), kind="stable")]
            selected = np.r_[own, candidates[:max(0, self["target_fold"]-len(own))]]
            index.extend(selected); bins.extend([c]*len(selected))
        index = np.asarray(index, int); heads = {k: v[index].copy() for k, v in data.headers.items()}
        heads["origin_cdp"] = cdp[index]; heads["cdp"] = np.asarray(bins)
        if "cdpx" in heads:
            centres = {c: float(np.mean(data.header("cdpx")[cdp == c])) for c in np.unique(cdp)}
            heads["cdpx"] = np.array([centres[c] for c in bins])
        out = SeismicData(data.traces[index], data.dt, heads, data.t0, data.name+" flex", data.domain, list(data.history), dict(data.meta))
        out.meta["origin_trace_ids"] = data.trace_ids[index].tolist()
        out.meta["flex_binning"] = "Borrowing changes analysis bins; actual source/receiver coordinates are preserved."
        return out


@register
class BinEnlarge(Process):
    name, label, category = "bin_enlarge", "Enlarge CMP bins", "Geometry"
    description = "Group consecutive CMP bin numbers, retaining original CMP labels and actual source/receiver coordinates."
    params = (Param("factor", "int", 2, "Original bins per new bin", minimum=2, maximum=100),)
    def run(self, data, context):
        c = data.header("cdp", None); out = data.copy()
        new = np.floor((c-c.min())/self["factor"])+1
        out.set_header("origin_cdp", c); out.set_header("cdp", new)
        for key in ("cdpx", "cdpy"):
            if key in data.headers:
                pos = data.header(key).copy()
                for b in np.unique(new):
                    idx = new == b; pos[idx] = np.mean(pos[idx])
                out.set_header(key, pos)
        return out


@register
class FloatingDatum(Process):
    name, label, category = "floating_datum", "Floating datum and residual flat-datum statics", "Statics"
    description = "Smooth the 2D surface by distance, then shift to that floating datum or from saved floating levels to a flat datum. Positive statics delay samples."
    params = (Param("mode", "choice", "surface to floating", "Mode", choices=("surface to floating", "floating to flat", "surface to flat")),
              Param("length", "float", 200, "Surface smoothing sigma", unit="m", minimum=0),
              Param("datum", "float", 0, "Final datum elevation", unit="m"),
              Param("velocity", "float", 2000, "Replacement velocity", unit="m/s", minimum=1),
              Param("apply", "bool", True, "Apply statics"))
    def run(self, data, context):
        if np.ptp(np.r_[data.header("sy"), data.header("gy")]) > .01:
            raise ValueError("floating surface smoothing currently requires a 2D line in x")
        sx, gx = data.header("sx", None), data.header("gx", None)
        se, ge = data.header("selev", None), data.header("gelev", None)
        x, inverse = np.unique(np.r_[sx, gx], return_inverse=True)
        elev = np.bincount(inverse, weights=np.r_[se, ge])/np.bincount(inverse)
        if len(x) < 2:
            floated = elev
        else:
            grid = np.linspace(x[0], x[-1], max(2, min(100000, len(x)*4)))
            surface = np.interp(grid, x, elev)
            sigma = self["length"]/(grid[1]-grid[0])
            floated = np.interp(x, grid, gaussian_filter1d(surface, sigma, mode="nearest") if sigma > 0 else surface)
        sf, gf = np.interp(sx, x, floated), np.interp(gx, x, floated)
        if self["mode"] == "floating to flat":
            source, receiver = data.header("sdel", None), data.header("gdel", None)
            target_s = target_g = self["datum"]
        else:
            source, receiver = se-data.header("sdepth"), ge
            target_s, target_g = (sf, gf) if self["mode"] == "surface to floating" else (self["datum"], self["datum"])
        ss, gs = -(source-target_s)/self["velocity"], -(receiver-target_g)/self["velocity"]
        out = data.copy(); out.set_header("sdel", target_s); out.set_header("gdel", target_g)
        out.set_header("sstat", ss*1000); out.set_header("gstat", gs*1000)
        if self["apply"]:
            out.traces = shift_traces(data.traces, data.dt, ss+gs)
            out.set_header("tstat", data.header("tstat")+1000*(ss+gs))
        context.results[self.name] = dict(x=x, surface=elev, floating=floated, source_static=ss, receiver_static=gs)
        return out


@register
class NearSurfaceZone(Process):
    name, label, category = "nszone_shift", "Extract and shift a near-surface time zone", "Statics"
    description = "Select a time gate, optionally flatten it using a millisecond trace header, and preserve absolute output timing."
    params = (Param("tmin", "float", 0, "Start", unit="s"), Param("tmax", "float", .2, "End", unit="s"),
              Param("shift_header", "str", "", "Header to remove (ms; empty = no shift)"))
    def run(self, data, context):
        out = data
        if self["shift_header"]:
            out = data.copy(shift_traces(data.traces, data.dt, -data.header(self["shift_header"], None)/1000))
        return out.time_window(self["tmin"], self["tmax"])


@register
class StreamerMotion(Process):
    name, label, category = "streamer_motion", "Moving-streamer compensation (constant velocity)", "Statics"
    description = "Resample the exact constant-velocity hyperbola for a receiver moving along x during recording; speed is signed in the x direction. Flat 2D approximation."
    params = (Param("speed", "float", 2.5, "Receiver speed along x", unit="m/s"),
              Param("velocity", "float", 1500, "Propagation velocity", unit="m/s", minimum=10))
    def run(self, data, context):
        v, u = self["velocity"], self["speed"]
        if abs(u) >= v or data.domain != "time":
            raise ValueError("streamer compensation requires time data and receiver speed below propagation velocity")
        if np.ptp(np.r_[data.header("sy"), data.header("gy")]) > .01:
            raise ValueError("streamer motion currently requires a 2D line along x")
        h = data.header("gx", None)-data.header("sx", None)
        b = 1-(u/v)**2; a = h*u/v**2
        sample = (a[:, None]+np.sqrt(a[:, None]**2+b*data.times[None]**2))/b
        out = np.stack([np.interp(q, data.times, tr, left=0, right=0) for q, tr in zip(sample, data.traces)])
        out[data.times[None] < np.abs(h[:, None])/v] = 0
        return data.copy(out.astype(np.float32))


@register
class GPRFirstBreakRemoval(Process):
    name, label, category = "gpr_first_break_removal", "GPR background removal / time-zero alignment", "Editing"
    description = "Subtract a moving mean/median trace in a nanosecond gate, or align the gate's peak magnitude to a target time."
    params = (Param("mode", "choice", "median subtraction", "Mode", choices=("mean subtraction", "median subtraction", "time-zero alignment")),
              Param("width", "int", 21, "Trace window", minimum=1, maximum=501),
              Param("start_ns", "float", 0, "Gate start", unit="ns"), Param("end_ns", "float", 30, "Gate end", unit="ns"),
              Param("target_ns", "float", 0, "Aligned peak time", unit="ns"))
    def run(self, data, context):
        if data.dt >= 1e-6 or data.domain != "time":
            raise ValueError("this tool expects GPR data with sub-microsecond sampling")
        mask = (data.times >= self["start_ns"]*1e-9) & (data.times <= self["end_ns"]*1e-9)
        if not mask.any():
            raise ValueError("GPR gate is outside the record")
        if self["mode"] == "time-zero alignment":
            time = data.times[np.flatnonzero(mask)[np.argmax(np.abs(data.traces[:, mask]), axis=1)]]
            out = data.copy(shift_traces(data.traces, data.dt, self["target_ns"]*1e-9-time))
            out.set_header("gpr_shift_ns", self["target_ns"]-time*1e9)
            return out
        width = self["width"] | 1
        background = (uniform_filter1d(data.traces, width, axis=0, mode="nearest") if self["mode"].startswith("mean")
                      else median_filter(data.traces, size=(width, 1), mode="nearest"))
        out = data.traces.copy(); out[:, mask] -= background[:, mask]
        return data.copy(out)
