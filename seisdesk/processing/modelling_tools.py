"""Layer impulse responses, impedance conversion and diffraction synthetics."""
import numpy as np
from scipy.fft import next_fast_len

from ..core.synthetic import events_to_traces, ricker
from .base import Param, Process, register


def goupillaud_response(reflections, layer_time, dt, nt, primaries_only=False):
    """1D pressure reflection response with internal multiples and a transparent top.

    Equal two-way-time layers. Interface recursion is R=(r+q R_next)/(1+r q R_next).
    Primaries-only returns the unweighted reflection series (the usual convolutional
    synthetic); the full response includes transmission loss and internal multiples.
    """
    r = np.asarray(reflections, float)
    if r.size == 0 or np.any(np.abs(r) >= 1) or not np.all(np.isfinite(r)):
        raise ValueError("reflection coefficients must be finite and strictly between -1 and 1")
    if primaries_only:
        out = np.zeros(nt)
        for j, value in enumerate(r):
            position = (j+1)*layer_time/dt; k = int(np.floor(position)); frac = position-k
            if k < nt:
                out[k] += value*(1-frac)
            if k+1 < nt:
                out[k+1] += value*frac
        return out
    n = next_fast_len(8*nt)
    alpha = np.log(1e8)/(n*dt)
    q = np.exp(-(alpha+2j*np.pi*np.fft.rfftfreq(n, dt))*layer_time)
    response = np.zeros(q.size, complex)
    for value in r[::-1]:
        response = (value+q*response)/(1+value*q*response)
    return np.fft.irfft(q*response, n)[:nt]*np.exp(alpha*np.arange(nt)*dt)


@register
class Goupillaud(Process):
    name, label, category = "goupillaud", "1D Goupillaud impulse response with internal multiples", "Modelling"
    description = "Equal two-way-time layers, normal-incidence pressure response, transmission loss and internal multiples. Transparent upper boundary; no free-surface multiples. Primaries-only returns the convolutional reflection series."
    params = (Param("reflections", "floats", ".2,-.15,.25", "Interface reflection coefficients"),
              Param("layer_time", "float", .05, "Two-way time of each layer", unit="s", minimum=.000001),
              Param("primaries_only", "bool", False, "Unweighted primaries only"),
              Param("f0", "float", 0, "Ricker peak (0 = impulse)", unit="Hz", minimum=0))
    def run(self, data, context):
        if data.domain != "time" or data.t0 != 0:
            raise ValueError("impulse modelling requires a time record starting at zero")
        response = goupillaud_response(self["reflections"], self["layer_time"], data.dt, data.nt, self["primaries_only"])
        if self["f0"]:
            w, c = ricker(self["f0"], data.dt)
            response = np.convolve(response, w)[c:c+data.nt]
        context.results[self.name] = dict(response=response, time=data.times)
        return data.copy(np.tile(response, (data.ntr, 1)).astype(np.float32))


@register
class ImpedanceReflectivity(Process):
    name, label, category = "impedance_reflectivity", "Impedance to reflectivity and back", "Conversion"
    description = "Normal-incidence r[i]=(Z[i]-Z[i-1])/(Z[i]+Z[i-1]); r[0]=0. Inversion starts from the supplied impedance and integrates the reflection coefficients."
    params = (Param("mode", "choice", "impedance to reflectivity", "Direction", choices=("impedance to reflectivity", "reflectivity to impedance")),
              Param("initial", "float", 4000000, "Initial acoustic impedance", minimum=1))
    def run(self, data, context):
        a = data.traces.astype(float)
        if self["mode"] == "impedance to reflectivity":
            if not np.all(np.isfinite(a)) or np.any(a <= 0):
                raise ValueError("acoustic impedance must be finite and positive")
            out = np.zeros_like(a); out[:, 1:] = np.diff(a, axis=1)/(a[:, 1:]+a[:, :-1])
        else:
            if np.any(np.abs(a) >= 1) or not np.all(np.isfinite(a)):
                raise ValueError("reflection coefficients must lie strictly inside (-1,1)")
            if np.any(np.abs(a[:, 0]) > 1e-8):
                raise ValueError("the first reflectivity sample must be zero because initial impedance is supplied separately")
            log_imp = np.log(self["initial"])+np.cumsum(np.log1p(a)-np.log1p(-a), axis=1)
            if np.any(log_imp > np.log(np.finfo(np.float32).max)):
                raise ValueError("integrated impedance overflows; check low-frequency bias in the reflectivity")
            out = np.exp(log_imp)
        return data.copy(out.astype(np.float32))


@register
class ScattererSynthetics(Process):
    name, label, category = "scatterer_synthetics", "Point / infinite-line scatterer shot records", "Modelling"
    description = "Homogeneous-medium diffraction times and geometrical spreading: 3D point 1/(rs rg), or 2D infinite-line 1/sqrt(rs rg). Relative coefficients, zero-phase Ricker; no radiation pattern or multiples."
    params = (Param("scatterers", "str", "0,0,500,1", "Scatterers: x,y,depth,amplitude; …"),
              Param("velocity", "float", 2000, "Velocity", unit="m/s", minimum=10),
              Param("geometry", "choice", "3D point", "Scattering", choices=("3D point", "2D infinite line")),
              Param("f0", "float", 25, "Ricker peak", unit="Hz", minimum=.1))
    def run(self, data, context):
        points = np.asarray([[float(x) for x in row.split(',')] for row in self["scatterers"].split(';') if row.strip()])
        if points.ndim != 2 or points.shape[1] != 4 or not np.all(np.isfinite(points)) or np.any(points[:, 2] <= 0):
            raise ValueError("give x,y,positive depth,amplitude for every scatterer")
        sx, gx = data.header("sx", None), data.header("gx", None)
        sy, gy = data.header("sy"), data.header("gy")
        sz, gz = data.header("sdepth")-data.header("selev"), -data.header("gelev")
        if self["geometry"] == "2D infinite line":
            if np.ptp(np.r_[sy, gy, points[:, 1]]) > 1e-6:
                raise ValueError("2D infinite-line scattering requires a single x-z plane")
            sy, gy = np.zeros_like(sy), np.zeros_like(gy); points[:, 1] = 0
        rs = np.sqrt((sx[:, None]-points[:, 0])**2+(sy[:, None]-points[:, 1])**2+(sz[:, None]-points[:, 2])**2)
        rg = np.sqrt((gx[:, None]-points[:, 0])**2+(gy[:, None]-points[:, 1])**2+(gz[:, None]-points[:, 2])**2)
        times = (rs+rg)/self["velocity"]
        power = 1 if self["geometry"] == "3D point" else .5
        amplitude = points[:, 3]/np.maximum(rs*rg, 1)**power
        w, c = ricker(self["f0"], data.dt)
        values = events_to_traces(times-data.t0, amplitude, data.nt, data.dt, w, c)
        return data.copy(values)
