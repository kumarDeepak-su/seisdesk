"""Chebyshev acoustic propagation and small-model anisotropic P-SV/ray synthetics."""
import numpy as np
from scipy.fft import dct

from ..core.models import LayeredModel
from ..core.synthetic import events_to_traces, ricker
from .base import Param, Process, register
from .modelling import (ElasticFD, _run_fd, _finish, _model, _stations, _time_axis,
                         _depth_limit, _fd_grid, _nodes, _shots, _sponge, _model_params, flat_reflections)


def chebyshev_propagators(v, spacing, dt):
    """cos(dt sqrt(A)) and sin(dt sqrt(A))/sqrt(A), A=-v Laplacian v.

    A is self-adjoint positive on the periodic Fourier grid after p=v*u. The
    spectral bound uses max(v)^2*(kx_max^2+kz_max^2); Chebyshev coefficients are
    computed at Gauss nodes, with a polynomial order above the propagation phase.
    """
    v = np.asarray(v, float)
    kz = 2*np.pi*np.fft.fftfreq(v.shape[0], spacing)
    kx = 2*np.pi*np.fft.fftfreq(v.shape[1], spacing)
    k2 = kz[:, None]**2+kx[None]**2
    bound = v.max()**2*k2.max()
    degree = max(20, int(np.ceil(dt*np.sqrt(bound)))+20)
    if degree > 1000:
        raise ValueError("REM polynomial order exceeds 1000; reduce the output step or coarsen the grid")
    count = 2*(degree+1)
    x = np.cos(np.pi*(np.arange(count)+.5)/count)
    omega = np.sqrt(bound*(x+1)/2)
    cosine = dct(np.cos(dt*omega), type=2)/count
    sine = dct(dt*np.sinc(dt*omega/np.pi), type=2)/count
    cosine[0] /= 2; sine[0] /= 2

    def operator(a):
        return v[..., None]*np.fft.ifft2(k2[..., None]*np.fft.fft2(v[..., None]*a, axes=(0, 1)), axes=(0, 1)).real

    def advance(u, velocity):
        previous = np.stack([u, velocity], axis=-1)
        current = 2*operator(previous)/bound-previous
        c = cosine[0]*previous+cosine[1]*current
        s = sine[0]*previous+sine[1]*current
        for n in range(2, degree+1):
            following = 4*operator(current)/bound-2*current-previous
            c += cosine[n]*following; s += sine[n]*following
            previous, current = current, following
        return c[..., 0]+s[..., 1], c[..., 1]-operator(s[..., :1])[..., 0]
    return advance


def rem_shot(vp, spacing, dt, nt, source, rec_iz, rec_ix, f0, absorbing, progress=None):
    """Constant-density Fourier/Chebyshev acoustic shot, sponge borders and Ricker volume injection."""
    advance = chebyshev_propagators(vp, spacing, dt)
    u = np.zeros(vp.shape); velocity = np.zeros_like(u)
    sponge = _sponge(*vp.shape, absorbing).astype(float)
    delay = int(np.ceil(1.5/f0/dt)); steps = nt+delay
    time = (np.arange(steps+1)-delay)*dt
    a = np.pi*f0*time
    # Pressure source derivative in p_tt = v² Laplacian(p) + q'(t).
    force = -2*(np.pi*f0)*a*(3-2*a*a)*np.exp(-a*a)
    result = np.zeros((len(rec_iz), nt))
    for n in range(steps):
        velocity[source] += .5*dt*force[n]/(spacing*spacing*vp[source])
        u, velocity = advance(u, velocity)
        velocity[source] += .5*dt*force[n+1]/(spacing*spacing*vp[source])
        u *= sponge; velocity *= sponge
        sample = n+1-delay
        if 0 <= sample < nt:
            result[:, sample] = (vp*u)[rec_iz, rec_ix]
        if progress is not None and n % 10 == 0:
            progress(n/steps)
    return result


@register
class RapidExpansion(Process):
    name, label, category = "rem_acoustic", "Acoustic rapid expansion (Fourier / Chebyshev)", "Modelling"
    description = "Constant-density pressure modelling with Fourier spatial derivatives and a Chebyshev time propagator. Large stable time steps, but output sampling must resolve the source. Sponge edges; no free surface; small 2D CPU grids."
    params = _model_params(15) + (Param('absorbing', 'int', 24, 'Sponge width', unit='cells', minimum=8, maximum=100),)
    def run(self, data, context):
        model = _model(self['model'], context)
        model, sx, sz, gx, gz = _stations(data, model, self['topography'], self['receiver_depth'])
        if np.ptp(model.surface_at(np.r_[sx, gx])) > .001:
            raise ValueError("REM currently requires a flat recording surface")
        nt, dt = _time_axis(data, self['tmax'], self['dt_ms'])
        if dt*self['f0'] > .1:
            raise ValueError("use at least ten output samples per peak-frequency period")
        h = self['dx'] if self['dx'] else 10
        x, z, vp, _, _, _ = _fd_grid(model, sx, sz, gx, gz, h, _depth_limit(model, self['zmax']), self['absorbing'], False)
        if vp.size > 100000 or vp.min()/(2.5*self['f0']*h) < 3:
            raise ValueError("REM needs at most 100,000 grid cells and at least three points per shortest wavelength")
        shots, groups = _shots(sx, sz); out = np.zeros((data.ntr, nt))
        for k in range(len(shots)):
            idx = np.flatnonzero(groups == k)
            si, sj = _nodes(sx[idx[:1]], sz[idx[:1]], x, z, h)
            ri, rj = _nodes(gx[idx], gz[idx], x, z, h)
            out[idx] = rem_shot(vp, h, dt, nt, (si[0], sj[0]), ri, rj, self['f0'], self['absorbing'],
                               lambda f, k=k: context.report((k+f)/len(shots)))
        return _finish(data, out, dt, 'REM acoustic', self.name)


@register
class ElasticVTI(ElasticFD):
    name, label = 'fd_elastic_vti', 'Elastic VTI finite-difference modelling (P-SV)'
    description = "2D staggered-grid P-SV modelling using Thomsen epsilon/delta stiffnesses, sponge borders and an optional flat free surface. Gamma affects SH waves and is outside this P-SV solver. Relative amplitudes."
    params = ElasticFD.params + (Param('epsilon', 'float', .1, 'Thomsen epsilon', minimum=-.1, maximum=.4),
                                 Param('delta', 'float', .05, 'Thomsen delta', minimum=-.1, maximum=.4))
    def run(self, data, context):
        model = _model(self['model'], context)
        if self['free_surface'] and np.ptp(model.surface_at(np.r_[data.header('sx'), data.header('gx')])) > .001:
            raise ValueError("the VTI free-surface implementation currently requires a flat surface")
        out, dt = _run_fd(self, data, context, elastic=True)
        return _finish(data, out, dt, 'VTI elastic '+self['component'], self.name)


def ti_flat_reflections(model, sx, sz, gx, gz, epsilon, delta):
    """Snell shooting through weak-VTI phase-slowness curves; isotropic interface amplitudes."""
    if not isinstance(model, LayeredModel) or not model.is_flat:
        raise ValueError("TI ray synthetics require flat homogeneous layers")
    vp, _, _ = model.properties(); n = len(vp)
    eps, delt = (np.broadcast_to(np.asarray(value, float), (n,)) for value in (epsilon, delta))
    if np.all(eps == 0) and np.all(delt == 0):
        return flat_reflections(model, sx, sz, gx, gz)
    theta = np.linspace(0, np.deg2rad(89.8), 8193)
    sine, cosine = np.sin(theta), np.cos(theta)
    tables = []
    for v, e, d in zip(vp, eps, delt):
        speed = v*(1+d*sine*sine*cosine*cosine+e*sine**4)
        derivative = v*(.5*d*np.sin(4*theta)+e*(np.sin(2*theta)-.5*np.sin(4*theta)))
        p = sine/speed; horizontal = speed*sine+derivative*cosine; vertical = speed*cosine-derivative*sine
        live = vertical > .001*v
        p, horizontal, vertical = p[live], horizontal[live], vertical[live]
        if np.any(np.diff(p) <= 0):
            raise ValueError("non-convex phase-slowness curve; reduce anisotropy")
        tables.append((p, horizontal/vertical, 1/vertical))
    times, amplitudes = flat_reflections(model, sx, sz, gx, gz)
    tops = np.array([layer.top_z[0] for layer in model.layers]); offset = np.abs(gx-sx)
    for k in range(1, n):
        down = np.maximum(tops[1:k+1]-np.maximum(tops[:k], sz[:, None]), 0)
        up = np.maximum(tops[1:k+1]-np.maximum(tops[:k], gz[:, None]), 0)
        thick = down+up
        lo = np.zeros(len(offset)); hi = np.full(len(offset), min(t[0][-1] for t in tables[:k]))
        for _ in range(52):
            p = (lo+hi)/2
            distance = sum(thick[:, j]*np.interp(p, *tables[j][:2]) for j in range(k))
            far = distance > offset; hi = np.where(far, p, hi); lo = np.where(far, lo, p)
        p = (lo+hi)/2
        distance = sum(thick[:, j]*np.interp(p, *tables[j][:2]) for j in range(k))
        time = sum(thick[:, j]*np.interp(p, tables[j][0], tables[j][2]) for j in range(k))
        valid = (sz < tops[k]) & (gz < tops[k]) & (np.abs(distance-offset) < .1)
        times[:, k-1] = np.where(valid, time, np.nan)
        amplitudes[~valid, k-1] = 0
    return times, amplitudes


@register
class TIRaySynthetics(Process):
    name, label, category = 'ti_ray_synthetics', 'Layered weak-VTI P-wave ray synthetics', 'Modelling'
    description = "Primary P reflections by Snell shooting on weak-VTI phase-slowness curves and group velocities. Epsilon/delta may be constants or one value per layer. Flat layers, isotropic reflection amplitudes, no multiples or converted waves."
    params = _model_params() + (Param('epsilon', 'floats', '.1', 'Thomsen epsilon by layer'),
                                Param('delta', 'floats', '.05', 'Thomsen delta by layer'))
    def run(self, data, context):
        model = _model(self['model'], context)
        model, sx, sz, gx, gz = _stations(data, model, self['topography'], self['receiver_depth'])
        if np.ptp(np.r_[data.header('sy'), data.header('gy')]) > .001:
            raise ValueError("TI ray synthetics require a 2D line in x")
        if any(np.any(np.abs(self[key]) > .2) for key in ('epsilon', 'delta')):
            raise ValueError("weak-VTI ray modelling requires |epsilon| and |delta| <= 0.2")
        time, amplitude = ti_flat_reflections(model, sx, sz, gx, gz, self['epsilon'], self['delta'])
        nt, dt = _time_axis(data, self['tmax'], self['dt_ms']); w, c = ricker(self['f0'], dt)
        out = events_to_traces(time, amplitude, nt, dt, w, c)
        return _finish(data, out, dt, 'TI P-ray synthetic', self.name)
