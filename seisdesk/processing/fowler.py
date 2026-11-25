"""Constant-velocity NMO/DMO stacks and coherence-based velocity selection."""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

from ..core.dataset import SeismicData
from ..core.velocity import VelocityPicks
from .base import Param, Process, create, register
from .nmo import nmo_correct


@register
class Fowler(Process):
    name, label, category = "fowler", "Fowler analysis (constant-velocity NMO + DMO stacks)", "Velocity"
    description = "Make a bank of constant-velocity NMO/DMO stacks; select coherent velocities at each CMP/time. Output the velocity panels or selected stack. Optional eta affects NMO only; DMO remains isotropic."
    params = (Param('vmin', 'float', 1500, 'Minimum velocity', minimum=10),
              Param('vmax', 'float', 3500, 'Maximum velocity', minimum=10),
              Param('nv', 'int', 15, 'Velocity trials', minimum=3, maximum=101),
              Param('offset_bin', 'float', 100, 'DMO offset bin', unit='m', minimum=1),
              Param('fmax', 'float', 60, 'DMO maximum frequency', unit='Hz', minimum=1),
              Param('eta', 'float', 0, 'NMO eta (DMO is isotropic)', minimum=-.1, maximum=.5),
              Param('stretch', 'float', .5, 'NMO stretch limit', minimum=0),
              Param('window', 'float', .04, 'Coherence window', unit='s', minimum=.001),
              Param('output', 'choice', 'selected stack', 'Output', choices=('selected stack', 'velocity panels')),
              Param('write_picks', 'bool', True, 'Write selected velocity picks'),
              Param('pick_step', 'float', .1, 'Time between picks', unit='s', minimum=.01))
    def run(self, data, context):
        if data.domain != 'time' or self['vmax'] <= self['vmin']:
            raise ValueError("Fowler analysis requires time gathers and an increasing velocity range")
        velocities = np.linspace(self['vmin'], self['vmax'], self['nv'])
        groups = list(data.iter_gathers('cdp')); cdps = np.array([g[0] for g in groups])
        if len(groups) < 3:
            raise ValueError("Fowler DMO analysis needs at least three CMPs")
        if len(velocities) * len(cdps) * data.nt > 20_000_000:
            raise ValueError("Fowler scan exceeds 20 million panel samples; subset CMPs/time or reduce velocity trials")
        stacks = np.zeros((len(velocities), len(cdps), data.nt)); scores = np.zeros_like(stacks)
        nwin = max(1, int(round(self['window']/data.dt)))
        for iv, velocity in enumerate(velocities):
            context.report(iv/len(velocities))
            corrected = nmo_correct(data.traces, data.times, data.header('offset', None),
                                     np.full(data.traces.shape, velocity), self['eta'], self['stretch'])
            # A local context prevents nested progress from jumping backwards between trials.
            from .base import Context
            inner = Context(log=context.log, cancelled=context.cancelled)
            moved = create('dmo', offset_bin=self['offset_bin'], fmax=self['fmax'])(data.copy(corrected), inner)
            for ic, (_, idx) in enumerate(groups):
                values = moved.traces[idx].astype(float)
                stacks[iv, ic] = np.mean(values, axis=0)
                energy = np.sum(values*values, axis=0)
                numerator = uniform_filter1d(np.sum(values, axis=0)**2, nwin)
                denominator = uniform_filter1d(len(values)*energy, nwin)
                scores[iv, ic] = numerator/np.maximum(denominator, 1e-30)
                scores[iv, ic, denominator < denominator.max()*1e-4] = 0
        smoothed = gaussian_filter(scores, (0.6, .7, max(1, nwin/4)))
        best = np.argmax(smoothed, axis=0)
        confidence = np.take_along_axis(smoothed, best[None], axis=0)[0]
        image = np.take_along_axis(stacks, best[None], axis=0)[0]
        field = velocities[best]
        if self['write_picks']:
            picks = VelocityPicks()
            step = max(1, int(round(self['pick_step']/data.dt)))
            for ic, c in enumerate(cdps):
                for it in range(0, data.nt, step):
                    if confidence[ic, it] >= .2:
                        picks.add(c, data.times[it], field[ic, it], self['eta'])
            context.velocity = picks
        context.results[self.name] = dict(cdp=cdps, velocities=velocities, stacks=stacks, semblance=scores,
                                          selected_velocity=field, confidence=confidence, times=data.times)
        first = np.array([idx[0] for _, idx in groups]); heads = {k: a[first] for k, a in data.headers.items()}
        heads['offset'] = np.zeros(len(cdps))
        if self['output'] == 'velocity panels':
            heads = {k: np.tile(a, len(velocities)) for k, a in heads.items()}
            heads['velocity_trial'] = np.repeat(velocities, len(cdps))
            traces = stacks.reshape(-1, data.nt)
        else:
            traces = image
        result = SeismicData(traces.astype(np.float32), data.dt, heads, data.t0, data.name+' Fowler', 'time', list(data.history))
        result.meta['stacked'] = True
        return result
