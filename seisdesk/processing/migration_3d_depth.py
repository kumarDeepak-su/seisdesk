"""Small-volume layered 3D depth migration and common-azimuth log-stretch DMO."""
import numpy as np
from scipy.fft import next_fast_len

from ..core.velocity import dix, time_to_depth_axis
from .base import Param, Process, register, resolve_velocity
from .migration_3d import Cube


def layered_time_table(depths, velocities, radii):
    """Single-leg diving-free rays from a flat datum in v(z), by Snell shooting."""
    z, v, radii = map(lambda a: np.asarray(a, float), (depths, velocities, radii))
    if z[0] != 0 or np.any(np.diff(z) <= 0) or np.any(v <= 0) or np.any(radii < 0):
        raise ValueError("ray table needs increasing depths from zero, positive velocity and nonnegative radii")
    out = np.zeros((len(z), len(radii)))
    for iz in range(1, len(z)):
        thick = np.diff(z[:iz+1]); velocity = .5*(v[:iz]+v[1:iz+1])
        lo, hi = np.zeros(len(radii)), np.full(len(radii), (1-1e-12)/velocity.max())
        for _ in range(52):
            p = (lo+hi)/2
            sine = p[:, None]*velocity
            distance = np.sum(thick*sine/np.sqrt(1-sine*sine), axis=1)
            far = distance > radii
            hi = np.where(far, p, hi); lo = np.where(far, lo, p)
        cosine = np.sqrt(1-((lo+hi)[:, None]/2*velocity)**2)
        out[iz] = np.sum(thick/(velocity*cosine), axis=1)
    return out


def kirchhoff_depth_3d(data, x, y, depths, velocity, aperture, max_angle, dr=5, progress=None):
    sx, sy, gx, gy = (data.header(k, None) for k in ('sx', 'sy', 'gx', 'gy'))
    maximum_radius = np.hypot(max(np.max(np.r_[sx, gx]), x.max())-min(np.min(np.r_[sx, gx]), x.min()),
                              max(np.max(np.r_[sy, gy]), y.max())-min(np.min(np.r_[sy, gy]), y.min()))
    radii = np.linspace(0, maximum_radius+dr, int(np.ceil(maximum_radius/dr))+2)
    if len(radii)*len(depths) > 2_000_000:
        raise ValueError("traveltime table exceeds two million entries; increase the radial or depth spacing")
    table = layered_time_table(depths, velocity, radii)
    nfft = next_fast_len(2*data.nt)
    omega = 2*np.pi*np.fft.rfftfreq(nfft, data.dt)
    filtered = np.fft.irfft(np.fft.rfft(data.traces, nfft, axis=1)*(-1j*omega), nfft, axis=1)[:, :data.nt]
    image = np.zeros((len(x), len(depths)))
    tangent = np.tan(np.deg2rad(max_angle))
    for i, trace in enumerate(filtered):
        if progress is not None:
            progress(i/data.ntr)
        distance = np.hypot(x-(sx[i]+gx[i])/2, y-(sy[i]+gy[i])/2)
        cols = np.flatnonzero(distance <= aperture)
        rs, rg = np.hypot(x[cols]-sx[i], y[cols]-sy[i]), np.hypot(x[cols]-gx[i], y[cols]-gy[i])
        for iz in range(1, len(depths)):
            ts, tg = np.interp(rs, radii, table[iz]), np.interp(rg, radii, table[iz])
            values = np.interp(ts+tg, data.times, trace, left=0, right=0)
            z = depths[iz]
            obliquity = z*z/(np.hypot(z, rs)*np.hypot(z, rg))
            spreading = 1/np.maximum(velocity[0]*np.sqrt(ts*tg), dr)
            live = distance[cols] <= z*tangent
            image[cols, iz] += values*obliquity*spreading*live
    return image


@register
class KirchhoffPSDM3D(Process):
    name, label, category = "kirchhoff_psdm_3d", "3D Kirchhoff depth migration (layered v(z))", "Depth migration"
    description = "Pre/post-stack small-volume migration with layered Snell-ray traveltime tables, anti-causal time derivative and approximate spreading/obliquity. Flat recording datum; one velocity column; relative amplitudes."
    params = (Param("velocity", "velocity", "2000", "RMS velocity, or 'project model'"),
              Param("model_x", "float", 0, "Depth-model column position", unit="m"),
              Param("dz", "float", 10, "Depth interval", unit="m", minimum=1),
              Param("zmax", "float", 2000, "Maximum depth", unit="m", minimum=10),
              Param("dr", "float", 5, "Traveltime radial spacing", unit="m", minimum=.5),
              Param("aperture", "float", 1000, "Midpoint aperture radius", unit="m", minimum=1),
              Param("max_angle", "float", 65, "Maximum midpoint angle", unit="deg", minimum=1, maximum=85))
    def run(self, data, context):
        if data.domain != 'time' or any(k not in data.headers for k in ('iline', 'xline', 'cdpx', 'cdpy')):
            raise ValueError("3D PSDM requires time traces with inline/crossline and map bin centres")
        if np.ptp(np.r_[data.header('selev'), data.header('gelev')]) > .01 or np.any(data.header('sdepth') != 0):
            raise ValueError("datum source and receiver traces to one flat recording level before 3D PSDM")
        _, first = np.unique(np.column_stack([data.header('iline'), data.header('xline')]), axis=0, return_index=True)
        x, y = data.header('cdpx')[first], data.header('cdpy')[first]
        z = np.arange(0, self['zmax']+self['dz']*.01, self['dz'])
        if len(x)*len(z)*data.ntr > 300_000_000:
            raise ValueError("3D summation exceeds the small-volume limit; crop the survey or coarsen the image")
        if self['velocity'] == 'project model':
            model = context.depth_model
            if model is None or model.z0 != 0 or z[-1] > model.z[-1]:
                raise ValueError("the project depth model must start at the recording datum and cover the output depth")
            velocity = model.at(z, np.full(len(z), self['model_x']))
        else:
            picks = resolve_velocity(self['velocity'], context)
            time = np.maximum(data.times, 0); rms = picks.function(float(np.median(picks.cdps)), time)
            interval = dix(time, rms); depth = time_to_depth_axis(time, interval)
            velocity = np.interp(z, depth, interval)
        geometry = data
        if not all(k in data.headers for k in ('sx', 'sy', 'gx', 'gy')):
            if len(first) != data.ntr:
                raise ValueError("prestack migration needs all four source/receiver map coordinates")
            geometry = data.copy()
            for key, value in (('sx', data.header('cdpx')), ('gx', data.header('cdpx')),
                               ('sy', data.header('cdpy')), ('gy', data.header('cdpy'))):
                geometry.set_header(key, value)
        image = kirchhoff_depth_3d(geometry, x, y, z, velocity, self['aperture'], self['max_angle'], self['dr'], context.report)
        out = data.subset(first).copy(image.astype(np.float32), dt=self['dz'], t0=0, domain='depth', name=data.name+' 3D PSDM')
        for key, value in (('sx', x), ('gx', x), ('sy', y), ('gy', y), ('offset', 0)):
            out.set_header(key, value)
        out.meta.update(migrated='kirchhoff_psdm_3d', stacked=True)
        return out


def log_dmo_cube(cube, times, spacing, direction, half_offset):
    """High-frequency log-stretch DMO with the wavenumber projected onto offset azimuth."""
    if half_offset == 0:
        return cube.copy()
    # Apply the same multiplier as 2D continuation, but on the projected 3D wavenumber.
    # Flat spatial mean is exact, and the remaining low-Omega/large-dip region is tapered.
    positive = times > 0
    nlog = next_fast_len(2*len(times)); nfft = 2*nlog
    sigma = np.linspace(np.log(times[positive][0]), np.log(times[-1]), nlog); ts = np.exp(sigma)
    mean = cube.mean(axis=(0, 1), keepdims=True)
    values = np.stack([np.interp(ts, times, row) for row in (cube-mean).reshape(-1, len(times))]).reshape(*cube.shape[:2], nlog)
    spectrum = np.fft.fftn(np.fft.rfft(values, nfft, axis=2), axes=(0, 1))
    ki = 2*np.pi*np.fft.fftfreq(cube.shape[0], spacing[0])[:, None, None]
    kj = 2*np.pi*np.fft.fftfreq(cube.shape[1], spacing[1])[None, :, None]
    omega = 2*np.pi*np.fft.rfftfreq(nfft, sigma[1]-sigma[0])[None, None]
    k = ki*direction[0]+kj*direction[1]
    e = 2*k*half_offset/np.where(omega > 0, omega, 1)
    root = np.sqrt(1+e*e)
    logamp = .5*np.log((1+root)/(2*root))+.5*(1-root)
    phase = .5*(1-root+np.log((1+root)/2))
    taper = np.cos(.5*np.pi*np.clip((np.abs(e)-.8)/.2, 0, 1))**2
    multiplier = np.exp(np.clip(-logamp, 0, np.log(10))+1j*omega*phase)*taper
    multiplier = np.where(omega >= 2, multiplier, 0); multiplier[0, 0] = 1
    output = np.fft.irfft(np.fft.ifftn(spectrum*multiplier, axes=(0, 1)), nfft, axis=2)[..., :nlog]
    result = np.stack([np.interp(times, ts, row, left=0, right=0) for row in output.reshape(-1, nlog)]).reshape(cube.shape)
    return result+mean


@register
class DMO3D(Process):
    name, label, category = "dmo_3d", "3D DMO (common-offset / azimuth log-stretch f-k)", "Moveout"
    description = "DMO on NMO-corrected, regularly binned 3D common-offset/common-azimuth data. High-frequency log-stretch approximation, dip taper, relative amplitudes and periodic spatial edges; small volumes only."
    params = (Param('offset_bin', 'float', 100, 'Offset bin width', unit='m', minimum=1),
              Param('azimuth_bin', 'float', 10, 'Azimuth bin width', unit='deg', minimum=.1, maximum=30))
    def run(self, data, context):
        if data.domain != 'time' or data.t0 < 0 or np.count_nonzero(data.times > 0) < 8:
            raise ValueError("3D DMO needs a nonnegative time axis with at least eight positive samples")
        il, xl = data.header('iline', None), data.header('xline', None)
        design = np.column_stack([np.ones(data.ntr), il, xl])
        mapping, _, rank, _ = np.linalg.lstsq(design, np.column_stack([data.header('cdpx', None), data.header('cdpy', None)]), rcond=None)
        if rank < 3:
            raise ValueError("3D DMO needs at least two inlines and two crosslines")
        axes = mapping[1:]; lengths = np.linalg.norm(axes, axis=1); axes = axes/lengths[:, None]
        if abs(axes[0] @ axes[1]) > .01:
            raise ValueError("3D bin axes must be orthogonal")
        dx, dy = data.header('gx', None)-data.header('sx', None), data.header('gy', None)-data.header('sy', None)
        azimuth = np.rad2deg(np.arctan2(dy, dx)) % 180; offset = np.hypot(dx, dy)
        labels = np.column_stack([np.rint(offset/self['offset_bin']), np.rint(azimuth/self['azimuth_bin'])])
        _, inverse = np.unique(labels, axis=0, return_inverse=True); out = data.traces.copy()
        for group in np.unique(inverse):
            context.report(group/(inverse.max()+1))
            idx = np.flatnonzero(inverse == group); part = data.subset(idx); grid = Cube(part)
            if grid.values.size > 500_000:
                raise ValueError("3D DMO exceeds the 500,000-sample per-offset-volume limit; crop the survey")
            angle = .5*np.angle(np.mean(np.exp(2j*np.deg2rad(azimuth[idx]))))
            direction = axes @ np.array([np.cos(angle), np.sin(angle)])
            corrected = log_dmo_cube(grid.values, data.times, (lengths[0]*grid.il_step, lengths[1]*grid.xl_step),
                                      direction, float(offset[idx].mean())/2)
            ii = np.rint((il[idx]-il[idx].min())/grid.il_step).astype(int)
            jj = np.rint((xl[idx]-xl[idx].min())/grid.xl_step).astype(int)
            out[idx] = corrected[ii, jj]
        return data.copy(out)
