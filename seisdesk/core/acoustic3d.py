"""3D acoustic propagation and its exact discrete adjoint, CPU or optional CuPy.

u[n+1] = D(2u[n] - D u[n-1] + dt²(v² L u[n] + source[n])).
L is the symmetric seven-point Laplacian; D is a sponge with zero outer edges.
Source samples are acceleration forcing. Trilinear source/receiver interpolation
is transposed exactly. Block recomputation uses disk checkpoints; the complete
space-time wavefield is never required in RAM. Coordinates are z,y,x in metres,
time in seconds, velocity in m/s. This engine is isotropic constant-density
acoustic; it does not claim elastic or anisotropic inversion.
"""
from __future__ import annotations

import itertools
from functools import wraps
from pathlib import Path
import tempfile

import numpy as np


def array_module(device, device_index=0):
    if device == "cpu":
        return np
    if device != "cuda":
        raise ValueError("device must be cpu or cuda")
    from .compute import cuda_module
    return cuda_module(device_index)


def _on_device(method):
    @wraps(method)
    def execute(self, *args, **kwargs):
        if self.xp is np:
            return method(self, *args, **kwargs)
        with self.xp.cuda.Device(self.device_index):
            return method(self, *args, **kwargs)
    return execute


def laplacian(values, spacing, xp=np):
    out = xp.zeros_like(values)
    for axis, interval in enumerate(spacing):
        centre = [slice(None)]*values.ndim
        lower, upper = centre.copy(), centre.copy()
        centre[axis] = slice(1, -1)
        lower[axis], upper[axis] = slice(None, -2), slice(2, None)
        out[tuple(centre)] += (values[tuple(lower)]-2*values[tuple(centre)]+values[tuple(upper)])/interval**2
        # Zero-valued ghost nodes keep L symmetric even at the sponge's outside edge.
        for edge, neighbour in ((0, 1), (-1, -2)):
            a, b = [slice(None)]*values.ndim, [slice(None)]*values.ndim
            a[axis], b[axis] = edge, neighbour
            out[tuple(a)] += (values[tuple(b)]-2*values[tuple(a)])/interval**2
    return out


class Sampling:
    def __init__(self, coordinates, shape, spacing, origin, padding, xp=np):
        coordinates = np.asarray(coordinates, float).reshape(-1, len(shape))
        index = (coordinates-np.asarray(origin))/np.asarray(spacing)
        if not np.isfinite(index).all() or np.any(index < -1e-6) or np.any(index > np.asarray(shape)-1+1e-6):
            raise ValueError("sampling coordinates are outside the velocity model")
        index = np.clip(index, 0, np.asarray(shape)-1)
        low = np.minimum(np.floor(index).astype(int), np.asarray(shape)-2)
        fraction = index-low
        padded_shape = tuple(n+2*padding for n in shape)
        self.indices, self.weights = [], []
        for corner in itertools.product((0, 1), repeat=len(shape)):
            position = low + np.array(corner) + padding
            weight = np.prod(np.where(corner, fraction, 1-fraction), axis=1)
            self.indices.append(xp.asarray(np.ravel_multi_index(position.T, padded_shape)))
            self.weights.append(xp.asarray(weight))
        self.xp, self.count = xp, len(coordinates)

    def take(self, field):
        flat = field.ravel()
        return sum(flat[index]*weight for index, weight in zip(self.indices, self.weights))

    def add(self, field, values):
        flat = field.ravel()
        for index, weight in zip(self.indices, self.weights):
            self.xp.add.at(flat, index, values*weight)


class Acoustic3D:
    def __init__(self, volume, dt, *, padding=12, device="cpu", device_index=0, dtype="float32", block=24):
        self.volume, self.dt = volume, float(dt)
        self.padding, self.block = int(padding), int(block)
        if padding < 2 or block < 1 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive dt/block and at least two sponge cells are required")
        self.device_index = int(device_index)
        self.xp = xp = array_module(device, device_index)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("wavefield dtype must be float32 or float64")
        stable = .9 / (float(volume.values.max())*np.sqrt(np.sum(1/np.asarray(volume.spacing)**2)))
        if dt > stable:
            raise ValueError(f"time step {dt:g} exceeds the stability limit {stable:g} s")
        self.v = xp.asarray(np.pad(volume.values, padding, mode="edge"), dtype=self.dtype)
        self.shape = self.v.shape
        damp = np.ones(self.shape, dtype=self.dtype)
        for axis, n in enumerate(self.shape):
            distance = np.minimum(np.arange(n), np.arange(n)[::-1])
            profile = np.exp(-.18*((np.maximum(padding-distance, 0)/padding)**2))
            profile[[0, -1]] = 0
            size = [1]*3; size[axis] = n
            damp *= profile.reshape(size)
        self.damp = xp.asarray(damp)

    def host(self, array):
        return np.asarray(array) if self.xp is np else self.xp.asnumpy(array)

    def sampling(self, coordinates):
        return Sampling(coordinates, self.volume.values.shape, self.volume.spacing,
                        self.volume.origin, self.padding, self.xp)

    def advance(self, previous, current, forcing, source):
        lap = laplacian(current, self.volume.spacing, self.xp)
        acc = self.v*self.v*lap
        source.add(acc, forcing)
        return self.damp*(2*current-self.damp*previous+self.dt**2*acc), lap

    @_on_device
    def forward(self, source_coordinates, receiver_coordinates, wavelet, *, checkpoints=None, report=None):
        wavelet = np.asarray(wavelet, dtype=self.dtype)
        if wavelet.ndim == 1:
            wavelet = wavelet[:, None]
        src, rec = self.sampling(source_coordinates), self.sampling(receiver_coordinates)
        if wavelet.ndim != 2 or wavelet.shape[1] != src.count or not np.isfinite(wavelet).all():
            raise ValueError("one finite source time series per source coordinate is required")
        nt = len(wavelet)
        previous = self.xp.zeros(self.shape, dtype=self.dtype)
        current = self.xp.zeros_like(previous)
        data = np.zeros((nt, rec.count), dtype=self.dtype)
        if checkpoints is not None:
            Path(checkpoints).mkdir(parents=True, exist_ok=True)
        for n in range(nt):
            if report and n % self.block == 0:
                report(n/max(nt, 1))
            if checkpoints is not None and n % self.block == 0:
                # Separate files permit mapping one checkpoint pair at a time.
                np.save(Path(checkpoints)/f"{n}-previous.npy", self.host(previous))
                np.save(Path(checkpoints)/f"{n}-current.npy", self.host(current))
            data[n] = self.host(rec.take(current))
            following, _ = self.advance(previous, current, self.xp.asarray(wavelet[n]), src)
            previous, current = current, following
        return data

    @_on_device
    def adjoint(self, source_coordinates, receiver_coordinates, wavelet, residual, checkpoints,
                *, image=False, report=None):
        """J_v^T residual and source adjoint; optionally zero-lag RTM image."""
        xp = self.xp
        wavelet = np.asarray(wavelet, dtype=self.dtype)
        if wavelet.ndim == 1:
            wavelet = wavelet[:, None]
        residual = np.asarray(residual, dtype=self.dtype)
        src, rec = self.sampling(source_coordinates), self.sampling(receiver_coordinates)
        nt = len(wavelet)
        if residual.shape != (nt, rec.count):
            raise ValueError("residual shape differs from the forward recording geometry")
        next_adj, next2_adj = xp.zeros(self.shape, dtype=self.dtype), xp.zeros(self.shape, dtype=self.dtype)
        gradient, imaging, illumination = (xp.zeros(self.shape, dtype=self.dtype) for _ in range(3))
        source_adjoint = np.zeros((nt, src.count), dtype=self.dtype)
        for begin in reversed(range(0, nt, self.block)):
            if report:
                report((nt-begin)/max(nt, 1))
            end = min(nt, begin+self.block)
            previous = xp.asarray(np.load(Path(checkpoints)/f"{begin}-previous.npy", allow_pickle=False))
            current = xp.asarray(np.load(Path(checkpoints)/f"{begin}-current.npy", allow_pickle=False))
            waves = []
            for n in range(begin, end):
                waves.append(current)
                following, _ = self.advance(previous, current, xp.asarray(wavelet[n]), src)
                previous, current = current, following
            for n in reversed(range(begin, end)):
                current = waves[n-begin]
                weighted = self.damp*next_adj
                gradient += 2*self.dt**2*self.v*weighted*laplacian(current, self.volume.spacing, xp)
                source_adjoint[n] = self.host(self.dt**2*src.take(weighted))
                adj = 2*weighted + self.dt**2*laplacian(self.v*self.v*weighted, self.volume.spacing, xp)
                adj -= self.damp*self.damp*next2_adj
                rec.add(adj, xp.asarray(residual[n]))
                if image:
                    imaging += current*adj
                    illumination += current*current
                next2_adj, next_adj = next_adj, adj
        # Edge padding replicates model boundary values; fold its derivative back.
        grad = self.host(gradient)
        p = self.padding
        for axis in range(3):
            grad = np.moveaxis(grad, axis, 0)
            core = grad[p:-p].copy()
            core[0] += grad[:p].sum(axis=0)
            core[-1] += grad[-p:].sum(axis=0)
            grad = np.moveaxis(core, 0, axis)
        crop = (slice(p, -p),)*3
        return dict(gradient=grad, source_adjoint=source_adjoint,
                    image=self.host(imaging[crop]), illumination=self.host(illumination[crop]))

    def objective_gradient(self, source, receiver, wavelet, observed, work_dir=None, report=None):
        with tempfile.TemporaryDirectory(prefix="wave-checkpoints-", dir=work_dir) as checkpoints:
            predicted = self.forward(source, receiver, wavelet, checkpoints=checkpoints,
                                     report=(lambda f: report(.5*f)) if report else None)
            residual = predicted - np.asarray(observed)
            if residual.shape != predicted.shape or not np.isfinite(residual).all():
                raise ValueError("observed data must match the predicted finite time series")
            result = self.adjoint(source, receiver, wavelet, residual, checkpoints,
                                  report=(lambda f: report(.5+.5*f)) if report else None)
        return .5*float(np.sum(residual.astype(float)**2)), result["gradient"], predicted
