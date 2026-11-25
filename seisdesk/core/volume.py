"""A true v(z,y,x) model with explicit metre coordinates and portable storage."""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class VelocityVolume:
    values: np.ndarray
    spacing: tuple = (10., 10., 10.)  # dz, dy, dx in metres
    origin: tuple = (0., 0., 0.)     # z0, y0, x0; z positive down
    reference_elevation: float = 0.

    def __post_init__(self):
        self.values = np.asarray(self.values)
        if self.values.dtype not in (np.dtype("float32"), np.dtype("float64")):
            self.values = self.values.astype(np.float32)
        self.spacing, self.origin = tuple(map(float, self.spacing)), tuple(map(float, self.origin))
        if self.values.ndim != 3 or min(self.values.shape) < 3:
            raise ValueError("velocity must have shape (nz,ny,nx), with at least three nodes per axis")
        if not np.isfinite(self.values).all() or np.any(self.values <= 0):
            raise ValueError("velocity values must be finite and positive, in m/s")
        if len(self.spacing) != 3 or len(self.origin) != 3 or min(self.spacing) <= 0:
            raise ValueError("give three positive spacings and three origins, in z,y,x order")
        if not np.isfinite([*self.spacing, *self.origin, self.reference_elevation]).all():
            raise ValueError("model coordinates must be finite")

    @property
    def axes(self):
        return tuple(o + np.arange(n)*d for o, d, n in zip(self.origin, self.spacing, self.values.shape))

    def save(self, path):
        path = Path(path)
        if path.suffix.lower() == ".npz":
            np.savez(path, values=self.values, spacing=self.spacing, origin=self.origin,
                     reference_elevation=self.reference_elevation)
        else:
            path.mkdir(parents=True, exist_ok=True)
            np.save(path/"velocity.npy", self.values)
            (path/"volume.json").write_text(json.dumps(dict(spacing=self.spacing, origin=self.origin,
                                                           reference_elevation=self.reference_elevation)), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_dir():
            metadata = json.loads((path/"volume.json").read_text(encoding="utf-8"))
            return cls(np.load(path/"velocity.npy", mmap_mode="r", allow_pickle=False), **metadata)
        with np.load(path, allow_pickle=False) as data:
            return cls(data["values"], tuple(data["spacing"]), tuple(data["origin"]),
                       float(data["reference_elevation"]) if "reference_elevation" in data else 0.)

    def coordinates(self, data):
        """Source/receiver positions in z,y,x order, including source depth."""
        src = np.column_stack([self.reference_elevation-data.header("selev")+data.header("sdepth"),
                               data.header("sy", None), data.header("sx", None)])
        rec = np.column_stack([self.reference_elevation-data.header("gelev"),
                               data.header("gy", None), data.header("gx", None)])
        for coords in (src, rec):
            relative = (coords-np.asarray(self.origin))/np.asarray(self.spacing)
            if not np.isfinite(relative).all() or np.any(relative < -1e-5) or np.any(
                    relative > np.asarray(self.values.shape)-1+1e-5):
                raise ValueError("source/receiver geometry lies outside the 3D model; check units and elevation datum")
        return src, rec

    def dataset(self, values=None, name="3D volume"):
        from .dataset import SeismicData
        values = self.values if values is None else np.asarray(values)
        if values.shape != self.values.shape:
            raise ValueError("output volume shape differs from the model grid")
        z, y, x = self.axes
        xx, yy = np.meshgrid(x, y)
        iy, ix = np.indices((len(y), len(x)))
        headers = dict(cdpx=xx.ravel(), cdpy=yy.ravel(), sx=xx.ravel(), gx=xx.ravel(),
                       sy=yy.ravel(), gy=yy.ravel(), iline=iy.ravel()+1, xline=ix.ravel()+1,
                       cdp=np.arange(xx.size)+1)
        out = SeismicData(np.moveaxis(values, 0, -1).reshape(-1, len(z)), self.spacing[0], headers,
                          t0=self.origin[0], domain="depth", name=name)
        out.meta["volume"] = dict(shape=list(values.shape), spacing=self.spacing, origin=self.origin,
                                   reference_elevation=self.reference_elevation)
        return out
