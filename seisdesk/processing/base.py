"""Process base class, parameter schema, registry and the context shared by the steps of a flow.

A process declares its parameters as ``Param`` objects; the GUI builds its form from them and a flow stores the values
as plain JSON. ``run(data, context)`` returns a new ``SeismicData``; side products (velocity picks, pick sets, a depth
model, tables) live in the ``Context`` so that later steps — and the GUI — can use them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.dataset import SeismicData
from ..core.picks import PickSet
from ..core.velocity import DepthModel, VelocityPicks

REGISTRY: dict[str, type["Process"]] = {}
CATEGORIES = ("Input/Output", "Geometry", "Editing", "Vibroseis", "Gain", "Filtering", "Deconvolution", "Wavelet",
              "Multiples", "Statics", "Datuming", "Interpolation", "Velocity", "Moveout", "Stack", "Time migration",
              "Depth migration", "Tomography", "Inversion", "AVO", "Converted waves", "Conversion", "Attributes",
              "Multicomponent", "Modelling", "Picking", "Utility")


@dataclass
class Param:
    name: str
    kind: str          # float | int | bool | choice | str | text | floats | key | keys | file | infile | directory | velocity | picks
    default: Any
    label: str = ""
    help: str = ""
    choices: tuple = ()
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""

    def coerce(self, value):
        """Convert a GUI or JSON value to the parameter's Python type, with range checks."""
        if value is None:
            return None
        k = self.kind
        if k == "float":
            v = float(value)
        elif k == "int":
            v = int(round(float(value)))
        elif k == "bool":
            v = value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
        elif k == "choice":
            v = str(value)
            if self.choices and v not in self.choices:
                raise ValueError(f"{self.name}: {v!r} is not one of {self.choices}")
        elif k == "floats":
            if isinstance(value, str):
                v = [float(s) for s in value.replace(";", ",").split(",") if s.strip()]
            else:
                v = [float(s) for s in value]
        elif k == "keys":
            v = [s.strip() for s in value.split(",") if s.strip()] if isinstance(value, str) else list(value)
        else:
            v = str(value)
        if k in ("float", "int"):
            if self.minimum is not None and v < self.minimum:
                raise ValueError(f"{self.name} must be >= {self.minimum}")
            if self.maximum is not None and v > self.maximum:
                raise ValueError(f"{self.name} must be <= {self.maximum}")
        return v


@dataclass
class Context:
    """Objects shared by the steps of a flow and the GUI."""
    velocity: VelocityPicks | None = None
    depth_model: DepthModel | None = None
    picks: dict = field(default_factory=dict)          # name -> PickSet
    results: dict = field(default_factory=dict)        # free-form outputs (tables, panels)
    work_dir: str = "."
    log: Callable[[str], None] = print
    progress: Callable[[float], None] | None = None
    cancelled: Callable[[], bool] = lambda: False

    def report(self, fraction: float) -> None:
        if self.progress is not None:
            self.progress(float(min(max(fraction, 0.0), 1.0)))
        if self.cancelled():
            raise InterruptedError("cancelled by the user")


class Process:
    name = ""
    label = ""
    category = "Utility"
    description = ""
    params: tuple = ()

    def __init__(self, **values):
        known = {p.name: p for p in self.params}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {unknown}")
        self.values = {}
        for p in self.params:
            self.values[p.name] = p.coerce(values.get(p.name, p.default))

    def __getitem__(self, key):
        return self.values[key]

    def run(self, data: SeismicData, context: Context) -> SeismicData:      # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.label}: " + ", ".join(f"{k}={v}" for k, v in self.values.items())

    def __call__(self, data: SeismicData, context: Context | None = None) -> SeismicData:
        ctx = context if context is not None else Context()
        out = self.run(data, ctx)
        out.add_history(self.describe())
        return out


def register(cls: type[Process]) -> type[Process]:
    if not cls.name:
        raise ValueError("process without a name")
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate process name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def create(name: str, **values) -> Process:
    if name not in REGISTRY:
        raise KeyError(f"unknown process {name!r}")
    return REGISTRY[name](**values)


# ------------------------------------------------------------------------------------------------ resolvers
def resolve_velocity(value: str, context: Context) -> VelocityPicks:
    """A velocity parameter: 'project' (the picks in the context), a number (constant), 't:v,t:v,...' (one
    function, times in seconds) or the path of a picks file."""
    text = str(value).strip()
    if text.lower() in ("", "project", "current", "picks"):
        if context.velocity is None or len(context.velocity) == 0:
            raise ValueError("no velocity picks in the project; pick velocities or give a constant/function/file")
        return context.velocity
    try:
        return VelocityPicks.constant(float(text))
    except ValueError:
        pass
    if ":" in text and not os.path.exists(text):
        pairs = [p.split(":") for p in text.replace(";", ",").split(",") if p.strip()]
        return VelocityPicks.from_function([float(a) for a, _ in pairs], [float(b) for _, b in pairs])
    path = text if os.path.isabs(text) else os.path.join(context.work_dir, text)
    if not os.path.exists(path):
        raise FileNotFoundError(f"velocity file {path}")
    return VelocityPicks.load(path)


def resolve_picks(name: str, context: Context) -> PickSet:
    """A pick-set parameter: the name of a pick set in the context or the path of a saved pick file."""
    if name in context.picks:
        return context.picks[name]
    path = name if os.path.isabs(name) else os.path.join(context.work_dir, name)
    if os.path.exists(path):
        ps = PickSet.load(path)
        context.picks[ps.name] = ps
        return ps
    raise KeyError(f"no pick set named {name!r} (pick it in the viewer or give a file)")


def trace_positions(data: SeismicData) -> np.ndarray:
    """Position along the line (m) of every trace: CMP x, else the source-receiver midpoint, else CMP number times
    the bin size of the 2D geometry."""
    cx = data.header("cdpx")
    if np.ptp(cx) > 0:
        return cx
    mx = 0.5 * (data.header("sx") + data.header("gx"))
    if np.ptp(mx) > 0:
        return mx
    cdp = data.header("cdp")
    bin_size = data.meta.get("geometry", {}).get("bin_size")
    if bin_size and np.ptp(cdp) > 0:
        return (cdp - cdp.min()) * float(bin_size)
    raise ValueError("trace positions unknown: run a geometry step (CMP x or source/receiver x)")


def model_grid(x: np.ndarray) -> tuple[float, float, int]:
    """Origin, spacing and size of an increasing regular grid covering the positions x."""
    ux = np.unique(np.asarray(x, dtype=np.float64))
    if ux.size < 2:
        return float(ux[0]) if ux.size else 0.0, 25.0, 2
    dx = float(np.median(np.diff(ux)))
    nx = int(round((ux[-1] - ux[0]) / dx)) + 1
    return float(ux[0]), dx, max(nx, 2)


def pad_to_time_zero(data: SeismicData) -> SeismicData:
    """Prepend zero samples so the first sample sits at t = 0 (f-k, phase-shift and split-step migrations assume
    records that start at time zero, which a recording delay or a time window breaks)."""
    n0 = int(round(data.t0 / data.dt))
    if n0 <= 0:
        return data
    return data.copy(np.pad(data.traces, ((0, 0), (n0, 0))), t0=data.t0 - n0 * data.dt)


def cdp_values(data: SeismicData) -> np.ndarray:
    cdp = data.header("cdp")
    if not np.any(cdp):
        raise ValueError("CMP numbers (cdp) are not set; run a geometry step first")
    return cdp


def taper_window(n: int, frac: float) -> np.ndarray:
    """Cosine taper of total length n with ``frac`` of the length tapered at each end."""
    w = np.ones(n)
    m = int(round(frac * n))
    if m > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
        w[:m] = ramp
        w[-m:] = ramp[::-1]
    return w
