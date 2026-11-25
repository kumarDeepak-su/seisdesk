"""A processing flow: an ordered list of steps (process name + parameter values), saved as JSON."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from ..core.dataset import SeismicData
from ..processing import base
from ..processing.base import Context


@dataclass
class FlowStep:
    process: str
    params: dict = field(default_factory=dict)
    enabled: bool = True
    note: str = ""

    def build(self):
        return base.create(self.process, **self.params)


@dataclass
class Flow:
    steps: list = field(default_factory=list)
    name: str = "flow"

    def add(self, process: str, **params) -> "Flow":
        base.create(process, **params)                     # validates the name and the values now
        self.steps.append(FlowStep(process, params))
        return self

    def run(self, data: SeismicData, context: Context | None = None, start: int = 0, stop: int | None = None,
            keep_intermediate: bool = False):
        """Run the enabled steps start..stop-1; returns the final dataset (and the intermediates if asked)."""
        ctx = context if context is not None else Context()
        steps = self.steps[start:stop]
        n = max(len(steps), 1)
        outer = ctx.progress
        current = data
        intermediates = []
        for i, step in enumerate(steps):
            if not step.enabled:
                continue
            proc = step.build()
            t0 = time.perf_counter()
            ctx.log(f"[{i + 1}/{len(steps)}] {proc.label} ...")
            if outer is not None:                     # a step's own progress maps into its slice of the flow
                ctx.progress = lambda f, i=i: outer((i + f) / n)
            try:
                current = proc(current, ctx)
            finally:
                ctx.progress = outer
            ctx.log(f"    done in {time.perf_counter() - t0:.1f} s: {current.ntr} traces x {current.nt} samples")
            if keep_intermediate:
                intermediates.append(current)
            ctx.report((i + 1) / n)
        return (current, intermediates) if keep_intermediate else current

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "steps": [asdict(s) for s in self.steps]}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Flow":
        d = json.loads(text)
        return cls([FlowStep(**s) for s in d.get("steps", [])], d.get("name", "flow"))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Flow":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())
