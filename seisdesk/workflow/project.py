"""A project: named datasets, the shared processing context (velocity picks, pick sets, depth model) and saved flows.

A saved project is a folder::

    project.json            names, flows, settings
    datasets/<n>.npz        traces, sample axis and headers of every dataset
    velocity.txt            velocity picks (if any)
    depth_model.npz         interval-velocity depth model (if any)
    picks/<name>.json       pick sets
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from ..core.dataset import SeismicData
from ..core.picks import PickSet
from ..core.velocity import DepthModel, VelocityPicks
from ..processing.base import Context
from .flow import Flow
from .results import load_results, save_results

FORMAT_VERSION = 1


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _file_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return stem or "data"


def save_dataset(data: SeismicData, path: str) -> None:
    arrays = {"traces": data.traces, "dt": np.float64(data.dt), "t0": np.float64(data.t0),
              "trace_ids": data.trace_ids}
    for key, values in data.headers.items():
        arrays["h_" + key] = values
    info = {"name": data.name, "domain": data.domain, "history": data.history, "meta": _json_safe(data.meta)}
    arrays["info"] = np.array(json.dumps(info))
    np.savez(path, **arrays)


def load_dataset(path: str) -> SeismicData:
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z["info"]))
        headers = {k[2:]: z[k] for k in z.files if k.startswith("h_")}
        return SeismicData(z["traces"], float(z["dt"]), headers, float(z["t0"]), info.get("name", "data"),
                           info.get("domain", "time"), list(info.get("history", [])), dict(info.get("meta", {})),
                           z["trace_ids"] if "trace_ids" in z.files else None)


class Project:
    def __init__(self, path: str | None = None):
        self.path = path
        self.datasets: dict[str, SeismicData] = {}
        self.context = Context(work_dir=path or os.getcwd())
        self.flows: dict[str, Flow] = {}
        self.settings: dict = {}

    # ------------------------------------------------------------------ datasets
    def unique_name(self, name: str) -> str:
        base = name.strip() or "data"
        out, k = base, 2
        while out in self.datasets:
            out = f"{base} ({k})"
            k += 1
        return out

    def add_dataset(self, data: SeismicData, name: str | None = None) -> str:
        key = self.unique_name(name or data.name)
        data.name = key
        self.datasets[key] = data
        return key

    def remove_dataset(self, name: str) -> None:
        self.datasets.pop(name, None)

    def rename_dataset(self, old: str, new: str) -> str:
        if old not in self.datasets:
            raise KeyError(old)
        data = self.datasets.pop(old)
        key = self.unique_name(new)
        data.name = key
        self.datasets[key] = data
        return key

    # ------------------------------------------------------------------ pick sets
    def pick_set(self, name: str, gather_key: str = "fldr", x_key: str = "offset") -> PickSet:
        if name not in self.context.picks:
            self.context.picks[name] = PickSet(gather_key, x_key, name)
        return self.context.picks[name]

    # ------------------------------------------------------------------ files
    def save(self, path: str | None = None) -> str:
        path = path or self.path
        if not path:
            raise ValueError("no project folder given")
        os.makedirs(os.path.join(path, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(path, "picks"), exist_ok=True)
        save_results(self.context.results, os.path.join(path, "results.npz"))
        files = {}
        used = set()
        for name, data in self.datasets.items():
            stem = _file_stem(name)
            k = 2
            while stem.lower() in used:
                stem = f"{_file_stem(name)}_{k}"
                k += 1
            used.add(stem.lower())
            save_dataset(data, os.path.join(path, "datasets", stem + ".npz"))
            files[name] = stem + ".npz"
        for old in os.listdir(os.path.join(path, "datasets")):
            if old.endswith(".npz") and old not in files.values():
                os.remove(os.path.join(path, "datasets", old))
        vel = os.path.join(path, "velocity.txt")
        if self.context.velocity is not None and len(self.context.velocity):
            self.context.velocity.save(vel)
        elif os.path.exists(vel):
            os.remove(vel)
        dm = os.path.join(path, "depth_model.npz")
        if self.context.depth_model is not None:
            self.context.depth_model.save(dm)
        elif os.path.exists(dm):
            os.remove(dm)
        pick_files = {}
        used_picks = set()
        for name, ps in self.context.picks.items():
            stem = _file_stem(name)
            k = 2
            while stem.lower() in used_picks:
                stem = f"{_file_stem(name)}_{k}"
                k += 1
            used_picks.add(stem.lower())
            ps.save(os.path.join(path, "picks", stem + ".json"))
            pick_files[name] = stem + ".json"
        for old in os.listdir(os.path.join(path, "picks")):
            if old.endswith(".json") and old not in pick_files.values():
                os.remove(os.path.join(path, "picks", old))
        doc = {"format": FORMAT_VERSION, "datasets": files, "picks": pick_files,
               "flows": {name: json.loads(flow.to_json()) for name, flow in self.flows.items()},
               "settings": _json_safe(self.settings), "results": "results.npz"}
        with open(os.path.join(path, "project.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
        self.path = path
        self.context.work_dir = path
        return path

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(os.path.join(path, "project.json"), encoding="utf-8") as f:
            doc = json.load(f)
        proj = cls(path)
        if doc.get("results"):
            proj.context.results = load_results(os.path.join(path, doc["results"]))
        for name, fname in doc.get("datasets", {}).items():
            data = load_dataset(os.path.join(path, "datasets", fname))
            data.name = name
            proj.datasets[name] = data
        vel = os.path.join(path, "velocity.txt")
        if os.path.exists(vel):
            proj.context.velocity = VelocityPicks.load(vel)
        dm = os.path.join(path, "depth_model.npz")
        if os.path.exists(dm):
            proj.context.depth_model = DepthModel.load(dm)
        for name, fname in doc.get("picks", {}).items():
            ps = PickSet.load(os.path.join(path, "picks", fname))
            ps.name = name
            proj.context.picks[name] = ps
        for name, flow in doc.get("flows", {}).items():
            proj.flows[name] = Flow.from_json(json.dumps(flow))
        proj.settings = dict(doc.get("settings", {}))
        return proj
