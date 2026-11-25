"""Portable diagnostic snapshots: typed JSON plus arrays, without pickle."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def save_results(results: dict, path: str | Path) -> None:
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("diagnostic object arrays cannot be saved")
            key = f"a{len(arrays)}"
            arrays[key] = value
            return ["array", key]
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, dict):
            return ["dict", [[encode(k), encode(v)] for k, v in value.items()]]
        if isinstance(value, (list, tuple)):
            return ["tuple" if isinstance(value, tuple) else "list", [encode(v) for v in value]]
        if isinstance(value, complex):
            return ["complex", [value.real, value.imag]]
        if value is None or isinstance(value, (str, int, float, bool)):
            return ["scalar", value]
        raise ValueError(f"unsupported diagnostic value: {type(value).__name__}")

    arrays["tree"] = np.array(json.dumps(encode(results)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".results-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_results(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        def decode(node):
            kind, value = node
            if kind == "array":
                return archive[value].copy()
            if kind == "dict":
                return {decode(k): decode(v) for k, v in value}
            if kind in ("list", "tuple"):
                items = [decode(v) for v in value]
                return tuple(items) if kind == "tuple" else items
            if kind == "complex":
                return complex(*value)
            if kind == "scalar":
                return value
            raise ValueError(f"unknown diagnostic storage type {kind!r}")

        result = decode(json.loads(str(archive["tree"])))
    if not isinstance(result, dict):
        raise ValueError("diagnostic snapshot must contain a mapping")
    return result
