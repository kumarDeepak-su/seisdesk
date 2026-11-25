"""The process and parameter reference, generated from the registry (shown under Help and appended to the manual)."""

from __future__ import annotations

import html

from .base import CATEGORIES, REGISTRY

KINDS = {"float": "number", "int": "whole number", "bool": "yes/no", "choice": "choice", "str": "text",
         "floats": "list of numbers", "key": "header name", "keys": "list of header names", "file": "output file",
         "infile": "input file", "velocity": "velocity", "picks": "pick set"}


def _rows():
    by_cat = {}
    for cls in REGISTRY.values():
        by_cat.setdefault(cls.category, []).append(cls)
    for cat in list(CATEGORIES) + sorted(set(by_cat) - set(CATEGORIES)):
        for cls in sorted(by_cat.get(cat, []), key=lambda c: c.label):
            yield cat, cls


def _default(p) -> str:
    v = p.default
    if isinstance(v, (list, tuple)):
        v = ", ".join(str(x) for x in v)
    if p.kind == "float" and isinstance(v, float) and abs(v) >= 1e29:
        return "auto"
    return "" if v == "" else str(v)


def _range(p) -> str:
    if p.choices:
        return " | ".join(p.choices)
    parts = []
    if p.minimum is not None:
        parts.append(f">= {p.minimum:g}")
    if p.maximum is not None:
        parts.append(f"<= {p.maximum:g}")
    return ", ".join(parts)


def process_reference_markdown() -> str:
    out = ["# Process and parameter reference", "",
           "Every processing step, as listed in the flow editor, with its parameters, defaults and allowed values. "
           "The step name in brackets is the one used in saved flows and by `seisdesk-cli`.", ""]
    current = None
    for cat, cls in _rows():
        if cat != current:
            out += [f"## {cat}", ""]
            current = cat
        out += [f"### {cls.label} (`{cls.name}`)", "", cls.description, ""]
        if cls.params:
            out += ["| Parameter | Type | Default | Allowed | Unit |", "|---|---|---|---|---|"]
            for p in cls.params:
                cells = [f"{p.label or p.name} (`{p.name}`)", KINDS.get(p.kind, p.kind),
                         _default(p), _range(p), p.unit]
                # Literal pipes in choices/defaults must not become Markdown column separators.
                out.append("| " + " | ".join(str(cell).replace("|", "&#124;").replace("\n", " ")
                                               for cell in cells) + " |")
            out.append("")
    return "\n".join(out)


def process_reference_html() -> str:
    parts = ["<h2>Process and parameter reference</h2>",
             "<p>Every processing step, as listed in the flow editor, with its parameters, defaults and allowed "
             "values. The name in brackets is the one used in saved flows and by <tt>seisdesk-cli</tt>.</p>"]
    current = None
    for cat, cls in _rows():
        if cat != current:
            parts.append(f"<h3>{html.escape(cat)}</h3>")
            current = cat
        parts.append(f"<p><b>{html.escape(cls.label)}</b> <tt>({cls.name})</tt><br>{html.escape(cls.description)}</p>")
        if cls.params:
            rows = "".join(
                f"<tr><td>{html.escape(p.label or p.name)} <tt>({p.name})</tt></td><td>{KINDS.get(p.kind, p.kind)}</td>"
                f"<td>{html.escape(_default(p))}</td><td>{html.escape(_range(p))}</td><td>{html.escape(p.unit)}</td></tr>"
                for p in cls.params)
            parts.append("<table border='1' cellspacing='0' cellpadding='3'><tr><th>Parameter</th><th>Type</th>"
                         f"<th>Default</th><th>Allowed</th><th>Unit</th></tr>{rows}</table>")
    return "\n".join(parts)
