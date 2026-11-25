"""Command-line use of SeisDesk: inspect files, list processes, run saved flows in batch, make synthetic data.

    seisdesk-cli info line.sgy
    seisdesk-cli processes [name]
    seisdesk-cli run flow.json input.sgy output.sgy [--velocity picks.txt] [--picks name=file.json]
                 [--depth-model model.npz] [--save-velocity out.txt]
    seisdesk-cli synthetic marine|land out.sgy
"""

from __future__ import annotations

import argparse
import os
import sys


def _read(path: str):
    from .core.segy_io import read_any
    return read_any(path)


def _write(data, path: str) -> None:
    from .core.segy_io import write_segy, write_su
    (write_su if path.lower().endswith(".su") else write_segy)(data, path)


def cmd_info(args) -> int:
    data = _read(args.file)
    print(data.summary())
    keys = [k for k in data.headers if (data.headers[k] != 0).any()]
    print("non-zero headers:", ", ".join(keys))
    return 0


def cmd_import(args) -> int:
    from .core.segy_io import read_any
    from .workflow.project import Project
    from .core.sps import apply_sps
    options = dict(target_dt=args.dt_ms/1000 if args.dt_ms else None, window=args.window, max_mb=args.max_mb)
    data = read_any(args.input, **options)
    if any((args.sps_source, args.sps_receiver, args.sps_relation)):
        if not all((args.sps_source, args.sps_receiver, args.sps_relation)):
            raise ValueError("SPS geometry requires all three S/R/X files")
        data = apply_sps(data, args.sps_source, args.sps_receiver, args.sps_relation,
                         coordinate_factor=args.coordinate_factor, tape=args.tape)
    project = Project()
    project.datasets[data.name] = data
    project.save(args.output)
    print(data.summary())
    return 0


def cmd_processes(args) -> int:
    from .processing import REGISTRY
    if args.name:
        cls = REGISTRY.get(args.name)
        if cls is None:
            print(f"unknown process {args.name!r}")
            return 1
        print(f"{cls.name}: {cls.label} [{cls.category}]\n  {cls.description}")
        for p in cls.params:
            extra = f" ({p.unit})" if p.unit else ""
            choices = f" one of {list(p.choices)}" if p.choices else ""
            print(f"  {p.name:18s} {p.kind:9s} default {p.default!r}{extra}{choices}  {p.label}")
        return 0
    for cls in sorted(REGISTRY.values(), key=lambda c: (c.category, c.name)):
        print(f"{cls.category:16s} {cls.name:22s} {cls.label}")
    return 0


def cmd_run(args) -> int:
    from .core.picks import PickSet
    from .core.velocity import DepthModel, VelocityPicks
    from .processing.base import Context
    from .workflow.flow import Flow

    flow = Flow.load(args.flow)
    ctx = Context(work_dir=args.work_dir or os.path.dirname(os.path.abspath(args.output)))
    if args.velocity:
        ctx.velocity = VelocityPicks.load(args.velocity)
    for item in args.picks or []:
        name, path = item.split("=", 1) if "=" in item else (None, item)
        ps = PickSet.load(path)
        ctx.picks[name or ps.name] = ps
    if args.depth_model:
        ctx.depth_model = DepthModel.load(args.depth_model)
    data = _read(args.input)
    print(data.summary())
    out = flow.run(data, ctx)
    _write(out, args.output)
    print(f"wrote {args.output}: {out.ntr} traces x {out.nt} samples")
    if args.save_velocity and ctx.velocity is not None:
        ctx.velocity.save(args.save_velocity)
    return 0


def cmd_synthetic(args) -> int:
    from .core import synthetic
    data = synthetic.marine_line(noise=0.02) if args.kind == "marine" else synthetic.land_refraction_line(
        noise=0.02, source_statics_ms=8.0, receiver_statics_ms=8.0)
    _write(data, args.output)
    print(data.summary())
    return 0


def cmd_su(args) -> int:
    from .backends.su import SUBackend, run_pipeline
    backend = SUBackend(args.backend, args.bin_dir, args.distribution)
    if args.command == "su-programs":
        print(backend.documentation(args.program) if args.program else "\n".join(backend.programs()))
        return 0
    pipeline = args.pipeline
    if args.pipeline_file:
        with open(args.pipeline_file, encoding="utf-8") as stream:
            pipeline = stream.read()
    report = run_pipeline(pipeline, args.input, args.output, backend, timeout=args.timeout,
                          work_dir=args.work_dir)
    print(f"Completed. Job record: {report['record']}")
    return 0


def cmd_volume_job(args) -> int:
    import json
    from .processing import Context, create
    from .workflow.project import save_dataset
    with open(args.parameters, encoding="utf-8") as stream:
        parameters = json.load(stream)
    parameters.update(input_file=os.path.abspath(args.input), job_directory=os.path.abspath(args.output), mode=args.mode)
    context = Context(work_dir=os.path.dirname(os.path.abspath(args.parameters)))
    output = create("volume_file_job", **parameters).run(None, context)
    save_dataset(output, os.path.join(args.output, "preview.npz"))
    print(f"Completed 3D {args.mode}; job record: {os.path.abspath(args.output)}/job.json")
    return 0


def cmd_remote(args) -> int:
    import json
    from pathlib import Path
    from .backends.remote import RemoteClient, RemoteProfile, load_record, prepare_job
    if args.command == "remote-submit":
        profile = RemoteProfile(**json.loads(Path(args.profile).read_text(encoding="utf-8")))
        parameters = json.loads(Path(args.parameters).read_text(encoding="utf-8"))
        record = prepare_job(parameters, profile, args.directory, work_dir=Path(args.parameters).resolve().parent,
                             report=lambda s: print(s, flush=True))
        print("Saved job record: "+record["record_file"], flush=True)
        RemoteClient(profile).submit(record, report=lambda s: print(s, flush=True))
    else:
        record = load_record(args.record)
        client = RemoteClient(RemoteProfile(**record["profile"]))
        if args.command == "remote-fetch":
            client.fetch(record, args.directory, report=lambda s: print(s, flush=True))
        else:
            getattr(client, args.command.removeprefix("remote-"))(record)
    print(json.dumps(record, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seisdesk-cli", description="SeisDesk command line")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("info", help="summary of a SEG-Y or SU file")
    p.add_argument("file")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("import", help="import field data with explicit resampling and optional SPS into a project")
    p.add_argument("input")
    p.add_argument("output", help="project path (preserves original trace timing and provenance)")
    p.add_argument("--dt-ms", type=float, default=0)
    p.add_argument("--window", choices=("strict", "intersection", "union", "zero"), default="strict")
    p.add_argument("--max-mb", type=float, default=1024)
    p.add_argument("--sps-source"); p.add_argument("--sps-receiver"); p.add_argument("--sps-relation")
    p.add_argument("--coordinate-factor", type=float)
    p.add_argument("--tape", default="")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("processes", help="list processing steps, or the parameters of one")
    p.add_argument("name", nargs="?")
    p.set_defaults(fn=cmd_processes)
    p = sub.add_parser("run", help="run a saved flow on a file")
    p.add_argument("flow")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--velocity")
    p.add_argument("--picks", action="append", help="name=file.json (repeatable)")
    p.add_argument("--depth-model")
    p.add_argument("--save-velocity")
    p.add_argument("--work-dir")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("synthetic", help="write a synthetic test line")
    p.add_argument("kind", choices=("marine", "land"))
    p.add_argument("output")
    p.set_defaults(fn=cmd_synthetic)
    p = sub.add_parser("volume-job", help="stream a survey into 3D modelling, RTM or FWI")
    p.add_argument("parameters", help="JSON parameters; paths are relative to this file")
    p.add_argument("input", help="SEG-Y or SU survey")
    p.add_argument("output", help="job directory")
    p.add_argument("--mode", choices=("model", "rtm", "fwi"), default="rtm")
    p.set_defaults(fn=cmd_volume_job)
    p = sub.add_parser("remote-submit", help="bundle inputs/source and submit a volume job to SSH/Slurm")
    p.add_argument("parameters", help="volume job parameters JSON, including input_file and model")
    p.add_argument("profile", help="SSH/Slurm profile JSON")
    p.add_argument("directory", help="local directory for durable job records and bundles")
    p.set_defaults(fn=cmd_remote)
    for name in ("status", "cancel", "fetch"):
        p = sub.add_parser("remote-"+name, help=name+" a saved SSH/Slurm job")
        p.add_argument("record", help="saved remote-job.json")
        if name == "fetch":
            p.add_argument("directory", help="download results into a new subfolder here")
        p.set_defaults(fn=cmd_remote)
    for name in ("su-run", "su-programs"):
        p = sub.add_parser(name, help="run or inspect installed Seismic Unix programs")
        p.add_argument("--backend", choices=("local", "wsl"), default="local")
        p.add_argument("--bin-dir", default="")
        p.add_argument("--distribution", default="")
        if name == "su-run":
            p.add_argument("input")
            p.add_argument("output")
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument("--pipeline")
            group.add_argument("--pipeline-file")
            p.add_argument("--timeout", type=float, default=3600)
            p.add_argument("--work-dir")
        else:
            p.add_argument("program", nargs="?")
        p.set_defaults(fn=cmd_su)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
