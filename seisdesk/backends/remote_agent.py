"""Slurm-side job control. Metadata operations never import scientific packages.

Executed as a file by the desktop SSH client; only ``run`` performs computation,
and that entry point requires a Slurm allocation.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".part")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def command(args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def receipt(root):
    value = read_json(root/"submission.json")
    if not value or not re.fullmatch(r"[0-9]+", str(value.get("job_id", ""))):
        raise ValueError("no confirmed Slurm job ID; inspect the submission log before trying again")
    return value


def submit(root):
    existing = read_json(root/"submission.json")
    if existing:
        return receipt(root)
    # Keep this marker even if the SSH connection breaks while sbatch runs.
    # An uncertain submission must be recovered, never submitted a second time.
    with (root/"submission.lock").open("x", encoding="utf-8") as stream:
        stream.write("Submission started. Do not resubmit this bundle.\n")
    profile = read_json(root/"profile.json")
    bundle = root/"bundle"
    script = root/"run.sh"
    lines = ["#!/bin/bash", "set -euo pipefail", "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
             "export PYTHONUNBUFFERED=1"]
    if profile.get("modules"):
        lines.append("module load "+shlex.join(profile["modules"]))
    lines.append("exec "+shlex.join([profile["python"], str(bundle/"code/seisdesk/backends/remote_agent.py"),
                                     "run", str(root)]))
    script.write_text("\n".join(lines)+"\n", encoding="utf-8")
    args = ["sbatch", "--parsable", "--job-name=sd-"+root.name[-12:], "--nodes=1", "--ntasks=1",
            "--cpus-per-task="+str(profile["cpus"]), "--mem="+str(profile["memory_gb"])+"G",
            "--time="+profile["walltime"], "--chdir="+str(root),
            "--output="+str(root/"slurm.log"), "--error="+str(root/"slurm.log")]
    for field in ("partition", "account"):
        if profile.get(field):
            args.append("--"+field+"="+profile[field])
    if profile.get("gpus"):
        args.append("--gres=gpu:"+str(profile["gpus"]))
    args.append(str(script))
    try:
        response = command(args)
        identifier = response.split(";")[0].strip()
        if not re.fullmatch(r"[0-9]+", identifier):
            raise RuntimeError(f"unrecognised sbatch response: {response}")
        result = dict(job_id=identifier, submitted=time.time(), remote_directory=str(root), state="SUBMITTED")
        save_json(root/"submission.json", result)
        return result
    except BaseException as exc:
        save_json(root/"submission-error.json", dict(error=str(exc), time=time.time()))
        raise


def status(root):
    saved = read_json(root/"submission.json")
    if not saved:
        return dict(state="SUBMISSION_UNCONFIRMED", error=read_json(root/"submission-error.json"),
                    log="Submission is unconfirmed. Check this job's directory and Slurm before retrying.")
    result = dict(receipt(root))
    identifier = result["job_id"]
    # Some Slurm sites return nonzero after the job leaves squeue. Accounting
    # is the authoritative fallback; do not call that a failed compute job.
    queue = subprocess.run(["squeue", "--noheader", "--jobs="+identifier, "--format=%T|%R"],
                           capture_output=True, text=True, timeout=45)
    if queue.returncode == 0 and queue.stdout.strip():
        fields = queue.stdout.strip().splitlines()[0].split("|", 1)
        result.update(state=fields[0], reason=fields[1] if len(fields) > 1 else "")
    else:
        text = command(["sacct", "--noheader", "--parsable2", "--allocations", "--jobs="+identifier,
                        "--format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,NodeList"])
        rows = [line.split("|") for line in text.splitlines()]
        row = next((row for row in rows if len(row) >= 6 and row[0] == identifier), None)
        if row:
            result.update(zip(("job_id", "state", "exit_code", "elapsed", "cpus", "nodes"), row[:6]))
        else:
            result.update(state="ACCOUNTING_PENDING", reason="Not in queue; accounting record is not available yet")
    for label, name in (("application", "run.json"), ("results", "results.json")):
        result[label] = read_json(root/name)
    result["log"] = ""
    for name in ("slurm.log", "run.log"):
        path = root/name
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16_384))
                result["log"] += name+":\n"+stream.read().decode("utf-8", errors="replace")+"\n"
    return result


def cancel(root):
    saved = receipt(root)
    command(["scancel", saved["job_id"]])
    save_json(root/"cancel-request.json", dict(job_id=saved["job_id"], time=time.time()))
    return dict(job_id=saved["job_id"], state="CANCEL_REQUESTED")


def run(root):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("scientific execution requires a Slurm compute allocation")
    bundle = root/"bundle"
    sys.path.insert(0, str(bundle/"code"))
    from seisdesk.backends.job_bundle import write_bundle
    parameters = read_json(bundle/"parameters.json")
    packages = {}
    for name in ("numpy", "scipy", "segyio", "cupy-cuda12x", "distributed"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = dict(state="running", started=time.time(), host=platform.node(),
                  job_id=os.environ["SLURM_JOB_ID"], python=sys.version, packages=packages,
                  source=read_json(bundle/"manifest.json")["metadata"])
    save_json(root/"run.json", record)
    child = None
    def terminated(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise InterruptedError(f"Slurm signal {signum}")
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    code = 1
    try:
        environment = dict(os.environ, PYTHONPATH=str(bundle/"code"))
        args = [sys.executable, "-m", "seisdesk.cli", "volume-job", str(bundle/"parameters.json"),
                str(bundle/parameters["input_file"]), str(bundle/"output"), "--mode", parameters.get("mode", "rtm")]
        with (root/"run.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(args, cwd=bundle, env=environment, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        record.update(state="completed" if code == 0 else "failed", exit_code=code)
    except BaseException as exc:
        record.update(state="cancelled" if isinstance(exc, InterruptedError) else "failed", error=str(exc))
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
    finally:
        record["finished"] = time.time()
        save_json(root/"run.json", record)
    # Packaging is also compute-node work. A hard Slurm kill can prevent this;
    # status/log retrieval remains available even when no result archive exists.
    files = [("run.json", root/"run.json"), ("input-manifest.json", bundle/"manifest.json")]
    if (root/"run.log").exists():
        files.append(("run.log", root/"run.log"))
    # Retain original inputs and compute source as well as staged/predicted shots.
    # Otherwise the downloaded parameter file would point at absent input files.
    files += [(p.relative_to(bundle).as_posix(), p) for p in sorted(bundle.rglob("*"))
              if p.is_file() and not p.is_symlink() and p != bundle/"manifest.json"
              and not any(part == "__pycache__" or part.startswith("checkpoints-") for part in p.parts)]
    archive = write_bundle(root/"results.zip", files, kind="volume-results",
                           metadata=dict(remote_root=str(bundle), state=record["state"], job_id=record["job_id"]))
    save_json(root/"results.json", {key: archive[key] for key in ("size", "sha256")})
    return code


def main():
    action, directory = sys.argv[1:3]
    root = Path(directory).resolve(strict=True)
    if action == "run":
        return run(root)
    result = {"submit": submit, "status": status, "cancel": cancel}[action](root)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
