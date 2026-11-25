"""File-streamed Seismic Unix pipelines on a local host or Windows WSL.

Commands are argv lists and pipes are connected by Python, not evaluated by a
shell. Existing shell scripts can be explicit stages: ``bash my-flow.sh``.
Scripts receive SU on stdin and must write SU to stdout for GUI use. Parameters
containing filenames use paths in the execution host's filesystem.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass


def parse_pipeline(text: str) -> list[list[str]]:
    """Parse literal arguments and pipes; never expand variables or run a shell."""
    if text.lstrip().startswith("["):
        stages = json.loads(text)
    else:
        lexer = shlex.shlex(text, posix=True, punctuation_chars="|&;<>()")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        stages = [[]]
        for token in lexer:
            if token == "|":
                stages.append([])
            elif token and all(c in "|&;<>()" for c in token):
                raise ValueError("only pipes are supported here; use an explicit bash script for shell syntax")
            else:
                stages[-1].append(token)
    if not isinstance(stages, list) or not stages or any(
        not isinstance(stage, list) or not stage or any(not isinstance(arg, str) or "\0" in arg for arg in stage)
        or not stage[0] for stage in stages
    ):
        raise ValueError("give a pipeline or a JSON list of nonempty command argument lists")
    return stages


@dataclass
class SUBackend:
    kind: str = "local"
    bin_dir: str = ""
    distribution: str = ""

    def __post_init__(self):
        if self.kind not in ("local", "wsl"):
            raise ValueError("SU backend must be local or wsl")
        if self.kind == "wsl" and os.name != "nt":
            raise ValueError("the WSL backend is available on Windows; use local on Linux")
        if self.kind == "local" and not self.bin_dir:
            root = os.environ.get("CWPROOT")
            found = shutil.which("sugain")
            self.bin_dir = str(Path(root) / "bin") if root else (str(Path(found).parent) if found else "")
        if self.kind == "wsl" and not self.bin_dir.startswith("/"):
            raise ValueError("give the Linux SU bin directory, for example /opt/cwp/bin")

    def environment(self):
        env = os.environ.copy()
        if self.bin_dir and self.kind == "local":
            env["CWPROOT"] = str(Path(self.bin_dir).resolve().parent)
            env["PATH"] = str(Path(self.bin_dir).resolve()) + os.pathsep + env.get("PATH", "")
        return env

    def wrap(self, argv: list[str]) -> list[str]:
        if self.kind == "local":
            return argv
        prefix = ["wsl.exe"]
        if self.distribution:
            prefix += ["--distribution", self.distribution]
        return prefix + ["--exec", "env", "CWPROOT=" + self.bin_dir.rstrip("/").rsplit("/", 1)[0],
                         "PATH=" + self.bin_dir + ":/usr/local/bin:/usr/bin:/bin", *argv]

    def command(self, stage: list[str]) -> list[str]:
        program, *args = stage
        if self.kind == "local":
            found = shutil.which(program, path=self.environment().get("PATH"))
            if found is None:
                raise FileNotFoundError(f"program {program!r} not found; configure the SU bin directory")
            return [found, *args]
        # SU and utility executables resolve in the explicitly configured Linux PATH.
        return self.wrap([program, *args])

    def programs(self) -> list[str]:
        if self.kind == "local":
            if not self.bin_dir or not Path(self.bin_dir).is_dir():
                raise FileNotFoundError("SU bin directory not found; install SU and configure its bin directory")
            return sorted(p.name for p in Path(self.bin_dir).iterdir() if p.is_file() and os.access(p, os.X_OK))
        done = subprocess.run(self.wrap(["find", self.bin_dir, "-maxdepth", "1", "-type", "f", "-executable",
                                         "-printf", "%f\n"]), capture_output=True, timeout=20)
        if done.returncode:
            raise RuntimeError(done.stderr.decode(errors="replace"))
        return sorted(done.stdout.decode(errors="replace").splitlines())

    def documentation(self, program: str) -> str:
        if not re.fullmatch(r"[\w.+-]+", program):
            raise ValueError("select one installed program")
        # SU self documentation is normally printed on stderr when stdin is empty.
        with tempfile.TemporaryDirectory(prefix="seisdesk-sudoc-") as directory:
            output = Path(directory) / "doc"
            with output.open("wb") as stream:
                proc = subprocess.Popen(self.command([program]), stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT, env=self.environment(),
                                        **_process_options())
                deadline = time.monotonic() + 5
                while proc.poll() is None:
                    if time.monotonic() > deadline or output.stat().st_size > 1_000_000:
                        _stop([proc])
                        break
                    time.sleep(.05)
            return output.read_bytes()[:1_000_000].decode(errors="replace")


def _process_options():
    return {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}


def _stop(processes):
    for proc in processes:
        if proc.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
    for proc in processes:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def run_pipeline(stages, input_path, output_path, backend=None, *, timeout=3600,
                 cancelled=lambda: False, log=print, progress=None, work_dir=None):
    """Stream a file through installed programs, atomically publish output on success.

    No seismic array is loaded into the controlling process. Each stage's exit
    status is checked (including upstream failures). Logs and a JSON job record
    are kept beside the output even when a job fails or is cancelled.
    """
    stages = parse_pipeline(stages) if isinstance(stages, str) else parse_pipeline(json.dumps(stages))
    backend = backend or SUBackend()
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("input and output must be different files")
    if not input_path.is_file():
        raise FileNotFoundError(str(input_path))
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    commands = [backend.command(stage) for stage in stages]
    job_dir = Path(tempfile.mkdtemp(prefix=output_path.name + ".job-", dir=output_path.parent))
    partial = job_dir / "output.part"
    report = {"commands": commands, "backend": backend.kind, "input": str(input_path),
              "output": str(output_path), "state": "running", "logs": [], "returncodes": []}
    processes, errors = [], []
    started = time.monotonic()
    try:
        with input_path.open("rb") as source, partial.open("wb") as target:
            previous = source
            for i, command in enumerate(commands):
                if cancelled():
                    raise InterruptedError("SU job cancelled")
                err_path = job_dir / f"stage-{i+1}.log"
                err = err_path.open("wb")
                errors.append(err)
                report["logs"].append(str(err_path))
                log(f"SU [{i+1}/{len(commands)}]: {shlex.join(stages[i])}")
                proc = subprocess.Popen(command, stdin=previous,
                                        stdout=target if i == len(commands)-1 else subprocess.PIPE,
                                        stderr=err, env=backend.environment(), cwd=work_dir,
                                        **_process_options())
                processes.append(proc)
                if previous is not source:
                    previous.close()
                previous = proc.stdout
            while True:
                codes = [proc.poll() for proc in processes]
                if cancelled():
                    raise InterruptedError("SU job cancelled")
                if any(code is not None and code != 0 for code in codes):
                    raise RuntimeError("an SU pipeline stage failed")
                if all(code is not None for code in codes):
                    break
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"SU pipeline exceeded {timeout:g} seconds")
                if progress:
                    progress(sum(code is not None for code in codes) / len(codes))
                time.sleep(.05)
        os.replace(partial, output_path)
        report["state"] = "completed"
        return report
    except BaseException as exc:
        _stop(processes)
        report["state"] = "cancelled" if isinstance(exc, InterruptedError) else "failed"
        report["error"] = str(exc)
        if isinstance(exc, RuntimeError):
            for stream in errors:
                stream.flush()
            detail = "\n".join(Path(p).read_bytes()[-8000:].decode(errors="replace") for p in report["logs"])
            raise RuntimeError(f"{exc}\n{detail}\nJob logs: {job_dir}") from exc
        raise
    finally:
        for stream in errors:
            stream.close()
        for proc in processes:
            if proc.stdout is not None:
                proc.stdout.close()
        if partial.exists():
            partial.unlink()
        report["returncodes"] = [proc.returncode for proc in processes]
        report["elapsed_seconds"] = time.monotonic() - started
        report["record"] = str(job_dir / "job.json")
        (job_dir / "job.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
