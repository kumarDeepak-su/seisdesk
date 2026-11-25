"""Desktop SSH/Slurm client with portable inputs and durable, recoverable job records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile
import time
import uuid

from .job_bundle import atomic_json, build_volume_bundle, extract_bundle, rebase_paths


@dataclass
class RemoteProfile:
    host: str = "athena"
    remote_root: str = ""
    python: str = "python3"
    modules: list = field(default_factory=list)
    partition: str = ""
    account: str = ""
    cpus: int = 2
    memory_gb: int = 8
    gpus: int = 0
    walltime: str = "00:30:00"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", self.host):
            raise ValueError("use an SSH host alias or user@hostname, without command options")
        if (not self.remote_root.startswith("/") or self.remote_root == "/"
                or ".." in PurePosixPath(self.remote_root).parts):
            raise ValueError("remote job root must be an absolute directory below /")
        for value in [self.remote_root, self.python, self.partition, self.account, *self.modules]:
            if not isinstance(value, str) or any(ord(c) < 32 for c in value):
                raise ValueError("remote settings must be single-line text")
        if not self.python or self.python.startswith("-"):
            raise ValueError("choose a Python executable, not a command with options")
        if not isinstance(self.modules, list) or any(not m or m.startswith("-") for m in self.modules):
            raise ValueError("module names must be a list without command options")
        for value in (self.partition, self.account):
            if value and not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
                raise ValueError("invalid Slurm partition or account")
        if type(self.cpus) is not int or not 1 <= self.cpus <= 256:
            raise ValueError("CPU count must be between 1 and 256")
        if type(self.memory_gb) is not int or not 1 <= self.memory_gb <= 4096:
            raise ValueError("memory must be between 1 and 4096 GiB")
        if type(self.gpus) is not int or self.gpus not in (0, 1):
            raise ValueError("this remote launcher currently supports zero or one GPU per job")
        if not re.fullmatch(r"(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]", self.walltime):
            raise ValueError("wall time must be [days-]hours:minutes:seconds")
        return self


def save_record(record):
    atomic_json(record["record_file"], record)


def load_record(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    record["record_file"] = str(Path(path).resolve())
    return record


def prepare_job(parameters, profile, directory, *, work_dir=".", cancelled=lambda: False, report=lambda s: None):
    profile.validate()
    workers = int(parameters.get("workers", 1))
    if workers > profile.cpus:
        raise ValueError("shot workers exceed the requested CPUs")
    if float(parameters.get("memory_gb", 4)) * workers > profile.memory_gb:
        raise ValueError("worker memory budgets exceed the Slurm memory request")
    if parameters.get("device", "cpu") == "cuda" and (profile.gpus != 1 or workers != 1):
        raise ValueError("CUDA remote jobs need one GPU and one local shot worker")
    token = uuid.uuid4().hex
    directory = Path(directory).resolve()/token
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(format=1, token=token, profile=asdict(profile), state="PREPARING", created=time.time(),
                  remote_directory=str(PurePosixPath(profile.remote_root)/token),
                  record_file=str(directory/"remote-job.json"))
    save_record(record)
    try:
        bundle = build_volume_bundle(parameters, directory/"inputs.zip", work_dir=work_dir,
                                     cancelled=cancelled, report=report)
        record.update(state="PREPARED", bundle={key: bundle[key] for key in ("path", "size", "sha256")})
    except BaseException as exc:
        record.update(state="PREPARATION_FAILED", error=str(exc))
        raise
    finally:
        save_record(record)
    return record


# Only our own hash-verified source is executed. Its extraction implementation
# checks every name, inventory entry and content hash before publishing bundle/.
BOOTSTRAP = """import hashlib,json,pathlib,sys,zipfile
p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
h=hashlib.sha256()
with (p/'inputs.zip').open('rb') as f:
 for chunk in iter(lambda:f.read(4194304),b''): h.update(chunk)
if h.hexdigest()!=expected: raise ValueError('uploaded input hash mismatch')
with zipfile.ZipFile(p/'inputs.zip') as z:
 source=z.read('code/seisdesk/backends/job_bundle.py')
namespace={'__name__':'bundle_bootstrap'}
exec(compile(source,'job_bundle.py','exec'),namespace)
namespace['extract_bundle'](p/'inputs.zip',p/'bundle',kind='volume-input')
namespace['atomic_json'](p/'profile.json',json.loads(sys.argv[3]))
print(json.dumps({'state':'STAGED'}))
"""


class RemoteClient:
    def __init__(self, profile):
        self.profile = profile.validate()

    def ssh_command(self, command):
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "--", self.profile.host,
                "bash -lc "+shlex.quote(command)]

    def python_command(self, args):
        command = shlex.join([self.profile.python, *map(str, args)])
        if self.profile.modules:
            command = "module load "+shlex.join(self.profile.modules)+" >&2 && "+command
        return command

    def _run(self, command, *, input_file=None, output_file=None, timeout=180, cancelled=lambda: False):
        """File streams keep multi-GB transfers out of desktop memory; no local shell."""
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(self.ssh_command(command), stdin=input_file or subprocess.DEVNULL,
                                       stdout=output_file or output, stderr=errors,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if cancelled():
                        raise InterruptedError("SSH operation cancelled; refresh the saved job to recover its remote state")
                    if time.monotonic()-started > timeout:
                        raise TimeoutError("SSH timed out; refresh the saved job before resubmitting")
                    time.sleep(.1)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
            errors.seek(0)
            if process.returncode:
                raise RuntimeError(errors.read().decode("utf-8", errors="replace").strip()
                                   or f"SSH exited with code {process.returncode}")
            output.seek(0)
            return output.read()

    def _directory(self, record):
        if not re.fullmatch(r"[0-9a-f]{32}", record.get("token", "")):
            raise ValueError("invalid remote job token")
        expected = str(PurePosixPath(self.profile.remote_root)/record["token"])
        if record.get("remote_directory") != expected or record.get("profile") != asdict(self.profile):
            raise ValueError("the saved job belongs to a different remote profile")
        return expected

    def _action(self, record, action, **kwargs):
        root = self._directory(record)
        helper = str(PurePosixPath(root)/"bundle/code/seisdesk/backends/remote_agent.py")
        response = self._run(self.python_command([helper, action, root]), **kwargs)
        return json.loads(response)

    def check(self, cancelled=lambda: False):
        script = ("import json,sys,shutil,importlib.metadata as m; "
                  "print(json.dumps(dict(python=sys.version,executable=sys.executable,"
                  "sbatch=shutil.which('sbatch'),packages={n:m.version(n) for n in ['numpy','scipy','segyio']})))")
        result = json.loads(self._run(self.python_command(["-c", script]), cancelled=cancelled))
        if not result["sbatch"]:
            raise ValueError("sbatch is not available on the selected host")
        return result

    def submit(self, record, *, cancelled=lambda: False, report=lambda s: None):
        root = self._directory(record)
        if record["state"] != "PREPARED":
            raise ValueError("only a newly prepared job can be submitted; refresh this job to recover its state")
        record.update(state="TRANSFERRING")
        save_record(record)
        try:
            report("Creating a new remote job directory")
            self._run("mkdir -p -- "+shlex.quote(self.profile.remote_root)+" && mkdir -- "+shlex.quote(root),
                      cancelled=cancelled)
            report("Uploading inputs and compute source")
            with open(record["bundle"]["path"], "rb") as stream:
                self._run("cat > "+shlex.quote(root+"/inputs.zip.part")+" && mv -- "+
                          shlex.quote(root+"/inputs.zip.part")+" "+shlex.quote(root+"/inputs.zip"),
                          input_file=stream, timeout=24*3600, cancelled=cancelled)
            report("Verifying input hashes and staging the portable bundle")
            self._run(self.python_command(["-c", BOOTSTRAP, root, record["bundle"]["sha256"],
                                            json.dumps(asdict(self.profile))]), timeout=24*3600, cancelled=cancelled)
            # Persist before sbatch: after any connection loss the user can refresh,
            # and the remote receipt/lock prevents accidental duplicate submissions.
            record["state"] = "SUBMITTING"
            save_record(record)
            report("Submitting to Slurm")
            record.update(self._action(record, "submit", cancelled=cancelled))
            record.pop("error", None)
            return record
        except BaseException as exc:
            record.update(error=str(exc), state="SUBMISSION_UNCONFIRMED")
            raise
        finally:
            save_record(record)

    def status(self, record, *, cancelled=lambda: False):
        result = self._action(record, "status", cancelled=cancelled)
        record.pop("error", None)
        record.pop("reason", None)  # A former queue reason is not a completed job's failure cause.
        record.update(result, checked=time.time())
        save_record(record)
        return record

    def cancel(self, record, *, cancelled=lambda: False):
        result = self._action(record, "cancel", cancelled=cancelled)
        record.update(result)
        save_record(record)
        return record

    def fetch(self, record, destination, *, cancelled=lambda: False, report=lambda s: None):
        self.status(record, cancelled=cancelled)
        result = record.get("results")
        if not result:
            raise ValueError("no completed result archive is available; status and logs remain accessible")
        root = self._directory(record)
        destination = Path(destination).resolve()/record["token"]
        if destination.exists():
            raise FileExistsError(f"results already exist: {destination}; choose another download directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".download-", suffix=".zip", dir=destination.parent)
        try:
            report(f"Downloading {result['size']:,} bytes")
            with os.fdopen(fd, "wb") as stream:
                self._run("cat -- "+shlex.quote(root+"/results.zip"), output_file=stream,
                          timeout=24*3600, cancelled=cancelled)
            if Path(temporary).stat().st_size != result["size"]:
                raise ValueError("downloaded result size differs from the remote record")
            report("Verifying and unpacking results")
            manifest = extract_bundle(temporary, destination, expected_sha256=result["sha256"],
                                      kind="volume-results", cancelled=cancelled)
            remote_bundle = manifest["metadata"]["remote_root"]
            # Keep the original manifest and a rebasing record for provenance.
            for path in destination.rglob("*.json"):
                if path.name not in ("manifest.json", "input-manifest.json"):
                    value = json.loads(path.read_text(encoding="utf-8"))
                    atomic_json(path, rebase_paths(value, remote_bundle, destination))
            atomic_json(destination/"path-map.json", dict(remote_root=remote_bundle, local_root=str(destination)))
            record["local_results"] = str(destination)
            save_record(record)
            return record
        finally:
            Path(temporary).unlink(missing_ok=True)
