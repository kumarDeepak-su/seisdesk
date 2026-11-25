"""Portable-job integrity, scheduler semantics and real subprocess compute checks."""
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import zipfile

import numpy as np
import pytest

from seisdesk.backends import remote_agent
from seisdesk.backends.job_bundle import (atomic_json, build_volume_bundle, extract_bundle,
                                          rebase_paths, sha256, write_bundle)
from seisdesk.backends.remote import RemoteClient, RemoteProfile, load_record, prepare_job
from seisdesk.core.dataset import SeismicData
from seisdesk.core.segy_io import write_su
from seisdesk.core.volume import VelocityVolume
from seisdesk.workflow.project import load_dataset


def inputs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    model = directory/"model with spaces.npz"
    VelocityVolume(np.full((9, 8, 10), 1900., dtype=np.float32), (10, 10, 10)).save(model)
    survey = directory/"input with spaces.su"
    data = SeismicData(np.zeros((3, 50), dtype=np.float32), .001,
                       dict(sx=20, sy=30, selev=-20, gx=[10, 50, 70], gy=[20, 35, 50], gelev=-20, fldr=1,
                            tracf=[1, 2, 3], offset=[-10, 30, 50]))
    write_su(data, str(survey))
    return dict(input_file=survey.name, model=model.name, mode="model", device="cpu", workers=1,
                memory_gb=.2, frequency=45, source_delay=.015, source_amplitude=1e6, padding=3, block=11,
                vmax=2400, vmin=1000, allow_coarse_grid=True)


def test_bundle_runs_after_original_inputs_are_removed(tmp_path, monkeypatch):
    params = inputs(tmp_path/"original")
    archive = build_volume_bundle(params, tmp_path/"inputs.zip", work_dir=tmp_path/"original")
    root = tmp_path/"remote"
    extract_bundle(archive["path"], root/"bundle", expected_sha256=archive["sha256"], kind="volume-input")
    for path in (tmp_path/"original").iterdir():
        path.unlink()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute allocation"):
        remote_agent.run(root)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert remote_agent.run(root) == 0
    assert json.loads((root/"run.json").read_text())["state"] == "completed"
    receipt = json.loads((root/"results.json").read_text())
    result = tmp_path/"download"
    manifest = extract_bundle(root/"results.zip", result, expected_sha256=receipt["sha256"], kind="volume-results")
    preview = load_dataset(result/"output/preview.npz")
    assert preview.ntr == 3 and np.isfinite(preview.traces).all() and np.linalg.norm(preview.traces) > 0
    portable = json.loads((result/"parameters.json").read_text())
    assert (result/portable["input_file"]).is_file() and (result/portable["model"]).is_file()
    assert (result/"code/seisdesk/cli.py").is_file() and (result/"input-manifest.json").is_file()
    saved = json.loads((result/"output/job.json").read_text())
    local = rebase_paths(saved, manifest["metadata"]["remote_root"], result)
    assert Path(local["final_model"]).is_file()
    for shot in local["shot_results"]:
        assert Path(shot["files"]["predicted"]).is_file()


def test_bundle_rejects_modified_data_and_unsafe_archive_members(tmp_path):
    item = tmp_path/"data.txt"; item.write_text("known bytes")
    bundle = write_bundle(tmp_path/"bundle.zip", [("nested/data.txt", item)], kind="test")
    with pytest.raises(ValueError, match="SHA-256"):
        extract_bundle(bundle["path"], tmp_path/"wrong", expected_sha256="0"*64)
    with zipfile.ZipFile(bundle["path"]) as reader:
        manifest = reader.read("manifest.json")
    with zipfile.ZipFile(tmp_path/"changed.zip", "w") as writer:
        writer.writestr("nested/data.txt", b"other bytes")
        writer.writestr("manifest.json", manifest)
    with pytest.raises(ValueError, match="hash mismatch"):
        extract_bundle(tmp_path/"changed.zip", tmp_path/"changed")
    assert not (tmp_path/"changed").exists()
    for index, name in enumerate(("../escape", "/absolute", "C:/drive", "a\\b", "a/../b", "AUX.txt")):
        path = tmp_path/f"unsafe-{index}.zip"
        with zipfile.ZipFile(path, "w") as writer:
            info = zipfile.ZipInfo("placeholder")
            info.filename = name  # Preserve raw backslashes even on a Windows writer.
            writer.writestr(info, b"unsafe")
        with pytest.raises(ValueError, match="unsafe"):
            extract_bundle(path, tmp_path/f"out-{index}")
    extract_bundle(bundle["path"], tmp_path/"good")
    with pytest.raises(FileExistsError):
        extract_bundle(bundle["path"], tmp_path/"good")


def test_profile_quotes_shell_characters_and_rejects_options(tmp_path):
    profile = RemoteProfile(remote_root="/scratch/path with 'quotes' and $(echo BAD)",
                            python="/venv with spaces/bin/python", modules=["Python/3.10.4"])
    client = RemoteClient(profile)
    command = client.ssh_command(client.python_command(["-c", "print('literal')"]))
    assert command[:2] == ["ssh", "-T"] and command[-2] == "athena"
    shell = shlex.split(command[-1])
    assert shell[:2] == ["bash", "-lc"]
    assert "'/venv with spaces/bin/python'" in shell[2]
    for changes in ({"host": "-oProxyCommand=bad"}, {"remote_root": "relative"}, {"account": "ok\n--nodes=9"},
                    {"gpus": 2}, {"cpus": 0}, {"walltime": "10:99:00"}, {"modules": ["-bad"]}):
        with pytest.raises(ValueError):
            RemoteProfile(**(asdict(profile) | changes)).validate()


def test_submission_is_recoverable_and_never_automatically_duplicated(tmp_path, monkeypatch):
    params = inputs(tmp_path/"inputs")
    profile = RemoteProfile(remote_root="/scratch/jobs")
    record = prepare_job(params, profile, tmp_path/"records", work_dir=tmp_path/"inputs")
    client = RemoteClient(profile)
    commands = []
    def disconnected(command, **kwargs):
        commands.append(command)
        if "remote_agent.py" in command:
            raise TimeoutError("connection dropped after sbatch")
        return b"{}"
    monkeypatch.setattr(client, "_run", disconnected)
    with pytest.raises(TimeoutError):
        client.submit(record)
    saved = load_record(record["record_file"])
    assert saved["state"] == "SUBMISSION_UNCONFIRMED" and saved["token"] == record["token"]
    count = len(commands)
    with pytest.raises(ValueError, match="newly prepared"):
        client.submit(saved)
    assert len(commands) == count
    saved["reason"] = "Resources"
    monkeypatch.setattr(client, "_run", lambda *a, **k: b'{"job_id":"12345","state":"RUNNING"}')
    client.status(saved)
    assert load_record(saved["record_file"])["job_id"] == "12345"
    assert "reason" not in saved


def test_scheduler_accounting_handles_queue_disappearance_and_nonzero_exit(tmp_path, monkeypatch):
    atomic_json(tmp_path/"submission.json", dict(job_id="123", state="SUBMITTED"))
    calls = []
    def execute(args, **kwargs):
        calls.append(args)
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 1, "", "Invalid job id specified")
        if args[0] == "sacct":
            return subprocess.CompletedProcess(args, 0, "123|FAILED|2:0|00:00:02|2|node001\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(remote_agent.subprocess, "run", execute)
    status = remote_agent.status(tmp_path)
    assert status["state"] == "FAILED" and status["exit_code"] == "2:0"
    remote_agent.cancel(tmp_path)
    assert calls[-1] == ["scancel", "123"]
    atomic_json(tmp_path/"submission.json", dict(job_id="123; touch bad"))
    with pytest.raises(ValueError, match="confirmed Slurm"):
        remote_agent.cancel(tmp_path)


def test_scheduler_receipt_prevents_second_submission(tmp_path, monkeypatch):
    profile = RemoteProfile(remote_root="/scratch/jobs")
    atomic_json(tmp_path/"profile.json", asdict(profile))
    calls = []
    def execute(args):
        calls.append(args)
        return "90210;cluster"
    monkeypatch.setattr(remote_agent, "command", execute)
    first = remote_agent.submit(tmp_path)
    assert remote_agent.submit(tmp_path) == first
    assert first["job_id"] == "90210" and len(calls) == 1
    assert "--nodes=1" in calls[0] and "--cpus-per-task=2" in calls[0]


def test_failure_still_returns_logs_and_diagnostics(tmp_path, monkeypatch):
    params = inputs(tmp_path/"original")
    params["vmin"], params["vmax"] = 4000, 1000  # Rejected on the compute host.
    bundle = build_volume_bundle(params, tmp_path/"bad.zip", work_dir=tmp_path/"original")
    root = tmp_path/"remote"
    extract_bundle(bundle["path"], root/"bundle")
    monkeypatch.setenv("SLURM_JOB_ID", "54321")
    assert remote_agent.run(root) != 0
    assert json.loads((root/"run.json").read_text())["state"] == "failed"
    assert (root/"results.zip").is_file() and (root/"run.log").stat().st_size > 0
