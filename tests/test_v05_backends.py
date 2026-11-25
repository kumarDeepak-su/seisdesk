import json
from pathlib import Path
import sys

import numpy as np
import pytest

from seisdesk.backends.su import SUBackend, parse_pipeline, run_pipeline
from seisdesk.workflow.project import Project
from seisdesk.workflow.results import load_results, save_results


def test_literal_pipeline_and_shell_rejection():
    assert parse_pipeline('sugain tpow=2 | sufilter "f=5,10,60,80"') == [
        ["sugain", "tpow=2"], ["sufilter", "f=5,10,60,80"]]
    assert parse_pipeline('[["tool", "a|b", "path with spaces"]]') == [["tool", "a|b", "path with spaces"]]
    for value in ("sugain > output.su", "sugain && echo bad", "sugain |", "[]", '["sugain"]'):
        with pytest.raises(ValueError):
            parse_pipeline(value)


def test_streamed_pipeline_space_arguments_and_stage_failure(tmp_path):
    source = tmp_path / "input with spaces.su"
    target = tmp_path / "output.su"
    source.write_bytes(bytes(range(256))*8192)
    code = "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); print(sys.argv[1], file=sys.stderr)"
    stage = [sys.executable, "-c", code, "literal $HOME; `echo danger`"]
    report = run_pipeline([stage, stage], source, target, timeout=30, log=lambda s: None)
    assert target.read_bytes() == source.read_bytes()
    assert report["returncodes"] == [0, 0] and report["state"] == "completed"
    assert "literal $HOME; `echo danger`" in Path(report["logs"][0]).read_text()
    record = json.loads(Path(report["record"]).read_text())
    assert record["state"] == "completed"
    with pytest.raises(RuntimeError, match="deliberate failure"):
        run_pipeline([[sys.executable, "-c", "import sys; print('deliberate failure',file=sys.stderr);sys.exit(7)"],
                      stage], source, target, log=lambda s: None)
    assert target.read_bytes() == source.read_bytes()  # Failed jobs never replace prior output.


def test_pipeline_timeout_and_cancellation(tmp_path):
    source = tmp_path / "in.su"
    source.write_bytes(b"input")
    stage = [[sys.executable, "-c", "import time; time.sleep(30)"]]
    with pytest.raises(TimeoutError):
        run_pipeline(stage, source, tmp_path/"out.su", timeout=.15, log=lambda s: None)
    calls = []
    def cancelled():
        calls.append(1)
        return len(calls) >= 3
    with pytest.raises(InterruptedError):
        run_pipeline(stage, source, tmp_path/"out.su", cancelled=cancelled, log=lambda s: None)
    assert not (tmp_path/"out.su").exists()
    assert sorted(json.loads(p.read_text())["state"] for p in tmp_path.glob("*.job-*/job.json")) == ["cancelled", "failed"]


def test_backend_program_resolution():
    backend = SUBackend("local", str(Path(sys.executable).parent))
    assert Path(backend.command([sys.executable, "--version"])[0]).is_file()
    with pytest.raises(FileNotFoundError):
        backend.command(["this-seismic-program-does-not-exist"])


def test_diagnostic_arrays_persist_without_type_loss(tmp_path):
    results = {"tomo": {"velocity": np.arange(24, dtype=np.float32).reshape(4, 6),
                        "rms": [.3, .1], "coverage": np.array([True, False]),
                        "mapping": {1.0: (np.float64(3), np.int32(5))}},
               "wave": np.array([1+2j, 3-4j]), "nan": float("nan")}
    project = Project()
    project.context.results = results
    project.pick_set("a/b").new_phase("direct")
    project.pick_set("a b").new_phase("reflection")
    project.save(str(tmp_path))
    back = Project.load(str(tmp_path))
    assert back.context.results["tomo"]["velocity"].dtype == np.float32
    np.testing.assert_array_equal(back.context.results["wave"], results["wave"])
    assert back.context.results["tomo"]["mapping"][1.0] == (3., 5)
    assert np.isnan(back.context.results["nan"])
    assert back.context.picks["a/b"].phase_order == ["direct"]
    assert back.context.picks["a b"].phase_order == ["reflection"]
    with pytest.raises(ValueError, match="object arrays"):
        save_results({"bad": np.array([object()])}, tmp_path/"results.npz")
    assert "tomo" in load_results(tmp_path/"results.npz")
