"""I/O-only client validation: scientific jobs execute exclusively in Slurm.

The fixture must already have been generated on a compute node. The cancellation
case submits the normal runner with Slurm's --hold flag, then tests real scancel.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seisdesk.backends.job_bundle import atomic_json
from seisdesk.backends.remote import RemoteClient, RemoteProfile, prepare_job


class HeldClient(RemoteClient):
    def _action(self, record, action, **kwargs):
        if action != "submit":
            return super()._action(record, action, **kwargs)
        root = self._directory(record)
        script = """import sys,json
from pathlib import Path
root=Path(sys.argv[1]); sys.path.insert(0,str(root/'bundle/code'))
from seisdesk.backends import remote_agent as agent
execute=agent.command
agent.command=lambda argv:execute(argv[:1]+['--hold']+argv[1:]) if argv[0]=='sbatch' else execute(argv)
print(json.dumps(agent.submit(root)))
"""
        return json.loads(self._run(self.python_command(["-c", script, root]), **kwargs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("fixture")
    parser.add_argument("destination")
    parser.add_argument("--cases", default="cpu,failure,cancel,cuda")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    root, fixture = Path(args.destination).resolve(), Path(args.fixture).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = RemoteProfile(**json.loads(Path(args.profile).read_text(encoding="utf-8")))
    params = json.loads((fixture/"parameters.json").read_text(encoding="utf-8"))
    report_file = root/"lifecycle-report.json"
    evidence = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else dict(cases={})
    evidence["connection"] = RemoteClient(base).check()
    atomic_json(root/"lifecycle-report.json", evidence)
    for case in args.cases.split(","):
        if case not in ("cpu", "failure", "cancel", "cuda"):
            raise ValueError("unknown validation case: "+case)
        profile = replace(base, gpus=1 if case == "cuda" else 0)
        settings = dict(params, device="cuda" if case == "cuda" else "cpu")
        if case == "failure":
            settings.update(vmin=4000, vmax=1000)
        client = (HeldClient if case == "cancel" else RemoteClient)(profile)
        print("Starting "+case, flush=True)
        record = prepare_job(settings, profile, root/"jobs", work_dir=fixture)
        client.submit(record, report=lambda s: print(s, flush=True))
        print(f"{case}: submitted {record['job_id']}", flush=True)
        if case == "cancel":
            client.status(record)
            if record["state"] != "PENDING" or "held" not in record.get("reason", "").lower():
                raise AssertionError("cancellation fixture was not held by Slurm")
            client.cancel(record)
        deadline = time.monotonic()+args.timeout
        while True:
            client.status(record)
            print(f"{case}: {record['state']} {record.get('reason', '')}", flush=True)
            state = record["state"].split()[0].rstrip("+")
            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
                break
            if time.monotonic() > deadline:
                # Do not strand a validation allocation in the queue.
                client.cancel(record)
                raise TimeoutError("validation allocation did not finish within the requested timeout")
            time.sleep(30)
        expected = "CANCELLED" if case == "cancel" else "FAILED" if case == "failure" else "COMPLETED"
        evidence.setdefault("attempts", []).append(dict(case=case, **record))
        evidence["cases"][case] = record
        atomic_json(root/"lifecycle-report.json", evidence)
        if state != expected:
            raise AssertionError(f"{case}: expected {expected}, got {record['state']}: {record.get('log', '')}")
        if case != "cancel":
            client.fetch(record, root/"results", report=lambda s: print(s, flush=True))
            result = Path(record["local_results"])
            if case != "failure":
                job = json.loads((result/"output/job.json").read_text(encoding="utf-8"))
                assert job["params"]["device"] == settings["device"]
                for shot in job["shot_results"]:
                    assert Path(shot["files"]["predicted"]).is_file()
                portable = json.loads((result/"parameters.json").read_text(encoding="utf-8"))
                assert (result/portable["model"]).is_file() and (result/portable["input_file"]).is_file()
                assert (result/"output/preview.npz").is_file()
        evidence["cases"][case] = record
        atomic_json(root/"lifecycle-report.json", evidence)
    print("All requested lifecycle cases passed", flush=True)


if __name__ == "__main__":
    main()
