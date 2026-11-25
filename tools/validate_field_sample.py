"""Reproducible software-level checks on a field sample against installed SU.

The constant velocity is an operator comparison, not an interpreted field model.
This report must not be described as a full-survey or geological validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from seisdesk.backends.su import SUBackend, run_pipeline
from seisdesk.core.segy_io import read_any, read_su, write_su
from seisdesk.core.streaming import iter_file_gathers
from seisdesk.processing import Context, create


def compare(a, b):
    if a.shape != b.shape:
        raise ValueError(f"comparison shapes differ: {a.shape}, {b.shape}")
    x, y = a.astype(float).ravel(), b.astype(float).ravel()
    return dict(correlation=float(np.corrcoef(x, y)[0, 1]),
                relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(y), 1e-30)))


def main(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = read_any(args.input)
    report = dict(input=str(Path(args.input).resolve()), sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                  ntr=data.ntr, nt=data.nt, dt=data.dt, scope="field-sample software/operator comparison",
                  full_survey_validated=False, geological_model_validated=False)
    report["headers"] = {key: dict(min=float(data.header(key).min()), max=float(data.header(key).max()),
                                    unique=int(len(np.unique(data.header(key)))))
                         for key in ("fldr", "tracf", "cdp", "offset", "sx", "sy", "gx", "gy")}
    count = 0
    for index, gather in iter_file_gathers(args.input):
        np.testing.assert_array_equal(gather.traces, data.traces[index])
        count += len(index)
    assert count == data.ntr
    report["streamed_read_exact"] = True
    source = root/"input.su"; write_su(data, str(source))
    engine = SUBackend("local", args.su_bin)
    run_pipeline("sugain scale=2", source, root/"scaled.su", engine)
    scaled = read_su(str(root/"scaled.su"))
    np.testing.assert_allclose(scaled.traces, 2*data.traces, rtol=1e-6, atol=1e-10)
    report["real_su_scaling_exact"] = True
    context = Context(work_dir=str(root))
    if not np.any(data.header("cdp")) or not np.any(data.header("offset")):
        report["nmo_stack"] = "not run: required field geometry headers are absent"
    else:
        # Give both implementations identical filtered input and velocity.
        band = create("bandpass", corners=[4, 8, 55, 70])(data.sort(["cdp", "offset"]), context)
        write_su(band, str(root/"band.su"))
        native_nmo = create("nmo", velocity="2000", stretch_limit=0)(band, context)
        native_stack = create("stack", normalisation="live")(native_nmo, context)
        run_pipeline("sunmo vnmo=2000 smute=1000000 lmute=0 sscale=0", root/"band.su", root/"su-nmo.su", engine)
        run_pipeline("sustack", root/"su-nmo.su", root/"su-stack.su", engine)
        reference_nmo, reference_stack = read_su(str(root/"su-nmo.su")), read_su(str(root/"su-stack.su"))
        nmo, stack = compare(native_nmo.traces, reference_nmo.traces), compare(native_stack.traces, reference_stack.traces)
        report["nmo_stack"] = dict(velocity_m_s=2000, stretch_muting="disabled", nmo=nmo, stack=stack,
                                   note="SU sinc and native linear interpolation need not be sample-identical",
                                   correlation_acceptance=.95)
        assert nmo["correlation"] >= .95 and stack["correlation"] >= .95
        write_su(native_stack, str(root/"native-stack.su"))
    report["elapsed_seconds"] = time.monotonic()-started
    report["status"] = "passed"
    (root/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input"); parser.add_argument("output"); parser.add_argument("--su-bin", required=True)
    main(parser.parse_args())
