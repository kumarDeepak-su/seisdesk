"""Integration with an actual native-endian CWP/SU installation."""
import os

import numpy as np
import pytest

from seisdesk.backends.su import SUBackend, run_pipeline
from seisdesk.core.segy_io import read_su, write_su

pytestmark = pytest.mark.skipif(not os.environ.get("SEISDESK_SU_BIN"), reason="requires installed CWP/SU programs")


def test_real_su_pipeline_scaling_selection_and_documentation(tmp_path, marine):
    engine = SUBackend("local", os.environ["SEISDESK_SU_BIN"])
    source, output = tmp_path/"input.su", tmp_path/"output.su"
    write_su(marine, str(source))
    shot = float(marine.header("fldr")[0])
    result = run_pipeline([["suwind", "key=fldr", f"min={shot:g}", f"max={shot:g}"],
                           ["sugain", "scale=2"]], source, output, engine)
    actual = read_su(str(output))
    expected = marine.subset(np.flatnonzero(marine.header("fldr") == shot))
    np.testing.assert_allclose(actual.traces, 2*expected.traces, rtol=1e-6)
    np.testing.assert_array_equal(actual.header("fldr"), expected.header("fldr"))
    assert "sugain" in engine.programs()
    assert "SUGAIN" in engine.documentation("sugain").upper()
    assert result["returncodes"] == [0, 0]


def test_real_su_upstream_error_is_reported(tmp_path, marine):
    engine = SUBackend("local", os.environ["SEISDESK_SU_BIN"])
    source, output = tmp_path/"input.su", tmp_path/"output.su"
    write_su(marine, str(source))
    output.write_bytes(b"previous output")
    with pytest.raises(RuntimeError):
        run_pipeline("sufilter f=50,10,80,100 | sugain scale=1", source, output, engine)
    assert output.read_bytes() == b"previous output"
