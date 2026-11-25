"""A real SU pipeline as a reusable, parameterised processing step."""
from pathlib import Path
import tempfile

from ..backends.su import SUBackend, run_pipeline
from ..core.segy_io import read_su, write_su
from .base import Param, Process, register


@register
class SUPipeline(Process):
    name, label, category = "su_pipeline", "Seismic Unix pipeline (installed SU)", "Input/Output"
    description = ("Run installed SU programs on a local host or through Windows WSL. Literal arguments and pipes "
                   "are supported, e.g. 'sugain tpow=2 | sufilter f=5,10,60,80'. For an existing script use "
                   "'bash /path/flow.sh'; it must read SU from stdin and write SU to stdout. Output gets new trace "
                   "identities because external programs may stack, duplicate or change traces. For files larger "
                   "than memory use the SU file-job window or seisdesk-cli su-run.")
    params = (Param("pipeline", "text", "sugain tpow=2", "Pipeline"),
              Param("backend", "choice", "local", "Execution host", choices=("local", "wsl")),
              Param("bin_dir", "str", "", "SU bin directory", help="Linux path for WSL, e.g. /opt/cwp/bin"),
              Param("distribution", "str", "", "WSL distribution (empty = default)"),
              Param("byteorder", "choice", "little", "SU executable file convention", choices=("little", "big"),
                    help="Choose big for a SUXDR build; native x86 SU normally uses little."),
              Param("timeout", "float", 3600, "Maximum runtime", unit="s", minimum=1))

    def run(self, data, context):
        if data.domain != "time":
            raise ValueError("the SU bridge requires time-domain data")
        backend = SUBackend(self["backend"], self["bin_dir"], self["distribution"])
        Path(context.work_dir).mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="su-", dir=context.work_dir))
        source, target = directory / "input.su", directory / "output.su"
        byteorder = "<" if self["byteorder"] == "little" else ">"
        write_su(data, str(source), byteorder=byteorder)
        report = run_pipeline(self["pipeline"], source, target, backend, timeout=self["timeout"],
                              cancelled=context.cancelled, log=context.log, progress=context.report,
                              work_dir=context.work_dir)
        out = read_su(str(target), name=data.name + " SU", byteorder=byteorder)
        out.history = list(data.history)
        out.meta["su_job"] = report
        context.results[self.name] = report
        return out
