"""``python -m seisdesk`` (and the installed SeisDesk program) opens the GUI; ``--self-test`` checks the build."""

from __future__ import annotations

import sys


def main() -> int:
    import multiprocessing
    multiprocessing.freeze_support()
    if "--check-gpu-runtime" in sys.argv:
        from seisdesk.backends.gpu_runtime_check import main as check_gpu
        return check_gpu()
    if "--self-test" in sys.argv:
        import os
        import tempfile
        import traceback
        out = os.path.join(tempfile.gettempdir(), "seisdesk_selftest.txt")
        try:
            os.remove(out)
        except OSError:
            pass
        try:
            from seisdesk.gui.app import main as gui_main
            return gui_main(sys.argv)
        except BaseException:                       # noqa: BLE001 - a windowed exe must not hang on an error dialog
            with open(out, "w", encoding="utf-8") as f:
                f.write("SeisDesk self-test FAILED\n" + traceback.format_exc())
            return 1
    from seisdesk.gui.app import main as gui_main
    return gui_main(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
