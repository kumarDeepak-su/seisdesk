"""Make bundled CUDA DLL directories visible before importing CuPy on Windows."""
import os
from pathlib import Path
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    root = Path(sys._MEIPASS)
    directories = {p.parent for p in (root/"nvidia").rglob("*.dll")}
    directories.update(p.parent for p in (root/"cupy/.data").rglob("*.dll"))
    # Keep handles alive for the whole process, including frozen worker children.
    sys._seisdesk_cuda_dll_handles = [os.add_dll_directory(str(p)) for p in sorted(directories)]
