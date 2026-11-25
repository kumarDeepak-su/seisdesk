# PyInstaller spec for the SeisDesk Windows program (one folder: dist/SeisDesk/SeisDesk.exe).
# Build from the repository root:  pyinstaller --noconfirm packaging/seisdesk.spec
import os
import importlib.metadata
import json
from pathlib import Path
import zipfile

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = os.path.abspath(os.path.join(SPECPATH, ".."))
icon = os.path.join(SPECPATH, "seisdesk.ico")

datas = [(os.path.join(root, "seisdesk", "docs", "USER_GUIDE.md"), os.path.join("seisdesk", "docs"))]
# Frozen Python bytecode cannot be executed by a Linux compute host. Ship the
# exact build's Python source for portable remote jobs, without data or secrets.
snapshot = Path(root)/"build"/"compute_source.zip"
snapshot.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((Path(root)/"seisdesk").rglob("*.py")):
        archive.write(path, path.relative_to(root).as_posix())
datas.append((str(snapshot), "seisdesk"))
if os.path.isdir(os.path.join(root, "seisdesk", "docs", "images")):
    datas.append((os.path.join(root, "seisdesk", "docs", "images", "*.png"), os.path.join("seisdesk", "docs", "images")))
datas += collect_data_files("pyqtgraph", excludes=["**/examples/*", "**/tests/*"])
hidden = collect_submodules("seisdesk") + ["segyio._segyio"]
# Numba needs real source files for its cached dispersion kernels in the installed app.
datas += collect_data_files("disba", include_py_files=True)
for package in ("disba", "numba", "llvmlite"):
    datas += copy_metadata(package)
hidden += collect_submodules("disba")
gpu_build = os.environ.get("SEISDESK_BUILD_CUDA") == "1"
runtime_hooks = []
build_info = dict(cuda=gpu_build, packages={})
if gpu_build:
    # CuPy's compiled scalar module imports graphlib at runtime; modulegraph
    # cannot see that dependency inside the extension binary.
    hidden.append("graphlib")
    for package in ("cupy", "cupy_backends", "cupyx", "cuda.pathfinder"):
        hidden += collect_submodules(package, on_error="warn once")
        datas += collect_data_files(package)
    for name in ("cupy-cuda12x", "cuda-pathfinder", "cuda-toolkit"):
        datas += copy_metadata(name)
        build_info["packages"][name] = importlib.metadata.version(name)
    # Preserve NVIDIA's wheel layout, metadata, headers and licences. Pathfinder
    # uses this layout to locate matching runtime/NVRTC DLLs and include files.
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if not name.lower().replace("_", "-").startswith("nvidia-"):
            continue
        build_info["packages"][name] = distribution.version
        for member in distribution.files or ():
            if ".." in member.parts or str(member).endswith(".pyc"):
                continue
            path = Path(distribution.locate_file(member))
            if path.is_file():
                datas.append((str(path), str(Path(str(member)).parent)))
    runtime_hooks.append(os.path.join(SPECPATH, "cuda_runtime_hook.py"))
manifest = Path(root)/"build"/"build-info.json"
manifest.write_text(json.dumps(build_info, indent=2), encoding="utf-8")
datas.append((str(manifest), "seisdesk"))

a = Analysis(
    [os.path.join(root, "seisdesk", "__main__.py")],
    pathex=[root],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=runtime_hooks,
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "IPython", "pytest", "pytestqt",
              "pyqtgraph.examples", "pyqtgraph.opengl", "OpenGL"],
    noarchive=False,
    module_collection_mode={"disba": "py"},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SeisDesk",
    console=False,
    icon=icon if os.path.exists(icon) else None,
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SeisDesk")
