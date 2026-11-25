"""Portable, hash-checked compute bundles. This module uses only the standard library."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import zipfile


CHUNK = 4 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def checked_name(name):
    """Accept only portable relative file names, including on a Windows recipient."""
    parts = PurePosixPath(name).parts
    if (not parts or name != "/".join(parts) or any(p in (".", "..") for p in parts)
            or name.startswith("/") or any(c in name for c in "\\:\x00")
            or any(ord(c) < 32 for c in name)
            or any(p.endswith((".", " ")) for p in parts)
            or any(re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?", p) for p in parts)):
        raise ValueError(f"unsafe bundle name: {name!r}")
    return name


def write_bundle(destination, files, *, kind, metadata=None, cancelled=lambda: False, report=lambda s: None):
    """Stream files into an atomic ZIP64 archive; hashes cover the actual archived bytes."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    manifest = dict(format=1, kind=kind, metadata=metadata or {}, files={})
    names = set()
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for name, source in files:
                checked_name(name)
                if name.casefold() in names or name == "manifest.json":
                    raise ValueError(f"duplicate or reserved bundle name: {name}")
                names.add(name.casefold())
                source = Path(source)
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f"bundle input must be a regular file: {source}")
                report(f"Bundling {name}")
                digest, size = hashlib.sha256(), 0
                with source.open("rb") as reader, archive.open(name, "w", force_zip64=True) as writer:
                    while True:
                        if cancelled():
                            raise InterruptedError("bundle creation cancelled")
                        chunk = reader.read(CHUNK)
                        if not chunk:
                            break
                        writer.write(chunk); digest.update(chunk); size += len(chunk)
                manifest["files"][name] = dict(size=size, sha256=digest.hexdigest())
            archive.writestr("manifest.json", json.dumps(manifest, allow_nan=False))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(destination), size=destination.stat().st_size, sha256=sha256(destination), manifest=manifest)


def extract_bundle(source, destination, *, expected_sha256=None, kind=None, cancelled=lambda: False):
    """Verify every member before publishing a fresh extraction directory; never overwrite a job."""
    source, destination = Path(source), Path(destination)
    if expected_sha256 and sha256(source) != expected_sha256:
        raise ValueError("bundle SHA-256 mismatch")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".extract-", dir=destination.parent))
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            # ZipInfo normalises backslashes on Windows and truncates NULs.
            # Inspect the original stored name before either transformation.
            names = [checked_name(info.orig_filename) for info in infos]
            if any(info.filename != name for info, name in zip(infos, names)):
                raise ValueError("unsafe bundle name normalisation")
            if len(set(n.casefold() for n in names)) != len(names):
                raise ValueError("duplicate bundle members")
            if any(info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) for info in infos):
                raise ValueError("bundle contains directory or symbolic link members")
            if "manifest.json" not in names:
                raise ValueError("bundle is missing its manifest")
            if archive.getinfo("manifest.json").file_size > 32 * 1024 * 1024:
                raise ValueError("bundle manifest is too large")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != 1 or (kind is not None and manifest.get("kind") != kind):
                raise ValueError("unsupported bundle format or kind")
            if set(names) != set(manifest["files"]) | {"manifest.json"}:
                raise ValueError("bundle inventory differs from manifest")
            total = sum(info.file_size for info in infos)
            if total > shutil.disk_usage(temporary).free:
                raise OSError("not enough free disk space to extract the bundle")
            for name, item in manifest["files"].items():
                info = archive.getinfo(name)
                if item["size"] != info.file_size:
                    raise ValueError(f"bundle size mismatch: {name}")
                target = temporary.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open(name) as reader, target.open("xb") as writer:
                    while True:
                        if cancelled():
                            raise InterruptedError("bundle extraction cancelled")
                        chunk = reader.read(CHUNK)
                        if not chunk:
                            break
                        digest.update(chunk); writer.write(chunk)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"bundle content hash mismatch: {name}")
            atomic_json(temporary/"manifest.json", manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def source_files():
    """Source installs and frozen desktop builds ship the same compute Python files."""
    package = Path(__file__).resolve().parents[1]
    snapshot = package/"compute_source.zip"
    if snapshot.is_file():
        # The snapshot is produced by our packaging spec, and contains only Python source.
        with tempfile.TemporaryDirectory(prefix="seisdesk-source-") as directory:
            with zipfile.ZipFile(snapshot) as archive:
                for info in archive.infolist():
                    checked_name(info.filename)
                    if not info.filename.startswith("seisdesk/") or not info.filename.endswith(".py"):
                        raise ValueError("invalid packaged compute source")
                    target = Path(directory)/info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(info))
            yield from (("code/"+p.relative_to(directory).as_posix(), p)
                        for p in sorted(Path(directory).rglob("*.py")))
    elif not getattr(sys, "frozen", False):
        yield from (("code/seisdesk/"+p.relative_to(package).as_posix(), p)
                    for p in sorted(package.rglob("*.py")) if not p.is_symlink())
    else:
        raise FileNotFoundError("this installation is missing its compute source snapshot")


def build_volume_bundle(parameters, destination, *, work_dir=".", cancelled=lambda: False, report=lambda s: None):
    from .. import __version__
    params = dict(parameters)
    if params.get("mode", "rtm") not in ("model", "rtm", "fwi"):
        raise ValueError("unsupported remote volume operation")
    if params.get("scheduler"):
        raise ValueError("clear the desktop Dask scheduler address before submitting a remote job")
    files = []
    for field in ("input_file", "model", "wavelet"):
        value = params.get(field, "")
        if not value and field == "wavelet":
            continue
        if not value:
            raise ValueError(f"choose {field} before submitting a remote job")
        path = Path(value)
        path = path if path.is_absolute() else Path(work_dir)/path
        if path.is_symlink():
            raise ValueError(f"select the actual file instead of a symbolic link: {path}")
        if path.is_dir() and field == "model":
            name = "inputs/model"
            files.extend((name+"/"+f, path/f) for f in ("volume.json", "velocity.npy"))
        else:
            name = "inputs/"+field+path.suffix.lower()
            files.append((name, path))
        params[field] = name
    params.update(job_directory="output", scheduler="")
    with tempfile.TemporaryDirectory(prefix="seisdesk-parameters-") as directory:
        settings = Path(directory)/"parameters.json"
        atomic_json(settings, params)
        def members():
            yield "parameters.json", settings
            yield from source_files()
            yield from files
        source_reported = False
        def progress(message):
            nonlocal source_reported
            if message.startswith("Bundling code/"):
                if not source_reported:
                    report("Bundling SeisDesk compute source")
                    source_reported = True
            else:
                report(message)
        return write_bundle(destination, members(), kind="volume-input", metadata=dict(version=__version__),
                            cancelled=cancelled, report=progress)


def rebase_paths(value, remote_root, local_root):
    """Only rewrite paths inside this job, preserving labels and unrelated strings."""
    remote_root = str(remote_root).rstrip("/")
    if isinstance(value, str) and (value == remote_root or value.startswith(remote_root+"/")):
        return str(Path(local_root).joinpath(*PurePosixPath(value[len(remote_root):].lstrip("/")).parts))
    if isinstance(value, dict):
        return {key: rebase_paths(item, remote_root, local_root) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase_paths(item, remote_root, local_root) for item in value]
    return value
