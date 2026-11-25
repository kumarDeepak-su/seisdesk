"""Download the small published land benchmarks, checking exact content hashes.

Downloads only data. --extract uses libarchive's tar (macOS/Windows) or a supplied
--tar executable; GNU tar does not support RAR. No acquired scripts are run.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import urllib.request


SOURCES = (
    dict(filename="sirente-rfl1.rar", bytes=1125807,
         url="https://data.mendeley.com/public-files/datasets/9rrjbnvmkm/files/38884248-f088-4bd8-b4a9-f6c8ca8782c1/file_downloaded",
         sha256="6da41e85c0f0256eab1e6322724bc9537ed24b27499f2df6525d74657b0ba0b3",
         citation="Torrese, Rossi, Ori, Rainone and Ormo (2019), DOI 10.17632/9rrjbnvmkm.1",
         licence="CC BY 4.0", landing_page="https://data.mendeley.com/datasets/9rrjbnvmkm/1"),
    dict(filename="koenigsee.sgt", bytes=9844,
         url="https://raw.githubusercontent.com/gimli-org/example-data/master/traveltime/koenigsee.sgt",
         sha256="cf8f6c8c79fd0f60984eefc61aff67b567b0c875f8f5c663fe904cdbf0d0414a",
         citation="pyGIMLi development team, Koenigsee field example",
         landing_page="https://www.pygimli.org/_examples_auto/2_seismics/plot_04_koenigsee/"),
)


def fetch(root, source):
    target = root/source["filename"]
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"existing {target} has a different hash; move it aside before downloading")
        return target
    request = urllib.request.Request(source["url"], headers={"User-Agent": "SeisDesk-field-validation/0.6"})
    temporary = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(dir=root, delete=False) as out:
            temporary = Path(out.name)
            content = response.read(source["bytes"]+1)
            if len(content) != source["bytes"] or hashlib.sha256(content).hexdigest() != source["sha256"]:
                raise ValueError(f"{source['filename']}: downloaded content differs from the validated sample")
            out.write(content)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def main(args):
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        fetch(root, source)
    if args.extract:
        tar = args.tar or shutil.which("bsdtar") or shutil.which("tar")
        if not tar:
            raise ValueError("install libarchive/bsdtar or extract sirente-rfl1.rar with your archive manager")
        archive = root/"sirente-rfl1.rar"
        listing = subprocess.run([tar, "-tf", str(archive)], capture_output=True, text=True, check=True).stdout
        members = [PurePosixPath(line.strip().replace("\\", "/")) for line in listing.splitlines() if line.strip()]
        for member in members:
            if member.is_absolute() or ".." in member.parts or member.parts[0] != "RFL 1_raw_data_SEG2":
                raise ValueError("unexpected path in the pinned Sirente archive")
            if len(member.parts) > 1 and (len(member.parts) != 2 or member.suffix.upper() != ".DAT"):
                raise ValueError("unexpected non-waveform member in the pinned Sirente archive")
        subprocess.run([tar, "-xf", str(archive), "-C", str(root)], check=True)
    (root/"land-sources.json").write_text(json.dumps(dict(sources=SOURCES,
         scope="waveform/operator and held-out pick tests, not geological ground truth",
         sirente_geometry="Original coordinate headers disagree with the published acquisition description"),
         indent=2), encoding="utf-8")
    print(f"Verified small land datasets in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--tar", help="path to libarchive/bsdtar with RAR support")
    main(parser.parse_args())
