"""Fetch bounded public-data samples and record provenance; never run downloaded scripts.

Alaska source: https://wiki.seg.org/wiki/Alaska_2D_land_line_31-81 (USGS).
Marine source: SEG Mobil AVO Viking Graben line 12; the Zenodo mirror is used
only if the original SEG bucket is unavailable. This is data preparation, not
an assertion of processing accuracy.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tarfile
import urllib.request


def response(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "SeisDesk-validation/0.5"}), timeout=45)


def main(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    report = {"land": {}, "marine": {}, "scientific_accuracy": "not assessed by downloading data"}
    errors = []
    for url in ("https://s3.amazonaws.com/open.source.geoscience/open_data/alaska/alaska31-81.tar.gz",
                "https://www.dropbox.com/s/ee1r2riltd0li94/open_lib_demo_2012.tar.gz?dl=1"):
        try:
            archive = root/"alaska.tar.gz"
            with response(url) as stream, archive.open("wb") as target:
                count = 0
                while True:
                    block = stream.read(1024*1024)
                    if not block:
                        break
                    count += len(block)
                    if count > 256*1024**2:
                        raise ValueError("archive exceeds the 256 MiB download budget")
                    target.write(block)
            with tarfile.open(archive) as source:
                members = source.getmembers()
                inventory = [{"name": item.name, "bytes": item.size} for item in members if item.isfile()]
                # Extract only data and text, to fresh numbered paths; no links,
                # absolute paths, parent traversals, binaries or downloaded code.
                extracted = []
                for index, item in enumerate(members):
                    if not item.isfile() or item.size > 512*1024**2:
                        continue
                    suffix = Path(item.name).suffix.lower()
                    if suffix not in (".su", ".sgy", ".segy", ".txt", ".par"):
                        continue
                    target = root/f"alaska-{index:05d}{suffix}"
                    with source.extractfile(item) as stream, target.open("wb") as out:
                        while block := stream.read(1024*1024):
                            out.write(block)
                    extracted.append({"original": item.name, "path": str(target.resolve()), "bytes": item.size})
            report["land"] = {"url": url, "credit": "U.S. Geological Survey; scripts by Karl Schleicher",
                              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                              "inventory": inventory, "extracted": extracted}
            break
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
    for url in ("https://s3.amazonaws.com/open.source.geoscience/open_data/Mobil_Avo_Viking_Graben_Line_12/seismic.segy",
                "https://zenodo.org/records/23481108/files/seismic.segy?download=1"):
        try:
            with response(url) as stream:
                header = stream.read(3600)
                if len(header) != 3600:
                    raise ValueError("missing SEG-Y reel header")
                ns, fmt = struct.unpack_from(">HH", header, 3220)[0], struct.unpack_from(">H", header, 3224)[0]
                if fmt not in (1, 5) or not 1 <= ns <= 65535:
                    raise ValueError("expected fixed-length IBM/IEEE float SEG-Y")
                record_bytes = 240+4*ns
                payload = stream.read(record_bytes*2400)  # First 20 shots of the published 120-channel dataset.
                count = len(payload)//record_bytes
                if count < 120:
                    raise ValueError("too few complete field traces")
                content = header+payload[:count*record_bytes]
            path = root/"viking-first-2400.sgy"
            path.write_bytes(content)
            report["marine"] = dict(url=url, path=str(path.resolve()), traces=count, samples=ns,
                                     sha256=hashlib.sha256(content).hexdigest(),
                                     scope="first complete traces only; no claim of full-survey validation",
                                     credit="Mobil; Keys and Foster, SEG Open File Publication 4")
            break
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
    report["download_errors"] = errors
    (root/"sources.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "land"}, indent=2))
    print("Land extracted:", report["land"].get("extracted", []))


if __name__ == "__main__":
    main(sys.argv[1])
