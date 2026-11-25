"""Assemble existing test images without running SeisDesk or changing any pixels.

Only standard-library file I/O is used. Exact duplicates share one gallery entry;
different renders are retained with their original paths and SHA-256 hashes.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import struct
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "dist/test-image-gallery"
TEMPLATES = Path(__file__).with_name("gallery")
ALIASES = {
    "land-refraction-koenigsee.png": "koenigsee-validation.png",
    "joint-interface-inversion.png": "v06_interface_inversion.png",
    "remote-host-settings.png": "v07_remote_host.png",
    "remote-job-controls.png": "v07_remote_jobs.png",
    "workstation-gpu-controls.png": "v08_workstation_devices.png",
}
SECTIONS = [
    ("validation", "Field and compute validation"),
    ("diagnostics", "Synthetic diagnostics and model views"),
    ("examples", "Picking and processing demonstrations"),
    ("controls", "Software controls"),
    ("earlier", "Other saved renders"),
]
TITLES = {
    "m12_picking_phases_fits.png": "Multiple refraction arrivals and fitted branches",
    "m16_reflection_picking.png": "Multiple reflection arrivals and fitted hyperbolas",
    "m21_stack_time_migration.png": "Stack and time-migration comparison",
    "m54_instantaneous_attributes.png": "Instantaneous seismic attributes",
    "m56_harlan_eigenimage.png": "Harlan and eigenimage filtering",
    "koenigsee-validation.png": "Koenigsee land refraction: held-out picks",
    "athena-cpu-cuda-predictions.png": "Athena CPU / CUDA prediction comparison",
    "v04_tomography.png": "First-arrival tomography",
    "v04_grm.png": "Generalised reciprocal method",
    "v04_fwi.png": "2D acoustic full-waveform inversion",
    "v04_mva.png": "Migration velocity update",
    "v04_velocity_slice.png": "Velocity semblance slice",
    "v04_fowler.png": "Fowler trial stacks",
    "v04_welltie.png": "Wavelet extraction and well tie",
    "v04_multitrace.png": "Well synthetic and nearby seismic traces",
    "v04_deviation.png": "Deviated well trajectory",
    "v04_wellavo.png": "Well-log AVO synthetic",
    "v04_avo_crossplot.png": "AVO intercept / gradient crossplot",
    "v04_statistics.png": "Trace statistics",
    "v04_panel_spectrum.png": "Panel wavenumber spectrum",
    "v04_shear_rotation.png": "Four-component shear rotation",
    "v05_joint_tomography.png": "Joint refraction / reflection tomography",
    "v05_model_builder.png": "3D velocity-model builder",
    "v05_volume_slices.png": "3D velocity-model slices",
    "v05_volume_job.png": "3D survey job settings",
    "v05_su_workbench.png": "Seismic Unix workbench",
    "v06_branch_editor.png": "Multiple arrival branches and shared interfaces",
    "v06_interface_inversion.png": "Joint interface-depth / velocity inversion",
    "v06_seg2_import.png": "SEG-2 timing and import settings",
    "v07_remote_host.png": "SSH / Slurm host settings",
    "v07_remote_jobs.png": "Remote job monitoring and results",
    "v08_workstation_devices.png": "Workstation CPU / CUDA controls",
    "v09_arrival_controls.png": "3D arrival phase and interface controls",
    "v09_arrival_qc.png": "3D per-arrival residuals and convergence",
    "v09_velocity_slices.png": "3D recovered velocity slices",
    "v09_uncertainty_slices.png": "3D conditional velocity uncertainty",
    "v09_reconstruction_coverage.png": "Survey reconstruction fold maps",
    "v09_synthetic_tomography.png": "Synthetic 3D multi-arrival tomography",
    "v09_synthetic_reconstruction.png": "Synthetic 5D plane-wave reconstruction",
    "v09_curved_3d.png": "Withheld curved-event reconstruction: 3D",
    "v09_curved_5d.png": "Withheld curved-event reconstruction: 5D",
    "v10_field_import.png": "Public SEG-D reader and analytical resampling checks",
    "v10_layered_surface_waves.png": "Layered Rayleigh / Love inversion and conditional refits",
    "v10_resample_import.png": "Mixed-rate common-axis import controls",
    "v10_segd_import.png": "SEG-D channel and descaling controls",
    "v10_sps_geometry.png": "SPS survey geometry controls",
    "v10_surface_inversion.png": "Layered dispersion fit and residuals",
    "v11_interface_controls.png": "Shared 3D boundary controls and bounds",
    "v11_interface_depths.png": "Fitted 3D boundary depths and sections",
    "v11_interface_uncertainty.png": "Conditional 3D interface uncertainty",
    "v11_joint_arrival_qc.png": "Arrival residuals after 3D boundary updates",
    "v11_interface_recovery.png": "3D boundary recovery from independent plane reflections",
}
DESCRIPTIONS = {
    "koenigsee-validation.png": "Field-pick validation: 571 fitted and 143 held-out observations. Held-out RMS improves from 3.61 to 0.69 ms. Velocity is masked outside ray coverage; no velocity ground truth is supplied.",
    "athena-cpu-cuda-predictions.png": "Small controlled synthetic execution check. CPU job 3489878 and CUDA job 3489905 agree on all 510 predicted samples. This is not a field image or a survey-scale physics benchmark.",
    "v05_joint_tomography.png": "Controlled synthetic direct and reflected arrival inversion with fixed assigned interfaces.",
    "v05_volume_slices.png": "Orthogonal views of an illustrative constructed 3D velocity model; not a recovered field model.",
    "v06_seg2_import.png": "Interface preview for trace timing, DELAY alignment and the explicit SKEW option.",
    "v08_workstation_devices.png": "Interface preview with illustrative GPU information. Actual A100 hardware evidence is recorded separately in validation/v0.8.0.",
    "v09_synthetic_tomography.png": "Controlled 96-arrival, three-phase example with fixed interfaces. RMS improves from 5.350 to 0.080 ms; coverage and conditional uncertainty are shown separately.",
    "v09_synthetic_reconstruction.png": "Favourable periodic plane-wave example. This is not an estimate of field-data reconstruction accuracy.",
    "v09_curved_3d.png": "Independent curved-event example with approximately 30% withheld traces. Withheld relative waveform error is 21.08%; recorded samples remain exact.",
    "v09_curved_5d.png": "Independent curved-event example with approximately 30% withheld traces. Withheld relative waveform error is 21.89%; recorded samples remain exact.",
    "v10_field_import.png": "Actual SmartSolo SEG-D trace and an analytical mixed-rate timing example. All samples in the nine-trace specimen match the independent reader; vendor navigation and timestamps are not inferred.",
    "v10_layered_surface_waves.png": "Noisy synthetic Rayleigh/Love picks and 30 conditional measurement-error refits. Fixed model topology, mode assignments, Vp/Vs and density; not total geological uncertainty.",
    "v10_surface_inversion.png": "Windows GUI using the noisy synthetic Rayleigh/Love example and eight conditional noise refits.",
    "v11_interface_recovery.png": "Independent analytical plane-reflection times with measurement noise, a fixed known velocity and 20 conditional timing refits. Ray-grid depth resolution and unilluminated areas limit interpretation.",
    "v11_interface_depths.png": "Windows interface view using analytical plane-reflection times with noise. Initial/final boundaries and orthogonal sections are shown; this is not a field interpretation.",
    "v11_interface_uncertainty.png": "Windows preview with three diagnostic refits. Uncertainty is masked without final boundary contacts; zero spread can result from coarse ray-grid resolution.",
}
CONTROL_NAMES = {
    "v05_model_builder.png", "v05_volume_job.png", "v05_su_workbench.png",
    "v06_branch_editor.png", "v06_seg2_import.png", "v07_remote_host.png",
    "v07_remote_jobs.png", "v08_workstation_devices.png",
    "v09_arrival_controls.png", "v10_resample_import.png", "v10_segd_import.png", "v10_sps_geometry.png",
    "v11_interface_controls.png",
}
VALIDATION_NAMES = {"koenigsee-validation.png", "athena-cpu-cuda-predictions.png", "v10_field_import.png",
                    "v10_layered_surface_waves.png", "v09_synthetic_tomography.png", "v09_synthetic_reconstruction.png",
                    "v09_curved_3d.png", "v09_curved_5d.png", "v11_interface_recovery.png"}


def priority(path):
    value = path.as_posix()
    if "/earlier/" in value:
        return 50
    for text, rank in (("v0.11.0-5ba6dd4", 1300), ("v0.11.0", 1200), ("v0.10.0", 1150), ("v0.9.0", 1100),
                       ("v0.8.0-45b7663", 1000), ("v0.7.0-c73ee6d", 900),
                       ("v0.7.0-9b8ee7d", 850), ("v0.6.0-de5fbc6", 800),
                       ("v0.6.0-30a888d", 700), ("v0.5.0-f3d7971", 600),
                       ("v0.8.0-7758b6a", 500), ("validation/", 400),
                       ("dist/result-images/", 300)):
        if text in value:
            return rank
    return 100


def revision(path):
    found = re.search(r"v0\.\d+\.0-([0-9a-f]+)", path)
    return found.group(1) if found else "saved documentation image"


def write_html(path, content):
    path.write_text(content)
    scripts = [body for attributes, body in re.findall(r"<script\b([^>]*)>(.*?)</script>", content, re.S)
               if not attributes.strip()]
    path.with_suffix(path.suffix + ".js").write_text("\n".join(scripts).strip() + "\n")


def main():
    guide = (ROOT / "seisdesk/docs/USER_GUIDE.md").read_text()
    captions = {}
    for caption, filename in re.findall(r"!\[([^\]]+)\]\(images/([^\)]+)\)", guide):
        captions.setdefault(filename, " ".join(caption.split()))
    provenance = {}
    for path in sorted((ROOT / "validation").glob("v*/image-manifest.json")):
        document = json.loads(path.read_text())
        items = document.get("images", []) if isinstance(document, dict) else document
        for entry in items:
            if isinstance(entry, dict) and entry.get("sha256") and entry.get("source"):
                provenance[entry["sha256"]] = entry["source"]
    paths = []
    for directory in ("dist", "validation", "seisdesk/docs"):
        paths += [p for p in (ROOT / directory).rglob("*.png") if not p.is_relative_to(DEST)]
    groups = defaultdict(list)
    for path in sorted(paths):
        groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    entries = []
    for digest, sources in groups.items():
        source = max(sources, key=lambda p: (priority(p.relative_to(ROOT)), p.as_posix()))
        name = ALIASES.get(source.name, source.name)
        raw = source.read_bytes()
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG: {source}")
        width, height = struct.unpack(">II", raw[16:24])
        section = ("validation" if name in VALIDATION_NAMES
                   else "examples" if re.match(r"m\d", name)
                   else "controls" if name in CONTROL_NAMES else "diagnostics")
        kind = ("Field-pick validation" if name == "koenigsee-validation.png" else
                "Field-reader / analytical timing checks" if name == "v10_field_import.png" else
                "Synthetic compute validation" if section == "validation" else
                "Interface example" if section == "controls" else
                "Synthetic / manual demonstration" if section == "examples" else
                "Synthetic diagnostic or model view")
        title = TITLES.get(name, re.sub(r"^m\d+[a-z]?_", "", Path(name).stem).replace("_", " ").capitalize())
        entries.append(dict(id=digest[:12], sha256=digest, filename=name,
                            image=f"images/{digest[:12]}-{name}", title=title, kind=kind, section=section,
                            caption=DESCRIPTIONS.get(name, captions.get(name, "Software interface example using illustrative inputs.")),
                            width=width, height=height, bytes=len(raw),
                            representative_source=source.relative_to(ROOT).as_posix(),
                            sources=[p.relative_to(ROOT).as_posix() for p in sources],
                            source_revision=provenance.get(digest, revision(source.relative_to(ROOT).as_posix())),
                            rank=priority(source.relative_to(ROOT))))
    by_name = defaultdict(list)
    for entry in entries:
        by_name[entry["filename"]].append(entry)
    for name, variants in by_name.items():
        preferred = max(variants, key=lambda e: e["rank"])
        for entry in variants:
            if entry is not preferred:
                entry["section"] = "earlier"
                older = entry["rank"] < preferred["rank"] or "/earlier/" in entry["representative_source"]
                entry["title"] += " — earlier render" if older else " — alternate render"
                entry["caption"] += f" {'Earlier' if older else 'Additional'} render from {entry['source_revision']}; retained for completeness."
                if name == "v08_workstation_devices.png" and "v0.8.0-7758b6a" in entry["representative_source"]:
                    entry["caption"] = "Earlier failed preview with missing-font glyphs. Superseded by the readable workstation preview; not a current GUI reference."
                    entry["kind"] = "Superseded font-rendering failure"
    order = {name: index for index, (name, _) in enumerate(SECTIONS)}
    entries.sort(key=lambda e: (order[e["section"]], e["filename"], -e["rank"]))
    (DEST / "images").mkdir(parents=True, exist_ok=True)
    for entry in entries:
        target = DEST / entry["image"]
        shutil.copy2(ROOT / entry["representative_source"], target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]
    counts = Counter(e["section"] for e in entries)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), original_file_count=len(paths),
                    unique_image_count=len(entries), distinct_views=len(by_name),
                    section_counts=dict(counts), pixels_changed=False,
                    search_roots=["dist", "validation", "seisdesk/docs"], images=entries)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cards = []
    nav = []
    escape = html.escape
    for key, label in SECTIONS:
        nav.append(f'<a href="#{key}">{escape(label)} ({counts[key]})</a>')
        cards.append(f'<section id="{key}"><h2>{escape(label)} <span>{counts[key]}</span></h2><div class="grid">')
        for entry in (e for e in entries if e["section"] == key):
            cards.append(f'''<figure>
<a class="preview" href="{entry['image']}" data-image-id="{entry['id']}" aria-label="Enlarge {escape(entry['title'], quote=True)}">
<img src="{entry['image']}" alt="{escape(entry['title'], quote=True)}" width="{entry['width']}" height="{entry['height']}" loading="lazy"></a>
<figcaption><p class="kind">{escape(entry['kind'])}</p><h3>{escape(entry['title'])}</h3><p>{escape(entry['caption'])}</p>
<details><summary>Original files and provenance</summary><p>{escape(entry['source_revision'])} · {entry['width']} × {entry['height']} pixels</p><ul>{''.join('<li>'+escape(p)+'</li>' for p in entry['sources'])}</ul><p class="digest">SHA-256: {entry['sha256']}</p></details></figcaption></figure>''')
        cards.append("</div></section>")
    template = (TEMPLATES / "index.html.in").read_text()
    page = (template.replace("__IMAGE_COUNT__", str(len(entries))).replace("__VIEW_COUNT__", str(len(by_name)))
            .replace("__FILE_COUNT__", str(len(paths))).replace("__NAV__", "\n".join(nav))
            .replace("__CARDS__", "\n".join(cards))
            .replace("__IMAGE_DATA__", json.dumps(entries).replace("<", "\\u003c")))
    write_html(DEST / "index.html", page)
    selected_names = ["koenigsee-validation.png", "v06_interface_inversion.png", "m12_picking_phases_fits.png",
                      "m16_reflection_picking.png", "v04_fwi.png", "v08_workstation_devices.png"]
    short_labels = ["Field refraction", "Interface inversion", "Refraction picks", "Reflection picks", "Acoustic FWI", "GPU controls"]
    selected = []
    for name, label in zip(selected_names, short_labels):
        item = next(e for e in entries if e["filename"] == name and e["section"] != "earlier")
        selected.append(dict(id=item["id"], label=label, title=item["title"], kind=item["kind"], caption=item["caption"],
                             src="data:image/png;base64," + base64.b64encode((DEST / item["image"]).read_bytes()).decode()))
    fragment = (TEMPLATES / "selected.html.in").read_text().replace("__SELECTED_DATA__", json.dumps(selected).replace("<", "\\u003c"))
    assert len(fragment.encode()) < 1_000_000, "Inline preview exceeds 1 MB"
    write_html(DEST / "selected-test-images.html", fragment)
    (DEST / "READ-ME.txt").write_text(
        f"SeisDesk test image collection\n\n{len(entries)} distinct saved images, {len(by_name)} views, {len(paths)} original file locations.\n"
        "Open index.html in a browser. Click an image to enlarge it; Open original displays its full resolution.\n"
        "Field validation, synthetic demonstrations, interface examples and earlier render variants are labelled separately.\n"
        "Every PNG is an exact copy. manifest.json contains source paths, dimensions and SHA-256 hashes.\n"
        "The images do not establish completion or field validation of every processing method.\n")
    archive_path = ROOT / "dist/seisdesk-test-images.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (DEST / "index.html", DEST / "manifest.json", DEST / "READ-ME.txt"):
            archive.write(path, "seisdesk-test-images/" + path.relative_to(DEST).as_posix())
        for entry in entries:
            archive.write(DEST / entry["image"], "seisdesk-test-images/" + entry["image"])
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(entries) + 3
    print(json.dumps(dict(original_files=len(paths), unique_images=len(entries), distinct_views=len(by_name),
                          categories=dict(counts), gallery=str(DEST / "index.html"),
                          inline_bytes=len(fragment.encode()), archive_bytes=archive_path.stat().st_size), indent=2))


if __name__ == "__main__":
    main()
