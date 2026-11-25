"""Build the PDF user manual: seisdesk/docs/USER_GUIDE.md + the generated process reference -> HTML (pandoc) ->
PDF (headless Chrome).

    python tools/build_manual.py dist        # writes dist/SeisDesk_User_Manual.pdf (and .html)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(ROOT, "seisdesk", "docs")


def find(candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def main(out_dir: str) -> int:
    sys.path.insert(0, ROOT)
    from seisdesk import __version__
    from seisdesk.processing.reference import process_reference_markdown

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(DOCS, "USER_GUIDE.md"), encoding="utf-8") as f:
        guide = f.read()
    reference = process_reference_markdown().replace("# Process and parameter reference",
                                                     "# Appendix: process and parameter reference", 1)
    src = os.path.join(DOCS, "_manual_build.md")
    with open(src, "w", encoding="utf-8") as f:
        f.write(guide.rstrip() + "\n\n" + reference + "\n")
    pandoc = find([shutil.which("pandoc"), r"C:\Program Files\Pandoc\pandoc.exe"])
    chrome = find([r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                   r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                   shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome"),
                   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"])
    if pandoc is None or chrome is None:
        print(f"pandoc: {pandoc}, chrome: {chrome}: cannot build the PDF")
        return 1
    html = os.path.abspath(os.path.join(out_dir, "SeisDesk_User_Manual.html"))
    pdf = os.path.abspath(os.path.join(out_dir, "SeisDesk_User_Manual.pdf"))
    subprocess.run([pandoc, src, "--standalone", "--embed-resources", "--toc", "--toc-depth=2",
                    "--metadata", "title=SeisDesk User Manual", "--metadata", f"subtitle=Version {__version__}",
                    "--css", os.path.join(ROOT, "packaging", "manual.css"), "--resource-path", DOCS, "-o", html],
                   check=True)
    os.remove(src)
    url = "file:///" + html.replace("\\", "/").lstrip("/")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-sandbox",
                    f"--print-to-pdf={pdf}", url], check=True, timeout=300)
    print("wrote", pdf, os.path.getsize(pdf), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "dist"))
