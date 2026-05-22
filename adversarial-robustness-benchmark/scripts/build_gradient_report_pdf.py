"""Convert results/gradient/REPORT.md (+ figures/) into a single PDF.

Pure-Python: markdown -> HTML -> PDF via xhtml2pdf (no external binaries).
Run after `scripts/build_gradient_report.py` (which writes the markdown).

Usage:
    python scripts/build_gradient_report_pdf.py
    python scripts/build_gradient_report_pdf.py --input results/gradient/REPORT.md --output results/gradient/REPORT.pdf
"""

from __future__ import annotations

import argparse
import os
import sys

import markdown
from xhtml2pdf import pisa


_CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #111; line-height: 1.45; }
h1 { font-size: 18pt; border-bottom: 1px solid #999; padding-bottom: 4pt; margin-top: 0; }
h2 { font-size: 14pt; margin-top: 18pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 12pt; margin-top: 14pt; }
p { margin: 4pt 0; }
ul, ol { margin: 4pt 0 4pt 18pt; }
li { margin: 2pt 0; }
code { font-family: "Courier New", monospace; font-size: 9.5pt; background: #f3f3f3; padding: 1pt 3pt; border-radius: 2pt; }
pre { background: #f3f3f3; padding: 6pt; font-size: 9pt; }
table { border-collapse: collapse; margin: 8pt 0; width: 100%; }
th, td { border: 1px solid #888; padding: 3pt 6pt; font-size: 9.5pt; text-align: left; }
th { background: #eee; }
img { max-width: 100%; }
em { color: #555; }
figcaption, .caption { font-size: 9pt; color: #555; font-style: italic; }
hr { border: none; border-top: 1px solid #ccc; margin: 12pt 0; }
"""


def _link_callback(uri: str, rel: str) -> str:
    """Resolve relative image URIs against the markdown file's directory."""
    if os.path.isabs(uri) and os.path.exists(uri):
        return uri
    base = _link_callback.base_dir  # type: ignore[attr-defined]
    candidate = os.path.normpath(os.path.join(base, uri))
    return candidate


def md_to_pdf(input_md: str, output_pdf: str) -> None:
    with open(input_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_CSS}</style></head><body>{html_body}</body></html>"
    )

    _link_callback.base_dir = os.path.dirname(os.path.abspath(input_md))  # type: ignore[attr-defined]

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)) or ".", exist_ok=True)
    with open(output_pdf, "wb") as out:
        result = pisa.CreatePDF(html, dest=out, link_callback=_link_callback)
    if result.err:
        raise RuntimeError(f"xhtml2pdf reported {result.err} error(s) building {output_pdf}")


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_in = os.path.join(repo_root, "results", "gradient", "REPORT.md")
    default_out = os.path.join(repo_root, "results", "gradient", "REPORT.pdf")

    p = argparse.ArgumentParser()
    p.add_argument("--input", default=default_in, help="Path to REPORT.md")
    p.add_argument("--output", default=default_out, help="Output PDF path")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input markdown not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    md_to_pdf(args.input, args.output)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"wrote {args.output}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
