"""Render a Markdown report to PDF: Markdown -> HTML (python-markdown) -> PDF (headless Chrome).

Usage: python make_pdf.py REPORT.md REPORT.pdf
"""
import subprocess
import sys
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 18pt; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
h1, h2, h3 { font-weight: 700; page-break-after: avoid; }
p, li { margin: 3pt 0; }
ul { margin: 3pt 0 6pt 0; padding-left: 18pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #f0efec; }
code { font-family: Menlo, monospace; font-size: 8.5pt; background: #f4f4f4; padding: 0 2pt; }
a { color: #1a4f9c; text-decoration: none; }
"""

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    body = markdown.markdown(src.read_text(), extensions=["tables", "sane_lists"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    tmp = dst.with_suffix(".html")
    tmp.write_text(html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={dst.resolve()}", tmp.resolve().as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp.unlink()
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
