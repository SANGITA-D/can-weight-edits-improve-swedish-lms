"""Convert the Claude Doc's XML view (saved from a docs `read`) into Markdown, so the report files match the reviewed doc.

Usage: python doc_to_markdown.py <saved-read-result.txt> <out.md>
"""
import json
import sys
import xml.etree.ElementTree as ET

FMT = {"bold": "**", "italic": "*", "code": "`"}


def runs(el, fmt=(), href=None):
    """Flatten inline content into (text, fmt, href) runs."""
    out = []
    if el.text:
        out.append((el.text, fmt, href))
    for ch in el:
        if ch.tag in FMT:
            out += runs(ch, fmt + (ch.tag,), href)
        elif ch.tag == "link":
            out += runs(ch, fmt, ch.get("href"))
        elif ch.tag == "date":
            out.append(("19 November 2026" if ch.get("value") == "2026-11-19" else ch.get("value"), fmt, href))
        elif ch.tag == "mention":
            out.append(("Sangita Dutta", fmt, href))
        else:
            out += runs(ch, fmt, href)
        if ch.tail:
            out.append((ch.tail, fmt, href))
    return out


def render(rs, strip_bold=False):
    merged = []
    for t, f, h in rs:
        f = tuple(x for x in f if not (strip_bold and x == "bold"))
        if merged and merged[-1][1] == f and merged[-1][2] == h:
            merged[-1] = (merged[-1][0] + t, f, h)
        else:
            merged.append((t, f, h))
    s = ""
    for t, f, h in merged:
        lead = t[: len(t) - len(t.lstrip())]
        trail = t[len(t.rstrip()):]
        core = t.strip()
        if core:
            for x in ("code", "bold", "italic"):
                if x in f:
                    core = f"{FMT[x]}{core}{FMT[x]}"
            if h:
                core = f"[{core}]({h})"
        s += lead + core + trail
    return s


def para(p, strip_bold=False):
    return render(runs(p), strip_bold)


def block(el, depth=0):
    ind = "    " * depth
    if el.tag == "paragraph":
        h = el.get("heading")
        if h:
            return "#" * int(h) + " " + para(el, strip_bold=True) + "\n"
        return para(el) + "\n"
    if el.tag == "list":
        lines = []
        for i, item in enumerate(el.findall("listItem"), 1):
            marker = f"{i}." if el.get("kind") == "ordered" else "-"
            first = True
            for ch in item:
                if ch.tag == "paragraph":
                    text = para(ch)
                    lines.append(f"{ind}{marker} {text}" if first else f"{ind}    {text}")
                    first = False
                elif ch.tag == "list":
                    lines.append(block(ch, depth + 1).rstrip("\n"))
        return "\n".join(lines) + "\n"
    if el.tag == "table":
        rows = el.findall("row")
        out = []
        for r_i, row in enumerate(rows):
            cells = [" ".join(para(p) for p in c.findall("paragraph")).replace("|", "\\|") for c in row.findall("cell")]
            out.append("| " + " | ".join(cells) + " |")
            if r_i == 0:
                out.append("|" + "---|" * len(cells))
        return "\n".join(out) + "\n"
    return ""


def main():
    src, dst = sys.argv[1], sys.argv[2]
    raw = json.load(open(src))
    text = next(x["text"] for x in raw if x["text"].lstrip().startswith("{"))
    obj, _ = json.JSONDecoder().raw_decode(text)
    root = ET.fromstring(obj["data"]["xml"])
    md = "\n".join(block(el) for el in root)
    open(dst, "w").write(md)
    print(f"wrote {dst}: {len(md.split())} words")


if __name__ == "__main__":
    main()
