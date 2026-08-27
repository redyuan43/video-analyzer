#!/usr/bin/env python3
"""Render a Markdown document as a mobile-first PDF."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter
from weasyprint import HTML


MERMAID_BLOCK_RE = re.compile(r"(^```mermaid[ \t]*\n)(.*?)(^```[ \t]*$)", re.M | re.S)

MOBILE_CSS = """
@page {
  size: 96mm 170mm;
  margin: 6mm 5mm;
}

html {
  font-size: 14pt;
}

body {
  color: #111827;
  font-family: "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
    "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif;
  line-height: 1.62;
  margin: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
}

h1, h2, h3, h4 {
  color: #0f172a;
  font-weight: 750;
  line-height: 1.28;
  margin: 1.05em 0 0.45em;
  page-break-after: avoid;
}

h1 {
  font-size: 1.55rem;
  margin-top: 0;
}

h2 {
  border-bottom: 0.4pt solid #cbd5e1;
  font-size: 1.28rem;
  padding-bottom: 0.18em;
}

h3 {
  font-size: 1.12rem;
}

p, ul, ol, blockquote, table, pre {
  margin: 0.55em 0;
}

ul, ol {
  padding-left: 1.35em;
}

li + li {
  margin-top: 0.2em;
}

a {
  color: #0f5cc0;
  text-decoration: none;
}

blockquote {
  border-left: 3pt solid #94a3b8;
  color: #475569;
  margin-left: 0;
  padding: 0.1em 0 0.1em 0.75em;
}

code {
  background: #f1f5f9;
  border-radius: 3pt;
  font-family: "JetBrains Mono", "Noto Sans Mono CJK SC", "SFMono-Regular",
    Consolas, monospace;
  font-size: 0.82em;
  padding: 0.06em 0.22em;
}

pre {
  background: #0f172a;
  border-radius: 5pt;
  color: #e2e8f0;
  font-family: "JetBrains Mono", "Noto Sans Mono CJK SC", "SFMono-Regular",
    Consolas, monospace;
  font-size: 0.72rem;
  line-height: 1.48;
  overflow-wrap: anywhere;
  padding: 0.65em;
  white-space: pre-wrap;
}

pre code {
  background: transparent;
  color: inherit;
  font-size: inherit;
  padding: 0;
}

img, svg {
  display: block;
  height: auto;
  margin: 0.65em auto;
  max-width: 100%;
}

.final-image-page {
  align-items: center;
  break-after: page;
  break-before: page;
  display: flex;
  height: 158mm;
  justify-content: center;
  margin: 0;
  page-break-after: always;
  page-break-before: always;
  page-break-inside: avoid;
}

.final-image-page img {
  height: auto;
  margin: 0 auto;
  max-height: 154mm;
  max-width: 100%;
  object-fit: contain;
}

.mobile-flowchart {
  margin: 0.8em 0 1em;
}

.flow-node {
  background: #eef2ff;
  border: 1pt solid #8b5cf6;
  border-radius: 5pt;
  color: #111827;
  display: table;
  font-weight: 650;
  line-height: 1.42;
  margin: 0 auto;
  max-width: 88%;
  min-height: 2.3em;
  padding: 0.55em 0.7em;
  page-break-inside: avoid;
  text-align: center;
}

.flow-node.decision {
  background: #fff7ed;
  border-color: #f59e0b;
}

.flow-index {
  color: #64748b;
  display: block;
  font-size: 0.72em;
  font-weight: 700;
  margin-bottom: 0.15em;
}

.flow-arrow {
  color: #334155;
  font-size: 1.4rem;
  font-weight: 800;
  line-height: 1;
  margin: 0.22em 0;
  page-break-after: avoid;
  page-break-before: avoid;
  text-align: center;
}

.flow-row {
  align-items: stretch;
  display: flex;
  flex-wrap: wrap;
  gap: 0.45em;
  justify-content: center;
  page-break-inside: avoid;
}

.flow-row .flow-node {
  display: block;
  flex: 1 1 38%;
  margin: 0;
  max-width: 100%;
  min-width: 8.5em;
}

table {
  border-collapse: collapse;
  display: table;
  font-size: 0.78rem;
  table-layout: fixed;
  width: 100%;
}

th, td {
  border: 0.5pt solid #cbd5e1;
  padding: 0.38em 0.42em;
  vertical-align: top;
  word-break: break-word;
}

th {
  background: #f8fafc;
  font-weight: 700;
}

hr {
  border: 0;
  border-top: 0.5pt solid #cbd5e1;
  margin: 1em 0;
}

.toc ul {
  list-style: none;
  padding-left: 0.8em;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Markdown to a mobile-first PDF")
    parser.add_argument("input_md", help="Prepared Markdown file")
    parser.add_argument("output_pdf", help="Output PDF path")
    parser.add_argument("--title", default="", help="Optional document title")
    return parser.parse_args()


def render_markdown(text: str) -> str:
    return markdown.markdown(
        normalize_markdown_tables(text),
        extensions=[
            "markdown.extensions.extra",
            "markdown.extensions.sane_lists",
            "markdown.extensions.toc",
            "markdown.extensions.codehilite",
        ],
        extension_configs={
            "markdown.extensions.codehilite": {
                "guess_lang": False,
                "noclasses": False,
            }
        },
        output_format="html5",
    )


def normalize_markdown_tables(text: str) -> str:
    normalized: list[str] = []
    for line in str(text or "").splitlines():
        split_lines = split_inline_markdown_table_line(line)
        normalized.extend(split_lines)
    return "\n".join(normalized)


def split_inline_markdown_table_line(line: str) -> list[str]:
    match = re.match(r"^(.+?[：:])\s+(\|.+)$", line)
    if not match:
        return [line]
    rows = [row.strip() for row in re.split(r"(?<=\|)\s+(?=\|)", match.group(2).strip()) if row.strip()]
    if len(rows) < 2 or not is_markdown_table_separator(rows[1]):
        return [line]
    return [match.group(1).strip(), "", *rows]


def parse_markdown_table_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return [cell.strip() for cell in value.split("|")]


def is_markdown_table_separator(line: str) -> bool:
    value = line.strip()
    if "|" not in value:
        return False
    cells = parse_markdown_table_row(value)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def wrap_final_images(body: str) -> str:
    final_image_re = re.compile(
        r"<p>\s*(<img\b(?=[^>]*\bsrc=\"[^\"]*baoyu_images/final/[^\"]+\.png\")[^>]*>)\s*</p>",
        re.I,
    )
    return final_image_re.sub(r'<section class="final-image-page">\1</section>', body)


def normalize_mermaid(diagram: str) -> str:
    return re.sub(
        r'(?<![\w])([A-Za-z][A-Za-z0-9_]*)\[([^"\]\n][^\]\n]*)\]',
        lambda match: f'{match.group(1)}["{match.group(2).replace(chr(34), chr(92) + chr(34))}"]',
        diagram,
    )


def parse_mermaid_node(node: str) -> tuple[str, str | None, str]:
    node = node.strip().rstrip(";")
    match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)(?:([\[\{\(])(.*)([\]\}\)]))?$", node)
    if not match:
        return node, None, "box"
    node_id = match.group(1)
    label = match.group(3)
    shape = "decision" if match.group(2) == "{" else "box"
    if label is not None:
        label = label.strip()
        if len(label) >= 2 and label[0] == label[-1] == '"':
            label = label[1:-1]
    return node_id, label, shape


def mermaid_label_to_html(label: str) -> str:
    escaped = html.escape(label)
    return re.sub(r"&lt;br\s*/?&gt;", "<br>", escaped, flags=re.I)


def render_simple_mermaid_flowchart(diagram: str) -> str | None:
    lines = [
        line.strip()
        for line in diagram.splitlines()
        if line.strip() and not line.strip().startswith("%%")
    ]
    if not lines:
        return None
    first_line = lines[0].strip()
    if not re.match(r"^(flowchart\s+(TB|TD|LR|RL)|graph\s+(TB|TD|LR|RL))\b", first_line, re.I):
        return None

    labels: dict[str, str] = {}
    shapes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    node_order: list[str] = []
    standalone_nodes: list[str] = []
    unsupported_prefixes = re.compile(r"^(subgraph|end\b|direction\b|style\b|classDef\b|class\b)", re.I)

    def remember_node(node_id: str, label: str | None, shape: str) -> bool:
        if not node_id or re.search(r"[{}()[\]]", node_id):
            return False
        if node_id not in node_order:
            node_order.append(node_id)
        if label is not None:
            labels[node_id] = label
        shapes.setdefault(node_id, shape)
        return True

    for line in lines[1:]:
        if unsupported_prefixes.match(line) or re.fullmatch(r"[};\s]+", line):
            continue
        edge_match = re.match(r"^(.+?)\s*-->(?:\|.*?\|)?\s*(.+?)\s*;?$", line)
        if not edge_match:
            node_id, label, shape = parse_mermaid_node(line)
            if not remember_node(node_id, label, shape):
                return None
            if node_id not in standalone_nodes:
                standalone_nodes.append(node_id)
            continue
        left_id, left_label, left_shape = parse_mermaid_node(edge_match.group(1))
        right_id, right_label, right_shape = parse_mermaid_node(edge_match.group(2))
        if not remember_node(left_id, left_label, left_shape):
            continue
        if not remember_node(right_id, right_label, right_shape):
            continue
        edges.append((left_id, right_id))

    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, int] = {}
    connected_nodes = {node_id for edge in edges for node_id in edge}
    nodes = set(connected_nodes)
    standalone_only = [node_id for node_id in standalone_nodes if node_id not in connected_nodes]
    if not edges and not standalone_only:
        return None
    for left_id, right_id in edges:
        outgoing.setdefault(left_id, []).append(right_id)
        incoming[right_id] = incoming.get(right_id, 0) + 1
        incoming.setdefault(left_id, incoming.get(left_id, 0))
    starts = [node_id for node_id in nodes if incoming.get(node_id, 0) == 0]
    if edges and not starts:
        return None

    order_index = {node_id: index for index, node_id in enumerate(node_order)}
    queue = sorted(starts, key=lambda node_id: order_index.get(node_id, 10**9))
    remaining_incoming = dict(incoming)
    topological: list[str] = []
    while queue:
        node_id = queue.pop(0)
        topological.append(node_id)
        for next_id in outgoing.get(node_id, []):
            remaining_incoming[next_id] -= 1
            if remaining_incoming[next_id] == 0:
                queue.append(next_id)
                queue.sort(key=lambda item: order_index.get(item, 10**9))
    if edges and len(topological) != len(nodes):
        return None

    levels: dict[str, int] = {node_id: 0 for node_id in starts}
    for node_id in topological:
        base_level = levels.get(node_id, 0)
        for next_id in outgoing.get(node_id, []):
            levels[next_id] = max(levels.get(next_id, 0), base_level + 1)
    max_level = max(levels.values()) if edges else -1
    grouped: list[list[str]] = [[] for _ in range(max_level + 1)] if max_level >= 0 else []
    for node_id in topological:
        grouped[levels.get(node_id, 0)].append(node_id)

    flow_parts = ['<div class="mobile-flowchart" aria-label="流程图">']
    rendered_index = 1

    def append_row(row: list[str]) -> None:
        nonlocal rendered_index
        flow_parts.append('<div class="flow-row">')
        for node_id in row:
            label = mermaid_label_to_html(labels.get(node_id, node_id))
            shape = " decision" if shapes.get(node_id) == "decision" else ""
            flow_parts.append(
                f'<div class="flow-node{shape}"><span class="flow-index">{rendered_index}</span>{label}</div>'
            )
            rendered_index += 1
        flow_parts.append("</div>")

    for level, row in enumerate(grouped):
        if not row:
            return None
        append_row(row)
        if level < max_level:
            flow_parts.append('<div class="flow-arrow">↓</div>')
    if standalone_only:
        if grouped:
            flow_parts.append('<div class="flow-arrow">↓</div>')
        append_row(standalone_only)
    flow_parts.append("</div>")
    return "\n".join(flow_parts)


def render_mermaid_blocks(text: str, work_dir: Path) -> str:
    puppeteer_config: Path | None = None
    parts: list[str] = []
    last = 0
    count = 0
    for match in MERMAID_BLOCK_RE.finditer(text):
        count += 1
        parts.append(text[last : match.start()])
        diagram = match.group(2).strip()
        flow_html = render_simple_mermaid_flowchart(diagram)
        if flow_html is not None:
            parts.append(f"\n\n{flow_html}\n\n")
            last = match.end()
            continue
        if puppeteer_config is None:
            chrome = find_chrome()
            puppeteer_config = work_dir / "puppeteer-config.json"
            puppeteer_config.write_text(
                "{\n"
                f'  "executablePath": "{chrome}",\n'
                '  "headless": true,\n'
                '  "args": ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]\n'
                "}\n",
                encoding="utf-8",
            )
        stem = f"mermaid_{count:03d}"
        mmd_path = work_dir / f"{stem}.mmd"
        image_path = work_dir / f"{stem}.png"
        mmd_path.write_text(normalize_mermaid(diagram) + "\n", encoding="utf-8")
        try:
            subprocess.run(
                [
                    "npx",
                    "--yes",
                    "@mermaid-js/mermaid-cli",
                    "-q",
                    "-i",
                    str(mmd_path),
                    "-o",
                    str(image_path),
                    "-b",
                    "white",
                    "-s",
                    "2",
                    "-p",
                    str(puppeteer_config),
                ],
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            parts.append(f"\n\n```mermaid\n{diagram}\n```\n\n")
            last = match.end()
            continue
        parts.append(f"![Mermaid diagram {count}]({image_path.as_uri()})")
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def find_chrome() -> str:
    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("Chrome/Chromium not found; Mermaid rendering requires a browser")


def html_document(body: str, title: str) -> str:
    pygments_css = HtmlFormatter(style="default").get_style_defs(".codehilite")
    escaped_title = html.escape(title or "Video document")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>{MOBILE_CSS}</style>
  <style>{pygments_css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    input_md = Path(args.input_md).expanduser().resolve()
    output_pdf = Path(args.output_pdf).expanduser().resolve()
    if not input_md.is_file():
        raise SystemExit(f"Markdown file not found: {input_md}")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    text = input_md.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="video-doc-mermaid-") as mermaid_dir:
        text = render_mermaid_blocks(text, Path(mermaid_dir))
        body = render_markdown(text)
        body = wrap_final_images(body)
        HTML(string=html_document(body, args.title), base_url=str(input_md.parent)).write_pdf(output_pdf)
    if not output_pdf.is_file() or output_pdf.stat().st_size == 0:
        raise SystemExit(f"PDF output not written: {output_pdf}")
    print(output_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
