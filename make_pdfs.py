# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nbclient>=0.10",
#     "nbconvert>=7.16",
#     "playwright>=1.45",
# ]
# ///
"""Convert the course notebooks into A4 PDF handouts.

Each notebook is executed in a temporary copy of this folder with the
kernel from `.venv`, rendered to HTML with a print layout and printed to
PDF with Chrome. The notebooks in this folder are not changed.

Usage:
    uv run make_pdfs.py                          # all notebooks -> pdf/
    uv run make_pdfs.py 03_pandas_intro.ipynb    # selected notebooks
    uv run make_pdfs.py --no-execute             # keep the stored outputs

Chrome or Edge is used if installed, otherwise Playwright downloads
Chromium once. Fonts (IBM Plex) and formulas (MathJax) are loaded from
the internet; without a connection the PDFs use system fonts.

Not printed: the Colab/Kaggle badges, the Colab setup cell, error
messages and warnings (stderr). The terminal lists errors and warnings
instead; `--show-errors` prints them in the PDFs as well. Cells tagged
`remove-cell`, `remove-input` or `remove-output` are left out accordingly.
"""

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import AsyncKernelManager
from nbclient import NotebookClient
from nbconvert.filters.ansi import ansi2html
from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, get_lexer_by_name
from pygments.util import ClassNotFound

ROOT = Path(__file__).resolve().parent
COURSE = 'Python for Actuaries'
KERNEL_NAME = 'py4lju-pdf'
CELL_TIMEOUT = 600  # seconds

# Page layout in mm
A4_WIDTH, A4_HEIGHT = 210, 297
MARGIN_TOP, MARGIN_SIDE, MARGIN_BOTTOM = 18, 20, 20

# Sharper plots: PNGs with twice the resolution at the same size
SETUP_CELL = "%matplotlib inline\n%config InlineBackend.figure_formats = ['retina']"

BADGE = re.compile(
    r'\[!\[[^\]]*\]\([^)]*(?:colab-badge|open-in-kaggle)[^)]*\)\]\([^)]*\)'
    r'|!\[[^\]]*\]\([^)]*(?:colab-badge|open-in-kaggle)[^)]*\)'
)
HEADING = re.compile(r'<h([1-6])([^>]*)>(.*?)</h\1>', re.S)
EXERCISE = re.compile(r'^(?:Exercises?|Tasks?|Aufgaben?|Übung(?:en)?)\b')
TABLE_HEADER = re.compile(r'(<th[^>]*>)(.*?)(</th>)', re.S)
TAG = re.compile(r'<[^>]+>')

LEXER = PythonLexer(stripnl=False, ensurenl=False)
FORMATTER = HtmlFormatter(nowrap=True)

# Static font files: Chrome embeds them as TrueType (variable fonts become Type 3)
FONT_CDN = 'https://cdn.jsdelivr.net/npm/@fontsource'
FONT_VERSION = '5.2.7'
FONT_FACES = [
    ('IBM Plex Sans', 'ibm-plex-sans', 400, 'normal'),
    ('IBM Plex Sans', 'ibm-plex-sans', 400, 'italic'),
    ('IBM Plex Sans', 'ibm-plex-sans', 600, 'normal'),
    ('IBM Plex Sans', 'ibm-plex-sans', 600, 'italic'),
    ('IBM Plex Mono', 'ibm-plex-mono', 400, 'normal'),
    ('IBM Plex Mono', 'ibm-plex-mono', 400, 'italic'),
    ('IBM Plex Mono', 'ibm-plex-mono', 600, 'normal'),
]
FONT_SUBSETS = {
    'latin': 'U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, '
             'U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD',
    'latin-ext': 'U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+0304, U+0308, U+0329, '
                 'U+1D00-1DBF, U+1E00-1E9F, U+1EF2-1EFF, U+2020, U+20A0-20AB, U+20AD-20C0, U+2113, '
                 'U+2C60-2C7F, U+A720-A7FF',
}

MATHJAX = r"""<script>
window.MathJax = {
  tex: {inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']], processEscapes: true},
  svg: {fontCache: 'global'}
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>"""

WAIT_FOR_MATH = (
    'window.MathJax && MathJax.startup && MathJax.startup.promise'
    ' ? MathJax.startup.promise.then(() => true) : false'
)
WAIT_FOR_FONTS = 'document.fonts.ready.then(() => true)'
FONTS_LOADED = (
    "[...document.fonts].some(f => f.family.replace(/[\"']/g, '') === 'IBM Plex Sans'"
    " && f.status === 'loaded')"
)

# Runs in the browser before printing. Wide printed tables (DataFrames) are
# scaled down to the page width, at most to half size; other long lines wrap.
# HTML tables are always scaled to fit.
FIT_SCRIPT = r"""
window.fitBlocks = function () {
  const MIN_ZOOM = 0.5;
  const widest = lines => lines.reduce((max, line) => Math.max(max, line.length), 0);
  for (const pre of document.querySelectorAll('pre.output-text')) {
    pre.style.whiteSpace = 'pre';
    const style = getComputedStyle(pre);
    const inner = pre.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const probe = document.createElement('span');
    probe.textContent = '0'.repeat(100);
    pre.appendChild(probe);
    const charWidth = probe.getBoundingClientRect().width / 100;
    probe.remove();
    const capacity = Math.floor(inner / charWidth);
    const lines = pre.textContent.split('\n');
    if (widest(lines) <= capacity) continue;
    const tabular = lines.filter(line => line.length > capacity && /\S {2,}\S/.test(line));
    if (tabular.length) {
      const zoom = 0.99 * capacity / widest(tabular);
      if (zoom >= MIN_ZOOM) pre.style.zoom = zoom;
    }
    pre.style.whiteSpace = 'pre-wrap';
  }
  for (const table of document.querySelectorAll('.md table, .output-html table')) {
    const available = table.parentElement.clientWidth;
    const width = table.getBoundingClientRect().width;
    if (width > available) table.style.zoom = 0.99 * available / width;
  }
  // Text cells taller than half a page may break between code and output,
  // cells with plots only if they do not fit on one page
  const pageHeight = parseFloat(document.body.dataset.pageHeight) * 96 / 25.4;
  for (const cell of document.querySelectorAll('.cell')) {
    const height = cell.getBoundingClientRect().height;
    const limit = cell.querySelector('.output-image') ? 0.9 * pageHeight : pageHeight / 2;
    if (height > limit) cell.classList.add('breakable');
  }
};
"""

CSS = """
:root {
  --sans: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
  --mono: 'IBM Plex Mono', Menlo, Consolas, monospace;
  --ink: #1c2530;
  --muted: #5a6573;
  --rule: #d9dfe6;
  --accent: #1d5c96;
  --code-bg: #f4f6f9;
  --exercise: #c07a0c;
  --exercise-bg: #fcf6ea;
}
* { box-sizing: border-box; }
html {
  font-size: 10.5pt;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
body {
  margin: 0;
  background: #fff;
  color: var(--ink);
  font-family: var(--sans);
  line-height: 1.5;
  font-kerning: normal;
}

/* Title block */
.titleblock { margin: 0 0 18pt; padding-bottom: 14pt; border-bottom: 0.75pt solid var(--rule); }
.titleblock::before {
  content: "";
  display: block;
  width: 22mm;
  height: 3pt;
  margin-bottom: 12pt;
  background: var(--accent);
}
.eyebrow {
  margin: 0 0 5pt;
  color: var(--accent);
  font-size: 8pt;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
h1.title {
  margin: 0;
  font-size: 25pt;
  font-weight: 600;
  line-height: 1.15;
  letter-spacing: -0.01em;
  text-wrap: balance;
}
.subtitle { margin: 6pt 0 0; color: var(--muted); font-size: 13pt; }

/* Markdown */
h1, h2, h3, h4, h5, h6 {
  font-weight: 600;
  line-height: 1.25;
  text-wrap: balance;
  break-after: avoid;
  break-inside: avoid;
}
.md h1 {
  margin: 22pt 0 8pt;
  padding-bottom: 4pt;
  border-bottom: 1.25pt solid var(--accent);
  font-size: 18pt;
}
.md h2 { margin: 20pt 0 6pt; color: var(--accent); font-size: 14.5pt; }
.md h3 { margin: 16pt 0 5pt; font-size: 12pt; }
.md h4, .md h5, .md h6 { margin: 12pt 0 4pt; font-size: 10.5pt; }
.md h2.exercise, .md h3.exercise, .md h4.exercise {
  padding: 5pt 9pt;
  border-left: 3pt solid var(--exercise);
  border-radius: 0 3pt 3pt 0;
  background: var(--exercise-bg);
  color: var(--ink);
}
.md .exercise em { font-style: normal; }
.md { margin: 6pt 0; }
.md:has(+ .cell) > p:last-child { break-after: avoid; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; text-wrap: pretty; }
ul, ol { margin: 0 0 7pt; padding-left: 16pt; }
li { margin: 1.5pt 0; }
li:first-child { break-after: avoid; }
li:last-child { break-before: avoid; }
li > p { margin: 0; }
li > ul, li > ol { margin: 1.5pt 0 0; }
li::marker { color: var(--accent); }
a { color: var(--accent); text-decoration: none; }
strong { font-weight: 600; }
code {
  padding: 0.05em 0.3em;
  border-radius: 2pt;
  background: #eceff3;
  font-family: var(--mono);
  font-size: 0.86em;
}
blockquote {
  margin: 8pt 0;
  padding: 2pt 0 2pt 11pt;
  border-left: 3pt solid var(--rule);
  color: var(--muted);
}
hr { margin: 14pt 0; border: 0; border-top: 0.75pt solid var(--rule); }
img { max-width: 100%; height: auto; }
.md img { display: block; margin: 8pt auto; }
.md pre {
  margin: 0 0 7pt;
  padding: 6pt 9pt;
  border-radius: 3pt;
  background: var(--code-bg);
  break-inside: avoid;
}
.md .highlight pre { margin: 0; }
.md .highlight { margin: 0 0 7pt; }

/* Tables */
table {
  margin: 6pt 0 10pt;
  border: 0;
  border-collapse: collapse;
  font-size: 8.8pt;
  line-height: 1.35;
  font-variant-numeric: tabular-nums;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td {
  padding: 3pt 8pt;
  border: 0;
  border-bottom: 0.5pt solid var(--rule);
  text-align: left;
  vertical-align: top;
}
thead th { border-bottom: 1pt solid var(--ink); font-weight: 600; }
tbody tr:nth-child(even) { background: #f7f9fb; }
.output-html table { margin: 0; font-size: 8pt; }
.output-html th, .output-html td { padding: 2.5pt 6pt; }
table.dataframe td, table.dataframe thead th { text-align: right; }
table.dataframe tbody th { font-weight: 600; }

/* Code cells */
.cell { margin: 8pt 0 11pt; break-inside: avoid; }
.cell.breakable { break-inside: auto; }
.input, .output-image { break-inside: avoid; }
pre {
  margin: 0;
  font-family: var(--mono);
  font-size: 8.3pt;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  tab-size: 4;
}
pre code { padding: 0; background: none; font-size: inherit; }
.input {
  padding: 6pt 9pt;
  border-left: 2.5pt solid var(--accent);
  border-radius: 0 3pt 3pt 0;
  background: var(--code-bg);
}
.input .line { display: block; }
.output { margin-top: 5pt; }
.output > * + * { margin-top: 5pt; }
.output-text {
  padding: 1pt 0 1pt 9pt;
  border-left: 2.5pt solid var(--rule);
  overflow: hidden;
  color: #27313c;
  font-size: 8.1pt;
}
.output-error { border-left-color: #d64545; background: #fdf3f3; }
.output-image { text-align: center; }
.output-image img { display: block; max-width: 78%; margin: 0 auto; }

/* Syntax highlighting (Pygments token classes) */
pre .c, pre .c1, pre .ch, pre .cm, pre .cp, pre .cpf, pre .cs { color: #6e7781; font-style: italic; }
pre .k, pre .kc, pre .kd, pre .kn, pre .kp, pre .kr, pre .kt, pre .ow { color: #cf222e; }
pre .s, pre .s1, pre .s2, pre .sa, pre .sb, pre .sc, pre .sd, pre .dl, pre .sh, pre .sr, pre .ss, pre .sx { color: #0a3069; }
pre .se, pre .si { color: #0550ae; }
pre .m, pre .mb, pre .mf, pre .mh, pre .mi, pre .mo, pre .il { color: #0550ae; }
pre .nb, pre .bp { color: #0550ae; }
pre .nf, pre .fm, pre .nd { color: #8250df; }
pre .nc, pre .nn, pre .ne { color: #953800; }

/* ANSI colours in tracebacks */
.ansi-red-fg { color: #c62828; }
.ansi-green-fg { color: #2e7d32; }
.ansi-yellow-fg { color: #9a6700; }
.ansi-blue-fg { color: #1d5c96; }
.ansi-magenta-fg { color: #8e24aa; }
.ansi-cyan-fg { color: #00838f; }
.ansi-bold { font-weight: 600; }
"""


@dataclass
class Job:
    path: Path
    html_path: Path = None
    math: bool = False
    problems: list = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the course notebooks into A4 PDF handouts.')
    parser.add_argument('notebooks', nargs='*', type=Path,
                        help='notebooks to convert (default: all in this folder)')
    parser.add_argument('-o', '--outdir', type=Path, default=ROOT / 'pdf',
                        help='output folder (default: pdf/)')
    parser.add_argument('--no-execute', action='store_true',
                        help='do not execute, keep the stored outputs')
    parser.add_argument('--show-errors', action='store_true',
                        help='print error messages and warnings in the PDFs')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def find_python():
    for candidate in (ROOT / '.venv' / 'bin' / 'python', ROOT / '.venv' / 'Scripts' / 'python.exe'):
        if candidate.exists():
            return candidate
    sys.exit('No .venv found. Create it first: uv venv && uv pip install -r requirements.txt')


def write_kernel_spec(folder, python):
    """Kernel spec that starts the notebooks with the Python from .venv."""
    spec = folder / KERNEL_NAME
    spec.mkdir(parents=True)
    (spec / 'kernel.json').write_text(json.dumps({
        'argv': [str(python), '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Python (.venv)',
        'language': 'python',
        'env': {'PYDEVD_DISABLE_FILE_VALIDATION': '1'},
    }))
    return folder


def copy_course(target, outdir):
    """Copy this folder without hidden folders, _references and the PDFs."""
    def ignore(folder, names):
        if Path(folder) == ROOT:
            return [n for n in names if n.startswith(('.', '_')) or ROOT / n == outdir]
        return [n for n in names if n.startswith('.') or n == '__pycache__']
    shutil.copytree(ROOT, target, ignore=ignore)


def execute(nb, workdir, kernel_dir):
    nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP_CELL))
    km = AsyncKernelManager(
        kernel_name=KERNEL_NAME,
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]),
    )
    client = NotebookClient(
        nb,
        km=km,
        timeout=CELL_TIMEOUT,
        allow_errors=True,
        extra_arguments=['--HistoryManager.hist_file=:memory:', '--IPKernelApp.log_level=ERROR'],
        resources={'metadata': {'path': str(workdir)}},
    )
    try:
        client.execute(cleanup_kc=True)
    finally:
        setup = nb.cells.pop(0)
    if any(out.output_type == 'error' for out in setup.outputs):
        return ['setup: retina plots could not be enabled']
    return []


def first_line(text):
    return next((line.strip() for line in text.splitlines() if line.strip()), '')


def shorten(text, width):
    return text if len(text) <= width else text[:width - 1] + '…'


def find_problems(nb):
    """Errors and warnings in the outputs, one line each."""
    problems = []
    for number, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != 'code':
            continue
        where = f'cell {number} ({shorten(first_line(cell.source), 40)})'
        for out in merge_streams(cell.get('outputs', [])):
            if out['output_type'] == 'error':
                problems.append(f"{where}: {out['ename']}: {shorten(first_line(out['evalue']), 90)}")
            elif out['output_type'] == 'stream' and out['name'] == 'stderr':
                problems.append(f"{where}: stderr: {shorten(first_line(out['text']), 90)}")
    return problems


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def split_title(source):
    """Title (`# ...`), subtitle (a directly following `## ...`) and the rest."""
    lines = source.strip().split('\n')
    if not lines[0].startswith('# '):
        return None, None, source
    title = re.sub(r'\s+#+\s*$', '', lines[0][2:]).strip()
    subtitle = None
    if len(lines) > 1 and lines[1].startswith('## '):
        subtitle = re.sub(r'\s+#+\s*$', '', lines[1][3:]).strip()
        lines = lines[1:]
    return title, subtitle, '\n'.join(lines[1:])


def inline_markdown(text):
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', html.escape(text))


def plain_text(text):
    return text.replace('`', '')


def mark_exercise(match):
    level, attrs, inner = match.groups()
    if EXERCISE.match(html.unescape(TAG.sub('', inner)).strip()):
        attrs = ' class="exercise"' + attrs
    return f'<h{level}{attrs}>{inner}</h{level}>'


class Renderer(IPythonRenderer):
    """Shows code blocks in an unknown language (e.g. ```plaintext) as plain text.

    nbconvert would otherwise print the language name as the first line.
    """

    def block_code(self, code, info=None):
        lang = (info or '').strip().split(maxsplit=1)[:1]
        if lang and not lang[0].startswith('mermaid'):
            try:
                get_lexer_by_name(lang[0])
            except ClassNotFound:
                info = None
        return super().block_code(code, info)


def render_markdown(source, attachments, folder):
    renderer = Renderer(
        escape=False,
        exclude_anchor_links=True,
        embed_images=True,
        path=str(folder),
        attachments=attachments,
    )
    return HEADING.sub(mark_exercise, MarkdownWithMath(renderer=renderer).render(source))


def strip_blank_lines(text):
    return re.sub(r'^(?:[ \t]*\n)+', '', text).rstrip()


def render_code(source):
    """Highlighted code, one block per line with a hanging indent for wrapped lines."""
    lines = highlight(source.expandtabs(4), LEXER, FORMATTER).rstrip('\n').split('\n')
    spans = []
    for line in lines:
        text = html.unescape(TAG.sub('', line))
        hang = len(text) - len(text.lstrip(' ')) + 4
        style = f'padding-left:{hang}ch;text-indent:-{hang}ch'
        spans.append(f'<span class="line" style="{style}">{line or " "}</span>')
    return f'<pre class="input">{"".join(spans)}</pre>'


def merge_streams(outputs):
    """Join consecutive stream outputs of the same name."""
    merged = []
    for out in outputs:
        if (out['output_type'] == 'stream' and merged and merged[-1]['output_type'] == 'stream'
                and merged[-1]['name'] == out['name']):
            merged[-1] = {**merged[-1], 'text': merged[-1]['text'] + out['text']}
        else:
            merged.append(out)
    return merged


def clean_text(text):
    text = re.sub(r'[^\n]*\r(?!\n)', '', text.replace('\r\n', '\n'))
    return strip_blank_lines(text.expandtabs(8))


def text_block(text, extra_class=''):
    return f'<pre class="output-text{extra_class}">{ansi2html(text)}</pre>' if text else ''


def soften_headers(table_html):
    """Allow long column names like `number_of_prior_claims` to wrap after `_`."""
    return TABLE_HEADER.sub(lambda m: m[1] + m[2].replace('_', '_<wbr>') + m[3], table_html)


def render_data(out, folder):
    data = out['data']
    meta = out.get('metadata', {})
    if 'image/svg+xml' in data:
        return f'<div class="output-image">{data["image/svg+xml"]}</div>'
    for mime in ('image/png', 'image/jpeg'):
        if mime in data:
            size = meta.get(mime, {})
            attrs = ''.join(f' {key}="{round(float(size[key]))}"'
                            for key in ('width', 'height') if key in size)
            src = f'data:{mime};base64,{"".join(data[mime].split())}'
            return f'<div class="output-image"><img src="{src}"{attrs} alt=""></div>'
    if 'text/html' in data:
        return f'<div class="output-html">{soften_headers(data["text/html"])}</div>'
    if 'text/markdown' in data:
        return f'<div class="output-html md">{render_markdown(data["text/markdown"], {}, folder)}</div>'
    if 'text/latex' in data:
        return f'<div class="output-html">{html.escape(data["text/latex"])}</div>'
    if 'text/plain' in data:
        return text_block(clean_text(data['text/plain']))
    return ''


def render_outputs(outputs, folder, show_errors):
    parts = []
    for out in merge_streams(outputs):
        kind = out['output_type']
        if kind == 'stream':
            stderr = out['name'] == 'stderr'
            if stderr and not show_errors:
                continue
            parts.append(text_block(clean_text(out['text']), ' output-error' if stderr else ''))
        elif kind in ('execute_result', 'display_data'):
            parts.append(render_data(out, folder))
        elif kind == 'error' and show_errors:
            parts.append(text_block(clean_text('\n'.join(out['traceback'])), ' output-error'))
    parts = [part for part in parts if part]
    return f'<div class="output">{"".join(parts)}</div>' if parts else ''


def is_colab_setup(cell):
    return cell.cell_type == 'code' and "'google.colab' in sys.modules" in cell.source


def describe(path, title):
    """Eyebrow above the title and the footer text."""
    course = [] if title.startswith(COURSE) else [COURSE]
    number = re.match(r'(\d+[a-z]?)_', path.stem)
    variant = next((v.capitalize() for v in ('exercise', 'solution') if path.stem.endswith('_' + v)), None)
    eyebrow = course + ([f'Notebook {number[1]}'] if number else []) + ([variant] if variant else [])
    footer = course + [plain_text(title)] + ([variant] if variant else [])
    return ' · '.join(eyebrow), ' · '.join(footer)


def css_string(text):
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def font_faces():
    rules = []
    for family, package, weight, style in FONT_FACES:
        for subset, unicode_range in FONT_SUBSETS.items():
            url = f'{FONT_CDN}/{package}@{FONT_VERSION}/files/{package}-{subset}-{weight}-{style}.woff2'
            rules.append(
                f"@font-face {{ font-family: '{family}'; font-style: {style}; font-weight: {weight}; "
                f"font-display: block; src: url('{url}') format('woff2'); unicode-range: {unicode_range}; }}"
            )
    return '\n'.join(rules)


def page_css(footer):
    font = "7.5pt 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif"
    return f"""
@page {{
  size: A4;
  margin: {MARGIN_TOP}mm {MARGIN_SIDE}mm {MARGIN_BOTTOM}mm;
  @bottom-left {{ content: {css_string(footer)}; font: {font}; color: #7b8794; }}
  @bottom-right {{ content: counter(page) " / " counter(pages); font: {font}; color: #7b8794; }}
}}
body {{ width: {A4_WIDTH - 2 * MARGIN_SIDE}mm; }}"""


def render_notebook(nb, path, show_errors):
    """Return the HTML page and whether it contains formulas."""
    folder = path.parent
    cells = list(nb.cells)
    title, subtitle = None, None
    if cells and cells[0].cell_type == 'markdown':
        title, subtitle, rest = split_title(BADGE.sub('', cells[0].source))
        if title:
            cells[0] = nbformat.from_dict({**cells[0], 'source': rest})
    title = title or path.stem
    math = False
    body = []
    for cell in cells:
        tags = cell.metadata.get('tags', [])
        if 'remove-cell' in tags or cell.cell_type == 'raw' or is_colab_setup(cell):
            continue
        if cell.cell_type == 'markdown':
            source = BADGE.sub('', cell.source)
            if source.strip():
                math = math or '$' in source or '\\(' in source or '\\[' in source
                text = render_markdown(source, cell.get('attachments'), folder)
                body.append(f'<div class="md">{text}</div>')
            continue
        source = strip_blank_lines(cell.source)
        code = render_code(source) if source and 'remove-input' not in tags else ''
        outputs = '' if 'remove-output' in tags else render_outputs(cell.outputs, folder, show_errors)
        math = math or any('text/latex' in out.get('data', {}) for out in cell.outputs)
        if code or outputs:
            body.append(f'<div class="cell">{code}{outputs}</div>')

    eyebrow, footer = describe(path, title)
    subtitle_html = f'<p class="subtitle">{inline_markdown(subtitle)}</p>' if subtitle else ''
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(footer)}</title>
<style>{font_faces()}{CSS}{page_css(footer)}</style>
<script>{FIT_SCRIPT}</script>
{MATHJAX if math else ''}
</head>
<body data-page-height="{A4_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM}">
<header class="titleblock">
<p class="eyebrow">{html.escape(eyebrow)}</p>
<h1 class="title">{inline_markdown(title)}</h1>
{subtitle_html}
</header>
<main>
{chr(10).join(body)}
</main>
</body>
</html>
"""
    return page, math


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def launch_browser(playwright):
    for channel in ('chrome', 'msedge', None):
        try:
            return playwright.chromium.launch(channel=channel)
        except PlaywrightError:
            pass
    print('No Chrome or Edge found, downloading Chromium for Playwright (one time) ...')
    subprocess.run([sys.executable, '-m', 'playwright', 'install', 'chromium'], check=True)
    return playwright.chromium.launch()


def count_pages(pdf_path):
    return len(re.findall(rb'/Type\s*/Page[^s]', pdf_path.read_bytes()))


def print_pdfs(jobs, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fonts_missing = False
    with sync_playwright() as playwright:
        browser = launch_browser(playwright)
        page = browser.new_page()
        page.emulate_media(media='print')
        for job in jobs:
            page.goto(job.html_path.as_uri(), wait_until='networkidle', timeout=120_000)
            math_missing = job.math and not page.evaluate(WAIT_FOR_MATH)
            page.evaluate(WAIT_FOR_FONTS)
            fonts_missing = fonts_missing or not page.evaluate(FONTS_LOADED)
            page.evaluate('fitBlocks()')
            pdf_path = outdir / f'{job.path.stem}.pdf'
            page.pdf(
                path=str(pdf_path),
                format='A4',
                print_background=True,
                prefer_css_page_size=True,
                outline=True,
                tagged=True,
            )
            print(f'  {pdf_path.name:<40} {count_pages(pdf_path):>3} pages')
            if math_missing:
                job.problems.append('MathJax could not be loaded, formulas show as LaTeX source')
                print(f'      ! {job.problems[-1]}')
        browser.close()
    if fonts_missing:
        print('  ! IBM Plex could not be loaded (offline?), the PDFs use system fonts')


def main():
    args = parse_args()
    notebooks = [p.resolve() for p in args.notebooks] or sorted(ROOT.glob('*.ipynb'))
    missing = [str(p) for p in notebooks if not p.is_file()]
    if missing:
        sys.exit(f'Not found: {", ".join(missing)}')
    outdir = args.outdir.resolve()

    jobs, failed = [], []
    workdir = kernel_dir = None
    with tempfile.TemporaryDirectory(prefix='py4lju-pdf-') as tmp:
        tmp = Path(tmp)
        (tmp / 'html').mkdir()
        if args.no_execute:
            print(f'Rendering {len(notebooks)} notebooks with their stored outputs')
        else:
            python = find_python()
            kernel_dir = write_kernel_spec(tmp / 'kernels', python)
            workdir = tmp / 'work'
            copy_course(workdir, outdir)
            print(f'Executing {len(notebooks)} notebooks with {python}')

        for path in notebooks:
            job = Job(path)
            start = time.perf_counter()
            try:
                nb = nbformat.read(path, as_version=4)
                if not args.no_execute:
                    job.problems += execute(nb, workdir, kernel_dir)
                job.problems += find_problems(nb)
                page, job.math = render_notebook(nb, path, args.show_errors)
            except Exception as error:
                print(f'  {path.name:<40} FAILED: {type(error).__name__}: {error}')
                failed.append(path.name)
                continue
            job.html_path = tmp / 'html' / f'{path.stem}.html'
            job.html_path.write_text(page, encoding='utf-8')
            jobs.append(job)
            print(f'  {path.name:<40} {time.perf_counter() - start:5.1f} s')
            for problem in job.problems:
                print(f'      ! {problem}')

        if jobs:
            print(f'Printing PDFs to {outdir}')
            print_pdfs(jobs, outdir)

    with_problems = [job.path.name for job in jobs if job.problems]
    print(f'Done: {len(jobs)} PDFs in {outdir}')
    if with_problems:
        hidden = '' if args.show_errors else ' (not printed in the PDFs)'
        print(f'Errors or warnings{hidden} in: {", ".join(with_problems)}')
    if failed:
        print(f'Failed: {", ".join(failed)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
