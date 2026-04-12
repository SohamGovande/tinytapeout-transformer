#!/usr/bin/env python3

from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent

PALETTE = {
    "bg": "#fbfaf7",
    "paper": "#fffdfa",
    "ink": "#26212e",
    "muted": "#655f72",
    "line": "#bdb5c8",
    "stone": "#d7d3cb",
    "lavender": "#d8d5ff",
    "peach": "#ffd7ca",
    "butter": "#fff0bc",
    "mint": "#d8f2d7",
    "orchid": "#efdbff",
    "teal": "#6d9f9b",
    "coral": "#cc8b73",
    "sage": "#7c9a73",
}

FONT = '"Iowan Old Style", "Palatino Linotype", "Book Antiqua", "Times New Roman", serif'


def header(width: int, height: int, title: str, desc: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">{escape(desc)}</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8.5" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5 L0,10 z" fill="{PALETTE["ink"]}" />
    </marker>
    <marker id="arrow-muted" markerWidth="10" markerHeight="10" refX="8.5" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5 L0,10 z" fill="{PALETTE["muted"]}" />
    </marker>
    <style>
      text {{
        font-family: {FONT};
        fill: {PALETTE["ink"]};
      }}
      .title {{
        font-size: 28px;
        font-weight: 700;
      }}
      .subtitle {{
        font-size: 18px;
        font-style: italic;
        fill: {PALETTE["muted"]};
      }}
      .label {{
        font-size: 18px;
        font-weight: 700;
      }}
      .body {{
        font-size: 16px;
        fill: {PALETTE["muted"]};
      }}
      .token {{
        font-size: 20px;
        font-weight: 700;
      }}
      .tiny {{
        font-size: 14px;
        fill: {PALETTE["muted"]};
      }}
    </style>
  </defs>
  <rect width="{width}" height="{height}" fill="{PALETTE["bg"]}" />
"""


def footer() -> str:
    return "</svg>\n"


def rect(x, y, w, h, fill, stroke=PALETTE["line"], stroke_width=2, rx=14):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />\n'


def line(x1, y1, x2, y2, color=PALETTE["ink"], width=3, marker_end=None, dash=None):
    attrs = [
        f'x1="{x1}"',
        f'y1="{y1}"',
        f'x2="{x2}"',
        f'y2="{y2}"',
        f'stroke="{color}"',
        f'stroke-width="{width}"',
        'stroke-linecap="round"',
    ]
    if marker_end:
        attrs.append(f'marker-end="url(#{marker_end})"')
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    return f"<line {' '.join(attrs)} />\n"


def text(x, y, lines, klass="body", anchor="start"):
    if isinstance(lines, str):
        lines = [lines]
    out = [f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="hanging" class="{klass}">']
    for idx, line_text in enumerate(lines):
        dy = "0" if idx == 0 else "1.25em"
        out.append(f'<tspan x="{x}" dy="{dy}">{escape(line_text)}</tspan>')
    out.append("</text>\n")
    return "".join(out)


def save(name: str, width: int, height: int, title: str, desc: str, body: str) -> None:
    (OUT_DIR / name).write_text(header(width, height, title, desc) + body + footer(), encoding="utf-8")


def pe_cell(x: int, y: int, w: int, h: int, fill: str, label_text: str, formula: Optional[str] = None) -> str:
    body = rect(x, y, w, h, fill, stroke=PALETTE["ink"], stroke_width=1.8, rx=16)
    body += text(x + w / 2, y + 18, label_text, klass="label", anchor="middle")
    body += text(x + w / 2, y + 48, "MAC", klass="subtitle", anchor="middle")
    if formula:
        body += text(x + w / 2, y + 82, formula, klass="tiny", anchor="middle")
    return body


def token_box(x: int, y: int, label_text: str, fill: str, stroke: str = PALETTE["ink"]) -> str:
    body = rect(x, y, 62, 40, fill, stroke=stroke, stroke_width=1.6, rx=10)
    body += text(x + 31, y + 9, label_text, klass="token", anchor="middle")
    return body


def small_grid(x: int, y: int, cell_fill: str = PALETTE["paper"]) -> str:
    cell_w = 62
    cell_h = 40
    gap_x = 10
    gap_y = 14
    outer_w = cell_w * 2 + gap_x + 24
    outer_h = cell_h * 2 + gap_y + 24
    body = rect(x, y, outer_w, outer_h, PALETTE["paper"], stroke=PALETTE["line"], stroke_width=1.6, rx=18)
    for row in range(2):
        for col in range(2):
            cx = x + 12 + col * (cell_w + gap_x)
            cy = y + 12 + row * (cell_h + gap_y)
            body += rect(cx, cy, cell_w, cell_h, cell_fill, stroke=PALETTE["line"], stroke_width=1.2, rx=10)
    return body


def wavefront_panel(x: int, y: int, title_text: str, tokens: list[tuple[int, int, str, str]]) -> str:
    body = rect(x, y, 274, 194, PALETTE["paper"], stroke=PALETTE["line"], stroke_width=1.5, rx=18)
    body += text(x + 20, y + 16, title_text, klass="label")
    body += small_grid(x + 20, y + 40)
    for row, col, token, fill in tokens:
        tx = x + 32 + col * 72
        ty = y + 52 + row * 54
        body += token_box(tx, ty, token, fill)
    return body


def a_wavefront() -> None:
    width, height = 1290, 332
    body = []
    body.append(text(40, 28, "A wavefront", klass="title"))
    body.append(text(40, 62, "The controller injects rows from the left in a skewed pattern: FEED0, FEED1, FLUSH0, then zeros.", klass="subtitle"))
    body.append(wavefront_panel(40, 98, "FEED0", [(0, 0, "a₀₀", PALETTE["stone"])]))
    body.append(wavefront_panel(356, 98, "FEED1", [(0, 0, "a₀₁", PALETTE["lavender"]), (0, 1, "a₀₀", PALETTE["stone"]), (1, 0, "a₁₀", PALETTE["peach"])]))
    body.append(wavefront_panel(672, 98, "FLUSH0", [(0, 1, "a₀₁", PALETTE["lavender"]), (1, 0, "a₁₁", PALETTE["butter"]), (1, 1, "a₁₀", PALETTE["peach"])]))
    body.append(wavefront_panel(988, 98, "FLUSH1", [(1, 1, "a₁₁", PALETTE["butter"])]))
    body.append(line(314, 195, 356, 195, color=PALETTE["teal"], width=3.2, marker_end="arrow"))
    body.append(line(630, 195, 672, 195, color=PALETTE["teal"], width=3.2, marker_end="arrow"))
    body.append(line(946, 195, 988, 195, color=PALETTE["teal"], width=3.2, marker_end="arrow"))
    save(
        "systolic-a-wavefront.svg",
        width,
        height,
        "A operand wavefront through the 2 by 2 array",
        "Four panels show how a00, a01, a10, and a11 move across the array from left to right.",
        "".join(body),
    )


def b_wavefront() -> None:
    width, height = 1290, 332
    body = []
    body.append(text(40, 28, "B wavefront", klass="title"))
    body.append(text(40, 62, "Columns arrive from the top with the complementary skew, so the right pairs meet in each PE.", klass="subtitle"))
    body.append(wavefront_panel(40, 98, "FEED0", [(0, 0, "b₀₀", PALETTE["stone"])]))
    body.append(wavefront_panel(356, 98, "FEED1", [(0, 0, "b₁₀", PALETTE["peach"]), (0, 1, "b₀₁", PALETTE["lavender"]), (1, 0, "b₀₀", PALETTE["stone"])]))
    body.append(wavefront_panel(672, 98, "FLUSH0", [(0, 1, "b₁₁", PALETTE["butter"]), (1, 0, "b₁₀", PALETTE["peach"]), (1, 1, "b₀₁", PALETTE["lavender"])]))
    body.append(wavefront_panel(988, 98, "FLUSH1", [(1, 1, "b₁₁", PALETTE["butter"])]))
    body.append(line(314, 195, 356, 195, color=PALETTE["coral"], width=3.2, marker_end="arrow"))
    body.append(line(630, 195, 672, 195, color=PALETTE["coral"], width=3.2, marker_end="arrow"))
    body.append(line(946, 195, 988, 195, color=PALETTE["coral"], width=3.2, marker_end="arrow"))
    save(
        "systolic-b-wavefront.svg",
        width,
        height,
        "B operand wavefront through the 2 by 2 array",
        "Four panels show how b00, b01, b10, and b11 move downward through the array.",
        "".join(body),
    )


def output_formulas() -> None:
    width, height = 980, 420
    body = []
    body.append(text(54, 32, "Dot products per PE", klass="title"))
    body.append(text(54, 66, "Each processing element owns exactly one output entry of C.", klass="subtitle"))
    body.append(pe_cell(96, 116, 350, 106, PALETTE["stone"], "c₀₀", "a₀₀·b₀₀ + a₀₁·b₁₀"))
    body.append(pe_cell(524, 116, 350, 106, PALETTE["lavender"], "c₀₁", "a₀₀·b₀₁ + a₀₁·b₁₁"))
    body.append(pe_cell(96, 252, 350, 106, PALETTE["peach"], "c₁₀", "a₁₀·b₀₀ + a₁₁·b₁₀"))
    body.append(pe_cell(524, 252, 350, 106, PALETTE["butter"], "c₁₁", "a₁₀·b₀₁ + a₁₁·b₁₁"))
    save(
        "systolic-output-formulas.svg",
        width,
        height,
        "Output formulas for each processing element",
        "A 2 by 2 arrangement of output cells showing the dot product formula computed by each processing element.",
        "".join(body),
    )

def main() -> None:
    a_wavefront()
    b_wavefront()
    output_formulas()


if __name__ == "__main__":
    main()
