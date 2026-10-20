"""Builds Latent's brand files (docs/assets/brand) and README chart (docs/assets/charts).

    uv run --with fonttools --with brotli --with uharfbuzz python design/brand.py

The mark is a 5 x 5 halftone on the same grid as Mimic's bars (24 px pitch in a 128 box):
an L of dots so large they nearly touch, and noise that gets finer the further it sits from the L. A latent image is the picture
on exposed film before it is developed; the amber dot is the safelight, the one pixel developing now.
Text is converted to outlines, so the SVGs look the same on GitHub as here.
"""
import io
import math
import random
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "app" / "ui" / "fonts"
OUT = ROOT / "docs" / "assets" / "brand"

DARK = {"ground": "#111213", "tile": "#17181A", "ink": "#ECEDE9", "ink2": "#A9ABA6", "noise": "#7A7C7F", "light": "#FFAA2C"}
LIGHT = {"ground": "#F4F4F1", "tile": "#EAEAE6", "ink": "#141516", "ink2": "#4F5154", "noise": "#75777A", "light": "#C76A00"}

TAGLINE = "Generate and edit images on your own GPU"
ALT = "Latent: local image generation and editing on your own GPU"

# ---------- the mark ----------

# Radius of each dot by distance from the L (0 = on the L). Jitter keeps the noise from reading as a gradient.
RADII = {0: 10.0, 1: 6.2, 2: 4.6, 3: 3.3, 4: 2.2}  # the L nearly touches, like solid areas in print
ACCENT = (1, 3)  # column, row: the inner corner of the L, the next pixel to develop


def dots():
    """(cx, cy, r, kind, distance) for the 25 cells; kind is "ink", "noise" or "light"."""
    rng = random.Random(7)
    out = []
    for row in range(5):
        for col in range(5):
            d = min(col, 4 - row)
            r = RADII[d] if d == 0 else RADII[d] * rng.uniform(0.8, 1.15)
            kind = "ink" if d == 0 else "noise"
            if (col, row) == ACCENT:
                kind, r = "light", 7.0
            out.append((16 + 24 * col, 16 + 24 * row, round(r, 2), kind, d))
    return out


def mark_shapes(p, scale=1.0, dx=0.0, dy=0.0, animated=False):
    colour = {"ink": p["ink"], "noise": p["noise"], "light": p["light"]}
    parts = []
    for cx, cy, r, kind, d in dots():
        cls = f' class="d d{d}"' if animated and kind != "ink" else ""
        parts.append(f'<circle{cls} cx="{dx + cx * scale:.2f}" cy="{dy + cy * scale:.2f}" r="{r * scale:.2f}" fill="{colour[kind]}"/>')
    return "\n    ".join(parts)


# Noise dots swell and settle from the L outward, the way a print comes up in the developer tray.
SETTLE_CSS = """
    .d { transform-box: fill-box; transform-origin: 50% 50%; animation: settle 4.2s cubic-bezier(.23,1,.32,1) infinite; }
    .d1 { animation-delay: 0s; } .d2 { animation-delay: .14s; } .d3 { animation-delay: .28s; } .d4 { animation-delay: .42s; }
    @keyframes settle {
      0%   { transform: scale(1); }
      10%  { transform: scale(.35); }
      22%  { transform: scale(1.35); }
      34%  { transform: scale(.8); }
      48%  { transform: scale(1); }
      100% { transform: scale(1); }
    }
    @media (prefers-reduced-motion: reduce) { .d { animation: none; } }"""


def mark_svg(p, animated=False):
    style = f"\n  <style>{SETTLE_CSS}\n  </style>" if animated else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128" role="img" aria-label="Latent">
  <title>Latent</title>{style}
    {mark_shapes(p, animated=animated)}
</svg>
"""


def icon_svg(p):
    # App icon: the mark on a rounded tile, 22% corner like Mimic's.
    s = 1024 / 128 * 0.66
    off = (1024 - 128 * s) / 2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <rect x="0" y="0" width="1024" height="1024" rx="225.3" fill="{p['tile']}"/>
    {mark_shapes(p, scale=s, dx=off, dy=off)}
</svg>
"""


# ---------- text as outlines ----------

def text_path(font_file, text, size, wght, opsz=None):
    """Shape text with HarfBuzz (so kerning applies) and return (svg path d, width, cap height) at `size` px."""
    font = TTFont(FONTS / font_file)
    axes = {"wght": wght}
    if opsz is not None:
        axes["opsz"] = opsz
    font = instancer.instantiateVariableFont(font, axes)
    font.flavor = None
    buf = io.BytesIO()
    font.save(buf)
    hb_font = hb.Font(hb.Face(buf.getvalue()))
    b = hb.Buffer()
    b.add_str(text)
    b.guess_segment_properties()
    hb.shape(hb_font, b, {"kern": True, "liga": True})
    upm = font["head"].unitsPerEm
    k = size / upm
    glyphs = font.getGlyphSet()
    order = font.getGlyphOrder()
    pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
    x = 0
    for info, pos in zip(b.glyph_infos, b.glyph_positions):
        name = order[info.codepoint]
        glyphs[name].draw(TransformPen(pen, (k, 0, 0, -k, (x + pos.x_offset) * k, -pos.y_offset * k)))
        x += pos.x_advance
    return pen.getCommands(), x * k, font["OS/2"].sCapHeight * k


# ---------- banner ----------

def developing_strip(p, x0, x1, y_mid, rows=9, pitch=5.6, rmax=2.2):
    """A band of halftone dots: pure noise on the left, a developed horizon on the right."""
    rng = random.Random(3)
    cols = int((x1 - x0) / pitch)
    out = []
    for c in range(cols):
        t = c / (cols - 1)
        x = x0 + c * pitch
        horizon = 0.5 + 0.28 * math.sin(t * 9.0) + 0.12 * math.sin(t * 23.0 + 1.3)  # a hill line, 0..1 of the band
        for r in range(rows):
            v = r / (rows - 1)  # 0 top .. 1 bottom
            signal = 1.0 if v >= 1 - horizon else 0.08
            noise = rng.random()
            dev = min(1.0, max(0.0, (t - 0.12) / 0.7)) ** 1.4  # how developed this column is
            a = noise * (1 - dev) + signal * dev
            rad = max(0.45, rmax * a)
            out.append(f'<circle cx="{x:.1f}" cy="{y_mid + (r - (rows - 1) / 2) * pitch:.1f}" r="{rad:.2f}"/>')
    return f'<g fill="{p["ink2"]}" opacity="0.3">{"".join(out)}</g>'


def banner_svg(p, w=1280, h=400, radius=24, tag=TAGLINE):
    word_d, word_w, cap = text_path("newsreader.woff2", "Latent", 132, 500, opsz=72)
    tag_d, tag_w, _ = text_path("ibm-plex-sans.woff2", tag, 26, 400)
    # Size the mark so its dots span a little more than the cap height, with the L's foot on the baseline.
    top, bottom = 16 - RADII[0], 112 + RADII[0]
    s = cap * 1.3 / (bottom - top)
    mark = 128 * s
    gap = 34
    total = mark + gap + word_w
    mx = (w - total) / 2
    base = 214  # wordmark baseline
    my = base - bottom * s
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{ALT}">
  <title>Latent</title>
  <style>{SETTLE_CSS}
  </style>
  <rect width="{w}" height="{h}" rx="{radius}" fill="{p['ground']}"/>
  <g transform="translate(0 {(h - 400) / 2})">
  {developing_strip(p, 64, w - 64, 336)}
    {mark_shapes(p, scale=s, dx=mx, dy=my, animated=True)}
  <path fill="{p['ink']}" transform="translate({mx + mark + gap:.2f} {base})" d="{word_d}"/>
  <path fill="{p['ink2']}" transform="translate({(w - tag_w) / 2:.2f} 270)" d="{tag_d}"/>
  </g>
</svg>
"""


def social_svg(p):
    """GitHub's social preview (1280 x 640), still: render it to social-preview.png and upload it in the repo settings."""
    return banner_svg(p, 1280, 640, radius=0).replace("@keyframes settle", ".d { animation: none; } @keyframes settle")


# ---------- README chart ----------

def architecture_svg(p, line):
    """How Latent works, in the same layout as Mimic's chart. Plain text: GitHub falls back to the system sans."""
    font = "'IBM Plex Sans', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
    box, edge = ("#202124", "#2A2B2F") if p is DARK else ("#E0E0DB", "#D4D4CE")

    def panel(x, y, w, h, title, lines, accent=False):
        stroke = f'stroke="{p["light"]}" stroke-width="1.5"' if accent else f'stroke="{edge}" stroke-width="1"'
        body = "".join(f'<text x="{x + 16}" y="{y + 52 + 20 * i}" font-size="13" fill="{p["ink2"]}">{t}</text>' for i, t in enumerate(lines))
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{box}" {stroke}/>'
                f'<text x="{x + 16}" y="{y + 28}" font-size="15" font-weight="600" fill="{p["ink"]}">{title}</text>{body}')

    def arrow(x, y1, y2, label):
        return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{line}" stroke-width="1.5" marker-end="url(#a)" marker-start="url(#a)"/>'
                f'<text x="{x + 8}" y="{(y1 + y2) / 2 + 4}" font-size="12" fill="{line}">{label}</text>')

    parts = [
        panel(32, 32, 816, 76, "Interface", ["The window you use: prompt, sources, frames and the canvas. HTML, CSS and JavaScript in the system web view."]),
        arrow(440, 108, 146, "Tauri commands and events, live step previews"),
        panel(32, 146, 816, 120, "Rust core (Tauri 2)", [
            "Model manager: one model in memory at a time, idle unload, cancel between steps.",
            "Installer: a Python environment with PyTorch for your GPU or CPU, model weights from Hugging Face.",
            "Hardware and memory readout, sources, frames, saving with the recipe in the file."], accent=True),
        arrow(250, 266, 304, "JSON lines over stdin/stdout"),
        arrow(690, 266, 304, "JSON lines"),
        panel(32, 304, 500, 96, "Diffusion worker (diffusers)", [
            "Z-Image Turbo · FLUX.2 [klein] · Qwen-Image-Edit · SDXL",
            "Quantized and offloaded variants, so big models fit"]),
        panel(548, 304, 300, 96, "Tools worker", ["Upscale (Real-ESRGAN)", "Remove background (BiRefNet)"]),
        arrow(440, 400, 432, ""),
        panel(32, 432, 816, 60, "Your hardware", []),
    ]
    hw = f'<text x="180" y="460" font-size="13" fill="{p["ink2"]}">NVIDIA GPU (CUDA) · AMD GPU on Linux (ROCm) · Intel Arc (XPU) · CPU</text>'
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 524" width="880" height="524" role="img" aria-label="How Latent works" font-family="{font}">
<title>How Latent works</title>
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{line}"/></marker></defs>
<rect width="880" height="524" rx="16" fill="{p['ground']}"/>
{"".join(parts)}{hw}
</svg>
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    charts = ROOT / "docs" / "assets" / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    (charts / "architecture-dark.svg").write_text(architecture_svg(DARK, "#7A7C7F"), encoding="utf-8")
    (charts / "architecture-light.svg").write_text(architecture_svg(LIGHT, "#75777A"), encoding="utf-8")
    files = {
        "mark.svg": mark_svg(DARK),
        "mark-light.svg": mark_svg(LIGHT),
        "mark-animated.svg": mark_svg(DARK, animated=True),
        "icon.svg": icon_svg(DARK),
        "banner-dark.svg": banner_svg(DARK),
        "banner-light.svg": banner_svg(LIGHT),
        "social-preview.svg": social_svg(DARK),
    }
    for name, svg in files.items():
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"wrote {name} ({len(svg) // 1024} KB)")


if __name__ == "__main__":
    main()
