"""Draw the four repo diagrams in the endjin skin (diagram-design profile `endjin`), light and dark.

One layout per diagram; the theme only swaps tokens, so the two variants cannot drift.
Writes <slug>.html and <slug>-dark.html next to this file (or into the directory given).

    uv run python docs/diagrams/build_diagrams.py

Then export each HTML file to SVG (see README.md in this folder).
"""

import math
import pathlib
import sys

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent

SANS = "'Inter', system-ui, sans-serif"
MONO = "Consolas, Menlo, Inconsolata, 'DejaVu Sans Mono', monospace"
SERIF = "'Instrument Serif', serif"
FONTS = "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap"

THEMES = {
    "light": {"paper": "#fafafa", "ink": "#181a1d", "ink_rgb": "24,26,29", "muted": "#636871", "soft": "#858891",
              "accent": "#60991a", "accent_rgb": "96,153,26", "link": "#365ed6", "link_rgb": "54,94,214",
              "card": "#ffffff", "muted_rgb": "99,104,113"},
    "dark": {"paper": "#181a1d", "ink": "#f7f9fc", "ink_rgb": "247,249,252", "muted": "#a3a7af", "soft": "#858891",
             "accent": "#78c021", "accent_rgb": "120,192,33", "link": "#7f9cf0", "link_rgb": "127,156,240",
             "card": "#26292e", "muted_rgb": "163,167,175"},
}


def mono_w(text: str, size: float) -> float:
    return len(text) * size * 0.62


def up4(v: float) -> int:
    return int(math.ceil(v / 4) * 4)


class D:
    """One diagram being drawn in one theme."""

    def __init__(self, theme: str):
        self.t = THEMES[theme]
        self.parts: list[str] = []

    def add(self, s: str):
        self.parts.append(s)

    # ---- connectors (drawn first) ----
    def path(self, d: str, kind: str = "muted", dashed: bool = False, width: float = 1.2, marker: bool = True):
        t = self.t
        colour = {"muted": t["muted"], "accent": t["accent"], "link": t["link"]}[kind]
        mid = {"muted": "arrow", "accent": "arrow-accent", "link": "arrow-link"}[kind]
        dash = ' stroke-dasharray="4,3"' if dashed else ""
        mk = f' marker-end="url(#{mid})"' if marker else ""
        self.add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{dash}{mk}/>')

    def label(self, cx: float, cy: float, text: str, kind: str = "soft", anchor: str = "middle", fill: str | None = None):
        """A masked arrow label, 12px tall, centred on (cx, cy) - or starting at cx when anchor='start'."""
        t = self.t
        w = up4(mono_w(text, 8) + 8)
        x = cx - w / 2 if anchor == "middle" else cx
        colour = {"soft": t["muted"], "accent": t["accent"], "link": t["link"]}[kind]
        self.add(f'<rect x="{x:g}" y="{cy - 6:g}" width="{w}" height="12" rx="2" fill="{fill or t["paper"]}"/>')
        self.add(f'<text x="{x + w / 2:g}" y="{cy + 3:g}" fill="{colour}" font-size="8" font-family="{MONO}" '
                 f'text-anchor="middle" letter-spacing="0.06em">{text}</text>')
        return x, w

    # ---- nodes ----
    def node(self, x, y, w, h, kind, tag, name, sub, badge=None):
        t, ink = self.t, self.t["ink_rgb"]
        fill, stroke, tag_stroke, tag_text = {
            "focal": (f"rgba({t['accent_rgb']},0.08)", t["accent"], f"rgba({t['accent_rgb']},0.50)", t["accent"]),
            "backend": (t["card"], t["ink"], f"rgba({ink},0.40)", t["ink"]),
            "store": (f"rgba({ink},0.05)", t["muted"], f"rgba({t['muted_rgb']},0.50)", t["muted"]),
            "external": (f"rgba({ink},0.03)", f"rgba({ink},0.30)", f"rgba({ink},0.22)", t["soft"]),
        }[kind]
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{t["paper"]}"/>')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        if tag:
            tw = up4(mono_w(tag, 7) + 10)
            self.add(f'<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="none" stroke="{tag_stroke}" stroke-width="0.8"/>')
            self.add(f'<text x="{x + 8 + tw / 2:g}" y="{y + 17}" fill="{tag_text}" font-size="7" font-family="{MONO}" '
                     f'text-anchor="middle" letter-spacing="0.08em">{tag}</text>')
        if badge is not None:
            bw = up4(mono_w(badge, 8) + 8)
            self.add(f'<rect x="{x + w - 8 - bw}" y="{y + 8}" width="{bw}" height="12" rx="2" fill="{t["paper"]}" '
                     f'stroke="rgba({ink},0.22)" stroke-width="0.8"/>')
            self.add(f'<text x="{x + w - 8 - bw / 2:g}" y="{y + 17}" fill="{t["muted"]}" font-size="8" font-family="{MONO}" '
                     f'text-anchor="middle">{badge}</text>')
        cx = x + w / 2
        cy = y + h / 2 + (6 if tag or badge is not None else 0)
        self.add(f'<text x="{cx:g}" y="{cy + 1:g}" fill="{t["ink"]}" font-size="12" font-weight="600" font-family="{SANS}" '
                 f'text-anchor="middle">{name}</text>')
        if sub:
            self.add(f'<text x="{cx:g}" y="{cy + 16:g}" fill="{t["muted"]}" font-size="9" font-family="{MONO}" '
                     f'text-anchor="middle">{sub}</text>')

    # ---- legend ----
    def legend(self, y: int, width: int, items: list):
        t, ink = self.t, self.t["ink_rgb"]
        self.add(f'<line x1="40" y1="{y}" x2="{width - 40}" y2="{y}" stroke="rgba({ink},0.10)" stroke-width="0.8"/>')
        self.add(f'<text x="40" y="{y + 16}" fill="{t["muted"]}" font-size="8" font-family="{MONO}" letter-spacing="0.18em">LEGEND</text>')
        x = 40
        for kind, text in items:
            iy = y + 32
            if kind in ("focal", "backend", "store", "external"):
                fill, stroke = {
                    "focal": (f"rgba({t['accent_rgb']},0.08)", t["accent"]),
                    "backend": (t["card"], t["ink"]),
                    "store": (f"rgba({ink},0.05)", t["muted"]),
                    "external": (f"rgba({ink},0.03)", f"rgba({ink},0.30)"),
                }[kind]
                self.add(f'<rect x="{x}" y="{iy}" width="14" height="10" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
            else:
                colour, mid, dash = {
                    "arrow": (t["muted"], "arrow", ""),
                    "accent": (t["accent"], "arrow-accent", ""),
                    "link": (t["link"], "arrow-link", ""),
                    "return": (t["muted"], "arrow", ' stroke-dasharray="4,3"'),
                }[kind]
                self.add(f'<line x1="{x}" y1="{iy + 5}" x2="{x + 24}" y2="{iy + 5}" stroke="{colour}" stroke-width="1.2"{dash} '
                         f'marker-end="url(#{mid})"/>')
            tx = x + (20 if kind in ("focal", "backend", "store", "external") else 32)
            self.add(f'<text x="{tx}" y="{iy + 8}" fill="{t["muted"]}" font-size="8.5" font-family="{SANS}">{text}</text>')
            x = up4(tx + len(text) * 8.5 * 0.55 + 24)


def page(slug: str, theme: str, eyebrow: str, h1: str, title: str, desc: str, w: int, h: int, body: str) -> str:
    t = THEMES[theme]
    s = f"{slug}-dark" if theme == "dark" else slug
    markers = "".join(
        f'<marker id="{mid}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
        f'<polygon points="0 0, 8 3, 0 6" fill="{c}"/></marker>'
        for mid, c in (("arrow", t["muted"]), ("arrow-accent", t["accent"]), ("arrow-link", t["link"]))
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{h1}</title>
  <link href="{FONTS}" rel="stylesheet">
  <!-- Drawn with the diagram-design skill, endjin profile (tokens from endjin.com). Light and dark share one layout. -->
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: {SANS}; background: {t["paper"]}; color: {t["ink"]}; padding: 3rem 2rem; }}
    .frame {{ max-width: {w + 40}px; margin: 0 auto; }}
    .eyebrow {{ font-family: {MONO}; font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em;
               text-transform: uppercase; color: {t["muted"]}; margin-bottom: 0.5rem; }}
    h1 {{ font-family: {SERIF}; font-size: 1.75rem; font-weight: 400; letter-spacing: -0.02em; margin-bottom: 1.5rem; }}
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    svg {{ width: 100%; min-width: {w}px; display: block; }}
    @media print {{ .diagram-container {{ overflow-x: visible; }} svg {{ min-width: 0; }} }}
  </style>
</head>
<body>
  <div class="frame">
    <p class="eyebrow">{eyebrow}</p>
    <h1>{h1}</h1>
    <div class="diagram-container">
      <svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{s}-title {s}-desc">
        <title id="{s}-title">{title}</title>
        <desc id="{s}-desc">{desc}</desc>
        <defs>{markers}</defs>
        <rect width="100%" height="100%" fill="{t["paper"]}"/>
        {body}
      </svg>
    </div>
  </div>
</body>
</html>
"""


# ---------------------------------------------------------------- 1. pipeline
def pipeline(theme):
    d = D(theme)
    W, H = 960, 488
    # Arrows first. Columns x = 40, 280, 520, 760 (160 wide, 80 gaps); rows y = 40, 184, 328 (64 tall).
    # Paid calls rise to y = 72 and turn into the service, so Classify reaches both without crossing either.
    d.path("M 120,184 V 80 Q 120,72 128,72 H 280", "link")      # generate -> Foundry, misses only
    d.path("M 560,184 V 80 Q 560,72 552,72 H 440", "link")      # classify (DSPy) -> Foundry, misses only
    d.path("M 640,184 V 80 Q 640,72 648,72 H 760", "link")      # classify (Jev) -> Jev, misses only
    for cx in (120, 600):
        d.path(f"M {cx},248 V 328")                              # stage -> its cache
    d.path("M 200,216 H 280")                                    # generate -> dataset
    d.path("M 440,216 H 520")                                    # dataset -> classify
    d.path("M 680,216 H 760")                                    # classify -> explore
    d.path("M 680,240 H 712 Q 720,240 720,248 V 352 Q 720,360 728,360 H 760")  # classify -> compare
    d.label(204, 58, "TEXT · MISSES ONLY", "link")
    d.label(496, 58, "DSPY · MISSES ONLY", "link")
    d.label(704, 58, "JEV · MISSES ONLY", "link")
    for cx in (120, 600):
        d.label(cx + 8, 288, "CHECK + STORE", anchor="start")
    d.label(240, 202, "BATCH")
    d.label(480, 202, "LISTED ONLY")
    d.label(720, 202, "SENTENCES")
    # Nodes.
    d.node(280, 40, 160, 64, "external", "PAID", "Azure AI Foundry", "chat deployment")
    d.node(760, 40, 160, 64, "external", "PAID", "TypeSafe AI Jev", "26 questions each")
    d.node(40, 184, 160, 64, "backend", "01", "Generate", "generate-data")
    d.node(280, 184, 160, 64, "store", "FILES", "Dataset", "data/generated/")
    d.node(520, 184, 160, 64, "backend", "02", "Classify", "Jev or DSPy")
    d.node(760, 184, 160, 64, "backend", "03", "Explore", "notebook · dashboard")
    d.node(40, 328, 160, 64, "focal", "CACHE", "Review text cache", "review + prompt hash")
    d.node(520, 328, 160, 64, "focal", "CACHE", "Answer caches", "one per classifier")
    d.node(760, 328, 160, 64, "backend", "OPTIONAL", "Compare", "both · notebook 02")
    d.legend(420, W, [("focal", "Cache: paid work is kept"), ("backend", "Pipeline stage"), ("store", "Data on disk"),
                      ("external", "Paid service"), ("link", "Paid call"), ("arrow", "Data flow")])
    return page("pipeline", theme, "Architecture · retail review knowledge mining", "The pipeline and its caches",
                "Pipeline with cached paid services",
                "Generate, classify and explore stages run left to right. Sentences are classified by TypeSafe AI Jev "
                "or, through DSPy, by a foundation model on Azure AI Foundry, and an optional comparison runs both. "
                "Each paid service is reached only through a cache, so only cache misses are paid for.",
                W, H, "\n        ".join(d.parts))


# ---------------------------------------------------------------- 2. packages
def packages(theme):
    d = D(theme)
    W, H = 1160, 432
    # Ranks y = 40, 160, 280 (56 tall). Edges skipping rank 1 turn at y = 248, in the clear band above rank 2.
    d.path("M 192,96 V 120 Q 192,128 184,128 H 128 Q 120,128 120,136 V 160")    # app -> insights
    d.path("M 248,96 V 120 Q 248,128 256,128 H 312 Q 320,128 320,136 V 160")    # app -> charts
    d.path("M 120,216 V 280")                                                    # insights -> jev
    d.path("M 592,96 V 240 Q 592,248 584,248 H 528 Q 520,248 520,256 V 280")    # generator -> writer
    d.path("M 648,96 V 240 Q 648,248 656,248 H 744 Q 752,248 752,256 V 280")    # generator -> model
    d.path("M 912,96 V 240 Q 912,248 904,248 H 816 Q 808,248 808,256 V 280")    # wrangler -> model
    d.path("M 968,96 V 240 Q 968,248 976,248 H 1032 Q 1040,248 1040,256 V 280")  # wrangler -> features
    d.path("M 416,96 V 240 Q 416,248 408,248 H 184 Q 176,248 176,256 V 280")    # dspy -> jev (the question set)
    d.path("M 480,96 V 280")                                                     # dspy -> writer (Foundry settings)
    d.node(140, 40, 160, 56, "backend", None, "Streamlit app", "app/streamlit_app.py", "0 in")
    d.node(340, 40, 160, 56, "backend", None, "dspy_classifier", "DSPy · comparison", "0 in")
    d.node(540, 40, 160, 56, "backend", None, "retail_generator", "generate-data CLI", "0 in")
    d.node(860, 40, 160, 56, "backend", None, "review_wrangler", "load · split", "0 in")
    d.node(40, 160, 160, 56, "backend", None, "review_insights", "rollups · risk", "1 in")
    d.node(240, 160, 160, 56, "store", None, "review_charts", "Plotly figures", "1 in")
    d.node(40, 280, 160, 56, "store", None, "jev_classifier", "questions · cache", "2 in")
    d.node(440, 280, 160, 56, "store", None, "review_writer", "Foundry · text cache", "2 in")
    d.node(700, 280, 160, 56, "store", None, "retail_model", "schemas · storage", "2 in")
    d.node(960, 280, 160, 56, "store", None, "customer_features", "point in time", "1 in")
    d.legend(368, W, [("backend", "Imports other packages"), ("store", "Leaf: imports no project package"),
                      ("arrow", "Imports")])
    d.add(f'<text x="{W - 40}" y="{368 + 40}" fill="{d.t["muted"]}" font-size="8.5" font-family="{SANS}" '
          f'text-anchor="end">N in: how many packages import it</text>')
    return page("packages", theme, "Architecture · retail review knowledge mining", "Package dependencies",
                "Package dependencies",
                "Four entry points import downwards into five leaf packages; retail_model, jev_classifier and "
                "review_writer are each imported by two others, and there are no cycles.",
                W, H, "\n        ".join(d.parts))


# ---------------------------------------------------------------- 3. batch commit
def batch(theme):
    d = D(theme)
    t, ink = d.t, d.t["ink_rgb"]
    W, H = 960, 696
    xs = [112, 296, 480, 664, 848]
    # Fragment frame and divider first, so lifelines and messages sit on top.
    d.add(f'<rect x="64" y="224" width="832" height="368" rx="4" fill="rgba({ink},0.02)" stroke="rgba({ink},0.22)" stroke-width="1"/>')
    d.add(f'<rect x="64" y="224" width="40" height="16" rx="2" fill="{t["paper"]}" stroke="rgba({ink},0.22)" stroke-width="1"/>')
    d.add(f'<text x="84" y="236" fill="{t["muted"]}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.12em">ALT</text>')
    d.add(f'<text x="116" y="236" fill="{t["muted"]}" font-size="8" font-family="{MONO}" letter-spacing="0.04em">[--dry-run]</text>')
    d.add(f'<line x1="72" y1="296" x2="888" y2="296" stroke="rgba({ink},0.20)" stroke-width="1" stroke-dasharray="4,3"/>')
    d.add(f'<text x="76" y="312" fill="{t["muted"]}" font-size="8" font-family="{MONO}" letter-spacing="0.04em">[else]</text>')
    for x in xs:
        d.add(f'<line x1="{x}" y1="88" x2="{x}" y2="616" stroke="rgba({ink},0.20)" stroke-width="1" stroke-dasharray="3,3"/>')
    # Activation bars: incremental holds control throughout; review_writer while it writes text.
    d.add(f'<rect x="292" y="112" width="8" height="472" fill="rgba({ink},0.06)" stroke="{t["muted"]}" stroke-width="0.8"/>')
    d.add(f'<rect x="660" y="328" width="8" height="144" fill="rgba({ink},0.06)" stroke="{t["muted"]}" stroke-width="0.8"/>')

    def blend(hex_, rgb, a):
        p = [int(hex_[i:i + 2], 16) for i in (1, 3, 5)]
        q = [int(v) for v in rgb.split(",")]
        return "#" + "".join(f"{round(pv * (1 - a) + qv * a):02x}" for pv, qv in zip(p, q))
    framed = blend(t["paper"], ink, 0.02)

    def msg(x1, x2, y, text, kind="muted", dashed=False, label_x=None):
        end = x2 - 4 if x2 > x1 else x2 + 4
        start = x1 + 4 if x2 > x1 else x1 - 4
        d.path(f"M {start},{y} H {end}", kind, dashed)
        d.label(label_x if label_x is not None else (x1 + x2) / 2, y - 14, text, "accent" if kind == "accent" else
                ("link" if kind == "link" else "soft"), fill=framed if 224 < y < 592 else None)

    def self_msg(x, y, text):
        d.path(f"M {x + 4},{y} H {x + 32} Q {x + 40},{y} {x + 40},{y + 8} Q {x + 40},{y + 16} {x + 32},{y + 16} H {x + 4}")
        d.label(x + 48, y + 8, text, anchor="start", fill=framed if 224 < y < 592 else None)

    msg(112, 296, 120, "ADD-CUSTOMERS")
    msg(296, 480, 152, "READ MANIFEST")
    self_msg(296, 176, "GEN + ESTIMATE")
    msg(296, 112, 272, "ESTIMATE ONLY", dashed=True)
    msg(296, 664, 336, "BRIEFS", label_x=572)
    msg(664, 480, 368, "CACHE LOOKUP")
    msg(664, 848, 400, "MISSES ONLY", "link")
    msg(664, 480, 432, "APPEND CACHE")
    msg(664, 296, 464, "REVIEW_TEXTS", dashed=True, label_x=572)
    self_msg(296, 488, "INTEGRITY")
    msg(296, 480, 536, "BATCH FILES")
    msg(296, 480, 568, "SWAP MANIFEST", "accent")
    actors = [("CLI", "generate-data", "command", "backend"), ("PY", "incremental", "retail_generator", "backend"),
              ("FILES", "data/generated/", "manifest + batches", "store"), ("PY", "review_writer", "text cache", "backend"),
              ("PAID", "Azure AI Foundry", "chat deployment", "external")]
    for x, (tag, name, sub, kind) in zip(xs, actors):
        d.node(x - 72, 32, 144, 56, kind, tag, name, sub)
    d.legend(640, W, [("arrow", "Call"), ("return", "Return"), ("link", "Paid call"), ("accent", "The commit")])
    return page("batch-commit", theme, "Architecture · retail review knowledge mining", "How a batch is committed",
                "How a batch is committed",
                "A batch is generated and costed first; a dry run stops there, otherwise review text is written "
                "through the cache, the batch files are written, and swapping in the new manifest commits it.",
                W, H, "\n        ".join(d.parts))


# ---------------------------------------------------------------- 4. data model
def entity(d, x, y, w, name, fields, focal=False, eyebrow="ENTITY"):
    t, ink = d.t, d.t["ink_rgb"]
    h = 40 + 20 * len(fields) + 12
    if focal:
        body, head, stroke, rule, eb = (f"rgba({t['accent_rgb']},0.04)", f"rgba({t['accent_rgb']},0.10)", t["accent"],
                                        f"rgba({t['accent_rgb']},0.40)", t["accent"])
    else:
        body, head, stroke, rule, eb = (t["card"], f"rgba({ink},0.04)", t["ink"], f"rgba({ink},0.22)", t["muted"])
    d.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{t["paper"]}"/>')
    d.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{body}" stroke="{stroke}" stroke-width="1"/>')
    d.add(f'<path d="M {x + 0.5},{y + 40} V {y + 6} Q {x + 0.5},{y + 0.5} {x + 6},{y + 0.5} H {x + w - 6} '
          f'Q {x + w - 0.5},{y + 0.5} {x + w - 0.5},{y + 6} V {y + 40} Z" fill="{head}"/>')
    d.add(f'<line x1="{x}" y1="{y + 40}" x2="{x + w}" y2="{y + 40}" stroke="{rule}" stroke-width="1"/>')
    d.add(f'<text x="{x + 16}" y="{y + 16}" fill="{eb}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">{eyebrow}</text>')
    d.add(f'<text x="{x + 16}" y="{y + 33}" fill="{t["ink"]}" font-size="14" font-weight="600" font-family="{SANS}">{name}</text>')
    for i, (field, kind) in enumerate(fields):
        fy = y + 60 + 20 * i
        d.add(f'<text x="{x + 16}" y="{fy}" fill="{t["ink"]}" font-size="10" font-family="{MONO}">{field}</text>')
        d.add(f'<text x="{x + w - 16}" y="{fy}" fill="{t["muted"]}" font-size="10" font-family="{MONO}" text-anchor="end">{kind}</text>')
    return h


def card(d, cx, cy, text):
    t = d.t
    w = up4(len(text) * 10 * 0.62 + 6)
    d.add(f'<rect x="{cx - w / 2:g}" y="{cy - 6}" width="{w}" height="12" rx="2" fill="{t["paper"]}"/>')
    d.add(f'<text x="{cx:g}" y="{cy + 4}" fill="{t["muted"]}" font-size="10" font-weight="600" font-family="{MONO}" '
          f'text-anchor="middle">{text}</text>')


def data_model(theme):
    d = D(theme)
    W, H = 1120, 712
    m = d.t["muted"]

    def line(path):
        d.add(f'<path d="{path}" fill="none" stroke="{m}" stroke-width="1"/>')

    # Relationships first.
    line("M 240,100 H 320")                                                      # customers - orders
    line("M 520,100 H 600")                                                      # orders - order_lines
    line("M 880,100 H 800")                                                      # products - order_lines
    line("M 668,192 V 280")                                                      # order_lines - reviews
    line("M 140,152 V 392 Q 140,400 148,400 H 600")                              # customers - reviews
    line("M 960,172 V 232 Q 960,240 952,240 H 740 Q 732,240 732,248 V 280")     # products - reviews
    line("M 800,320 H 880")                                                      # reviews - review_texts
    line("M 800,420 H 832 Q 840,420 840,428 V 532 Q 840,540 848,540 H 880")     # reviews - review_truth
    for cx, cy, text in [(252, 100, "1"), (308, 100, "N"), (532, 100, "1"), (580, 100, "1..*"),
                         (868, 100, "1"), (812, 100, "N"), (668, 204, "1"), (668, 264, "0..1"),
                         (140, 164, "1"), (588, 400, "N"), (960, 184, "1"), (732, 266, "N"),
                         (812, 320, "1"), (860, 320, "0..1"), (812, 420, "1"), (868, 540, "1")]:
        card(d, cx, cy, text)
    entity(d, 40, 40, 200, "customers", [("# customer_id", "str"), ("country", "str"), ("signup_at", "datetime")])
    entity(d, 320, 40, 200, "orders", [("# order_id", "str"), ("→ customer_id", "str"), ("ordered_at", "datetime"),
                                       ("order_total", "float")])
    entity(d, 600, 40, 200, "order_lines", [("# order_line_id", "str"), ("→ order_id", "str"), ("→ product_id", "str"),
                                            ("quantity", "int"), ("unit_price", "float")])
    entity(d, 880, 40, 200, "products", [("# product_id", "str"), ("category", "str"), ("fuel_type", "str?"),
                                         ("list_price", "float")])
    entity(d, 600, 280, 200, "reviews", [("# review_id", "str"), ("→ order_line_id", "str"), ("→ customer_id", "str"),
                                         ("→ product_id", "str"), ("reviewed_at", "datetime"), ("rating", "1–5")],
           focal=True, eyebrow="ENTITY · CENTRAL")
    entity(d, 880, 280, 200, "review_texts", [("→ review_id", "str"), ("# prompt_hash", "str"), ("review_text", "str"),
                                              ("language", "str"), ("model", "str")], eyebrow="ENTITY · OPTIONAL")
    entity(d, 880, 480, 200, "review_truth", [("# → review_id", "str"), ("satisfaction", "0–1"), ("aspect", "str"),
                                              ("language", "str")], eyebrow="ENTITY · EVALUATION ONLY")
    d.legend(652, W, [("focal", "Central entity"), ("backend", "Entity")])
    d.add(f'<text x="{W - 40}" y="{652 + 40}" fill="{m}" font-size="8.5" font-family="{SANS}" text-anchor="end">'
          f'# key  ·  → foreign key  ·  1, N, 0..1, 1..* cardinality at each end</text>')
    return page("data-model", theme, "Data model · retail review knowledge mining", "The retail data model",
                "The retail data model",
                "Customers place orders of order lines for products; a review covers at most one order line, always "
                "has its generator truth, and has text once it has been written.",
                W, H, "\n        ".join(d.parts))


for fn, slug in ((pipeline, "pipeline"), (packages, "packages"), (batch, "batch-commit"), (data_model, "data-model")):
    for theme in ("light", "dark"):
        name = slug if theme == "light" else f"{slug}-dark"
        (OUT / f"{name}.html").write_text(fn(theme))
        print("wrote", name)
