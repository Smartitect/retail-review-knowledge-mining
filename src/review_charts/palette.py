"""
endjin's palette, assigned by the job each colour does.

Hex values are read from endjin.com's custom properties (`--c-grey-*`, the
brand greens, `--db-orange`) and the danger red used across endjin's web
palette. Roles:

- Sentiment is a polarity: brand deep green for positive, danger red for
  negative, a neutral grey for mixed.
- Risk is a state: danger red, amber, brand green. Every tier also has its own
  marker shape and a labelled legend entry, so colour never carries it alone.
- Structural Sankey nodes (category, product, issue) are grey, so the only
  colour in the diagram is the sentiment being traced.

Checked with the dataviz validator against endjin's light surface (#fafafa),
all pairs: sentiment green/red CVD dE 10.9, normal 29.2; risk red/amber/green
passes every hard gate. Amber (2.05:1) sits below 3:1 contrast, so it is never
unlabelled - the legend and hover name every tier. These colours are validated
for the light theme only, which is why `.streamlit/config.toml` pins it.
"""

INK = "#181a1d"
INK_MUTED = "#636871"
GREY = "#858891"
RULE = "#e2e6eb"
PAPER = "#fafafa"

BRAND = "#78bf20"
BRAND_DEEP = "#60991a"
DANGER = "#b3261e"
AMBER = "#e3a710"

SENTIMENT = {
    "positive": BRAND_DEEP,
    "mixed": GREY,
    "negative": DANGER,
}

RISK = {
    "At risk": {"color": DANGER, "symbol": "diamond"},
    "Frustrated": {"color": AMBER, "symbol": "triangle-up"},
    "Satisfied": {"color": BRAND_DEEP, "symbol": "circle"},
}

NEUTRAL_NODE = INK_MUTED
MUTED_INK = INK_MUTED
LINK_ALPHA = 0.35


def with_alpha(hex_colour: str, alpha: float) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def pretty(label: str) -> str:
    """`build_quality` -> `Build quality`; `no_issue` -> `No issue raised`."""
    return "No issue raised" if label == "no_issue" else label.replace("_", " ").capitalize()
