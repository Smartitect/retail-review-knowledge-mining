"""Plotly figures for the dashboard. Each takes a Polars frame and returns a Figure."""

import math

import plotly.graph_objects as go
import polars as pl

from .palette import (
    INK,
    LINK_ALPHA,
    MUTED_INK,
    NEUTRAL_NODE,
    PAPER,
    RISK,
    RULE,
    SENTIMENT,
    pretty,
    with_alpha,
)

LEVEL_ORDER = ["sentiment", "category", "product", "issue"]
SENTIMENT_ORDER = ["negative", "mixed", "positive"]
REVENUE_TICKS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def _style(figure: go.Figure, height: int) -> go.Figure:
    """endjin ink on a transparent ground, recessive grid, system sans."""
    axis = {"gridcolor": RULE, "linecolor": RULE, "zerolinecolor": RULE, "tickfont": {"color": MUTED_INK},
                "title_font": {"color": MUTED_INK}}
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAPER,
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "size": 12, "color": INK},
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": RULE, "font": {"color": INK}},
        xaxis=axis,
        yaxis=axis,
    )
    return figure


def _label(level: str, name: str) -> str:
    """Codes (sentiment, issue) are prettified; category and product names are shown as written."""
    return pretty(name) if level in ("sentiment", "issue") else name.capitalize() if level == "category" else name


def _nodes(links: pl.DataFrame) -> pl.DataFrame:
    """Every node once, sized by the customers flowing through it, in drawing order.

    A node's size is what flows *into* it, except at the first level, which has
    no inflow and is sized by what flows out.
    """
    inflow = links.group_by(level="target_level", name="target").agg(pl.col("reviews").sum())
    outflow = (
        links.filter(pl.col("source_level") == LEVEL_ORDER[0])
        .group_by(level="source_level", name="source")
        .agg(pl.col("reviews").sum())
    )
    return (
        pl.concat([outflow, inflow])
        .with_columns(
            level_rank=pl.col("level").replace_strict(LEVEL_ORDER, list(range(len(LEVEL_ORDER)))),
            sentiment_rank=pl.col("name").replace_strict(SENTIMENT_ORDER, [0, 1, 2], default=9),
            no_issue_last=pl.col("name") == "no_issue",
        )
        .sort("level_rank", "sentiment_rank", "no_issue_last", "reviews", descending=[False, False, False, True])
        .with_row_index("index")
    )


def sankey_figure(links: pl.DataFrame) -> go.Figure:
    """Sentiment -> category -> product -> issue, with links coloured by sentiment."""
    nodes = _nodes(links)
    index = {(r["level"], r["name"]): r["index"] for r in nodes.iter_rows(named=True)}
    node_colour = [
        SENTIMENT[r["name"]] if r["level"] == "sentiment" else NEUTRAL_NODE for r in nodes.iter_rows(named=True)
    ]
    rows = links.to_dicts()
    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            valueformat=",d",
            node={
                "label": [f"{_label(lvl, n)}  {c:,}" for lvl, n, c in nodes.select("level", "name", "reviews").rows()],
                "color": node_colour,
                "pad": 10,
                "thickness": 14,
                "line": {"width": 0},
                "hovertemplate": "%{label}<extra></extra>",
            },
            link={
                "source": [index[(r["source_level"], r["source"])] for r in rows],
                "target": [index[(r["target_level"], r["target"])] for r in rows],
                "value": [r["reviews"] for r in rows],
                "color": [with_alpha(SENTIMENT[r["sentiment"]], LINK_ALPHA) for r in rows],
                "customdata": [
                    [pretty(r["sentiment"]), _label(r["source_level"], r["source"]), _label(r["target_level"], r["target"])]
                    for r in rows
                ],
                "hovertemplate": "%{customdata[0]} reviews<br>%{customdata[1]} → %{customdata[2]}: "
                "<b>%{value}</b><extra></extra>",
            },
        )
    )
    return _style(figure, max(520, 28 * nodes.filter(pl.col("level") == "product").height))


def risk_scatter(customers: pl.DataFrame) -> go.Figure:
    """Tenure against lifetime revenue, one mark per customer, shaped and coloured by risk tier.

    Each customer is placed as at their most recent review: the tenure and
    lifetime value they had when they wrote it, and the risk it signals.

    Revenue runs over two orders of magnitude, so its axis is logarithmic.
    Dashed hairlines mark the medians of the customers shown, splitting the
    plot into newer/longer-standing and lower/higher-value quadrants.
    """
    figure = go.Figure()
    issue_names = {i: pretty(i) for i in customers["primary_issue"].unique()}
    # Satisfied first so the tiers that matter are drawn on top.
    for tier in reversed(list(RISK)):
        subset = customers.filter(pl.col("risk_tier") == tier)
        style = RISK[tier]
        figure.add_trace(
            go.Scatter(
                x=subset["tenure_days"],
                y=subset["lifetime_revenue"],
                mode="markers",
                name=f"{tier} ({subset.height})",
                legendrank=list(RISK).index(tier),
                marker={
                    "size": 11 if tier != "Satisfied" else 9,
                    "color": style["color"],
                    "symbol": style["symbol"],
                    "opacity": 1.0 if tier != "Satisfied" else 0.5,
                    "line": {"width": 1.5, "color": "rgba(255,255,255,0.9)"},
                },
                customdata=subset.select(
                    "review_id", "product_name", "country", "rating",
                    pl.col("peak_frustration").round(1),
                    pl.col("primary_issue").replace_strict(issue_names),
                    pl.col("review_text").str.slice(0, 110),
                ).rows(),
                hovertemplate=(
                    "<b>%{customdata[1]}</b> · %{customdata[2]}<br>"
                    "Tenure %{x:,.0f} days · lifetime revenue %{y:,.2f} (at this review)<br>"
                    "%{customdata[3]}★ · peak frustration %{customdata[4]} of 4 · %{customdata[5]}<br>"
                    "<i>%{customdata[6]}…</i><extra></extra>"
                ),
            )
        )
    if customers.height:
        line = {"line_dash": "dash", "line_width": 1, "line_color": MUTED_INK, "opacity": 0.6,
                "annotation_font_color": MUTED_INK, "annotation_font_size": 11}
        tenure, revenue = customers["tenure_days"].median(), customers["lifetime_revenue"].median()
        figure.add_vline(tenure, annotation_text=f"median {tenure:,.0f} days", **line)
        # On a log axis Plotly places shapes in data units but annotations in log10 units,
        # so the line and its label are added separately.
        figure.add_hline(revenue, **{k: v for k, v in line.items() if not k.startswith("annotation")})
        figure.add_annotation(x=1, xref="paper", y=math.log10(revenue), yref="y", showarrow=False,
                              xanchor="right", yanchor="bottom", text=f"median revenue {revenue:,.0f}",
                              font={"color": MUTED_INK, "size": 11})
    _style(figure, 540)
    figure.update_layout(
        margin={"t": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
        xaxis={"title": "Tenure at latest review (days)", "rangemode": "tozero", "showgrid": False},
        yaxis={"title": "Lifetime revenue at latest review (log scale)", "type": "log",
               "tickvals": REVENUE_TICKS, "ticktext": [f"{t:,}" for t in REVENUE_TICKS]},
        hovermode="closest",
    )
    return figure
