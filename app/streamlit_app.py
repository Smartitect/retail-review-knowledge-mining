"""
BBQ review insights: where sentiment comes from, and which customers are at risk.

Layout and state only. The review and customer views are built in
`review_insights` and the figures in `review_charts`; this file wires filters
to them.

    uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

import polars as pl
import streamlit as st

from review_charts import pretty, risk_scatter, sankey_figure
from review_insights import (
    NO_ISSUE,
    SENTIMENTS,
    assign_risk,
    customers,
    reviews,
    sankey_links,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "output" / "sentences_classified.parquet"

st.set_page_config(page_title="BBQ review insights", page_icon="🔥", layout="wide")


@st.cache_data
def load_reviews() -> tuple[pl.DataFrame, dict]:
    sentences = pl.scan_parquet(DATA)
    meta = sentences.select(
        pl.col("jev_model").first(), pl.col("question_set_version").first(),
        pl.col("text_source").unique().sort().str.join(", "),
    ).collect().row(0, named=True)
    return reviews(sentences).collect(), meta


if not DATA.exists():
    st.error(f"No classified reviews at `{DATA.relative_to(DATA.parents[2])}`. "
             "Run `notebooks/01_classify_reviews_with_jev.ipynb` first.")
    st.stop()

all_reviews, meta = load_reviews()

# --- settings: these change definitions, so they sit apart from the filters ---
with st.sidebar:
    st.header("Definitions")
    source = st.radio(
        "Review sentiment from",
        ["Review text (Jev)", "Star rating"],
        help="Jev never saw the star rating, so switching shows where the text and the stars disagree.",
    )
    sentiment_col = "text_sentiment" if source.startswith("Review") else "star_sentiment"
    st.subheader("Risk rules")
    at_risk_at = st.slider("At risk from peak frustration", 0.0, 4.0, 3.0, 0.25)
    frustrated_at = st.slider("Frustrated from peak frustration", 0.0, 4.0, 2.0, 0.25)
    st.caption("Frustration is scored 0 (none) to 4 (furious) per sentence; a review's peak is its "
               "most frustrated sentence. Any churn signal - returning it, a refund, switching brand, "
               "won't buy again - also counts as at risk. A customer is judged on their most recent review.")
    st.divider()
    st.caption(f"Classified by TypeSafe AI's {meta['jev_model']} · question set {meta['question_set_version']}"
               f" · review text: {meta['text_source']}")

st.title("BBQ review insights")
st.caption("Fictional customers, orders and reviews; every review sentence classified by TypeSafe AI's Jev. "
           "Tenure and lifetime value are as at the moment each review was written.")

# --- filters: one row above everything they scope ---
f1, f2, f3 = st.columns(3)
categories = f1.multiselect("Product category", sorted(all_reviews["product_category"].unique()), placeholder="All")
products = f2.multiselect(
    "Product",
    sorted(all_reviews.filter(pl.col("product_category").is_in(categories) if categories else pl.lit(True))["product_name"].unique()),
    placeholder="All",
)
countries = f3.multiselect("Country", sorted(all_reviews["country"].unique()), placeholder="All")

view = assign_risk(
    all_reviews.lazy().filter(
        pl.col("product_category").is_in(categories) if categories else pl.lit(True),
        pl.col("product_name").is_in(products) if products else pl.lit(True),
        pl.col("country").is_in(countries) if countries else pl.lit(True),
    ),
    at_risk_frustration=at_risk_at,
    frustrated_frustration=frustrated_at,
).collect()

if view.is_empty():
    st.info("No reviews match these filters.")
    st.stop()

people = customers(view.lazy()).collect()

# --- headline numbers ---
at_risk = people.filter(pl.col("risk_tier") == "At risk")
negative = view.filter(pl.col(sentiment_col) == "negative").height
k1, k2, k3, k4 = st.columns(4)
k1.metric("Reviews", f"{view.height:,}", f"from {people.height:,} customers", delta_color="off", delta_arrow="off",
          border=True)
k2.metric("Negative reviews", f"{negative:,}", f"{negative / view.height:.0%} of reviews", delta_color="off",
          delta_arrow="off", border=True)
k3.metric("Customers at risk", f"{at_risk.height:,}", f"{at_risk.height / people.height:.0%} of reviewers",
          delta_color="off", delta_arrow="off", border=True)
k4.metric("Lifetime revenue at risk", f"{at_risk['lifetime_revenue'].sum():,.0f}",
          f"{at_risk['lifetime_revenue'].sum() / people['lifetime_revenue'].sum():.0%} of reviewers' revenue",
          delta_color="off", delta_arrow="off", border=True)

REVIEW_COLUMNS = {
    "reviewed_at": st.column_config.DatetimeColumn("Reviewed", format="YYYY-MM-DD"),
    "customer_id": "Customer",
    "rating": st.column_config.NumberColumn("Stars", format="%d ★", width="small"),
    "product_name": "Product",
    "country": "Country",
    "tenure_days": st.column_config.NumberColumn("Tenure at review (days)", format="%d"),
    "lifetime_revenue": st.column_config.NumberColumn("Lifetime revenue at review", format="%.2f"),
    "peak_frustration": st.column_config.ProgressColumn("Peak frustration", min_value=0, max_value=4, format="%.1f"),
    "primary_issue": "Primary issue",
    "risk_tier": "Risk",
    "review_text": st.column_config.TextColumn("Review", width="large"),
}


def review_table(frame: pl.DataFrame):
    st.dataframe(
        frame.with_columns(pl.col("primary_issue").replace_strict(
            {i: pretty(i) for i in frame["primary_issue"].unique()})).select(*REVIEW_COLUMNS),
        column_config=REVIEW_COLUMNS, hide_index=True, width="stretch",
    )


flow_tab, risk_tab = st.tabs(["Sentiment flow", "Customers at risk"])

# --- Sankey: sentiment -> category -> product -> primary issue ---
with flow_tab:
    c1, c2 = st.columns([2, 1])
    shown = c1.pills("Sentiment", SENTIMENTS, default=SENTIMENTS, selection_mode="multi", format_func=pretty)
    hide_no_issue = c2.toggle("Hide reviews that raised no issue", value=False)
    flow = view.filter(pl.col(sentiment_col).is_in(shown or []))
    if hide_no_issue:
        flow = flow.filter(pl.col("primary_issue") != NO_ISSUE)

    if flow.is_empty():
        st.info("Nothing to trace for this selection.")
    else:
        st.plotly_chart(sankey_figure(sankey_links(flow.lazy(), sentiment_col)), width="stretch", theme=None)
        st.caption("Width is the number of reviews. Each review follows one path, ending at its "
                   "**primary issue**: the problem raised in its most frustrated sentence. "
                   "Hover a link for its count.")

        st.subheader("Reviews behind a flow")
        d1, d2 = st.columns(2)
        issue_counts = flow.group_by("primary_issue").len().sort("len", descending=True)
        issue = d1.selectbox("Primary issue", issue_counts["primary_issue"],
                             format_func=lambda i: f"{pretty(i)} ({issue_counts.filter(pl.col('primary_issue') == i)['len'][0]})")
        in_issue = flow.filter(pl.col("primary_issue") == issue)
        product = d2.selectbox("Product", ["All products", *sorted(in_issue["product_name"].unique())])
        if product != "All products":
            in_issue = in_issue.filter(pl.col("product_name") == product)
        review_table(in_issue.sort("peak_frustration", descending=True))

# --- scatter: tenure against lifetime revenue at each customer's latest review ---
with risk_tab:
    event = st.plotly_chart(
        risk_scatter(people), width="stretch", theme=None, key="risk_scatter",
        on_select="rerun", selection_mode=("points", "box", "lasso"),
    )
    st.caption("One mark per customer, placed at the tenure and lifetime value they had when they wrote their "
               "most recent review. Dashed lines are the medians of the customers shown. Box or lasso select to "
               "list them below.")

    picked = [p["customdata"][0] for p in event.selection.points] if event and event.selection.points else []
    if picked:
        st.subheader(f"{len(picked)} selected customer(s): their latest review")
        table = people.filter(pl.col("review_id").is_in(picked))
    else:
        st.subheader("At-risk customers, most valuable first: their latest review")
        table = at_risk
    review_table(table.sort("lifetime_revenue", descending=True))
