"""
BBQ review insights: where sentiment comes from, and which customers are at risk.

Layout and state only. The customer view is built in `review_insights` and the
figures in `review_charts`; this file wires filters to them.

    uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

import polars as pl
import streamlit as st

from review_charts import pretty, risk_scatter, sankey_figure
from review_insights import NO_ISSUE, SENTIMENTS, assign_risk, customers, sankey_links

DATA = Path(__file__).resolve().parents[1] / "data" / "output" / "sentences_classified.parquet"

st.set_page_config(page_title="BBQ review insights", page_icon="🔥", layout="wide")


@st.cache_data
def load_customers() -> tuple[pl.DataFrame, dict]:
    sentences = pl.scan_parquet(DATA)
    meta = sentences.select(pl.col("jev_model").first(), pl.col("question_set_version").first()).collect().row(0, named=True)
    return customers(sentences).collect(), meta


if not DATA.exists():
    st.error(f"No classified reviews at `{DATA.relative_to(DATA.parents[2])}`. "
             "Run `notebooks/01_classify_reviews_with_jev.ipynb` first.")
    st.stop()

all_customers, meta = load_customers()

# --- settings: these change definitions, so they sit apart from the filters ---
with st.sidebar:
    st.header("Definitions")
    source = st.radio(
        "Customer sentiment from",
        ["Review text (Jev)", "Star rating"],
        help="Jev never saw the star rating, so switching shows where the text and the stars disagree.",
    )
    sentiment_col = "text_sentiment" if source.startswith("Review") else "star_sentiment"
    st.subheader("Risk rules")
    at_risk_at = st.slider("At risk from peak frustration", 0.0, 4.0, 3.0, 0.25)
    frustrated_at = st.slider("Frustrated from peak frustration", 0.0, 4.0, 2.0, 0.25)
    st.caption("Frustration is scored 0 (none) to 4 (furious) per sentence; a customer's peak is their "
               "most frustrated sentence. Any churn signal - returning it, a refund, switching brand, "
               "won't buy again - also counts as at risk.")
    st.divider()
    st.caption(f"Classified by TypeSafe AI's {meta['jev_model']} · question set {meta['question_set_version']}")

st.title("BBQ review insights")
st.caption("Fictional customer reviews, split into sentences and classified by TypeSafe AI's Jev. "
           "One review is one customer.")

# --- filters: one row above everything they scope ---
f1, f2, f3 = st.columns(3)
categories = f1.multiselect("Product category", sorted(all_customers["product_category"].unique()), placeholder="All")
products = f2.multiselect(
    "Product",
    sorted(all_customers.filter(pl.col("product_category").is_in(categories) if categories else pl.lit(True))["product_name"].unique()),
    placeholder="All",
)
countries = f3.multiselect("Country", sorted(all_customers["country"].unique()), placeholder="All")

view = assign_risk(
    all_customers.lazy().filter(
        pl.col("product_category").is_in(categories) if categories else pl.lit(True),
        pl.col("product_name").is_in(products) if products else pl.lit(True),
        pl.col("country").is_in(countries) if countries else pl.lit(True),
    ),
    at_risk_frustration=at_risk_at,
    frustrated_frustration=frustrated_at,
).collect()

if view.is_empty():
    st.info("No customers match these filters.")
    st.stop()

# --- headline numbers ---
at_risk = view.filter(pl.col("risk_tier") == "At risk")
negative = view.filter(pl.col(sentiment_col) == "negative").height
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{view.height:,}", border=True)
k2.metric("Negative", f"{negative:,}", f"{negative / view.height:.0%} of customers", delta_color="off", delta_arrow="off", border=True)
k3.metric("At risk", f"{at_risk.height:,}", f"{at_risk.height / view.height:.0%} of customers", delta_color="off", delta_arrow="off", border=True)
k4.metric("Lifetime revenue at risk", f"{at_risk['lifetime_revenue'].sum():,.0f}",
          f"{at_risk['lifetime_revenue'].sum() / view['lifetime_revenue'].sum():.0%} of revenue", delta_color="off", delta_arrow="off", border=True)

REVIEW_COLUMNS = {
    "rating": st.column_config.NumberColumn("Stars", format="%d ★", width="small"),
    "product_name": "Product",
    "country": "Country",
    "days_as_customer": st.column_config.NumberColumn("Tenure (days)", format="%d"),
    "lifetime_revenue": st.column_config.NumberColumn("Lifetime revenue", format="%.2f"),
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
    hide_no_issue = c2.toggle("Hide customers who raised no issue", value=False)
    flow = view.filter(pl.col(sentiment_col).is_in(shown or []))
    if hide_no_issue:
        flow = flow.filter(pl.col("primary_issue") != NO_ISSUE)

    if flow.is_empty():
        st.info("Nothing to trace for this selection.")
    else:
        st.plotly_chart(sankey_figure(sankey_links(flow.lazy(), sentiment_col)), width="stretch", theme=None)
        st.caption("Width is the number of customers. Each customer follows one path, ending at their "
                   "**primary issue**: the problem raised in their most frustrated sentence. "
                   "Hover a link for its count.")

        st.subheader("Customers behind a flow")
        d1, d2 = st.columns(2)
        issue_counts = flow.group_by("primary_issue").len().sort("len", descending=True)
        issue = d1.selectbox("Primary issue", issue_counts["primary_issue"],
                             format_func=lambda i: f"{pretty(i)} ({issue_counts.filter(pl.col('primary_issue') == i)['len'][0]})")
        in_issue = flow.filter(pl.col("primary_issue") == issue)
        product = d2.selectbox("Product", ["All products", *sorted(in_issue["product_name"].unique())])
        if product != "All products":
            in_issue = in_issue.filter(pl.col("product_name") == product)
        review_table(in_issue.sort("peak_frustration", descending=True))

# --- scatter: tenure against lifetime revenue, by risk tier ---
with risk_tab:
    event = st.plotly_chart(
        risk_scatter(view), width="stretch", theme=None, key="risk_scatter",
        on_select="rerun", selection_mode=("points", "box", "lasso"),
    )
    st.caption("Dashed lines are the medians of the customers shown: top right is long-standing, "
               "high-value customers. Box or lasso select to list customers below.")

    picked = [p["customdata"][0] for p in event.selection.points] if event and event.selection.points else []
    if picked:
        st.subheader(f"{len(picked)} selected customer(s)")
        table = view.filter(pl.col("review_row").is_in(picked))
    else:
        st.subheader("At-risk customers, most valuable first")
        table = at_risk
    review_table(table.sort("lifetime_revenue", descending=True))
