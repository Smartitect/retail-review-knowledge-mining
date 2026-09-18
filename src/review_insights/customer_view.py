"""
One row per customer, and the flows and risk tiers the dashboard draws from it.

Each review is written by one customer, so a customer here is a `review_row`
with its demographics. Two things are derived from the sentence answers:

- **Sentiment.** `text_sentiment` comes from Jev: positive or negative when one
  kind of sentence outnumbers the other, mixed when they tie. `star_sentiment`
  comes from the rating (1-2 negative, 3 mixed, 4-5 positive). Jev never saw
  the rating, so the two can be compared.
- **Primary issue.** The problem category of the customer's *most frustrated*
  problem sentence. Giving every customer exactly one issue is what lets a
  Sankey conserve flow: each customer is one path from sentiment to issue.
  Customers who raise no problem end at `no_issue`.
"""

import itertools

import polars as pl

from jev_classifier import NOUL_THRESHOLD

from .review_insights import review_rollup

SENTIMENTS = ["negative", "mixed", "positive"]
NO_ISSUE = "no_issue"
RISK_TIERS = ["At risk", "Frustrated", "Satisfied"]

RAISES_PROBLEM = pl.col("problem_category") != "none"


def customers(sentences: pl.LazyFrame, threshold: float = NOUL_THRESHOLD) -> pl.LazyFrame:
    """One row per customer: demographics, both sentiments, primary issue and flags."""
    per_customer = sentences.group_by("review_row").agg(
        pl.col("gender", "age", "days_as_customer", "lifetime_revenue").first(),
        (pl.col("sentiment") == "positive").sum().alias("positive_sentences"),
        (pl.col("sentiment") == "negative").sum().alias("negative_sentences"),
        pl.col("problem_category")
        .filter(RAISES_PROBLEM)
        .sort_by(pl.col("frustration").filter(RAISES_PROBLEM), descending=True)
        .first()
        .alias("primary_issue"),
    )
    return (
        review_rollup(sentences, threshold)
        .join(per_customer, on="review_row", how="left")
        .with_columns(
            text_sentiment=pl.when(pl.col("positive_sentences") > pl.col("negative_sentences"))
            .then(pl.lit("positive"))
            .when(pl.col("negative_sentences") > pl.col("positive_sentences"))
            .then(pl.lit("negative"))
            .otherwise(pl.lit("mixed")),
            star_sentiment=pl.when(pl.col("rating") <= 2)
            .then(pl.lit("negative"))
            .when(pl.col("rating") == 3)
            .then(pl.lit("mixed"))
            .otherwise(pl.lit("positive")),
            primary_issue=pl.col("primary_issue").fill_null(NO_ISSUE),
        )
    )


def assign_risk(
    customers: pl.LazyFrame, *, at_risk_frustration: float = 3.0, frustrated_frustration: float = 2.0
) -> pl.LazyFrame:
    """Add `risk_tier`.

    At risk: a churn signal (returning, refund, switching, won't buy again), or
    peak frustration at or above `at_risk_frustration`. Frustrated: peak
    frustration at or above `frustrated_frustration`. Otherwise satisfied.
    """
    return customers.with_columns(
        risk_tier=pl.when(pl.col("churn_risk") | (pl.col("peak_frustration") >= at_risk_frustration))
        .then(pl.lit(RISK_TIERS[0]))
        .when(pl.col("peak_frustration") >= frustrated_frustration)
        .then(pl.lit(RISK_TIERS[1]))
        .otherwise(pl.lit(RISK_TIERS[2]))
    )


def sankey_links(customers: pl.LazyFrame, sentiment: str = "text_sentiment") -> pl.DataFrame:
    """Customer counts for each hop: sentiment -> category -> product -> primary issue.

    Every hop is also split by sentiment, so a link can be coloured by it and a
    negative customer can be followed all the way down. Nodes are identified by
    `(level, name)` pairs, because the same word could appear at two levels.
    """
    levels = [("sentiment", sentiment), ("category", "product_category"),
              ("product", "product_name"), ("issue", "primary_issue")]
    hops = [
        customers.group_by(pl.col(sentiment).alias("sentiment"), pl.col(src).alias("source"), pl.col(tgt).alias("target"))
        .agg(pl.len().alias("customers"))
        .with_columns(source_level=pl.lit(src_level), target_level=pl.lit(tgt_level))
        for (src_level, src), (tgt_level, tgt) in itertools.pairwise(levels)
    ]
    return (
        pl.concat(hops)
        .select("source_level", "source", "target_level", "target", "sentiment", "customers")
        .sort("source_level", "source", "target", "sentiment")
        .collect()
    )
