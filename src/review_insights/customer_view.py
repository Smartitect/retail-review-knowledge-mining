"""
One row per review and one row per customer, and the flows and risk tiers the dashboard draws.

A customer can now write several reviews, so there are two views:

- **Reviews** (`reviews`): one row per review, with the customer's point-in-time
  features as at that review. The Sankey counts these.
- **Customers** (`customers`): one row per customer, taken from their *most
  recent* review. A customer who complained and later wrote a glowing review
  is judged on the glowing one. The risk scatter plots these, at the tenure and
  lifetime value they had when they wrote it.

Derived per review:

- **Sentiment.** `text_sentiment` comes from Jev: positive or negative when one
  kind of sentence outnumbers the other, mixed when they tie. `star_sentiment`
  comes from the rating (1-2 negative, 3 mixed, 4-5 positive). Jev never saw
  the rating, so the two can be compared.
- **Primary issue.** The problem category of the most frustrated problem
  sentence. One issue per review lets the Sankey conserve flow; reviews raising
  no problem end at `no_issue`.
"""

import itertools

import polars as pl

from jev_classifier import NOUL_THRESHOLD

from .review_insights import review_rollup

SENTIMENTS = ["negative", "mixed", "positive"]
NO_ISSUE = "no_issue"
RISK_TIERS = ["At risk", "Frustrated", "Satisfied"]

RAISES_PROBLEM = pl.col("problem_category") != "none"


def reviews(sentences: pl.LazyFrame, threshold: float = NOUL_THRESHOLD) -> pl.LazyFrame:
    """One row per review: facts, point-in-time features, both sentiments, primary issue and flags."""
    per_review = sentences.group_by("review_id").agg(
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
        .join(per_review, on="review_id", how="left")
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


def assign_risk(frame: pl.LazyFrame, *, at_risk_frustration: float = 3.0,
                frustrated_frustration: float = 2.0) -> pl.LazyFrame:
    """Add `risk_tier` to reviews.

    At risk: a churn signal (returning, refund, switching, won't buy again), or
    peak frustration at or above `at_risk_frustration`. Frustrated: peak
    frustration at or above `frustrated_frustration`. Otherwise satisfied.
    """
    return frame.with_columns(
        risk_tier=pl.when(pl.col("churn_risk") | (pl.col("peak_frustration") >= at_risk_frustration))
        .then(pl.lit(RISK_TIERS[0]))
        .when(pl.col("peak_frustration") >= frustrated_frustration)
        .then(pl.lit(RISK_TIERS[1]))
        .otherwise(pl.lit(RISK_TIERS[2]))
    )


def customers(reviews: pl.LazyFrame) -> pl.LazyFrame:
    """One row per customer: their most recent review, with how many they have written."""
    return (
        reviews.sort("reviewed_at")
        .group_by("customer_id")
        .agg(pl.all().last(), pl.len().alias("reviews_written"))
        .sort("customer_id")
    )


def sankey_links(reviews: pl.LazyFrame, sentiment: str = "text_sentiment") -> pl.DataFrame:
    """Review counts for each hop: sentiment -> category -> product -> primary issue.

    Every hop is also split by sentiment, so a link can be coloured by it and a
    negative review can be followed all the way down. Nodes are identified by
    `(level, name)` pairs, because the same word could appear at two levels.
    """
    levels = [("sentiment", sentiment), ("category", "product_category"),
              ("product", "product_name"), ("issue", "primary_issue")]
    hops = [
        reviews.group_by(pl.col(sentiment).alias("sentiment"), pl.col(src).alias("source"), pl.col(tgt).alias("target"))
        .agg(pl.len().alias("reviews"))
        .with_columns(source_level=pl.lit(src_level), target_level=pl.lit(tgt_level))
        for (src_level, src), (tgt_level, tgt) in itertools.pairwise(levels)
    ]
    return (
        pl.concat(hops)
        .select("source_level", "source", "target_level", "target", "sentiment", "reviews")
        .sort("source_level", "source", "target", "sentiment")
        .collect()
    )
