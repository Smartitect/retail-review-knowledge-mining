"""
Turn sentence-level answers into things a product or support team can act on.

Every stage takes the classified sentences (one row per sentence per review,
with the Jev columns joined on) and returns a LazyFrame. Thresholds are
arguments, not constants buried in the logic: Jev returns probabilities and
confidences, and where the line sits is a business decision.
"""

import polars as pl

from jev_classifier import NOUL_THRESHOLD

FLAGS = ["churn_risk", "safety_concern", "suggestion", "competitor_mention"]

# Per-review facts carried through from the loader: the review itself, and the
# point-in-time customer features as at the moment it was written.
CARRIED = [
    "customer_id", "product_id", "product_name", "product_category", "rating", "reviewed_at", "review_text",
    "text_language", "text_model", "tenure_days", "lifetime_revenue", "order_count", "days_since_last_order",
    "previous_reviews", "age", "gender", "country",
]


def review_rollup(sentences: pl.LazyFrame, threshold: float = NOUL_THRESHOLD) -> pl.LazyFrame:
    """One row per review: its facts, peak and average frustration, problems and flags."""
    return (
        sentences.group_by("review_id")
        .agg(
            pl.col(*CARRIED).first(),
            pl.len().alias("sentences"),
            pl.col("frustration").max().alias("peak_frustration"),
            pl.col("frustration").mean().alias("mean_frustration"),
            pl.col("problem_category").filter(pl.col("problem_category") != "none").unique().sort().alias("problems"),
            pl.col("products_mentioned").list.explode(keep_nulls=False, empty_as_null=False).unique().sort().alias("products_mentioned"),
            *[(pl.col(f) >= threshold).any().alias(f) for f in FLAGS],
        )
        .sort("review_id")
    )


def frustration_by_rating(reviews: pl.LazyFrame) -> pl.LazyFrame:
    """The sanity check: Jev never saw the star rating, so frustration should fall as it rises."""
    return (
        reviews.group_by("rating")
        .agg(
            pl.len().alias("reviews"),
            pl.col("peak_frustration").mean().round(2),
            pl.col("mean_frustration").mean().round(2),
        )
        .sort("rating")
    )


def problems_by_product(sentences: pl.LazyFrame) -> pl.LazyFrame:
    """Sentences raising each kind of problem, per product, with how frustrated they were."""
    return (
        sentences.filter(pl.col("problem_category") != "none")
        .group_by("product_name", "problem_category")
        .agg(
            pl.len().alias("sentences"),
            pl.col("review_id").n_unique().alias("reviews"),
            pl.col("frustration").mean().round(2).alias("mean_frustration"),
        )
        .sort("sentences", descending=True)
    )


def cross_mentions(sentences: pl.LazyFrame) -> pl.LazyFrame:
    """Products named in a review of a *different* product - compatibility and bundling signals."""
    return (
        sentences.explode("products_mentioned", empty_as_null=False)
        .filter(pl.col("products_mentioned") != pl.col("product_name"))
        .select("product_name", pl.col("products_mentioned").alias("also_mentions"), "sentence")
        .unique()
        .sort("product_name", "also_mentions")
    )


def flagged(sentences: pl.LazyFrame, flag: str, threshold: float = NOUL_THRESHOLD) -> pl.LazyFrame:
    """Distinct sentences where a yes/no flag clears the threshold, most certain first."""
    return (
        sentences.filter(pl.col(flag) >= threshold)
        .group_by("product_name", "sentence")
        .agg(pl.col(flag).first().round(2), pl.len().alias("occurrences"), pl.col("frustration").first().round(2))
        .sort(flag, descending=True)
    )


def needs_review(sentences: pl.LazyFrame, min_confidence: float = 0.6) -> pl.LazyFrame:
    """Sentences where Jev could not separate the problem categories: send to a person."""
    return (
        sentences.filter(pl.col("problem_category_confidence") < min_confidence)
        .select("product_name", "sentence", "problem_category", pl.col("problem_category_confidence").round(2))
        .unique()
        .sort("problem_category_confidence")
    )
