"""
Load reviews from the retail dataset into one flat Polars frame: one row per review.

Each review is joined to its text, its product, and what we knew about the
customer at the moment it was written (`customer_features`): tenure, lifetime
value, orders so far, age, gender and country. Reviews themselves store none
of that, so these are point-in-time facts with no leakage from the future.

`review_key` hashes the product and the text. The same words about the same
product get the same answer from a classifier, so classification is keyed on
it and joined back. Template text repeats across reviews, which makes that
worth doing.

Reviews without text (the text step failed or has not run) are left out, and
the count is printed, so a gap is never silent.
"""

import hashlib
from pathlib import Path

import polars as pl

from customer_features import review_features
from retail_model import DEFAULT_DIR, read_dataset


def content_key(product_name: str, review_text: str) -> str:
    """A key that is stable across runs and Polars versions, unlike `Expr.hash`."""
    return hashlib.sha1(f"{product_name}\x1f{review_text}".encode()).hexdigest()[:12]


def load_reviews(directory: Path | str = DEFAULT_DIR) -> pl.LazyFrame:
    tables = read_dataset(directory)
    if "review_texts" not in tables:
        raise FileNotFoundError(f"{directory} has no review_texts: write review text before loading reviews")
    reviews = tables["reviews"]
    with_text = reviews.join(
        tables["review_texts"].select("review_id", "review_text", pl.col("language").alias("text_language"),
                                      "text_source"),
        on="review_id",
    )
    if missing := reviews.height - with_text.height:
        print(f"{missing} review(s) have no text yet and are left out.")
    frame = (
        with_text.join(tables["products"].select("product_id", "product_name", pl.col("category").alias("product_category")),
                       on="product_id")
        .join(review_features(reviews, tables["customers"], tables["orders"]).drop("customer_id"), on="review_id")
        .sort("review_id")
    )
    keys = [content_key(p, t) for p, t in frame.select("product_name", "review_text").iter_rows()]
    return (
        frame.with_columns(review_key=pl.Series(keys, dtype=pl.String))
        .select("review_id", "review_key", "customer_id", "product_id", "product_name", "product_category",
                "rating", "reviewed_at", "review_text", "text_language", "text_source",
                pl.exclude("review_id", "review_key", "customer_id", "product_id", "product_name", "product_category",
                           "rating", "reviewed_at", "review_text", "text_language", "text_source", "order_line_id",
                           "batch_id"))
        .lazy()
    )


def product_catalogue(directory: Path | str = DEFAULT_DIR) -> pl.DataFrame:
    """The master product list, named as the classifier expects."""
    products = read_dataset(directory)["products"]
    return products.select("product_name", pl.col("category").alias("product_category")).sort(
        "product_category", "product_name"
    )
