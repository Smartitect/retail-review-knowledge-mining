"""
Load the raw review JSON into one flat Polars frame: one row per review.

The source is an array of objects with a nested `demographics` object. Polars
flattens that into `demographics_*` columns, and two keys are added:

- `review_row` - the position in the file. It identifies a review.
- `review_key` - a hash of the product and the text. It identifies what a
  review *says*. The sample repeats the same text under different customers,
  and a model asked the same question about the same words gives the same
  answer, so classification is keyed on this and joined back by it.
"""

import hashlib
import json
from pathlib import Path

import polars as pl

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "input" / "product_reviews.json"


def content_key(product_name: str, review_text: str) -> str:
    """A key that is stable across runs and Polars versions, unlike `Expr.hash`."""
    return hashlib.sha1(f"{product_name}\x1f{review_text}".encode()).hexdigest()[:12]


def load_reviews(path: Path | str = DEFAULT_PATH) -> pl.LazyFrame:
    """The reviews, flattened, with `review_row` and `review_key` first."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = [content_key(r["product_name"], r["review_text"]) for r in records]
    return (
        pl.json_normalize(records, separator="_")
        .with_row_index("review_row")
        .with_columns(review_key=pl.Series(keys, dtype=pl.String))
        .rename(lambda name: name.removeprefix("demographics_"))
        .select("review_row", "review_key", pl.exclude("review_row", "review_key"))
        .lazy()
    )


def product_catalogue(reviews: pl.LazyFrame) -> pl.DataFrame:
    """The master product list: every product reviewed, with its category."""
    return (
        reviews.select("product_name", "product_category")
        .unique()
        .sort("product_category", "product_name")
        .collect()
    )
