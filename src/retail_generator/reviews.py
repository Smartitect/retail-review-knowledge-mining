"""
Reviews for a biased subset of purchases, and the briefs their text is written from.

Every order line gets a hidden satisfaction, drawn around its product's quality.
Whether it is reviewed follows the U-shaped `ReviewPropensity`, so the delighted
and the disappointed are over-represented, as in reality. A review is written
days or weeks after the order (sooner when unhappy), is about one aspect, and
is sometimes in the customer's own language.

Each line draws from its own random stream, so whether and when it is reviewed
never depends on how the dataset was batched. A review belongs to the time
slice its `reviewed_at` falls in, even if the order was placed in an earlier
one. That is why `generate_reviews` is given every order line, not just new ones.
"""

from datetime import datetime

import polars as pl

from retail_model import ReviewSchema, ReviewTruthSchema, polars_schema
from review_writer import ReviewBrief

from .config import GeneratorConfig
from .random_streams import rng

MIN_DELAY_DAYS = 1.0  # nobody reviews before it arrives


def _draws(lines: pl.DataFrame, seed: int, config: GeneratorConfig) -> pl.DataFrame:
    """The per-line random draws, each line from its own stream."""
    content = config.review_content
    rows = []
    for line_id, product_id, category, country in lines.select(
        "order_line_id", "product_id", "category", "country"
    ).iter_rows():
        r = rng(seed, "review", int(line_id[1:]))
        q = content.product_quality[product_id]
        k = content.satisfaction_concentration
        satisfaction = float(r.beta(q * k, (1 - q) * k))
        issues = content.issues[category]
        issue = list(issues)[r.choice(len(issues), p=[w / sum(issues.values()) for w in issues.values()])]
        praise = content.praises[r.integers(len(content.praises))]
        local = content.local_language.get(country)
        rows.append({
            "order_line_id": line_id,
            "satisfaction": satisfaction,
            "review_draw": float(r.random()),
            "rating_noise": float(r.standard_normal()),
            "delay_days": float(r.lognormal(0.0, content.delay_spread)) * content.median_delay_days,
            "aspect": issue if satisfaction < 0.5 else praise,
            "language": local if local and r.random() < content.local_language_share else "english",
        })
    return pl.DataFrame(rows, schema={"order_line_id": pl.String, "satisfaction": pl.Float64,
                                      "review_draw": pl.Float64, "rating_noise": pl.Float64,
                                      "delay_days": pl.Float64, "aspect": pl.String, "language": pl.String})


def generate_reviews(customers: pl.DataFrame, orders: pl.DataFrame, order_lines: pl.DataFrame,
                     products: pl.DataFrame, *, seed: int, until: datetime, since: datetime | None = None,
                     batch_id: int = 1, config: GeneratorConfig | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reviews written after `since` (exclusive) and up to `until` (inclusive), and their hidden truth."""
    config = config or GeneratorConfig()
    content, propensity = config.review_content, config.reviews
    lines = (
        order_lines.join(orders.select("order_id", "customer_id", "ordered_at"), on="order_id")
        .join(products.select("product_id", "category"), on="product_id")
        .join(customers.select("customer_id", "country"), on="customer_id")
        .filter(pl.col("ordered_at") <= until)
    )
    delay = pl.max_horizontal(
        pl.col("delay_days") * pl.when(pl.col("satisfaction") < 0.5).then(content.unhappy_delay_factor).otherwise(1.0),
        pl.lit(MIN_DELAY_DAYS),
    )
    reviewed = (
        lines.join(_draws(lines, seed, config), on="order_line_id")
        .with_columns(
            reviewed_at=pl.col("ordered_at") + pl.duration(seconds=(delay * 86400).round().cast(pl.Int64)),
            rating=propensity.rating(pl.col("satisfaction"), pl.col("rating_noise")),
        )
        .filter(
            pl.col("review_draw") < propensity.probability(pl.col("satisfaction")),
            pl.col("reviewed_at") <= until,
            pl.lit(True) if since is None else pl.col("reviewed_at") > since,
        )
        .with_columns(review_id=pl.lit("R") + pl.col("order_line_id").str.slice(1), batch_id=pl.lit(batch_id, pl.Int32))
        .sort("review_id")
    )
    return (
        ReviewSchema.validate(reviewed.select(polars_schema(ReviewSchema).keys()).cast(polars_schema(ReviewSchema))),
        ReviewTruthSchema.validate(reviewed.select(polars_schema(ReviewTruthSchema).keys())),
    )


def review_briefs(reviews: pl.DataFrame, truth: pl.DataFrame, products: pl.DataFrame) -> list[ReviewBrief]:
    """What to ask the review writer for each review."""
    rows = (
        reviews.join(truth, on="review_id")
        .join(products.select("product_id", "product_name", "category", "description"), on="product_id")
        .sort("review_id")
    )
    return [
        ReviewBrief(review_id=r["review_id"], product_name=r["product_name"], product_category=r["category"],
                    product_description=r["description"], rating=r["rating"], satisfaction=r["satisfaction"],
                    aspect=r["aspect"], language=r["language"])
        for r in rows.iter_rows(named=True)
    ]
