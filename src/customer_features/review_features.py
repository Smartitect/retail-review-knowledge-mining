"""
What we knew about the customer at the moment each review was written.

Reviews do not store customer facts: they are derived here from `customers`
and `orders`, as at `reviewed_at`, through `point_in_time`. So a customer's
second review sees the orders before it, but never one placed afterwards.

| feature | meaning at the review |
|---|---|
| `tenure_days` | days since signup |
| `lifetime_revenue` | total of orders placed before it |
| `order_count` | orders placed before it |
| `days_since_last_order` | days since the most recent earlier order |
| `previous_reviews` | this customer's earlier reviews |
| `age` | age in whole years |
| `gender`, `country` | as recorded; neither changes over time in this model |
"""

import polars as pl

from .point_in_time import point_in_time

SECONDS_PER_DAY = 86_400


def review_features(reviews: pl.DataFrame, customers: pl.DataFrame, orders: pl.DataFrame) -> pl.DataFrame:
    """One row per review: `review_id`, `customer_id` and the point-in-time features."""
    base = reviews.select("review_id", "customer_id", "reviewed_at").join(
        customers.select("customer_id", "signup_at", "date_of_birth", "gender", "country"), on="customer_id"
    )
    with_orders = point_in_time(
        base, orders, by="customer_id", at="reviewed_at", event_time="ordered_at",
        sums={"lifetime_revenue": "order_total"}, count="order_count", last_event="last_order_at",
    )
    with_reviews = point_in_time(
        with_orders, reviews.select("customer_id", "reviewed_at").rename({"reviewed_at": "earlier_review_at"}),
        by="customer_id", at="reviewed_at", event_time="earlier_review_at", count="previous_reviews",
    )
    return with_reviews.select(
        "review_id",
        "customer_id",
        ((pl.col("reviewed_at") - pl.col("signup_at")).dt.total_seconds() / SECONDS_PER_DAY).alias("tenure_days"),
        pl.col("lifetime_revenue").round(2),
        "order_count",
        ((pl.col("reviewed_at") - pl.col("last_order_at")).dt.total_seconds() / SECONDS_PER_DAY)
        .alias("days_since_last_order"),
        "previous_reviews",
        ((pl.col("reviewed_at").dt.date() - pl.col("date_of_birth")).dt.total_days() // 365.25)
        .cast(pl.Int32).alias("age"),
        "gender",
        "country",
    ).sort("review_id")
