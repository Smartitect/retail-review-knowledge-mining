from datetime import date, datetime

import polars as pl
import pytest

from customer_features import point_in_time, review_features
from retail_generator import (
    generate_customers,
    generate_orders,
    generate_reviews,
    load_products,
)

CUSTOMERS = pl.DataFrame({
    "customer_id": ["C0000001"], "signup_at": [datetime(2025, 1, 1)], "date_of_birth": [date(1980, 6, 1)],
    "gender": ["female"], "country": ["Canada"],
})


def orders(*rows):
    return pl.DataFrame([{"order_id": f"O{i:09d}", "customer_id": "C0000001", "ordered_at": t, "order_total": v}
                         for i, (t, v) in enumerate(rows)])


def reviews(*times):
    return pl.DataFrame([{"review_id": f"R{i:010d}", "customer_id": "C0000001", "reviewed_at": t}
                         for i, t in enumerate(times)])


def features(o, r):
    return review_features(r, CUSTOMERS, o).row(0, named=True) if r.height == 1 else review_features(r, CUSTOMERS, o)


def test_a_review_on_the_day_of_its_order_counts_that_order():
    f = features(orders((datetime(2025, 3, 1, 9), 100.0)), reviews(datetime(2025, 3, 1, 18)))
    assert f["lifetime_revenue"] == 100.0 and f["order_count"] == 1
    assert f["days_since_last_order"] == pytest.approx(9 / 24)


def test_orders_after_the_review_do_not_count():
    f = features(orders((datetime(2025, 2, 1), 50.0), (datetime(2025, 4, 1), 5000.0)), reviews(datetime(2025, 3, 1)))
    assert f["lifetime_revenue"] == 50.0 and f["order_count"] == 1


def test_an_order_at_exactly_the_review_time_does_not_count():
    at = datetime(2025, 3, 1, 12)
    f = features(orders((datetime(2025, 2, 1), 50.0), (at, 70.0)), reviews(at))
    assert f["lifetime_revenue"] == 50.0


def test_a_second_review_sees_the_first_ones_order_but_not_later_ones():
    o = orders((datetime(2025, 2, 1), 40.0), (datetime(2025, 5, 1), 60.0), (datetime(2025, 9, 1), 900.0))
    f = features(o, reviews(datetime(2025, 2, 10), datetime(2025, 6, 1)))
    assert f["lifetime_revenue"].to_list() == [40.0, 100.0]
    assert f["order_count"].to_list() == [1, 2]
    assert f["previous_reviews"].to_list() == [0, 1]


def test_tenure_and_age_are_measured_at_the_review():
    f = features(orders((datetime(2025, 1, 5), 10.0)), reviews(datetime(2025, 7, 1)))
    assert f["tenure_days"] == pytest.approx(181.0)
    assert f["age"] == 45


def test_no_earlier_orders_gives_zero_not_null():
    f = features(orders((datetime(2025, 9, 1), 10.0)), reviews(datetime(2025, 3, 1)))
    assert f["lifetime_revenue"] == 0 and f["order_count"] == 0 and f["days_since_last_order"] is None


def test_point_in_time_counts_simultaneous_events_together():
    left = pl.DataFrame({"k": ["a"], "at": [datetime(2025, 1, 3)]})
    events = pl.DataFrame({"k": ["a", "a", "a"], "t": [datetime(2025, 1, 1)] * 2 + [datetime(2025, 1, 2)],
                           "v": [1.0, 2.0, 4.0]})
    out = point_in_time(left, events, by="k", at="at", event_time="t", sums={"total": "v"}, count="n")
    assert out.row(0, named=True) == {"k": "a", "at": datetime(2025, 1, 3), "total": 7.0, "n": 3}


def test_future_data_cannot_change_any_feature():
    """The leakage test: append huge events after every review; no feature may move."""
    until = datetime(2026, 6, 30)
    products = load_products()
    c = generate_customers(1, 150, seed=5, as_of=until)
    o, lines = generate_orders(c, products, seed=5, until=until)
    r, _ = generate_reviews(c, o, lines, products, seed=5, until=until)
    before = review_features(r, c, o)

    last_review = r.group_by("customer_id").agg(pl.col("reviewed_at").max())
    future = last_review.select(
        pl.format("O9{}", pl.col("customer_id").str.slice(1, 7)).str.slice(0, 10).alias("order_id"),
        "customer_id",
        (pl.col("reviewed_at") + pl.duration(seconds=1)).alias("ordered_at"),
        pl.lit(1_000_000.0).alias("order_total"),
    )
    future_reviews = last_review.select(
        pl.lit("R9999999999").alias("review_id"), "customer_id",
        (pl.col("reviewed_at") + pl.duration(days=1)).alias("reviewed_at"),
    ).head(1)
    after = review_features(pl.concat([r.select(future_reviews.columns), future_reviews]), c,
                            pl.concat([o.select(future.columns), future]))
    assert after.filter(pl.col("review_id") != "R9999999999").equals(before)


def test_features_match_a_brute_force_calculation():
    until = datetime(2026, 6, 30)
    products = load_products()
    c = generate_customers(1, 150, seed=9, as_of=until)
    o, lines = generate_orders(c, products, seed=9, until=until)
    r, _ = generate_reviews(c, o, lines, products, seed=9, until=until)
    brute = (
        r.join(o, on="customer_id").filter(pl.col("ordered_at") < pl.col("reviewed_at"))
        .group_by("review_id").agg(pl.col("order_total").sum().round(2).alias("expected"))
    )
    got = review_features(r, c, o).join(brute, on="review_id", how="left").with_columns(pl.col("expected").fill_null(0))
    assert (got["lifetime_revenue"] - got["expected"]).abs().max() < 0.01
