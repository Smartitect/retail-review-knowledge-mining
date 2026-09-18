from datetime import date, datetime

import pandera.polars as pa
import polars as pl
import pytest

from retail_model import (
    CustomerSchema,
    IntegrityError,
    OrderLineSchema,
    ReviewPropensity,
    check_integrity,
)

T0 = datetime(2025, 1, 1)


def tables(**overrides):
    t = {
        "customers": pl.DataFrame({"customer_id": ["C0000001"], "signup_at": [T0]}),
        "products": pl.DataFrame({"product_id": ["P001", "P002"]}),
        "orders": pl.DataFrame({"order_id": ["O000000001"], "customer_id": ["C0000001"],
                                "ordered_at": [datetime(2025, 2, 1)], "order_total": [30.0]}),
        "order_lines": pl.DataFrame({"order_line_id": ["L0000000001", "L0000000002"],
                                     "order_id": ["O000000001", "O000000001"], "product_id": ["P001", "P002"],
                                     "line_total": [10.0, 20.0]}),
        "reviews": pl.DataFrame({"review_id": ["R0000000001"], "customer_id": ["C0000001"], "product_id": ["P001"],
                                 "order_line_id": ["L0000000001"], "reviewed_at": [datetime(2025, 2, 10)]}),
    }
    return t | overrides


def test_a_consistent_dataset_passes():
    check_integrity(tables())


def test_rules_for_absent_tables_are_skipped():
    check_integrity({"customers": tables()["customers"]})


@pytest.mark.parametrize(("change", "rule"), [
    ({"orders": tables()["orders"].with_columns(customer_id=pl.lit("C9999999"))}, "order.customer_id exists"),
    ({"orders": tables()["orders"].with_columns(ordered_at=pl.lit(datetime(2024, 1, 1)))}, "order placed on or after signup"),
    ({"orders": tables()["orders"].with_columns(order_total=pl.lit(31.0))}, "order_total equals the sum of its lines"),
    ({"order_lines": tables()["order_lines"].with_columns(product_id=pl.lit("P999"))}, "order_line.product_id exists"),
    ({"reviews": tables()["reviews"].with_columns(product_id=pl.lit("P002"))},
     "review is by the customer who bought the line, for the product on it"),
    ({"reviews": tables()["reviews"].with_columns(reviewed_at=pl.lit(datetime(2025, 1, 15)))},
     "review written after the order it reviews"),
])
def test_each_broken_rule_is_named(change, rule):
    with pytest.raises(IntegrityError) as caught:
        check_integrity(tables(**change))
    assert rule in caught.value.failures


def test_line_total_must_be_quantity_times_price():
    lines = pl.DataFrame({"order_line_id": ["L0000000001"], "order_id": ["O000000001"], "product_id": ["P001"],
                          "quantity": pl.Series([2], dtype=pl.Int32), "unit_price": [9.99], "line_total": [19.98],
                          "batch_id": pl.Series([1], dtype=pl.Int32)})
    OrderLineSchema.validate(lines)
    with pytest.raises(pa.errors.SchemaError):
        OrderLineSchema.validate(lines.with_columns(line_total=pl.lit(25.0)))


def test_customer_ids_must_have_the_prefix():
    customer = pl.DataFrame({
        "customer_id": ["C0000001"], "first_name": ["Ann"], "last_name": ["Lee"], "email": ["ann@example.org"],
        "phone": ["1"], "street_address": ["1 Road"], "city": ["Town"], "postcode": ["AB1"],
        "country": ["Canada"], "locale": ["en_CA"], "gender": ["female"], "date_of_birth": [date(1980, 1, 1)],
        "signup_at": [T0], "batch_id": pl.Series([1], dtype=pl.Int32),
    })
    CustomerSchema.validate(customer)
    with pytest.raises(pa.errors.SchemaError):
        CustomerSchema.validate(customer.with_columns(customer_id=pl.lit("X1")))


def test_review_propensity_is_u_shaped():
    propensity = ReviewPropensity(base_rate=0.05, extremity_rate=0.4, sharpness=2)
    s = pl.DataFrame({"s": [0.0, 0.25, 0.5, 0.75, 1.0]})
    p = s.select(propensity.probability(pl.col("s")))["literal"].to_list()
    assert p[0] == pytest.approx(0.45) and p[4] == pytest.approx(0.45)
    assert p[2] == pytest.approx(0.05)
    assert p[1] == pytest.approx(p[3]) and p[2] < p[1] < p[0]


def test_rating_rises_with_satisfaction_and_stays_in_range():
    propensity = ReviewPropensity(rating_noise=0.0)
    s = pl.DataFrame({"s": [0.0, 0.3, 0.5, 0.8, 1.0], "noise": [0.0] * 5})
    ratings = s.select(propensity.rating(pl.col("s"), pl.col("noise")))["literal"].to_list()
    assert ratings == sorted(ratings) and ratings[0] == 1 and ratings[-1] == 5


def test_propensity_rejects_probabilities_above_one():
    with pytest.raises(ValueError, match="must both lie"):
        ReviewPropensity(base_rate=0.6, extremity_rate=0.6)
