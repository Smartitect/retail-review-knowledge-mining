from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from retail_generator import (
    SOUTHERN_HEMISPHERE,
    GeneratorConfig,
    generate_customers,
    generate_orders,
    load_products,
    seasonality,
)
from retail_model import check_integrity

UNTIL = datetime(2026, 6, 30)
PRODUCTS = load_products()


@pytest.fixture(scope="module")
def world():
    customers = generate_customers(1, 1500, seed=11, as_of=UNTIL)
    orders, lines = generate_orders(customers, PRODUCTS, seed=11, until=UNTIL)
    return customers, orders, lines


def test_catalogue_is_the_seventeen_original_products():
    assert PRODUCTS.height == 17
    assert set(PRODUCTS["category"]) == {"grills", "accessories", "consumables"}


def test_orders_are_consistent(world):
    customers, orders, lines = world
    check_integrity({"customers": customers, "products": PRODUCTS, "orders": orders, "order_lines": lines})


def test_same_seed_gives_identical_orders(world):
    customers, orders, _ = world
    again_orders, _ = generate_orders(customers.head(50), PRODUCTS, seed=11, until=UNTIL)
    assert_frame_equal(again_orders, orders.filter(pl.col("customer_id").is_in(customers.head(50)["customer_id"].implode())))


def test_generating_in_two_steps_equals_generating_once(world):
    customers, _, _ = world
    sample = customers.head(200)
    t1 = datetime(2025, 3, 1)
    first, first_lines = generate_orders(sample, PRODUCTS, seed=11, until=t1)
    rest, rest_lines = generate_orders(sample, PRODUCTS, seed=11, since=t1, until=UNTIL)
    once, once_lines = generate_orders(sample, PRODUCTS, seed=11, until=UNTIL)
    assert_frame_equal(pl.concat([first, rest]).sort("order_id"), once.sort("order_id"))
    assert_frame_equal(pl.concat([first_lines, rest_lines]).sort("order_line_id"), once_lines.sort("order_line_id"))


def test_nothing_is_sold_before_launch(world):
    _, orders, lines = world
    sold = lines.join(orders, on="order_id").join(PRODUCTS, on="product_id")
    assert sold.filter(pl.col("ordered_at").dt.date() < pl.col("launched_on")).is_empty()


def test_orders_peak_in_each_hemispheres_summer(world):
    customers, orders, _ = world
    by_month = (
        orders.join(customers.select("customer_id", "country"), on="customer_id")
        .filter(pl.col("ordered_at") >= datetime(2024, 1, 1), pl.col("ordered_at") < datetime(2026, 1, 1))
        .with_columns(south=pl.col("country").is_in(list(SOUTHERN_HEMISPHERE)), month=pl.col("ordered_at").dt.month())
        .group_by("south", "month").len()
    )

    def share(south, months):
        part = by_month.filter(pl.col("south") == south)
        return part.filter(pl.col("month").is_in(months))["len"].sum() / part["len"].sum()

    northern_summer, southern_summer = [6, 7, 8], [12, 1, 2]
    assert share(False, northern_summer) > share(False, southern_summer) * 1.8
    assert share(True, southern_summer) > share(True, northern_summer) * 1.4


def test_consumables_repeat_per_customer(world):
    _, orders, lines = world
    fuel = lines.filter(pl.col("product_id").is_in(["P013", "P014"])).join(orders, on="order_id")
    per_customer = fuel.group_by("customer_id").agg(pl.col("order_id").n_unique().alias("orders"))
    assert per_customer["orders"].median() >= 3


def test_grill_buyers_buy_the_matching_fuel_and_a_cover(world):
    _, orders, lines = world
    bought = lines.join(orders.select("order_id", "customer_id"), on="order_id").join(PRODUCTS, on="product_id")
    everyone = orders["customer_id"].unique()

    def buyers(product_id):
        return bought.filter(pl.col("product_id") == product_id)["customer_id"].unique()

    for fuel, consumable in (("pellet", "P013"), ("charcoal", "P014")):
        grill_buyers = bought.filter(pl.col("category") == "grills", pl.col("fuel_type") == fuel)["customer_id"].unique()
        rate = grill_buyers.is_in(buyers(consumable).implode()).mean()
        assert rate > 2 * (buyers(consumable).len() / everyone.len())

    grill_buyers = bought.filter(pl.col("category") == "grills")["customer_id"].unique()
    others = everyone.filter(~everyone.is_in(grill_buyers.implode()))
    covers = buyers("P007").implode()
    assert grill_buyers.is_in(covers).mean() > others.is_in(covers).mean()


def test_seasonality_flips_between_hemispheres():
    patterns = GeneratorConfig().orders
    july, january = datetime(2025, 7, 15), datetime(2025, 1, 15)
    assert seasonality(july, False, patterns) > seasonality(january, False, patterns)
    assert seasonality(january, True, patterns) > seasonality(july, True, patterns)


def test_black_friday_is_busier_than_early_november():
    patterns = GeneratorConfig().orders
    assert seasonality(datetime(2025, 11, 28), False, patterns) > seasonality(datetime(2025, 11, 5), False, patterns)
