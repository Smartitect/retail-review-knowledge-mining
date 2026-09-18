from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from retail_generator import (
    LOCALES,
    CustomerConfig,
    GeneratorConfig,
    generate_customers,
)

AS_OF = datetime(2026, 6, 30)


def customers(first=1, count=40, seed=7, **kwargs):
    return generate_customers(first, count, seed=seed, as_of=AS_OF, **kwargs)


def test_same_seed_gives_an_identical_frame():
    assert_frame_equal(customers(), customers())


def test_different_seeds_give_different_customers():
    assert customers(seed=1)["email"].to_list() != customers(seed=2)["email"].to_list()


def test_a_customer_is_the_same_whichever_batch_generates_it():
    whole = customers(first=1, count=10)
    part = customers(first=5, count=3)
    assert_frame_equal(part, whole.slice(4, 3))


def test_ids_follow_the_numbering():
    assert customers(first=998, count=3)["customer_id"].to_list() == ["C0000998", "C0000999", "C0001000"]


@pytest.mark.parametrize("country", sorted(LOCALES))
def test_every_country_generates_valid_customers(country):
    config = GeneratorConfig(customers=CustomerConfig(country_weights={country: 1.0}))
    frame = customers(count=5, config=config)
    assert set(frame["country"]) == {country}
    assert set(frame["locale"]) == {LOCALES[country]}
    assert not frame["street_address"].str.contains("\n").any()


def test_emails_only_use_reserved_example_domains():
    domains = customers(count=200)["email"].str.split("@").list.last().unique()
    assert set(domains) <= {"example.com", "example.org", "example.net"}


def test_signups_and_ages_are_in_range():
    config = GeneratorConfig()
    frame = customers(count=300)
    assert frame["signup_at"].min() >= config.start and frame["signup_at"].max() <= AS_OF
    age_years = (frame["signup_at"].dt.date() - frame["date_of_birth"]).dt.total_days() / 365.25
    assert age_years.min() >= config.customers.age_min - 0.01
    assert age_years.max() <= config.customers.age_max + 0.01


def test_country_weights_are_respected():
    config = GeneratorConfig(customers=CustomerConfig(country_weights={"Japan": 3.0, "Germany": 1.0}))
    counts = customers(count=800, config=config)["country"].value_counts()
    share = counts.filter(pl.col("country") == "Japan")["count"][0] / 800
    assert 0.68 < share < 0.82


def test_as_of_must_follow_the_start():
    with pytest.raises(ValueError, match="must be after"):
        generate_customers(1, 1, seed=1, as_of=datetime(2020, 1, 1))


def test_an_empty_batch_is_typed():
    assert customers(count=0).schema["signup_at"] == pl.Datetime("us")
