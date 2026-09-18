import json

import polars as pl
import pytest

from review_wrangler import (
    content_key,
    load_reviews,
    product_catalogue,
    split_sentences,
)


@pytest.fixture
def reviews_file(tmp_path):
    records = [
        {
            "rating": 2, "review_text": "Handle broke!  Cheap plastic. Going back to my old set.",
            "product_name": "Tongs", "product_category": "accessories",
            "demographics": {"gender": "female", "age": 32, "country": "Canada",
                             "days_as_customer": 10, "lifetime_revenue": 20.0},
        },
        {
            "rating": 5, "review_text": "Great?! Holds 5.5 degrees steady.",
            "product_name": "Grill", "product_category": "grills",
            "demographics": {"gender": "male", "age": 45, "country": "Japan",
                             "days_as_customer": 90, "lifetime_revenue": 999.5},
        },
    ]
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps(records))
    return path


def test_load_flattens_demographics_and_adds_keys(reviews_file):
    df = load_reviews(reviews_file).collect()
    assert df.columns[:2] == ["review_row", "review_key"]
    assert {"country", "age", "lifetime_revenue"} <= set(df.columns)
    assert df["review_key"][0] == content_key("Tongs", df["review_text"][0])


def test_content_key_depends_on_product_and_text():
    assert content_key("A", "x") == content_key("A", "x")
    assert content_key("A", "x") != content_key("B", "x")


def test_split_sentences(reviews_file):
    df = split_sentences(load_reviews(reviews_file)).collect()
    by_review = df.group_by("review_row", maintain_order=True).agg("sentence", "sentence_index")
    assert by_review["sentence"].to_list() == [
        ["Handle broke!", "Cheap plastic.", "Going back to my old set."],
        ["Great?!", "Holds 5.5 degrees steady."],
    ]
    assert by_review["sentence_index"].to_list() == [[0, 1, 2], [0, 1]]
    assert df["sentence_count"].to_list() == [3, 3, 3, 2, 2]


def test_catalogue_is_distinct_products(reviews_file):
    catalogue = product_catalogue(load_reviews(reviews_file))
    assert catalogue.to_dicts() == [
        {"product_name": "Tongs", "product_category": "accessories"},
        {"product_name": "Grill", "product_category": "grills"},
    ]
    assert isinstance(catalogue, pl.DataFrame)
