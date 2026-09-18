import asyncio
from datetime import datetime

import polars as pl
import pytest

from retail_generator import add_review_texts, build_dataset
from retail_model import write_dataset
from review_wrangler import (
    content_key,
    load_reviews,
    product_catalogue,
    split_sentences,
)
from review_writer import TemplateReviewWriter


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    tables = build_dataset(60, seed=4, as_of=datetime(2026, 6, 30))
    tables = asyncio.run(add_review_texts(tables, TemplateReviewWriter()))
    return write_dataset(tables, tmp_path_factory.mktemp("ds")), tables


def test_one_row_per_review_with_text_product_and_point_in_time_features(dataset):
    directory, tables = dataset
    reviews = load_reviews(directory).collect()
    assert reviews.height == tables["reviews"].height
    assert reviews["review_id"].is_unique().all()
    assert {"review_text", "product_name", "product_category", "tenure_days", "lifetime_revenue",
            "order_count", "age", "country"} <= set(reviews.columns)
    assert (reviews["lifetime_revenue"] > 0).all()  # every review follows at least one order


def test_review_key_hashes_product_and_text(dataset):
    directory, _ = dataset
    row = load_reviews(directory).collect().row(0, named=True)
    assert row["review_key"] == content_key(row["product_name"], row["review_text"])


def test_reviews_without_text_are_left_out_and_counted(dataset, tmp_path, capsys):
    _, tables = dataset
    partial = tables | {"review_texts": tables["review_texts"].slice(1)}
    reviews = load_reviews(write_dataset(partial, tmp_path)).collect()
    assert reviews.height == tables["reviews"].height - 1
    assert "1 review(s) have no text" in capsys.readouterr().out


def test_catalogue_is_named_for_the_classifier(dataset):
    directory, _ = dataset
    assert product_catalogue(directory).columns == ["product_name", "product_category"]


def test_split_sentences():
    reviews = pl.LazyFrame({"review_id": ["R1", "R2"],
                            "review_text": ["Handle broke!  Cheap plastic. Going back to my old set.",
                                            "Great?! Holds 5.5 degrees steady."]})
    df = split_sentences(reviews).collect()
    by_review = df.group_by("review_id", maintain_order=True).agg("sentence", "sentence_index")
    assert by_review["sentence"].to_list() == [
        ["Handle broke!", "Cheap plastic.", "Going back to my old set."],
        ["Great?!", "Holds 5.5 degrees steady."],
    ]
    assert by_review["sentence_index"].to_list() == [[0, 1, 2], [0, 1]]
    assert df["sentence_count"].to_list() == [3, 3, 3, 2, 2]
