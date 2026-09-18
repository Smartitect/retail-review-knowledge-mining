"""Incremental generation: batches, equivalence, idempotence, atomicity and the CLI. No live services."""

import asyncio
import json
from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from retail_generator import (
    GeneratorConfig,
    OrderPatterns,
    add_customers,
    advance,
    estimate,
    fill_texts,
    generate_orders,
    generate_reviews,
    init_dataset,
    load_manifest,
    load_products,
)
from retail_generator.cli import main
from retail_model import batch_file, read_dataset
from review_writer import ReviewBrief, TemplateReviewWriter

T1, T2 = datetime(2025, 12, 31), datetime(2026, 6, 30)
WRITER = TemplateReviewWriter()


def run(coro):
    return asyncio.run(coro)


def comparable(frame: pl.DataFrame, key: str) -> pl.DataFrame:
    return frame.drop("batch_id", "generated_at", strict=False).sort(key)


def build(directory, *counts, until=None):
    init_dataset(directory, seed=7, as_of=T1)
    for n in counts:
        run(add_customers(directory, count=n, writer=WRITER))
    if until:
        run(advance(directory, to=until, writer=WRITER))
    return read_dataset(directory)


KEYS = {"customers": "customer_id", "orders": "order_id", "order_lines": "order_line_id", "reviews": "review_id",
        "review_truth": "review_id", "review_texts": "review_id"}


def test_two_batches_of_customers_equal_one(tmp_path):
    split = build(tmp_path / "a", 40, 30)
    once = build(tmp_path / "b", 70)
    for name, key in KEYS.items():
        assert_frame_equal(comparable(split[name], key), comparable(once[name], key))


def test_advancing_time_equals_generating_straight_to_the_later_date(tmp_path):
    grown = build(tmp_path, 60, until=T2)
    products = load_products()
    customers = grown["customers"]  # the same people, signed up as they were
    orders, lines = generate_orders(customers, products, seed=7, until=T2)
    reviews, truth = generate_reviews(customers, orders, lines, products, seed=7, until=T2)
    for name, direct, key in (("orders", orders, "order_id"), ("order_lines", lines, "order_line_id"),
                              ("reviews", reviews, "review_id"), ("review_truth", truth, "review_id")):
        assert_frame_equal(comparable(grown[name], key), comparable(direct, key))


def test_an_advance_reviews_earlier_purchases_too(tmp_path):
    grown = build(tmp_path, 60, until=T2)
    advanced = grown["reviews"].filter(pl.col("batch_id") == 2).join(
        grown["order_lines"].select("order_line_id", "batch_id"), on="order_line_id", suffix="_line")
    assert (advanced["batch_id_line"] == 1).any()


def test_total_and_advance_are_idempotent(tmp_path):
    build(tmp_path, 50)
    again = run(add_customers(tmp_path, total=50, writer=WRITER))
    assert again.batch_id is None and "already has 50" in again.note
    run(advance(tmp_path, to=T2, writer=WRITER))
    assert run(advance(tmp_path, to=T2, writer=WRITER)).batch_id is None
    assert len(load_manifest(tmp_path).batches) == 2


def test_a_dry_run_writes_nothing(tmp_path):
    init_dataset(tmp_path, seed=7, as_of=T1)
    report = run(add_customers(tmp_path, count=30, dry_run=True))
    assert report.dry_run and report.counts["customers"] == 30 and report.estimate.reviews > 0
    assert load_manifest(tmp_path).batches == []
    assert not (tmp_path / "customers").exists()


def test_init_refuses_a_directory_of_tables_without_a_manifest(tmp_path):
    load_products().write_parquet(tmp_path / "products.parquet")
    with pytest.raises(ValueError, match="no manifest"):
        init_dataset(tmp_path, seed=7, as_of=T1)


def test_different_settings_are_refused(tmp_path):
    init_dataset(tmp_path, seed=7, as_of=T1)
    init_dataset(tmp_path, seed=7, as_of=T1)  # same settings: fine
    with pytest.raises(ValueError, match="different settings"):
        init_dataset(tmp_path, seed=8, as_of=T1)
    other = GeneratorConfig(orders=OrderPatterns(orders_per_year=9.0))
    with pytest.raises(ValueError, match="config differs"):
        run(add_customers(tmp_path, count=5, writer=WRITER, config=other))


def test_files_from_an_interrupted_batch_are_ignored_then_replaced(tmp_path):
    build(tmp_path, 30)
    orphan = batch_file(tmp_path, "customers", 2)
    orphan.parent.mkdir(exist_ok=True)
    pl.read_parquet(batch_file(tmp_path, "customers", 1)).write_parquet(orphan)  # a half-written batch 2
    assert read_dataset(tmp_path)["customers"].height == 30  # the manifest does not list batch 2 yet

    run(add_customers(tmp_path, count=10, writer=WRITER))
    customers = read_dataset(tmp_path)["customers"]
    assert customers.height == 40 and customers["customer_id"].is_unique().all()


class FlakyWriter(TemplateReviewWriter):
    """Fails for every other review, as a service under load might."""

    async def write(self, brief):
        if int(brief.review_id[-1]) % 2:
            raise TimeoutError("busy")
        return await super().write(brief)


def test_failed_text_is_retried_by_fill_texts(tmp_path):
    init_dataset(tmp_path, seed=7, as_of=T1)
    run(add_customers(tmp_path, count=40, writer=FlakyWriter()))
    data = read_dataset(tmp_path)
    assert data["review_texts"].height < data["reviews"].height

    report = run(fill_texts(tmp_path, writer=WRITER))
    data = read_dataset(tmp_path)
    assert report.batch_id == 2 and data["review_texts"].height == data["reviews"].height
    assert run(fill_texts(tmp_path, writer=WRITER)).batch_id is None


def test_estimate_scales_with_sentences():
    briefs = [ReviewBrief(f"R{i:010d}", "Tongs", "accessories", "", 3, 0.5, "performance") for i in range(10)]
    e = estimate(briefs, foundry_usd_per_million_input=1.0, foundry_usd_per_million_output=2.0)
    assert e.reviews == 10 and e.sentences == sum(b.sentences for b in briefs)
    assert e.jev_tokens == e.sentences * 2_750
    assert e.jev_usd == pytest.approx(e.jev_tokens / 1e6 * 0.042)
    assert e.foundry_usd == pytest.approx(e.foundry_input_tokens / 1e6 + 2 * e.foundry_output_tokens / 1e6)
    assert estimate(briefs).foundry_usd is None


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    d = str(tmp_path / "ds")
    assert main(["--dir", d, "init", "--seed", "3", "--as-of", "2025-12-31"]) == 0
    assert main(["--dir", d, "add-customers", "--count", "20", "--writer", "template"]) == 0
    assert main(["--dir", d, "advance", "--to", "2026-03-31", "--writer", "template"]) == 0
    assert main(["--dir", d, "status"]) == 0
    out = capsys.readouterr().out
    assert "Wrote batch 1 (add-customers)" in out and "Wrote batch 2 (advance)" in out and "20 customers" in out
    assert json.loads((tmp_path / "ds" / "manifest.json").read_text())["as_of"].startswith("2026-03-31")

    for name in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("retail_generator.cli.load_dotenv", lambda: None)
    assert main(["--dir", d, "add-customers", "--count", "5"]) == 1
    assert "not configured" in capsys.readouterr().err
