"""
A whole dataset in one call: customers, their orders, and the reviews they wrote.

Structured data is generated synchronously and deterministically. Review text
is a separate, asynchronous step, because it calls a model (Azure AI Foundry),
and is cached by prompt.
"""

from datetime import datetime
from pathlib import Path

import polars as pl

from review_writer import ReviewWriter, write_review_texts

from .catalogue import load_products
from .config import GeneratorConfig
from .customers import generate_customers
from .orders import generate_orders
from .reviews import generate_reviews, review_briefs


def build_dataset(customers: int, *, seed: int, as_of: datetime,
                  config: GeneratorConfig | None = None) -> dict[str, pl.DataFrame]:
    """Every table except `review_texts`, for customers 1..`customers`, as at `as_of`."""
    config = config or GeneratorConfig()
    products = load_products()
    people = generate_customers(1, customers, seed=seed, as_of=as_of, config=config)
    orders, lines = generate_orders(people, products, seed=seed, until=as_of, config=config)
    reviews, truth = generate_reviews(people, orders, lines, products, seed=seed, until=as_of, config=config)
    return {"customers": people, "products": products, "orders": orders, "order_lines": lines,
            "reviews": reviews, "review_truth": truth}


async def add_review_texts(tables: dict[str, pl.DataFrame], writer: ReviewWriter, *,
                           cache_path: Path | str | None = None, concurrency: int = 8) -> dict[str, pl.DataFrame]:
    """`tables` plus `review_texts`, written by `writer` for every review (cached by prompt)."""
    briefs = review_briefs(tables["reviews"], tables["review_truth"], tables["products"])
    texts = await write_review_texts(briefs, writer, cache_path=cache_path, concurrency=concurrency)
    return tables | {"review_texts": texts}
