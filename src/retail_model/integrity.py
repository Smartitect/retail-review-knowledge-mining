"""
The rules that span tables, which no single-frame schema can see.

`check_integrity` takes whichever tables exist and checks every rule whose
tables are all present, so it can run on a partial dataset (customers and
products before any orders) as well as a complete one. It raises one
`IntegrityError` listing every broken rule with a sample of offending keys,
rather than stopping at the first.
"""

import polars as pl

SAMPLE = 5


class IntegrityError(ValueError):
    def __init__(self, failures: dict[str, list]):
        self.failures = failures
        lines = [f"- {rule}: e.g. {keys}" for rule, keys in failures.items()]
        super().__init__("Referential integrity failed:\n" + "\n".join(lines))


def _orphans(child: pl.DataFrame, parent: pl.DataFrame, key: str) -> list:
    return child.join(parent.select(key), on=key, how="anti")[key].head(SAMPLE).to_list()


def _rules(t: dict[str, pl.DataFrame]):
    """(rule, required tables, function returning a sample of offending keys)."""
    yield "order.customer_id exists", ("orders", "customers"), lambda: _orphans(t["orders"], t["customers"], "customer_id")
    yield "order_line.order_id exists", ("order_lines", "orders"), lambda: _orphans(t["order_lines"], t["orders"], "order_id")
    yield "order_line.product_id exists", ("order_lines", "products"), lambda: _orphans(t["order_lines"], t["products"], "product_id")
    yield "every order has a line", ("orders", "order_lines"), lambda: _orphans(t["orders"], t["order_lines"], "order_id")
    yield "order placed on or after signup", ("orders", "customers"), lambda: (
        t["orders"].join(t["customers"].select("customer_id", "signup_at"), on="customer_id")
        .filter(pl.col("ordered_at") < pl.col("signup_at"))["order_id"].head(SAMPLE).to_list()
    )
    yield "order_total equals the sum of its lines", ("orders", "order_lines"), lambda: (
        t["orders"].join(t["order_lines"].group_by("order_id").agg(pl.col("line_total").sum().alias("lines")), on="order_id")
        .filter((pl.col("order_total") - pl.col("lines")).abs() >= 0.005)["order_id"].head(SAMPLE).to_list()
    )
    yield "review.customer_id exists", ("reviews", "customers"), lambda: _orphans(t["reviews"], t["customers"], "customer_id")
    yield "review.product_id exists", ("reviews", "products"), lambda: _orphans(t["reviews"], t["products"], "product_id")
    yield "review.order_line_id exists", ("reviews", "order_lines"), lambda: _orphans(t["reviews"], t["order_lines"], "order_line_id")
    yield "review is by the customer who bought the line, for the product on it", ("reviews", "order_lines", "orders"), lambda: (
        t["reviews"]
        .join(t["order_lines"].select("order_line_id", "order_id", pl.col("product_id").alias("line_product")), on="order_line_id")
        .join(t["orders"].select("order_id", pl.col("customer_id").alias("buyer"), "ordered_at"), on="order_id")
        .filter((pl.col("product_id") != pl.col("line_product")) | (pl.col("customer_id") != pl.col("buyer")))
        ["review_id"].head(SAMPLE).to_list()
    )
    yield "review written after the order it reviews", ("reviews", "order_lines", "orders"), lambda: (
        t["reviews"]
        .join(t["order_lines"].select("order_line_id", "order_id"), on="order_line_id")
        .join(t["orders"].select("order_id", "ordered_at"), on="order_id")
        .filter(pl.col("reviewed_at") <= pl.col("ordered_at"))["review_id"].head(SAMPLE).to_list()
    )
    yield "review_text.review_id exists", ("review_texts", "reviews"), lambda: _orphans(t["review_texts"], t["reviews"], "review_id")
    yield "review_truth.review_id exists", ("review_truth", "reviews"), lambda: _orphans(t["review_truth"], t["reviews"], "review_id")


def check_integrity(tables: dict[str, pl.DataFrame]) -> None:
    """Raise `IntegrityError` if any cross-table rule is broken; rules whose tables are absent are skipped."""
    failures = {
        rule: keys
        for rule, needs, find in _rules(tables)
        if all(name in tables for name in needs) and (keys := find())
    }
    if failures:
        raise IntegrityError(failures)
