"""The product catalogue: a committed reference table, validated on read."""

from pathlib import Path

import polars as pl

from retail_model import ProductSchema, polars_schema

PRODUCTS = Path(__file__).resolve().parents[2] / "reference_data" / "products.csv"


def load_products(path: Path | str = PRODUCTS) -> pl.DataFrame:
    schema = polars_schema(ProductSchema)
    frame = pl.read_csv(path, schema_overrides={k: v for k, v in schema.items() if k != "launched_on"},
                        try_parse_dates=True)
    return ProductSchema.validate(frame.select(schema.keys()).cast(schema))
