"""
The retail data model as pandera schemas: one per table.

    customers 1──* orders 1──* order_lines *──1 products
                                   │
                                   └──0..1 reviews 1──1 review_texts
                                                   1──1 review_truth

Keys are strings with a type prefix (`C…`, `P…`, `O…`, `L…`, `R…`) so an ID
read on its own says what it identifies. Timestamps are naive UTC.

Reviews hold only what a review is: who, what, which purchase, when, and the
stars. Customer facts such as tenure and lifetime value are derived from
orders as at the moment of the review, never stored on it. Review text lives in
its own table because it is produced by a separate, cached step, and the
generator's hidden truth (how satisfied the customer really was) lives in
another so no feature can read it by accident.

Cross-table rules (foreign keys, ordering in time, totals) are in
`integrity.py`: a pandera schema sees one frame at a time.
"""

import pandera.polars as pa
import polars as pl

COUNTRIES = [
    "Australia", "Brazil", "Canada", "China", "Costa Rica", "Germany", "Japan",
    "New Zealand", "South Africa", "Tanzania", "United Kingdom", "United States",
]
CATEGORIES = ["grills", "accessories", "consumables"]
FUEL_TYPES = ["gas", "pellet", "charcoal", "electric", "any"]
TEXT_SOURCES = ["foundry", "template"]


def _id(prefix: str, digits: int, **kwargs):
    return pa.Field(str_matches=rf"^{prefix}\d{{{digits}}}$", **kwargs)


class CustomerSchema(pa.DataFrameModel):
    customer_id: str = _id("C", 7, unique=True)
    first_name: str
    last_name: str
    email: str = pa.Field(str_matches=r"^[^@\s]+@[^@\s]+$")
    phone: str
    street_address: str
    city: str
    postcode: str
    country: str = pa.Field(isin=COUNTRIES)
    locale: str
    gender: str = pa.Field(isin=["female", "male"])
    date_of_birth: pl.Date
    signup_at: pl.Datetime
    batch_id: pl.Int32 = pa.Field(ge=1)

    class Config:
        strict = True


class ProductSchema(pa.DataFrameModel):
    """The catalogue: a curated reference table, so a new category is a finding, not an outage."""

    product_id: str = _id("P", 3, unique=True)
    sku: str = pa.Field(unique=True)
    product_name: str = pa.Field(unique=True)
    category: str = pa.Field(isin=CATEGORIES)
    description: str
    list_price: float = pa.Field(gt=0)
    fuel_type: str = pa.Field(isin=FUEL_TYPES, nullable=True)
    launched_on: pl.Date

    class Config:
        strict = True


class OrderSchema(pa.DataFrameModel):
    order_id: str = _id("O", 9, unique=True)
    customer_id: str = _id("C", 7)
    ordered_at: pl.Datetime
    order_total: float = pa.Field(gt=0)
    batch_id: pl.Int32 = pa.Field(ge=1)

    class Config:
        strict = True


class OrderLineSchema(pa.DataFrameModel):
    order_line_id: str = _id("L", 10, unique=True)
    order_id: str = _id("O", 9)
    product_id: str = _id("P", 3)
    quantity: pl.Int32 = pa.Field(ge=1)
    unit_price: float = pa.Field(gt=0)
    line_total: float = pa.Field(gt=0)
    batch_id: pl.Int32 = pa.Field(ge=1)

    class Config:
        strict = True

    @pa.dataframe_check
    def line_total_is_quantity_times_price(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(
            ((pl.col("quantity") * pl.col("unit_price")).round(2) - pl.col("line_total")).abs() < 0.005
        )


class ReviewSchema(pa.DataFrameModel):
    review_id: str = _id("R", 10, unique=True)
    customer_id: str = _id("C", 7)
    product_id: str = _id("P", 3)
    order_line_id: str = _id("L", 10, unique=True)  # at most one review per purchase
    reviewed_at: pl.Datetime
    rating: pl.Int8 = pa.Field(in_range={"min_value": 1, "max_value": 5})
    batch_id: pl.Int32 = pa.Field(ge=1)

    class Config:
        strict = True


class ReviewTextSchema(pa.DataFrameModel):
    """Generated text, cached by review and prompt so a rerun reuses it rather than regenerating."""

    review_id: str = _id("R", 10)
    prompt_hash: str
    review_text: str = pa.Field(str_length={"min_value": 1})
    language: str  # as written: a template is English whatever the brief asked for
    text_source: str = pa.Field(isin=TEXT_SOURCES)
    model: str
    generated_at: pl.Datetime

    class Config:
        strict = True


class ReviewTruthSchema(pa.DataFrameModel):
    """What the generator intended. For evaluating classifiers - never a feature."""

    review_id: str = _id("R", 10, unique=True)
    satisfaction: float = pa.Field(in_range={"min_value": 0.0, "max_value": 1.0})
    aspect: str
    language: str

    class Config:
        strict = True


TABLES = {
    "customers": CustomerSchema,
    "products": ProductSchema,
    "orders": OrderSchema,
    "order_lines": OrderLineSchema,
    "reviews": ReviewSchema,
    "review_texts": ReviewTextSchema,
    "review_truth": ReviewTruthSchema,
}


def polars_schema(model: type[pa.DataFrameModel]) -> dict[str, pl.DataType]:
    """The Polars dtypes a schema declares, in order - so even an empty frame is typed."""
    return {name: column.dtype.type for name, column in model.to_schema().columns.items()}
