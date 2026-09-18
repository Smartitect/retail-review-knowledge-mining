"""The retail data model: table schemas, cross-table integrity, and review propensity."""

from .integrity import IntegrityError, check_integrity
from .review_propensity import ReviewPropensity
from .storage import DEFAULT_DIR, read_dataset, write_dataset
from .tabular_schemas import (
    CATEGORIES,
    COUNTRIES,
    FUEL_TYPES,
    TABLES,
    TEXT_SOURCES,
    CustomerSchema,
    OrderLineSchema,
    OrderSchema,
    ProductSchema,
    ReviewSchema,
    ReviewTextSchema,
    ReviewTruthSchema,
    polars_schema,
)

__all__ = [
    "CATEGORIES",
    "COUNTRIES",
    "DEFAULT_DIR",
    "FUEL_TYPES",
    "TABLES",
    "TEXT_SOURCES",
    "CustomerSchema",
    "IntegrityError",
    "OrderLineSchema",
    "OrderSchema",
    "ProductSchema",
    "ReviewPropensity",
    "ReviewSchema",
    "ReviewTextSchema",
    "ReviewTruthSchema",
    "check_integrity",
    "polars_schema",
    "read_dataset",
    "write_dataset",
]
