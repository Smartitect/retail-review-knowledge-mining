"""The retail data model: table schemas, cross-table integrity, and review propensity."""

from .integrity import IntegrityError, check_integrity
from .review_propensity import ReviewPropensity
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
    "CATEGORIES", "COUNTRIES", "FUEL_TYPES", "TABLES", "TEXT_SOURCES", "CustomerSchema", "IntegrityError",
    "OrderLineSchema", "OrderSchema", "ProductSchema", "ReviewPropensity", "ReviewSchema", "ReviewTextSchema",
    "ReviewTruthSchema", "check_integrity", "polars_schema",
]
