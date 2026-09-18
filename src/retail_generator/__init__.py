"""Deterministic generation of the retail dataset: customers, orders and reviews."""

from .catalogue import PRODUCTS, load_products
from .config import (
    LOCALES,
    SOUTHERN_HEMISPHERE,
    CustomerConfig,
    GeneratorConfig,
    Holiday,
    OrderPatterns,
    ReviewContent,
)
from .customers import customer_id, generate_customers
from .dataset import add_review_texts, build_dataset
from .estimate import Estimate, estimate
from .incremental import (
    add_customers,
    advance,
    fill_texts,
    init_dataset,
    load_manifest,
    status,
)
from .orders import generate_orders, seasonality
from .random_streams import rng
from .reviews import generate_reviews, review_briefs

__all__ = [
    "LOCALES",
    "PRODUCTS",
    "SOUTHERN_HEMISPHERE",
    "CustomerConfig",
    "Estimate",
    "GeneratorConfig",
    "Holiday",
    "OrderPatterns",
    "ReviewContent",
    "add_customers",
    "add_review_texts",
    "advance",
    "build_dataset",
    "customer_id",
    "estimate",
    "fill_texts",
    "generate_customers",
    "generate_orders",
    "generate_reviews",
    "init_dataset",
    "load_manifest",
    "load_products",
    "review_briefs",
    "rng",
    "seasonality",
    "status",
]
