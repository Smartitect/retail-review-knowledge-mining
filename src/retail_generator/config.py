"""
Every knob the generator has, in one place.

Nothing else in the package hard-codes a rate, a weight or a date: change the
dataset by changing a `GeneratorConfig`. Defaults aim for a dataset that looks
like a plausible small online barbecue retailer.
"""

from dataclasses import dataclass, field
from datetime import datetime

from retail_model import ReviewPropensity

# Faker locale per country. Faker has no Costa Rican or South African English
# locale, so those fall back to the nearest one it does have.
LOCALES = {
    "Australia": "en_AU", "Brazil": "pt_BR", "Canada": "en_CA", "China": "zh_CN",
    "Costa Rica": "es_MX", "Germany": "de_DE", "Japan": "ja_JP", "New Zealand": "en_NZ",
    "South Africa": "zu_ZA", "Tanzania": "sw", "United Kingdom": "en_GB", "United States": "en_US",
}

SOUTHERN_HEMISPHERE = frozenset({"Australia", "Brazil", "New Zealand", "South Africa"})


@dataclass(frozen=True)
class CustomerConfig:
    country_weights: dict[str, float] = field(default_factory=lambda: dict.fromkeys(LOCALES, 1.0))
    female_share: float = 0.45
    age_min: int = 18
    age_mode: int = 42
    age_max: int = 78


@dataclass(frozen=True)
class GeneratorConfig:
    start: datetime = datetime(2023, 1, 1)  # noqa: DTZ001 - the model is naive UTC
    customers: CustomerConfig = field(default_factory=CustomerConfig)
    reviews: ReviewPropensity = field(default_factory=ReviewPropensity)
