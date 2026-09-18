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
class Holiday:
    """A short burst of extra orders around a date each year, e.g. Black Friday."""

    month: int
    day: int
    boost: float = 0.6
    width_days: float = 7.0


@dataclass(frozen=True)
class OrderPatterns:
    """How customers buy. Rates are per order unless they say otherwise."""

    # How often: a customer's everyday order rate is Gamma-distributed around this mean.
    orders_per_year: float = 2.5
    orders_per_year_shape: float = 2.0

    # When: order intensity x (1 + amplitude * cos(day of year - peak)). The peak is
    # mid-July in the north and six months later in the Southern Hemisphere.
    seasonal_amplitude: float = 0.6
    northern_peak_day_of_year: int = 196
    holidays: tuple[Holiday, ...] = (
        Holiday(month=6, day=16, boost=0.5),    # Father's Day (northern)
        Holiday(month=11, day=28, boost=0.7),   # Black Friday
        Holiday(month=12, day=12, boost=0.4),   # Christmas gifting
    )

    # Grills are bought rarely: on a first order, on a later order while the
    # customer has not bought one from us, or (much less often) as another grill.
    grill_first_order: float = 0.35
    grill_without_one: float = 0.10
    grill_again: float = 0.02
    grill_popularity: dict[str, float] = field(default_factory=lambda: {
        "P001": 0.8, "P002": 1.2, "P003": 1.5, "P004": 1.6, "P005": 0.9, "P006": 1.0,
    })
    # Customers who arrive already owning a grill from elsewhere still buy its fuel.
    owns_grill_elsewhere: float = 0.55
    elsewhere_fuel_weights: dict[str, float] = field(default_factory=lambda: {
        "gas": 0.4, "charcoal": 0.4, "pellet": 0.12, "electric": 0.08,
    })

    # Accessories occasionally; far more likely alongside a new grill.
    accessories_per_order: float = 0.35
    accessory_weights: dict[str, float] = field(default_factory=lambda: {
        "P007": 0.6, "P008": 1.4, "P009": 1.0, "P010": 0.8, "P011": 1.0, "P012": 1.2,
    })
    cover_with_new_grill: float = 0.45
    tool_with_new_grill: float = 0.50

    # Consumables repeatedly: fuel matches the grill (pellets for pellet
    # smokers; lump charcoal and lighters for charcoal), chips and rubs suit any.
    fuel_per_order: float = 0.55
    lighter_with_charcoal: float = 0.35
    chips_per_order: float = 0.12
    rub_per_order: float = 0.18
    # Fuel owners also restock on a cycle, more often in their barbecue season.
    replenish_every_days: float = 75.0
    replenish_shape: float = 4.0
    max_quantity: int = 3

    # Prices drift up over time, and off-season orders sometimes get a discount.
    price_drift_per_year: float = 0.04
    off_season_discount_chance: float = 0.25
    discount_range: tuple[float, float] = (0.10, 0.20)


@dataclass(frozen=True)
class ReviewContent:
    """What a review is about, when it is written, and in which language."""

    # Mean hidden satisfaction per product, calibrated from the original reviews'
    # mean stars as (stars - 1) / 4, so the BackYard King stays the dud it was.
    product_quality: dict[str, float] = field(default_factory=lambda: {
        "P001": 0.875, "P002": 0.35, "P003": 0.11, "P004": 0.775, "P005": 0.675, "P006": 0.375,
        "P007": 0.65, "P008": 0.75, "P009": 0.675, "P010": 0.575, "P011": 0.50, "P012": 0.59,
        "P013": 0.70, "P014": 0.60, "P015": 0.35, "P016": 0.525, "P017": 0.70,
    })
    # How tightly satisfaction clusters around the product's mean (Beta concentration).
    satisfaction_concentration: float = 5.0

    # What an unhappy review is mainly about. The labels match Jev's problem
    # categories, so review_truth can score how well Jev recovers them.
    issues: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "grills": {"performance": 3, "build_quality": 2.5, "assembly": 1, "ease_of_use": 1,
                   "cleaning_maintenance": 1, "shipping_delivery": 0.8, "customer_service": 0.8,
                   "price_value": 1, "safety": 0.4},
        "accessories": {"build_quality": 3, "ease_of_use": 2, "size_capacity": 1.5, "performance": 1.5,
                        "price_value": 1, "safety": 0.5},
        "consumables": {"performance": 3, "price_value": 1.5, "shipping_delivery": 0.8, "build_quality": 1},
    })
    praises: tuple[str, ...] = ("performance", "build_quality", "ease_of_use", "price_value")

    # Reviews come a few days to a few weeks after the order; unhappy customers write sooner.
    median_delay_days: float = 9.0
    delay_spread: float = 0.7
    unhappy_delay_factor: float = 0.6

    # Customers in non-English-speaking countries sometimes write in their own language.
    local_language: dict[str, str] = field(default_factory=lambda: {
        "Brazil": "portuguese", "China": "chinese", "Costa Rica": "spanish", "Germany": "german",
        "Japan": "japanese", "South Africa": "zulu", "Tanzania": "swahili",
    })
    local_language_share: float = 0.35


@dataclass(frozen=True)
class GeneratorConfig:
    start: datetime = datetime(2023, 1, 1)  # noqa: DTZ001 - the model is naive UTC
    customers: CustomerConfig = field(default_factory=CustomerConfig)
    orders: OrderPatterns = field(default_factory=OrderPatterns)
    reviews: ReviewPropensity = field(default_factory=ReviewPropensity)
    review_content: ReviewContent = field(default_factory=ReviewContent)
