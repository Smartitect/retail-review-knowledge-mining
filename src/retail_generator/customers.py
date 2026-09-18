"""
Fictional customers, generated deterministically with Faker.

Customer number *n* (1-based, global across batches) always comes out the
same for a given dataset seed, because it draws from its own random stream
and reseeds Faker from it. Names and addresses come from the Faker locale for
the customer's country; emails use Faker's reserved example domains, so no
generated address can reach a real inbox.
"""

from datetime import datetime, timedelta

import polars as pl
from faker import Faker

from retail_model import CustomerSchema, polars_schema

from .config import LOCALES, GeneratorConfig
from .random_streams import child_seed, rng

_fakers: dict[str, Faker] = {}


def _faker(locale: str) -> Faker:
    if locale not in _fakers:
        _fakers[locale] = Faker(locale)
    return _fakers[locale]


def customer_id(number: int) -> str:
    return f"C{number:07d}"


def _one_line(text: str) -> str:
    return ", ".join(part.strip() for part in text.splitlines() if part.strip())


def _customer(number: int, seed: int, config: GeneratorConfig, as_of: datetime) -> dict:
    r = rng(seed, "customer", number)
    c = config.customers
    countries = list(c.country_weights)
    weights = [c.country_weights[k] for k in countries]
    country = countries[r.choice(len(countries), p=[w / sum(weights) for w in weights])]
    locale = LOCALES[country]
    gender = "female" if r.random() < c.female_share else "male"
    span = (as_of - config.start).total_seconds()
    signup_at = config.start + timedelta(seconds=int(r.random() * span))
    age_days = int(r.triangular(c.age_min, c.age_mode, c.age_max) * 365.25)

    fake = _faker(locale)
    fake.seed_instance(child_seed(r))
    return {
        "customer_id": customer_id(number),
        "first_name": fake.first_name_female() if gender == "female" else fake.first_name_male(),
        "last_name": fake.last_name(),
        "email": fake.safe_email(),
        "phone": fake.phone_number(),
        "street_address": _one_line(fake.street_address()),
        "city": fake.city(),
        "postcode": fake.postcode(),
        "country": country,
        "locale": locale,
        "gender": gender,
        "date_of_birth": (signup_at - timedelta(days=age_days)).date(),
        "signup_at": signup_at,
    }


def generate_customers(first: int, count: int, *, seed: int, as_of: datetime, batch_id: int = 1,
                       config: GeneratorConfig | None = None) -> pl.DataFrame:
    """Customers numbered `first` .. `first + count - 1`, signed up between `config.start` and `as_of`."""
    config = config or GeneratorConfig()
    if as_of <= config.start:
        raise ValueError(f"as_of {as_of} must be after the dataset start {config.start}")
    rows = [_customer(n, seed, config, as_of) | {"batch_id": batch_id} for n in range(first, first + count)]
    return CustomerSchema.validate(pl.DataFrame(rows, schema=polars_schema(CustomerSchema)))
