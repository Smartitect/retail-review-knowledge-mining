"""
Customer order histories with line items, following `OrderPatterns`.

Each customer is simulated in time order from signup, using three random
streams of their own: when they order, what goes in the basket, and when they
restock fuel. Because every draw happens in time order, simulating to a later
date reproduces the earlier history exactly and then continues it. So
`generate_orders(..., since=t1, until=t2)` returns precisely the orders a
dataset generated to `t1` is missing to reach `t2`, and IDs stay stable:

    order_id       O + customer number (7 digits) + order number (2 digits)
    order_line_id  L + order_id digits (9) + line number (1 digit)

Buying patterns modelled:

- Seasonality: order intensity follows a yearly cosine peaking mid-July, or
  mid-January for Southern Hemisphere customers, plus holiday bursts.
- Grills are bought rarely; accessories occasionally, and far more often with a
  new grill (a cover, a tool); consumables repeatedly.
- Fuel matches the grill: pellets for pellet smokers, lump charcoal and
  lighters for charcoal. Customers may own a grill from elsewhere and still buy
  its fuel.
- Fuel owners also restock on a cycle, more often in their barbecue season.
- Prices drift up over time; off-season orders are sometimes discounted, which
  is why each line stores its own unit price.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from retail_model import OrderLineSchema, OrderSchema, polars_schema

from .config import SOUTHERN_HEMISPHERE, GeneratorConfig, OrderPatterns
from .random_streams import rng

MAX_ORDERS = 99
MAX_LINES = 9
DAYS_PER_YEAR = 365.25
TOOLS = ("P008", "P009", "P010", "P011", "P012")
PELLETS, CHARCOAL, CHIPS, LIGHTER, RUB, COVER = "P013", "P014", "P015", "P016", "P017", "P007"


def seasonality(when: datetime, southern: bool, patterns: OrderPatterns) -> float:
    """Order intensity relative to the yearly average (~1), at this moment."""
    day = when.timetuple().tm_yday + when.hour / 24
    peak = patterns.northern_peak_day_of_year + (DAYS_PER_YEAR / 2 if southern else 0)
    level = 1 + patterns.seasonal_amplitude * math.cos(2 * math.pi * (day - peak) / DAYS_PER_YEAR)
    for h in patterns.holidays:
        centre = datetime(when.year, h.month, h.day)  # noqa: DTZ001 - naive UTC throughout
        distance = min(abs((when - centre.replace(year=y)).total_seconds()) / 86400
                       for y in (when.year - 1, when.year, when.year + 1))
        level += h.boost * math.exp(-0.5 * (distance / h.width_days) ** 2)
    return level


def peak_seasonality(patterns: OrderPatterns) -> float:
    return 1 + patterns.seasonal_amplitude + sum(h.boost for h in patterns.holidays)


@dataclass
class _Customer:
    number: int
    signup_at: datetime
    southern: bool
    fuels: set[str] = field(default_factory=set)
    grills: set[str] = field(default_factory=set)
    orders: int = 0


def _weighted(r: np.random.Generator, weights: dict[str, float], allowed) -> str | None:
    options = [k for k in weights if k in allowed]
    if not options:
        return None
    w = np.array([weights[k] for k in options])
    return options[r.choice(len(options), p=w / w.sum())]


class _Simulator:
    def __init__(self, products: pl.DataFrame, config: GeneratorConfig, seed: int):
        self.p = config.orders
        self.config = config
        self.seed = seed
        self.price = dict(zip(products["product_id"], products["list_price"], strict=True))
        self.fuel = dict(zip(products["product_id"], products["fuel_type"], strict=True))
        self.category = dict(zip(products["product_id"], products["category"], strict=True))
        self.launched = dict(zip(products["product_id"], products["launched_on"], strict=True))

    def available(self, when: datetime) -> set[str]:
        return {pid for pid, day in self.launched.items() if day <= when.date()}

    def basket(self, c: _Customer, when: datetime, r: np.random.Generator, replenish: bool) -> dict[str, int]:
        p, live, lines = self.p, self.available(when), {}

        def add(pid, qty=1):
            if pid and pid in live and len(lines) < MAX_LINES:
                lines[pid] = lines.get(pid, 0) + qty

        def consumable_qty():
            return 1 + int(r.binomial(p.max_quantity - 1, 0.3))

        new_grill = None
        if not replenish:
            chance = p.grill_again if c.grills else (p.grill_first_order if c.orders == 0 else p.grill_without_one)
            if r.random() < chance:
                new_grill = _weighted(r, p.grill_popularity, live - c.grills)
                if new_grill:
                    add(new_grill)
                    c.grills.add(new_grill)
                    if self.fuel[new_grill] in ("pellet", "charcoal"):
                        c.fuels.add(self.fuel[new_grill])
            if new_grill and r.random() < p.cover_with_new_grill:
                add(COVER)
            if new_grill and r.random() < p.tool_with_new_grill:
                add(_weighted(r, p.accessory_weights, set(TOOLS) & live))
            for _ in range(int(r.poisson(p.accessories_per_order))):
                add(_weighted(r, p.accessory_weights, live - set(lines)))

        fuel_chance = 0.9 if (new_grill or replenish) else p.fuel_per_order
        if "pellet" in c.fuels and r.random() < fuel_chance:
            add(PELLETS, consumable_qty())
        if "charcoal" in c.fuels and r.random() < fuel_chance:
            add(CHARCOAL, consumable_qty())
            if r.random() < p.lighter_with_charcoal:
                add(LIGHTER, consumable_qty())
        if (c.fuels or c.grills) and r.random() < p.chips_per_order:
            add(CHIPS, consumable_qty())
        if r.random() < p.rub_per_order * (0.5 if replenish else 1):
            add(RUB)
        if not lines:  # a browse that became an order: one small item
            add(_weighted(r, p.accessory_weights | {CHIPS: 0.8, RUB: 1.0}, live))
        return lines

    def unit_price(self, pid: str, when: datetime, southern: bool, r: np.random.Generator) -> float:
        years = max(0.0, (when - self.config.start).days / DAYS_PER_YEAR)
        price = self.price[pid] * (1 + self.p.price_drift_per_year * years)
        if seasonality(when, southern, self.p) < 1 and r.random() < self.p.off_season_discount_chance:
            price *= 1 - r.uniform(*self.p.discount_range)
        return round(price, 2)

    def history(self, row: dict, until: datetime):
        """Every order this customer places up to `until`, in time order, as (when, lines)."""
        number = int(row["customer_id"][1:])
        c = _Customer(number, row["signup_at"], row["country"] in SOUTHERN_HEMISPHERE)
        times, basket, restock = (rng(self.seed, s, number) for s in ("order-times", "basket", "restock"))
        p = self.p

        if basket.random() < p.owns_grill_elsewhere:
            fuel = _weighted(basket, p.elsewhere_fuel_weights, p.elsewhere_fuel_weights)
            if fuel in ("pellet", "charcoal"):
                c.fuels.add(fuel)

        rate = times.gamma(p.orders_per_year_shape, p.orders_per_year / p.orders_per_year_shape)
        ceiling = peak_seasonality(p)

        def next_order(after: datetime) -> datetime:
            t = after
            while True:  # thinning: candidate arrivals at the peak rate, kept in proportion to the season
                t += timedelta(days=times.exponential(DAYS_PER_YEAR / max(rate * ceiling, 1e-9)))
                if t > until or times.random() * ceiling < seasonality(t, c.southern, p):
                    return t

        def next_restock(after: datetime) -> datetime:
            days = restock.gamma(p.replenish_shape, p.replenish_every_days / p.replenish_shape)
            return after + timedelta(days=days / max(seasonality(after, c.southern, p), 0.2))

        t_order = next_order(c.signup_at)
        t_restock = next_restock(c.signup_at) if c.fuels else None
        while c.orders < MAX_ORDERS:
            replenish = t_restock is not None and t_restock < t_order
            when = t_restock if replenish else t_order
            if when > until:
                return
            lines = self.basket(c, when, basket, replenish)
            yield when, c, [(pid, qty, self.unit_price(pid, when, c.southern, basket)) for pid, qty in lines.items()]
            c.orders += 1
            if replenish:
                t_restock = next_restock(when)
            else:
                t_order = next_order(when)
                if t_restock is None and c.fuels:
                    t_restock = next_restock(when)


def generate_orders(customers: pl.DataFrame, products: pl.DataFrame, *, seed: int, until: datetime,
                    since: datetime | None = None, batch_id: int = 1,
                    config: GeneratorConfig | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Orders and their lines placed after `since` (exclusive) and up to `until` (inclusive)."""
    sim = _Simulator(products, config or GeneratorConfig(), seed)
    orders, lines = [], []
    for row in customers.select("customer_id", "signup_at", "country").iter_rows(named=True):
        for when, c, items in sim.history(row, until):
            if since is not None and when <= since:
                continue
            order_id = f"O{c.number:07d}{c.orders:02d}"
            total = 0.0
            for i, (pid, qty, unit) in enumerate(items):
                line_total = round(qty * unit, 2)
                total += line_total
                lines.append({"order_line_id": f"L{order_id[1:]}{i}", "order_id": order_id, "product_id": pid,
                              "quantity": qty, "unit_price": unit, "line_total": line_total, "batch_id": batch_id})
            orders.append({"order_id": order_id, "customer_id": row["customer_id"], "ordered_at": when,
                           "order_total": round(total, 2), "batch_id": batch_id})
    return (
        OrderSchema.validate(pl.DataFrame(orders, schema=polars_schema(OrderSchema))),
        OrderLineSchema.validate(pl.DataFrame(lines, schema=polars_schema(OrderLineSchema))),
    )
