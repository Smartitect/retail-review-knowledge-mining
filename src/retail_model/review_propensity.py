"""
Who leaves a review, and what they rate it.

People rarely review a purchase that was merely fine. They review when they are
delighted or when something went wrong. Each purchase has a hidden
`satisfaction` in [0, 1], and the chance of a review is U-shaped over it:

    p(review) = base_rate + extremity_rate * |2s - 1| ** sharpness

At s = 0.5 only `base_rate` remains; at either extreme the chance rises to
`base_rate + extremity_rate`. `sharpness` controls how quickly it rises: 1 is a
V, higher values keep the middle flat and push reviews towards the extremes.

The corpus of reviews is therefore a biased sample of purchases, as in reality,
which is itself something the demo can show.
"""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class ReviewPropensity:
    base_rate: float = 0.03
    extremity_rate: float = 0.45
    sharpness: float = 2.5
    rating_noise: float = 0.35

    def __post_init__(self):
        if not 0 <= self.base_rate <= 1 or not 0 <= self.base_rate + self.extremity_rate <= 1:
            raise ValueError("base_rate and base_rate + extremity_rate must both lie in [0, 1]")
        if self.sharpness <= 0:
            raise ValueError("sharpness must be positive")

    def probability(self, satisfaction: pl.Expr) -> pl.Expr:
        """The chance a purchase with this satisfaction gets reviewed."""
        extremity = ((satisfaction * 2) - 1).abs().pow(self.sharpness)
        return pl.lit(self.base_rate) + pl.lit(self.extremity_rate) * extremity

    def rating(self, satisfaction: pl.Expr, noise: pl.Expr) -> pl.Expr:
        """Stars from satisfaction: 1 + 4s plus standard-normal `noise` scaled by `rating_noise`, rounded to 1..5."""
        return (
            (pl.lit(1.0) + satisfaction * 4 + noise * self.rating_noise)
            .round()
            .clip(1, 5)
            .cast(pl.Int8)
        )
