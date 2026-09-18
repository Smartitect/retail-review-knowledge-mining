"""
The one rule for every review-level feature: use only what happened before.

A model trained to predict something from a review must only see facts that
existed when the review was written. `point_in_time` enforces that for any
event table. For each row of `left`, it aggregates the same entity's events
that happened *strictly before* `left[at]`, with an as-of join on running
totals, so a later event can never leak in.

Events sharing a timestamp are collapsed first, so the as-of join never has to
choose between them.
"""

import polars as pl


def point_in_time(left: pl.DataFrame, events: pl.DataFrame, *, by: str, at: str, event_time: str,
                  sums: dict[str, str] | None = None, count: str | None = None,
                  last_event: str | None = None) -> pl.DataFrame:
    """`left` plus aggregates of `events` for the same `by`, strictly before `left[at]`.

    `sums` maps output column -> event column to total; `count` names the number
    of events; `last_event` names the time of the most recent one (null when
    there was none). Totals and counts are 0 when there were no earlier events.
    """
    sums = sums or {}
    per_moment = events.group_by(by, event_time).agg(
        *[pl.col(src).sum().alias(out) for out, src in sums.items()],
        pl.len().alias("__n"),
    )
    running = per_moment.sort(by, event_time).with_columns(
        *[pl.col(out).cum_sum().over(by) for out in sums],
        pl.col("__n").cum_sum().over(by),
        pl.col(event_time).alias("__last"),
    )
    joined = left.sort(at).join_asof(
        running.sort(event_time).rename({event_time: "__t"}),
        left_on=at, right_on="__t", by=by, strategy="backward", allow_exact_matches=False,
        check_sortedness=False,
    )
    extra = [pl.col(out).fill_null(0) for out in sums]
    if count:
        extra.append(pl.col("__n").fill_null(0).cast(pl.UInt32).alias(count))
    if last_event:
        extra.append(pl.col("__last").alias(last_event))
    return joined.with_columns(extra).drop("__n", "__last", "__t")
