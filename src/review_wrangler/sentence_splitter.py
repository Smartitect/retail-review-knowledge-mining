"""
Break reviews into sentences, entirely in Polars expressions.

A sentence ends at `.`, `!` or `?` (or a run of them) followed by whitespace,
or at the full-width `。`, `！` or `？` that Chinese and Japanese use with no
space after. That is deliberately simple: the sample has no abbreviations, decimals before
a space, quotations or line breaks, which are what trip a regex splitter. If
real data arrives with those, swap this stage for a proper segmenter; nothing
downstream depends on how the split was made.
"""

import polars as pl

BOUNDARY = "\x1f"  # a control character that never occurs in review text


def split_sentences(reviews: pl.LazyFrame) -> pl.LazyFrame:
    """One row per sentence, carrying every review column.

    Adds `sentence_index` (0-based within the review), `sentence_count` and
    `sentence`. Empty fragments are dropped before numbering, so the indices
    have no gaps.
    """
    return (
        reviews.with_columns(
            sentence=pl.col("review_text")
            .str.replace_all(r"\s+", " ")
            .str.replace_all(r"([.!?]+) |([。！？]+) ?", "${1}${2}" + BOUNDARY)
            .str.split(BOUNDARY)
        )
        .explode("sentence", empty_as_null=False)
        .with_columns(pl.col("sentence").str.strip_chars())
        .filter(pl.col("sentence").str.len_chars() > 0)
        .with_columns(
            sentence_index=pl.int_range(pl.len(), dtype=pl.UInt32).over("review_id"),
            sentence_count=pl.len().over("review_id").cast(pl.UInt32),
        )
    )
