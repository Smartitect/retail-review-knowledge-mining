"""
Write text for every brief, reusing anything already written for the same prompt.

The cache is a Parquet file of every text ever produced, keyed on `review_id`
and `prompt_hash`. The hash covers the model and the rendered prompt, so the
same seed gives the same briefs, the same hashes and the same cached text: the
dataset is reproducible even though a language model is not. Change the
writer, the model or anything in the brief and the hash changes, so that
review is written afresh; the old text stays in the cache in case you switch back.

A failed call is reported, not cached, and retried on the next run.
"""

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .prompt import ReviewBrief, prompt_hash
from .writers import ReviewWriter

TEXT_SCHEMA = {
    "review_id": pl.String,
    "prompt_hash": pl.String,
    "review_text": pl.String,
    "language": pl.String,
    "model": pl.String,
    "generated_at": pl.Datetime("us"),
}


def _read(cache_path: Path | None) -> pl.DataFrame:
    if cache_path is None or not Path(cache_path).exists():
        return pl.DataFrame(schema=TEXT_SCHEMA)
    return pl.read_parquet(cache_path).select(TEXT_SCHEMA.keys())


async def _write_one(writer: ReviewWriter, brief: ReviewBrief, key: str, gate: asyncio.Semaphore):
    async with gate:
        try:
            text = await writer.write(brief)
        except Exception as exc:  # noqa: BLE001 - any failure is reported and retried next run
            return brief.review_id, f"{type(exc).__name__}: {exc}"
        return {
            "review_id": brief.review_id, "prompt_hash": key, "review_text": text,
            "language": brief.language, "model": writer.model, "generated_at": datetime.now(UTC).replace(tzinfo=None),
        }


async def write_review_texts(
    briefs: list[ReviewBrief], writer: ReviewWriter, *, cache_path: Path | str | None = None,
    concurrency: int = 8,
) -> pl.DataFrame:
    """Text for every brief, one row per review, in `TEXT_SCHEMA`. Missing rows mean the call failed."""
    keys = {b.review_id: prompt_hash(b, writer.model) for b in briefs}
    wanted = pl.DataFrame({"review_id": list(keys), "prompt_hash": list(keys.values())},
                          schema={"review_id": pl.String, "prompt_hash": pl.String})
    cache = _read(cache_path)
    hits = cache.join(wanted, on=["review_id", "prompt_hash"], how="semi").unique(["review_id", "prompt_hash"])
    todo = [b for b in briefs if b.review_id not in set(hits["review_id"])]
    print(f"{len(todo)} review text(s) to write with {writer.model}, {hits.height} cached.")

    fresh = pl.DataFrame(schema=TEXT_SCHEMA)
    if todo:
        gate = asyncio.Semaphore(concurrency)
        started = time.perf_counter()
        results = await asyncio.gather(*(_write_one(writer, b, keys[b.review_id], gate) for b in todo))
        rows = [r for r in results if isinstance(r, dict)]
        failures = [r for r in results if not isinstance(r, dict)]
        fresh = pl.DataFrame(rows, schema=TEXT_SCHEMA)
        print(f"Wrote {fresh.height} in {time.perf_counter() - started:.1f}s; {len(failures)} failed.")
        for review_id, error in failures[:5]:
            print(f"  {review_id}: {error}")
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            pl.concat([cache, fresh]).write_parquet(cache_path)

    return pl.concat([hits, fresh]).join(wanted.select("review_id"), on="review_id", how="semi")
