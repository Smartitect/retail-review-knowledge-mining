"""
Send each sentence to Jev and turn the typed answers into one flat row.

Calls run concurrently, capped by a semaphore well inside the published rate
limit (1,200 requests a minute). Results are cached to Parquet keyed on
`review_key` + `sentence_index` + `question_set_version`, so re-running the
notebook only pays for sentences it has not seen under the current questions.

A failed call is never papered over: its row carries the error and nulls, is
not cached, and is retried on the next run.
"""

import asyncio
import json
import time
from pathlib import Path

import polars as pl
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from . import transcript
from .questions import (
    NOUL_THRESHOLD,
    QUESTION_SET_VERSION,
    build_questions,
    build_state,
    product_slug,
)

KEY = ["review_key", "sentence_index"]

CHOICES = ["problem_category", "language", "sentiment", "recommendation"]
NOULS = ["churn_risk", "safety_concern", "suggestion", "competitor_mention"]

RESULT_SCHEMA = {
    "review_key": pl.String,
    "sentence_index": pl.UInt32,
    "frustration": pl.Float64,
    "frustration_confidence": pl.Float64,
    **{c: pl.String for c in CHOICES},
    **{f"{c}_confidence": pl.Float64 for c in CHOICES},
    **{n: pl.Float64 for n in NOULS},
    "products_mentioned": pl.List(pl.String),
    "answers_json": pl.String,
    "jev_model": pl.String,
    "question_set_version": pl.String,
    "input_tokens": pl.Int64,
    "latency_ms": pl.Int64,
    "error": pl.String,
}


def flatten(sentence: dict, response, names: dict[str, str]) -> dict:
    """One sentence's answers as a flat row. `names` maps question name -> product name."""
    choices, scores, nouls = response.choices, response.scores, response.nouls
    row = {
        "frustration": scores["frustration"].score,
        "frustration_confidence": scores["frustration"].confidence,
        **{c: choices[c].choice for c in CHOICES},
        **{f"{c}_confidence": choices[c].confidence for c in CHOICES},
        **{n: nouls[n].noul for n in NOULS},
        "products_mentioned": sorted(
            product for question, product in names.items() if nouls[question].noul >= NOUL_THRESHOLD
        ),
        # Every distribution, kept for audit and for re-thresholding without re-asking.
        "answers_json": json.dumps({k: a.model_dump(mode="json") for k, a in response.answers.items()}),
        "jev_model": response.model,
        "input_tokens": response.usage.input_tokens,
    }
    return {**{k: sentence[k] for k in KEY}, **row}


def _request_body(state: dict, questions: dict) -> dict:
    """The request rebuilt from what was passed in, for when there is no wire to read."""
    return {"state": state, "questions": {k: q.model_dump(mode="json") for k, q in questions.items()}}


def _wire(response, state: dict, questions: dict) -> tuple[dict, dict, str | None]:
    """The request and response bodies as they crossed the wire, and the request ID.

    A response built without an HTTP exchange (a test stub) has no wire to read,
    so both sides are rebuilt from what was passed in and what came back.
    """
    try:
        raw = response.raw_http_response
    except TypeSafeError:
        return _request_body(state, questions), response.model_dump(mode="json"), None
    return json.loads(raw.request.content), json.loads(raw.content), raw.headers.get("x-typesafe-request-id")


async def _classify_one(client, sentence, questions, names, gate) -> dict:
    async with gate:
        state = build_state(sentence)
        started = time.perf_counter()
        try:
            response = await client.system_one(state=state, questions=questions)
            row = flatten(sentence, response, names)
            error = None
        except TypeSafeError as exc:
            response = None
            row = {k: sentence[k] for k in KEY}
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = round((time.perf_counter() - started) * 1000)
        if transcript.enabled():
            key = {k: sentence[k] for k in KEY}
            if response is None:
                transcript.exchange(key=key, sent=_request_body(state, questions), received=None, latency_ms=latency_ms, error=error)
            else:
                sent, received, request_id = _wire(response, state, questions)
                transcript.exchange(key=key, sent=sent, received=received, latency_ms=latency_ms,
                                    request_id=request_id)
        return {
            **row,
            "question_set_version": QUESTION_SET_VERSION,
            "latency_ms": latency_ms,
            "error": error,
        }


def _read_cache(cache_path: Path | None) -> pl.DataFrame:
    if cache_path is None or not Path(cache_path).exists():
        return pl.DataFrame(schema=RESULT_SCHEMA)
    return pl.read_parquet(cache_path).filter(
        pl.col("question_set_version") == QUESTION_SET_VERSION, pl.col("error").is_null()
    )


async def classify_sentences(
    sentences: pl.DataFrame,
    catalogue: pl.DataFrame,
    *,
    cache_path: Path | str | None = None,
    concurrency: int = 8,
    client: AsyncTypeSafeClient | None = None,
) -> pl.DataFrame:
    """Classify every distinct sentence, reusing cached answers.

    `sentences` needs the columns `build_state` reads plus the `KEY` columns;
    duplicates on `KEY` are asked once. Returns one row per distinct key,
    in `RESULT_SCHEMA`. Join it back to the sentences on `KEY`.
    """
    catalogue_rows = catalogue.select("product_name", "product_category").to_dicts()
    questions = build_questions(catalogue_rows)
    names = {f"mentions__{product_slug(p['product_name'])}": p["product_name"] for p in catalogue_rows}

    cached = _read_cache(cache_path)
    todo = sentences.unique(KEY, keep="first", maintain_order=True).join(cached.select(KEY), on=KEY, how="anti")
    print(f"{todo.height} sentence(s) to classify, {cached.height} cached.")

    fresh = pl.DataFrame(schema=RESULT_SCHEMA)
    if todo.height:
        owns_client = client is None
        client = client or AsyncTypeSafeClient(
            retry=RetryPolicy(max_retries=3, backoff_initial=0.5), timeout=30.0
        )
        gate = asyncio.Semaphore(concurrency)
        started = time.perf_counter()
        try:
            rows = await asyncio.gather(
                *(_classify_one(client, s, questions, names, gate) for s in todo.iter_rows(named=True))
            )
        finally:
            if owns_client:
                await client.aclose()
        fresh = pl.DataFrame(rows, schema=RESULT_SCHEMA)
        failed = fresh.filter(pl.col("error").is_not_null()).height
        print(f"Classified {fresh.height} in {time.perf_counter() - started:.1f}s; {failed} failed.")

    result = pl.concat([cached.select(RESULT_SCHEMA.keys()), fresh])
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(cache_path)
    return result
