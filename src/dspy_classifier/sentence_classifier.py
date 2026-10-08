"""
Send each sentence to a foundation model through DSPy, and return rows shaped like Jev's.

A drop-in alternative to `jev_classifier.classify_sentences`: the same sentences
in, the same `RESULT_SCHEMA` columns out, plus `output_tokens`, because a
reasoning model's hidden thinking is billed as output and dominates its cost.
`jev_model` holds the DSPy model name, so mixed results stay distinguishable.

One call per sentence answers all of Jev's questions at once. Calls run
concurrently behind a semaphore. Results are cached to Parquet keyed on
`review_key` + `sentence_index` + `question_set_version` + model, so a rerun
only pays for what it has not seen. A failed call is never papered over: its row
carries the error and nulls, is not cached, and is retried on the next run.
"""

import asyncio
import json
import time
from pathlib import Path

import dspy
import polars as pl

from jev_classifier import (
    KEY,
    NOUL_THRESHOLD,
    QUESTION_SET_VERSION,
    RESULT_SCHEMA,
    build_state,
    product_slug,
)
from jev_classifier.sentence_classifier import CHOICES, NOULS

from .signature import build_signature

DSPY_RESULT_SCHEMA = RESULT_SCHEMA | {"output_tokens": pl.Int64}
SCORE_TOP = 4.0  # frustration runs 0..4, as Jev's rubric does


def _unit(value) -> float:
    return min(max(float(value), 0.0), 1.0)


def flatten(sentence: dict, prediction: dspy.Prediction, names: dict[str, str], model: str) -> dict:
    """One sentence's answers as a flat row in `RESULT_SCHEMA`. `names` maps question name -> product name."""
    answers = {k: v for k, v in prediction.items()}
    usage = next(iter((prediction.get_lm_usage() or {}).values()), {})
    row = {
        "frustration": min(max(float(answers["frustration"]), 0.0), SCORE_TOP),
        "frustration_confidence": _unit(answers["frustration_confidence"]),
        **{c: answers[c] for c in CHOICES},
        **{f"{c}_confidence": _unit(answers[f"{c}_confidence"]) for c in CHOICES},
        **{n: _unit(answers[n]) for n in NOULS},
        "products_mentioned": sorted(p for q, p in names.items() if _unit(answers[q]) >= NOUL_THRESHOLD),
        "answers_json": json.dumps(answers, default=str),
        "jev_model": model,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }
    return {**{k: sentence[k] for k in KEY}, **row}


async def _classify_one(program, lm, sentence, names, gate) -> dict:
    async with gate:
        started = time.perf_counter()
        try:
            with dspy.context(lm=lm, adapter=dspy.JSONAdapter(), track_usage=True):
                prediction = await program.acall(state=build_state(sentence))
            row, error = flatten(sentence, prediction, names, lm.model), None
        except Exception as exc:  # noqa: BLE001 - any failure is recorded and retried next run
            row = {k: sentence[k] for k in KEY} | {"jev_model": lm.model}
            error = f"{type(exc).__name__}: {exc}"
        return {**row, "question_set_version": QUESTION_SET_VERSION,
                "latency_ms": round((time.perf_counter() - started) * 1000), "error": error}


def _read_cache(cache_path: Path | None, model: str) -> pl.DataFrame:
    if cache_path is None or not Path(cache_path).exists():
        return pl.DataFrame(schema=DSPY_RESULT_SCHEMA)
    return pl.read_parquet(cache_path).filter(
        pl.col("question_set_version") == QUESTION_SET_VERSION, pl.col("jev_model") == model, pl.col("error").is_null()
    )


async def classify_sentences(
    sentences: pl.DataFrame,
    catalogue: pl.DataFrame,
    *,
    lm: dspy.BaseLM,
    cache_path: Path | str | None = None,
    concurrency: int = 8,
) -> pl.DataFrame:
    """Classify every distinct sentence with `lm`, reusing cached answers.

    `sentences` needs the columns `build_state` reads plus the `KEY` columns;
    duplicates on `KEY` are asked once. Returns one row per distinct key, in
    `DSPY_RESULT_SCHEMA`. Join it back to the sentences on `KEY`.
    """
    catalogue_rows = catalogue.select("product_name", "product_category").to_dicts()
    program = dspy.Predict(build_signature(catalogue_rows))
    names = {f"mentions__{product_slug(p['product_name'])}": p["product_name"] for p in catalogue_rows}

    cached = _read_cache(cache_path, lm.model)
    todo = sentences.unique(KEY, keep="first", maintain_order=True).join(cached.select(KEY), on=KEY, how="anti")
    print(f"{todo.height} sentence(s) to classify with {lm.model}, {cached.height} cached.")

    fresh = pl.DataFrame(schema=DSPY_RESULT_SCHEMA)
    if todo.height:
        gate = asyncio.Semaphore(concurrency)
        started = time.perf_counter()
        rows = await asyncio.gather(*(_classify_one(program, lm, s, names, gate) for s in todo.iter_rows(named=True)))
        fresh = pl.DataFrame(rows, schema=DSPY_RESULT_SCHEMA)
        failed = fresh.filter(pl.col("error").is_not_null()).height
        print(f"Classified {fresh.height} in {time.perf_counter() - started:.1f}s; {failed} failed.")

    result = pl.concat([cached.select(DSPY_RESULT_SCHEMA.keys()), fresh])
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        keep = _previous(cache_path, lm.model)  # other models' answers stay in the same file
        pl.concat([keep, result.filter(pl.col("error").is_null())]).write_parquet(cache_path)
    return result


def _previous(cache_path: Path | str, model: str) -> pl.DataFrame:
    if not Path(cache_path).exists():
        return pl.DataFrame(schema=DSPY_RESULT_SCHEMA)
    return pl.read_parquet(cache_path).filter(
        (pl.col("jev_model") != model) | (pl.col("question_set_version") != QUESTION_SET_VERSION)
    ).select(DSPY_RESULT_SCHEMA.keys())
