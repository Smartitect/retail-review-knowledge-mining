"""
Compare two classifiers' answers on the same sentences: speed, cost, agreement and accuracy.

Every function takes result frames in (DSPY_)RESULT_SCHEMA, one row per
`KEY`, so it works for Jev, the DSPy classifier, or any other pair.

The sample is drawn by **review**, not by sentence: scoring a review's intended
issue against what was found needs every sentence of it. Reviews are stratified
by the language they were written in and their star rating, so the small
languages and the rare middle ratings are not swamped by English five-stars.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from jev_classifier import KEY, NOUL_THRESHOLD, build_state
from jev_classifier.sentence_classifier import CHOICES, NOULS

from .signature import build_signature

# What a review written in each language should be heard as. Zulu is not one of
# Jev's options, so the right answer for it is "other".
EXPECTED_LANGUAGE = {"zulu": "other"}


def stratified_sample(sentences: pl.DataFrame, target: int = 300, *, by=("text_language", "rating"),
                      seed: int = 42) -> pl.DataFrame:
    """Every sentence of a stratified sample of reviews, about `target` sentences in all.

    Each stratum gets reviews in proportion to its sentences, rounded down, and
    at least one review, so every language and rating appears. That floor costs
    about 200 sentences on the demo's 40 strata (8 languages by 5 ratings), so
    small targets come out larger: 300 gives about 370.
    """
    reviews = sentences.group_by("review_id", *by).agg(pl.len().alias("n")).sort("review_id")
    share = target / reviews["n"].sum()
    picked = (
        reviews.with_columns(
            rank=pl.int_range(pl.len()).shuffle(seed=seed).over(list(by)),
            quota=(pl.col("n").sum().over(list(by)) * share / pl.col("n").mean().over(list(by))).floor().clip(1),
        )
        .filter(pl.col("rank") < pl.col("quota"))
        .select("review_id")
    )
    return sentences.join(picked, on="review_id", how="semi").sort("review_id", "sentence_index")


@dataclass(frozen=True)
class Run:
    """One classifier's live run over the sample."""

    name: str
    results: pl.DataFrame
    seconds: float
    concurrency: int


def save_run(run: Run, path: Path | str) -> Run:
    """Keep a run, so the analysis can be repeated without paying for the calls again.

    The results go to `path` (Parquet) and the timing beside it (`.json`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run.results.write_parquet(path)
    path.with_suffix(".json").write_text(json.dumps({"name": run.name, "seconds": run.seconds,
                                                     "concurrency": run.concurrency}) + "\n")
    return run


def load_run(path: Path | str) -> Run:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    return Run(meta["name"], pl.read_parquet(path), meta["seconds"], meta["concurrency"])


def prompt_tokens(sentences: pl.DataFrame, catalogue: pl.DataFrame, chars_per_token: float = 4.0) -> int:
    """Roughly how many input tokens the DSPy classifier will send for these sentences.

    Each prompt is rendered exactly as it will be sent, then counted at about
    four characters a token: an estimate, since the tokenizer depends on the model.
    """
    import dspy

    signature = build_signature(catalogue.select("product_name", "product_category").to_dicts())
    adapter = dspy.JSONAdapter()
    chars = sum(
        len(message["content"])
        for sentence in sentences.unique(KEY).iter_rows(named=True)
        for message in adapter.format(signature, demos=[], inputs={"state": build_state(sentence)})
    )
    return round(chars / chars_per_token)


def speed(runs: list[Run], *, sentences_in_dataset: int) -> pl.DataFrame:
    """Latency per sentence and throughput, and how long the whole dataset would take at that rate."""
    rows = []
    for run in runs:
        ok = run.results.filter(pl.col("error").is_null())
        rows.append({
            "classifier": run.name,
            "sentences": run.results.height,
            "failed": run.results.height - ok.height,
            "p50_ms": ok["latency_ms"].median(),
            "p95_ms": ok["latency_ms"].quantile(0.95),
            "concurrency": run.concurrency,
            "sentences_per_s": round(run.results.height / run.seconds, 2),
            "dataset_minutes": round(sentences_in_dataset / (run.results.height / run.seconds) / 60, 1),
        })
    return pl.DataFrame(rows)


def cost(runs: list[Run], prices: dict[str, tuple[float | None, float | None]], *,
         sentences_per_1000_customers: int) -> pl.DataFrame:
    """Tokens and dollars per sentence. `prices` maps a run name to USD per million (input, output) tokens."""
    rows = []
    for run in runs:
        ok = run.results.filter(pl.col("error").is_null())
        tokens_in = ok["input_tokens"].mean()
        tokens_out = ok["output_tokens"].mean() if "output_tokens" in ok.columns else 0.0
        price_in, price_out = prices.get(run.name, (None, None))
        usd = (tokens_in * price_in + (tokens_out or 0) * (price_out or 0)) / 1e6 if price_in is not None else None
        rows.append({
            "classifier": run.name,
            "input_tokens": round(tokens_in or 0),
            "output_tokens": round(tokens_out or 0),
            "usd_per_sentence": usd,
            "usd_per_1000_customers": None if usd is None else round(usd * sentences_per_1000_customers, 2),
        })
    return pl.DataFrame(rows, schema_overrides={"usd_per_sentence": pl.Float64, "usd_per_1000_customers": pl.Float64})


def _paired(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    ok = pl.col("error").is_null()
    return a.filter(ok).join(b.filter(ok), on=KEY, suffix="_b")


def agreement(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """Per question, how often the two classifiers agree on the sentences both answered.

    Choices agree when they pick the same label; yes/no questions when both fall
    on the same side of the threshold; frustration when within one rubric level;
    product mentions when the sets are identical.
    """
    p = _paired(a, b)
    yes = lambda col: pl.col(col) >= NOUL_THRESHOLD
    checks = {
        "frustration (within 1)": (pl.col("frustration") - pl.col("frustration_b")).abs() <= 1,
        **{c: pl.col(c) == pl.col(f"{c}_b") for c in CHOICES},
        **{n: yes(n) == yes(f"{n}_b") for n in NOULS},
        "products_mentioned": pl.col("products_mentioned").list.sort() == pl.col("products_mentioned_b").list.sort(),
    }
    return pl.DataFrame({
        "question": list(checks),
        "agreement": [p.select(expr.mean()).item() for expr in checks.values()],
        "sentences": p.height,
    }).with_columns(pl.col("agreement").round(3))


def disagreements(a: pl.DataFrame, b: pl.DataFrame, sentences: pl.DataFrame, question: str, *,
                  names: tuple[str, str] = ("a", "b"), n: int = 5) -> pl.DataFrame:
    """Sentences where the two classifiers gave different answers to one Choice question."""
    p = _paired(a, b).filter(pl.col(question) != pl.col(f"{question}_b"))
    return (
        p.join(sentences.select(*KEY, "product_name", "sentence"), on=KEY)
        .select("product_name", "sentence", pl.col(question).alias(names[0]), pl.col(f"{question}_b").alias(names[1]))
        .head(n)
    )


def accuracy(results: pl.DataFrame, sentences: pl.DataFrame, truth: pl.DataFrame) -> dict[str, float]:
    """Score one classifier against what the generator intended (`review_truth`).

    - `language`: the share of sentences heard in the language their review was
      written in (Zulu should be heard as `other`).
    - `issue_found`: of the unhappy reviews, the share where the intended issue
      is among the problem categories found in any of its sentences.
    - `primary_issue`: of the unhappy reviews, the share where the intended issue
      is the problem raised in the review's most frustrated sentence.
    """
    rows = (
        sentences.select(*KEY, "review_id").join(results.filter(pl.col("error").is_null()), on=KEY)
        .join(truth.select("review_id", "satisfaction", "aspect", pl.col("language").alias("written")), on="review_id")
    )
    expected = pl.col("written").replace(EXPECTED_LANGUAGE)
    language = rows.select((pl.col("language") == expected).mean()).item()
    unhappy = rows.filter(pl.col("satisfaction") < 0.5)
    problem = pl.col("problem_category") != "none"
    per_review = unhappy.group_by("review_id").agg(
        pl.col("aspect").first(),
        pl.col("problem_category").filter(problem).alias("found"),
        pl.col("problem_category").filter(problem).sort_by(pl.col("frustration").filter(problem), descending=True)
        .first().alias("primary"),
    )
    return {
        "language": round(language, 3),
        "issue_found": round(per_review.select(pl.col("found").list.contains(pl.col("aspect")).mean()).item(), 3),
        "primary_issue": round(per_review.select((pl.col("primary") == pl.col("aspect")).mean()).item(), 3),
        "unhappy_reviews": per_review.height,
    }


def answers(results: pl.DataFrame, key: dict) -> dict:
    """Every answer one classifier gave for one sentence, decoded from `answers_json`."""
    row = results.filter(*[pl.col(k) == v for k, v in key.items()])
    return json.loads(row["answers_json"].item())
