"""The classifier against a stubbed client. No test calls the live API."""

import asyncio
import json

import polars as pl
from typesafe_sdk import SystemOneResponse, TypeSafeAPIConnectionError

from jev_classifier import (
    RESULT_SCHEMA,
    build_questions,
    build_state,
    classify_sentences,
)

CATALOGUE = pl.DataFrame({"product_name": ["Elite Tongs", "Pro Grill"], "product_category": ["accessories", "grills"]})

SENTENCE = {
    "review_key": "abc", "sentence_index": 0, "sentence_count": 2, "sentence": "Tongs broke.",
    "review_text": "Tongs broke. Fits my Pro Grill.", "product_name": "Elite Tongs",
    "product_category": "accessories", "rating": 1,
}


def response(**nouls):
    choice = lambda label: {"type": "choice", "choice": label, "confidence": 0.9, "probabilities": {label: 0.9}}
    answers = {
        "frustration": {"type": "score", "score": 2.5, "confidence": 0.8,
                        "legend": {"0": "calm", "1": "cross"}, "probabilities": {"0": 0.2, "1": 0.8}},
        "problem_category": choice("build_quality"), "language": choice("english"),
        "sentiment": choice("negative"), "recommendation": choice("none"),
        **{n: {"type": "noul", "noul": 0.1} for n in ["churn_risk", "safety_concern", "suggestion", "competitor_mention"]},
        **{n: {"type": "noul", "noul": p} for n, p in nouls.items()},
    }
    body = {"model": "jev-test", "usage": {"input_tokens": 100, "output_tokens": 5}, "answers": answers}
    return SystemOneResponse.model_validate_json(json.dumps(body))


class StubClient:
    def __init__(self, result):
        self.result, self.calls = result, []

    async def system_one(self, state, questions):
        self.calls.append((state, questions))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def run(client, sentences, **kwargs):
    return asyncio.run(classify_sentences(pl.DataFrame(sentences), CATALOGUE, client=client, **kwargs))


def test_one_mention_question_per_product():
    names = set(build_questions(CATALOGUE.to_dicts()))
    assert {"mentions__elite_tongs", "mentions__pro_grill"} <= names


def test_state_leaves_out_rating_and_demographics():
    state = build_state(SENTENCE | {"country": "Japan"})
    assert "Japan" not in json.dumps(state)
    assert "rating" not in json.dumps(state)
    assert state["sentence_position"] == "1 of 2"


def test_answers_become_one_flat_row():
    client = StubClient(response(mentions__elite_tongs=0.9, mentions__pro_grill=0.6))
    out = run(client, [SENTENCE])
    row = out.row(0, named=True)
    assert out.schema == pl.Schema(RESULT_SCHEMA)
    assert row["products_mentioned"] == ["Elite Tongs", "Pro Grill"]
    assert row["frustration"] == 2.5 and row["problem_category"] == "build_quality"
    assert row["jev_model"] == "jev-test" and row["error"] is None


def test_mentions_below_threshold_are_left_out():
    out = run(StubClient(response(mentions__elite_tongs=0.9, mentions__pro_grill=0.49)), [SENTENCE])
    assert out["products_mentioned"].to_list() == [["Elite Tongs"]]


def test_duplicate_sentences_are_asked_once():
    client = StubClient(response(mentions__elite_tongs=0.9, mentions__pro_grill=0.1))
    out = run(client, [SENTENCE, SENTENCE])
    assert len(client.calls) == 1 and out.height == 1


def test_a_failure_is_recorded_not_filled_in_and_not_cached(tmp_path):
    cache = tmp_path / "cache.parquet"
    out = run(StubClient(TypeSafeAPIConnectionError("down")), [SENTENCE], cache_path=cache)
    assert "TypeSafeAPIConnectionError" in out["error"][0]
    assert out["frustration"][0] is None

    retry = StubClient(response(mentions__elite_tongs=0.9, mentions__pro_grill=0.1))
    run(retry, [SENTENCE], cache_path=cache)
    assert len(retry.calls) == 1


def test_cached_sentences_are_not_asked_again(tmp_path):
    cache = tmp_path / "cache.parquet"
    run(StubClient(response(mentions__elite_tongs=0.9, mentions__pro_grill=0.1)), [SENTENCE], cache_path=cache)
    again = StubClient(response())
    out = run(again, [SENTENCE], cache_path=cache)
    assert again.calls == [] and out.height == 1
