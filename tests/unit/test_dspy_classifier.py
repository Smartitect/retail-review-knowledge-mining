"""The DSPy classifier against a stubbed language model. No test calls Azure AI Foundry."""

import asyncio
import json
import typing

import dspy
import polars as pl
import pytest
from dspy.lm15 import Message, Response, TextPart, Usage

from dspy_classifier import (
    DSPY_RESULT_SCHEMA,
    build_signature,
    classify_sentences,
    foundry_lm,
)
from jev_classifier import PROBLEM_CATEGORIES, RESULT_SCHEMA
from review_writer import ModelSettings

CATALOGUE = pl.DataFrame({"product_name": ["Elite Tongs", "Pro Grill"], "product_category": ["accessories", "grills"]})

SENTENCE = {
    "review_key": "abc", "sentence_index": 0, "sentence_count": 2, "sentence": "Tongs broke.",
    "review_text": "Tongs broke. Fits my Pro Grill.", "product_name": "Elite Tongs",
    "product_category": "accessories", "rating": 1, "country": "Germany",
}


def answer(**overrides):
    return {
        "frustration": 2.5, "frustration_confidence": 0.8,
        "problem_category": "build_quality", "problem_category_confidence": 0.9,
        "language": "english", "language_confidence": 0.99,
        "sentiment": "negative", "sentiment_confidence": 0.9,
        "recommendation": "none", "recommendation_confidence": 0.7,
        "churn_risk": 0.1, "safety_concern": 0.0, "suggestion": 0.0, "competitor_mention": 0.0,
        "mentions__elite_tongs": 0.95, "mentions__pro_grill": 0.6,
    } | overrides


class StubEngine:
    """A DSPy engine that answers with `answers` in turn, as raw JSON, and counts its calls.

    Given an exception instead, it raises it, as a failing deployment would.
    """

    def __init__(self, answers):
        self.answers = answers if isinstance(answers, Exception) else iter(answers)
        self.calls, self.prompts = 0, []

    def complete(self, request):
        self.calls += 1
        self.prompts.append(str(request.system) + "".join(m.text or "" for m in request.messages))
        if isinstance(self.answers, Exception):
            raise self.answers
        text = TextPart(json.dumps(next(self.answers)))
        return Response(id=None, model="stub", message=Message.assistant([text]), finish_reason="stop",
                        usage=Usage(input_tokens=1200, output_tokens=300, total_tokens=1500))


class AsyncStubEngine:
    def __init__(self, sync):
        self.sync = sync

    async def complete(self, request):
        return self.sync.complete(request)


def stub_lm(answers, model="stub-model"):
    engine = StubEngine(answers)
    lm = dspy.LM(f"openai/{model}", engine=engine, async_engine=AsyncStubEngine(engine), cache=False)
    lm.stub = engine
    return lm


def run(lm, sentences, **kwargs):
    return asyncio.run(classify_sentences(pl.DataFrame(sentences), CATALOGUE, lm=lm, **kwargs))


def test_every_jev_question_becomes_typed_output_fields():
    sig = build_signature(CATALOGUE.to_dicts())
    assert list(sig.input_fields) == ["state"]
    # 1 score + 4 choices, each with a confidence; 4 flags; one mention per product
    assert len(sig.output_fields) == 2 + 4 * 2 + 4 + CATALOGUE.height
    assert typing.get_args(sig.output_fields["problem_category"].annotation) == tuple(PROBLEM_CATEGORIES)
    assert sig.output_fields["mentions__pro_grill"].annotation is float


def test_answers_become_a_row_shaped_like_jevs():
    result = run(stub_lm([answer()]), [SENTENCE])
    assert set(RESULT_SCHEMA) <= set(result.columns) and result.schema == pl.Schema(DSPY_RESULT_SCHEMA)
    row = result.row(0, named=True)
    assert (row["frustration"], row["problem_category"], row["error"]) == (2.5, "build_quality", None)
    assert row["products_mentioned"] == ["Elite Tongs", "Pro Grill"]
    assert row["jev_model"] == "openai/stub-model"
    assert (row["input_tokens"], row["output_tokens"]) == (1200, 300)


def test_out_of_range_answers_are_clipped_to_their_scale():
    row = run(stub_lm([answer(frustration=6.0, sentiment_confidence=1.4, churn_risk=-0.2)]), [SENTENCE]).row(0, named=True)
    assert (row["frustration"], row["sentiment_confidence"], row["churn_risk"]) == (4.0, 1.0, 0.0)


def test_mentions_below_the_threshold_are_left_out():
    row = run(stub_lm([answer(mentions__pro_grill=0.3)]), [SENTENCE]).row(0, named=True)
    assert row["products_mentioned"] == ["Elite Tongs"]


def test_the_model_is_never_sent_the_rating_or_the_customer():
    lm = stub_lm([answer()])
    run(lm, [SENTENCE])
    prompt = lm.stub.prompts[-1]
    assert "Tongs broke" in prompt and "Germany" not in prompt and "rating" not in prompt


def test_duplicate_sentences_are_asked_once():
    lm = stub_lm([answer(), answer()])
    assert run(lm, [SENTENCE, SENTENCE]).height == 1 and lm.stub.calls == 1


def test_a_failure_is_recorded_not_filled_in_and_not_cached(tmp_path):
    cache = tmp_path / "answers.parquet"
    row = run(stub_lm(RuntimeError("deployment busy")), [SENTENCE], cache_path=cache).row(0, named=True)
    assert "deployment busy" in row["error"] and row["frustration"] is None
    retry = stub_lm([answer()])
    run(retry, [SENTENCE], cache_path=cache)
    assert retry.stub.calls == 1


def test_cached_sentences_are_not_asked_again(tmp_path):
    cache = tmp_path / "answers.parquet"
    run(stub_lm([answer()]), [SENTENCE], cache_path=cache)
    again = stub_lm([answer()])
    assert run(again, [SENTENCE], cache_path=cache).height == 1 and again.stub.calls == 0


def test_another_models_cached_answers_are_kept_but_not_reused(tmp_path):
    cache = tmp_path / "answers.parquet"
    run(stub_lm([answer()], model="model-a"), [SENTENCE], cache_path=cache)
    other = stub_lm([answer()], model="model-b")
    run(other, [SENTENCE], cache_path=cache)
    assert other.stub.calls == 1
    assert sorted(pl.read_parquet(cache)["jev_model"]) == ["openai/model-a", "openai/model-b"]


def test_foundry_lm_uses_the_v1_api_with_reasoning_model_settings():
    lm = foundry_lm(ModelSettings(endpoint="https://r.services.ai.azure.com/", api_key="k", api_version="v1",
                                  deployment="gpt-test"), reasoning_effort="low")
    assert lm.model == "openai/gpt-test"
    assert lm.kwargs["api_base"] == "https://r.services.ai.azure.com/openai/v1/"
    assert lm.kwargs["reasoning_effort"] == "low"


def test_foundry_lm_refuses_a_dated_api_version():
    with pytest.raises(ValueError, match="v1 API"):
        foundry_lm(ModelSettings(endpoint="https://r", api_key="k", api_version="2024-10-21", deployment="d"))
