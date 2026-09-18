"""Review text writers and cache. No test calls Azure AI Foundry: it is driven over a mock HTTP transport."""

import asyncio
import json

import httpx2
import pytest

from review_writer import (
    FoundryReviewWriter,
    ReviewBrief,
    TemplateReviewWriter,
    prompt_hash,
    render,
    write_review_texts,
)

BRIEF = ReviewBrief(review_id="R0000000001", product_name="GrillMaster Elite Tongs", product_category="accessories",
                    product_description="Stainless steel tongs.", rating=2, satisfaction=0.2, aspect="build_quality",
                    language="german")


def completion(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    return httpx2.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": f'"Text for {body["messages"][1]["content"][:9]}"'}}],
    })


def foundry(handler=completion, deployment="gpt-test"):
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return FoundryReviewWriter(endpoint="https://example.invalid/openai/v1/", api_key="k", deployment=deployment,
                               http_client=client)


def test_prompt_is_deterministic_and_hash_covers_model():
    assert render(BRIEF) == render(BRIEF)
    assert "German" in render(BRIEF) and "2 of 5" in render(BRIEF)
    assert prompt_hash(BRIEF, "a") != prompt_hash(BRIEF, "b")
    assert 2 <= BRIEF.sentences <= 7


def test_foundry_writer_sends_the_prompt_to_the_deployment():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return completion(request)

    text = asyncio.run(foundry(handler).write(BRIEF))
    assert seen["model"] == "gpt-test"
    assert seen["messages"][1]["content"] == render(BRIEF)
    assert text == "Text for Product:"  # surrounding quotes stripped


def test_foundry_writer_explains_missing_configuration(monkeypatch):
    for name in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="AZURE_FOUNDRY_ENDPOINT"):
        FoundryReviewWriter.from_env()


def test_template_writer_picks_the_same_product_nearest_rating(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps([
        {"product_name": "GrillMaster Elite Tongs", "rating": 1, "review_text": "Awful."},
        {"product_name": "GrillMaster Elite Tongs", "rating": 5, "review_text": "Great."},
        {"product_name": "Other", "rating": 2, "review_text": "Not these."},
    ]))
    writer = TemplateReviewWriter(path)
    assert asyncio.run(writer.write(BRIEF)) == "Awful."
    assert writer.language_written(BRIEF) == "english"


def test_template_writer_works_on_the_real_file():
    assert asyncio.run(TemplateReviewWriter().write(BRIEF))


def test_cache_reuses_text_and_regenerates_when_the_prompt_changes(tmp_path):
    cache = tmp_path / "cache.parquet"
    calls = []

    def counting(request):
        calls.append(1)
        return completion(request)

    first = asyncio.run(write_review_texts([BRIEF], foundry(counting), cache_path=cache))
    assert first.row(0, named=True)["language"] == "german" and first["text_source"][0] == "foundry"
    asyncio.run(write_review_texts([BRIEF], foundry(counting), cache_path=cache))
    assert len(calls) == 1

    asyncio.run(write_review_texts([BRIEF], foundry(counting, deployment="other"), cache_path=cache))
    assert len(calls) == 2


def test_failures_are_not_cached(tmp_path):
    cache = tmp_path / "cache.parquet"
    failed = asyncio.run(write_review_texts(
        [BRIEF], foundry(lambda r: httpx2.Response(400, json={"error": {"message": "bad"}})), cache_path=cache))
    assert failed.height == 0
    retried = asyncio.run(write_review_texts([BRIEF], foundry(), cache_path=cache))
    assert retried.height == 1
