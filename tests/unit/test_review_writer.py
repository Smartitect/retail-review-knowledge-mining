"""Review text writers and cache. No test calls Azure AI Foundry: it is driven over a mock HTTP transport."""

import asyncio
import json

import httpx2
import pytest

from review_writer import (
    FoundryReviewWriter,
    ModelSettings,
    ReviewBrief,
    model_settings,
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
    return FoundryReviewWriter(endpoint="https://example.invalid/", api_key="k", deployment=deployment,
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


FOUNDRY_ENV = ("KG_REFLECTION_MODEL_ENDPOINT", "KG_REFLECTION_MODEL_KEY", "KG_REFLECTION_MODEL_DEPLOYMENT",
               "KG_REFLECTION_MODEL_API_VERSION")


def test_foundry_writer_explains_missing_configuration(monkeypatch):
    for name in FOUNDRY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KG_REFLECTION_MODEL_KEY", "secret")
    with pytest.raises(RuntimeError, match="set KG_REFLECTION_MODEL_ENDPOINT, KG_REFLECTION_MODEL_DEPLOYMENT in .env"):
        FoundryReviewWriter.from_env()


def test_model_settings_come_from_the_prefixed_environment_variables(monkeypatch):
    monkeypatch.setenv("KG_REFLECTION_MODEL_ENDPOINT", "https://r.services.ai.azure.com/")
    monkeypatch.setenv("KG_REFLECTION_MODEL_KEY", "secret\n")
    monkeypatch.setenv("KG_REFLECTION_MODEL_API_VERSION", "2025-04-01-preview")
    monkeypatch.setenv("KG_REFLECTION_MODEL_DEPLOYMENT", "gpt-test")
    settings = model_settings()
    assert settings == ModelSettings(endpoint="https://r.services.ai.azure.com/", api_key="secret",
                                     api_version="2025-04-01-preview", deployment="gpt-test")
    assert "secret" not in repr(settings)


def test_another_model_role_reads_its_own_prefix_and_the_api_version_defaults_to_v1(monkeypatch):
    monkeypatch.setenv("KG_JUDGE_MODEL_ENDPOINT", "https://j.services.ai.azure.com/")
    monkeypatch.setenv("KG_JUDGE_MODEL_KEY", "k")
    monkeypatch.setenv("KG_JUDGE_MODEL_DEPLOYMENT", "gpt-judge")
    monkeypatch.delenv("KG_JUDGE_MODEL_API_VERSION", raising=False)
    assert model_settings("KG_JUDGE_MODEL") == ModelSettings(endpoint="https://j.services.ai.azure.com/", api_key="k",
                                                             api_version="v1", deployment="gpt-judge")


@pytest.mark.parametrize("pasted", [
    "https://r.services.ai.azure.com",
    "https://r.services.ai.azure.com/openai/v1/responses",
    "https://r.services.ai.azure.com/openai/deployments/gpt-test/chat/completions?api-version=2025-04-01-preview",
    "https://r.services.ai.azure.com/api/projects/demo",
])
def test_any_url_the_portal_shows_becomes_the_resource_root(monkeypatch, pasted):
    monkeypatch.setenv("KG_REFLECTION_MODEL_ENDPOINT", pasted)
    monkeypatch.setenv("KG_REFLECTION_MODEL_KEY", "k")
    monkeypatch.setenv("KG_REFLECTION_MODEL_DEPLOYMENT", "gpt-test")
    assert model_settings().endpoint == "https://r.services.ai.azure.com/"


def test_an_endpoint_that_is_not_a_url_is_refused(monkeypatch):
    monkeypatch.setenv("KG_REFLECTION_MODEL_ENDPOINT", "r.services.ai.azure.com")
    monkeypatch.setenv("KG_REFLECTION_MODEL_KEY", "k")
    monkeypatch.setenv("KG_REFLECTION_MODEL_DEPLOYMENT", "gpt-test")
    with pytest.raises(RuntimeError, match="is not a URL"):
        model_settings()


def test_foundry_writer_routes_by_api_version():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return completion(request)

    for api_version in ("v1", "2024-10-21"):
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        writer = FoundryReviewWriter.from_settings(
            ModelSettings(endpoint="https://r.services.ai.azure.com", api_key="k", api_version=api_version,
                          deployment="gpt-test"), http_client=client)
        asyncio.run(writer.write(BRIEF))
    assert urls[0] == "https://r.services.ai.azure.com/openai/v1/chat/completions"
    assert urls[1].startswith("https://r.services.ai.azure.com/openai/deployments/gpt-test/chat/completions")
    assert "api-version=2024-10-21" in urls[1]


def test_cache_reuses_text_and_regenerates_when_the_prompt_changes(tmp_path):
    cache = tmp_path / "cache.parquet"
    calls = []

    def counting(request):
        calls.append(1)
        return completion(request)

    first = asyncio.run(write_review_texts([BRIEF], foundry(counting), cache_path=cache))
    assert first.row(0, named=True)["language"] == "german" and first["model"][0] == "gpt-test"
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
