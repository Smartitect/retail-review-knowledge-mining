"""The transcript, driven through the real SDK over a mock HTTP transport. No test calls the live API."""

import asyncio
import io
import json

import httpx2
import polars as pl
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jev_classifier import classify_sentences, transcript

SECRET = "ts-test-key-do-not-print"

CATALOGUE = pl.DataFrame({"product_name": ["Elite Tongs"], "product_category": ["accessories"]})

SENTENCES = pl.DataFrame([
    {"review_key": "abc", "sentence_index": i, "sentence_count": 2, "sentence": s,
     "review_text": "Tongs broke. Going back to my old set.", "product_name": "Elite Tongs",
     "product_category": "accessories", "rating": 1}
    for i, s in enumerate(["Tongs broke.", "Going back to my old set."])
])


def answer_everything(request: httpx2.Request) -> httpx2.Response:
    """Answer each question in the request with the right type, as the API would."""
    questions = json.loads(request.content)["questions"]
    answers = {}
    for name, q in questions.items():
        if q["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.7}
        elif q["type"] == "choice":
            label = next(iter(q["criteria"]))
            answers[name] = {"type": "choice", "choice": label, "confidence": 0.9, "probabilities": {label: 0.9}}
        else:
            answers[name] = {"type": "score", "score": 1.0, "confidence": 0.8,
                             "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                             "probabilities": {str(i): 0.2 for i in range(len(q["criteria"]))}}
    body = {"model": "jev-test", "usage": {"input_tokens": 10, "output_tokens": 2}, "answers": answers}
    return httpx2.Response(200, json=body, headers={"x-typesafe-request-id": "req_test"})


def records(text: str) -> list[dict]:
    """Split a stream of pretty-printed JSON objects."""
    decoder, found, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        obj, i = decoder.raw_decode(text, i)
        found.append(obj)
    return found


def run(transport, **kwargs):
    client = AsyncTypeSafeClient(api_key=SECRET, transport=transport, retry=RetryPolicy(max_retries=0))
    return asyncio.run(classify_sentences(SENTENCES, CATALOGUE, client=client, concurrency=1, **kwargs))


@pytest.fixture
def stdout():
    stream = io.StringIO()
    transcript.log_to_stdout(stream)
    yield stream
    transcript.silence()


def test_off_by_default(capsys):
    transcript.silence()
    run(httpx2.MockTransport(answer_everything))
    assert "state" not in capsys.readouterr().out


def test_each_exchange_is_printed_as_it_crossed_the_wire(stdout):
    run(httpx2.MockTransport(answer_everything))
    first, second = sorted(records(stdout.getvalue()), key=lambda r: r["sentence"]["sentence_index"])
    assert first["sent"]["state"]["sentence"] == "Tongs broke."
    assert first["sent"]["model"] == "jev-latest"
    assert first["received"]["model"] == "jev-test"
    assert first["request_id"] == "req_test"
    assert second["sentence"] == {"review_key": "abc", "sentence_index": 1}


def test_questions_are_printed_once_then_elided(stdout):
    run(httpx2.MockTransport(answer_everything))
    first, second = records(stdout.getvalue())
    assert "frustration" in first["sent"]["questions"]
    assert second["sent"]["questions"].startswith("<elided")


def test_full_questions_prints_them_every_time():
    stream = io.StringIO()
    transcript.log_to_stdout(stream, full_questions=True)
    try:
        run(httpx2.MockTransport(answer_everything))
    finally:
        transcript.silence()
    assert all("frustration" in r["sent"]["questions"] for r in records(stream.getvalue()))


def test_the_key_is_never_printed(stdout):
    run(httpx2.MockTransport(answer_everything))
    assert SECRET not in stdout.getvalue()


def test_a_failed_call_is_printed_with_its_error(stdout):
    run(httpx2.MockTransport(lambda request: httpx2.Response(500, json={"error": "boom"})))
    first = records(stdout.getvalue())[0]
    assert first["received"] is None
    assert "Error" in first["error"]
    assert first["sent"]["state"]["sentence"] == "Tongs broke."
