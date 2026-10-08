"""Comparing two classifiers' answers: sampling, speed, cost, agreement and accuracy."""

import json

import polars as pl

from dspy_classifier import Run, accuracy, agreement, cost, speed, stratified_sample
from dspy_classifier.comparison import load_run, prompt_tokens, save_run
from dspy_classifier.sentence_classifier import DSPY_RESULT_SCHEMA


def results(rows):
    defaults = {
        "frustration": 0.0, "frustration_confidence": 0.9, "problem_category": "none", "language": "english",
        "sentiment": "positive", "recommendation": "none", "problem_category_confidence": 0.9,
        "language_confidence": 0.9, "sentiment_confidence": 0.9, "recommendation_confidence": 0.9,
        "churn_risk": 0.0, "safety_concern": 0.0, "suggestion": 0.0, "competitor_mention": 0.0,
        "products_mentioned": [], "answers_json": json.dumps({}), "jev_model": "m", "question_set_version": "v",
        "input_tokens": 1000, "latency_ms": 100, "error": None, "output_tokens": 500,
    }
    return pl.DataFrame([defaults | r for r in rows], schema=DSPY_RESULT_SCHEMA)


def sentences_for(reviews: dict[str, tuple[str, int, int]]) -> pl.DataFrame:
    """`reviews` maps review_id -> (language, rating, sentence count)."""
    return pl.DataFrame([
        {"review_id": rid, "review_key": f"k{rid}", "sentence_index": i, "text_language": lang, "rating": stars,
         "product_name": "Pro Grill", "sentence": f"{rid} sentence {i}"}
        for rid, (lang, stars, n) in reviews.items() for i in range(n)
    ], schema_overrides={"sentence_index": pl.UInt32})


def test_the_sample_takes_whole_reviews_from_every_stratum():
    reviews = {f"E{i}": ("english", 5, 4) for i in range(200)} | {"Z1": ("zulu", 1, 3), "G1": ("german", 3, 5)}
    sample = stratified_sample(sentences_for(reviews), target=80)
    assert {"zulu", "german", "english"} <= set(sample["text_language"])
    counts = sample.group_by("review_id").len().join(
        sentences_for(reviews).group_by("review_id").len(), on="review_id", suffix="_all")
    assert (counts["len"] == counts["len_all"]).all()  # whole reviews, never fragments
    assert 60 <= sample.height <= 100


def test_the_sample_is_reproducible():
    sentences = sentences_for({f"R{i}": ("english", 1 + i % 5, 3) for i in range(100)})
    assert stratified_sample(sentences, target=60).equals(stratified_sample(sentences, target=60))


def test_agreement_per_question():
    key = lambda i: {"review_key": "k", "sentence_index": i}
    a = results([key(0) | {"problem_category": "safety", "frustration": 3.0, "churn_risk": 0.9},
                 key(1) | {"problem_category": "none", "frustration": 0.0}])
    b = results([key(0) | {"problem_category": "safety", "frustration": 1.5, "churn_risk": 0.6},
                 key(1) | {"problem_category": "price_value", "frustration": 0.5}])
    table = dict(agreement(a, b).select("question", "agreement").iter_rows())
    assert table["problem_category"] == 0.5
    assert table["frustration (within 1)"] == 0.5
    assert table["churn_risk"] == 1.0  # both above the threshold


def test_failed_sentences_are_left_out_of_agreement():
    a = results([{"review_key": "k", "sentence_index": 0}, {"review_key": "k", "sentence_index": 1}])
    b = results([{"review_key": "k", "sentence_index": 0},
                 {"review_key": "k", "sentence_index": 1, "error": "Timeout", "problem_category": None}])
    assert agreement(a, b)["sentences"][0] == 1


def test_accuracy_against_the_generators_truth():
    sentences = sentences_for({"R1": ("english", 1, 2), "R2": ("zulu", 1, 1)})
    truth = pl.DataFrame({"review_id": ["R1", "R2"], "satisfaction": [0.1, 0.2], "aspect": ["safety", "assembly"],
                          "language": ["english", "zulu"]})
    res = results([
        {"review_key": "kR1", "sentence_index": 0, "problem_category": "performance", "frustration": 3.0},
        {"review_key": "kR1", "sentence_index": 1, "problem_category": "safety", "frustration": 2.0},
        {"review_key": "kR2", "sentence_index": 0, "problem_category": "assembly", "frustration": 2.0,
         "language": "other"},
    ])
    scores = accuracy(res, sentences, truth)
    assert scores["language"] == 1.0  # Zulu heard as "other" is right
    assert scores["issue_found"] == 1.0
    assert scores["primary_issue"] == 0.5  # R1's most frustrated sentence raised performance
    assert scores["unhappy_reviews"] == 2


def test_speed_and_cost():
    run = Run("m", results([{"review_key": "k", "sentence_index": i, "latency_ms": 100 * (i + 1)} for i in range(4)]),
              seconds=2.0, concurrency=8)
    s = speed([run], sentences_in_dataset=6000).row(0, named=True)
    assert (s["p50_ms"], s["sentences_per_s"], s["dataset_minutes"]) == (250.0, 2.0, 50.0)
    c = cost([run], {"m": (1.0, 4.0)}, sentences_per_1000_customers=6000).row(0, named=True)
    assert c["usd_per_sentence"] == (1000 * 1.0 + 500 * 4.0) / 1e6
    assert cost([run], {}, sentences_per_1000_customers=6000)["usd_per_sentence"][0] is None


def test_a_saved_run_comes_back_unchanged(tmp_path):
    run = Run("m", results([{"review_key": "k", "sentence_index": 0}]), seconds=12.5, concurrency=8)
    save_run(run, tmp_path / "m.parquet")
    back = load_run(tmp_path / "m.parquet")
    assert (back.name, back.seconds, back.concurrency) == ("m", 12.5, 8)
    assert back.results.equals(run.results)


def test_prompt_tokens_are_counted_per_distinct_sentence():
    catalogue = pl.DataFrame({"product_name": ["Pro Grill"], "product_category": ["grills"]})
    one = {"review_key": "k", "sentence_index": 0, "sentence_count": 1, "sentence": "Great grill.",
           "review_text": "Great grill.", "product_name": "Pro Grill", "product_category": "grills"}
    single = prompt_tokens(pl.DataFrame([one]), catalogue)
    assert single > 500  # the question set dominates the prompt
    assert prompt_tokens(pl.DataFrame([one, one]), catalogue) == single  # duplicates are asked once
    assert prompt_tokens(pl.DataFrame([one, one | {"sentence_index": 1}]), catalogue) > 1.9 * single
