from datetime import datetime

import polars as pl
import pytest

from review_insights import NO_ISSUE, assign_risk, customers, reviews, sankey_links


def sentence(review, customer, sentiment, problem, frustration, *, churn=0.1, rating=2, product="Tongs",
             category="accessories", day=1, revenue=50.0):
    return {
        "review_id": review, "customer_id": customer, "product_id": "P008", "product_name": product,
        "product_category": category, "rating": rating, "reviewed_at": datetime(2025, 1, day),
        "review_text": f"review {review}", "text_language": "english", "text_model": "stub",
        "tenure_days": 100.0 + day, "lifetime_revenue": revenue, "order_count": 3, "days_since_last_order": 5.0,
        "previous_reviews": 0, "age": 40, "gender": "female", "country": "UK",
        "sentiment": sentiment, "problem_category": problem, "frustration": frustration,
        "products_mentioned": [product], "churn_risk": churn, "safety_concern": 0.0,
        "suggestion": 0.0, "competitor_mention": 0.0,
    }


@pytest.fixture
def sentences():
    return pl.DataFrame([
        # R1 (customer A, day 1): negative, two problems - build quality is the more frustrated one
        sentence("R1", "A", "negative", "performance", 1.5),
        sentence("R1", "A", "negative", "build_quality", 3.2),
        sentence("R1", "A", "positive", "none", 0.0),
        # R2 (customer B): positive, no problem at all
        sentence("R2", "B", "positive", "none", 0.0, rating=5, product="Grill", category="grills"),
        sentence("R2", "B", "positive", "none", 0.0, rating=5, product="Grill", category="grills"),
        # R3 (customer A again, day 9, more revenue by then): one of each sentiment, a churn signal
        sentence("R3", "A", "positive", "none", 0.0, rating=3, day=9, revenue=80.0),
        sentence("R3", "A", "negative", "price_value", 1.0, churn=0.9, rating=3, day=9, revenue=80.0),
    ]).lazy()


def test_one_row_per_review_with_derived_columns(sentences):
    r = reviews(sentences).collect().sort("review_id")
    assert r["primary_issue"].to_list() == ["build_quality", NO_ISSUE, "price_value"]
    assert r["text_sentiment"].to_list() == ["negative", "positive", "mixed"]
    assert r["star_sentiment"].to_list() == ["negative", "positive", "mixed"]
    assert r["lifetime_revenue"].to_list() == [50.0, 50.0, 80.0]


def test_risk_tiers(sentences):
    tiers = assign_risk(reviews(sentences), at_risk_frustration=3.0, frustrated_frustration=2.0)
    assert tiers.collect().sort("review_id")["risk_tier"].to_list() == ["At risk", "Satisfied", "At risk"]
    relaxed = assign_risk(reviews(sentences), at_risk_frustration=4.0, frustrated_frustration=3.0)
    assert relaxed.collect().sort("review_id")["risk_tier"].to_list() == ["Frustrated", "Satisfied", "At risk"]


def test_a_customer_is_their_most_recent_review(sentences):
    people = customers(assign_risk(reviews(sentences))).collect().sort("customer_id")
    a = people.row(0, named=True)
    assert a["review_id"] == "R3" and a["reviews_written"] == 2
    assert a["lifetime_revenue"] == 80.0  # as at the latest review, not the first
    assert people["reviews_written"].to_list() == [2, 1]


def test_sankey_conserves_reviews_at_every_hop(sentences):
    links = sankey_links(reviews(sentences))
    per_hop = links.group_by("source_level").agg(pl.col("reviews").sum())
    assert per_hop["reviews"].to_list() == [3, 3, 3]
    last = links.filter(pl.col("target_level") == "issue").sort("target")
    assert last.select("sentiment", "source", "target").rows() == [
        ("negative", "Tongs", "build_quality"),
        ("positive", "Grill", NO_ISSUE),
        ("mixed", "Tongs", "price_value"),
    ]
