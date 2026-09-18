import polars as pl
import pytest

from review_insights import NO_ISSUE, assign_risk, customers, sankey_links


def sentence(row, sentiment, problem, frustration, churn=0.1, rating=2, product="Tongs", category="accessories"):
    return {
        "review_row": row, "rating": rating, "product_name": product, "product_category": category,
        "country": "UK", "review_text": f"review {row}", "gender": "female", "age": 30,
        "days_as_customer": 100 + row, "lifetime_revenue": 50.0 * (row + 1),
        "sentiment": sentiment, "problem_category": problem, "frustration": frustration,
        "products_mentioned": [product], "churn_risk": churn, "safety_concern": 0.0,
        "suggestion": 0.0, "competitor_mention": 0.0,
    }


@pytest.fixture
def sentences():
    return pl.DataFrame([
        # customer 0: negative, two problems - build quality is the more frustrated one
        sentence(0, "negative", "performance", 1.5),
        sentence(0, "negative", "build_quality", 3.2),
        sentence(0, "positive", "none", 0.0),
        # customer 1: positive, no problem at all
        sentence(1, "positive", "none", 0.0, rating=5, product="Grill", category="grills"),
        sentence(1, "positive", "none", 0.0, rating=5, product="Grill", category="grills"),
        # customer 2: one of each sentiment, a churn signal, mild frustration
        sentence(2, "positive", "none", 0.0, rating=3),
        sentence(2, "negative", "price_value", 1.0, churn=0.9, rating=3),
    ]).lazy()


def test_one_row_per_customer_with_derived_columns(sentences):
    c = customers(sentences).collect().sort("review_row")
    assert c["primary_issue"].to_list() == ["build_quality", NO_ISSUE, "price_value"]
    assert c["text_sentiment"].to_list() == ["negative", "positive", "mixed"]
    assert c["star_sentiment"].to_list() == ["negative", "positive", "mixed"]
    assert c["lifetime_revenue"].to_list() == [50.0, 100.0, 150.0]


def test_risk_tiers(sentences):
    tiers = assign_risk(customers(sentences), at_risk_frustration=3.0, frustrated_frustration=2.0)
    assert tiers.collect().sort("review_row")["risk_tier"].to_list() == ["At risk", "Satisfied", "At risk"]
    relaxed = assign_risk(customers(sentences), at_risk_frustration=4.0, frustrated_frustration=3.0)
    assert relaxed.collect().sort("review_row")["risk_tier"].to_list() == ["Frustrated", "Satisfied", "At risk"]


def test_sankey_conserves_customers_at_every_hop(sentences):
    links = sankey_links(customers(sentences))
    per_hop = links.group_by("source_level").agg(pl.col("customers").sum())
    assert per_hop["customers"].to_list() == [3, 3, 3]
    last = links.filter(pl.col("target_level") == "issue").sort("target")
    assert last.select("sentiment", "source", "target").rows() == [
        ("negative", "Tongs", "build_quality"),
        ("positive", "Grill", NO_ISSUE),
        ("mixed", "Tongs", "price_value"),
    ]
