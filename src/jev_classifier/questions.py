"""
The questions Jev is asked about every sentence.

Jev answers typed questions rather than writing text: a Choice picks one label
from those offered, a Score places the state on an ordered rubric, and a Noul
gives the probability that a statement is true. All the questions in one call
are evaluated independently and in parallel, so each one asks one narrow thing
and code does the combining.

Every Choice has an escape hatch (`none` / `other`), so a sentence that fits
no label is not forced into the nearest one.

Change a question here and results classified under the old wording are no
longer comparable: `QUESTION_SET_VERSION` is stored with every result so the
two cannot be mixed up.
"""

import re

from typesafe_sdk import Choice, Noul, Score

QUESTION_SET_VERSION = "2026-09-18.3"

# A Noul probability at or above this counts as "yes".
NOUL_THRESHOLD = 0.5

FRUSTRATION_LEVELS = [
    "Not frustrated: positive, neutral or purely factual.",
    "Mild: a minor gripe or slight disappointment, said calmly.",
    "Clearly frustrated: annoyed or let down by a real problem.",
    "Very frustrated: angry, feels cheated, or the problem ruined the experience.",
    "Furious: hostile or emphatic language, demands a refund, or vows never to buy again.",
]

PROBLEM_CATEGORIES = {
    "none": "No problem is raised: praise, a neutral fact, or a recommendation.",
    "build_quality": "Durability or materials: breaks, bends, rusts, chips, feels cheap, wears out.",
    "performance": "Does not do its job well: heat, temperature control, accuracy, burn, flavour, results.",
    "ease_of_use": "Awkward, fiddly or confusing to use day to day.",
    "assembly": "Hard to assemble, poor instructions, missing or misfitting parts at setup.",
    "cleaning_maintenance": "Hard to clean, ash or grease handling, upkeep.",
    "size_capacity": "Too big, too small, too short, not enough capacity, does not fit.",
    "safety": "A risk of injury, burns, fire, gas leaks or unsafe food.",
    "shipping_delivery": "Late, lost or damaged in transit, wrong item delivered.",
    "customer_service": "Support, returns, warranty or refund handling.",
    "price_value": "Poor value for money, overpriced.",
    "general_dissatisfaction": "Overall disappointment with no specific cause named ('worst grill ever', 'a disaster').",
    "other": "A problem that fits none of the other categories.",
}

SENTIMENT = {
    "positive": "Favourable about the product or experience.",
    "negative": "Unfavourable about the product or experience.",
    "mixed": "Both favourable and unfavourable in the same sentence.",
    "neutral": "Factual or descriptive with no clear opinion.",
}

RECOMMENDATION = {
    "recommends": "Recommends the product to others or says it is worth buying.",
    "warns_against": "Tells others not to buy it or to look elsewhere.",
    "none": "Makes no recommendation either way.",
}

LANGUAGES = {
    "english": None, "spanish": None, "portuguese": None, "french": None,
    "german": None, "italian": None, "dutch": None, "swahili": None,
    "afrikaans": None, "chinese": None, "japanese": None, "korean": None,
    "hindi": None, "arabic": None, "russian": None, "other": None,
}


def product_slug(name: str) -> str:
    """`GrillMaster Elite Tongs` -> `grillmaster_elite_tongs`, for question names."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def mention_question(product_name: str, product_category: str) -> Noul:
    return Noul(
        instructions=(
            f"Does this sentence mention or refer to the {product_name} ({product_category})? "
            "Yes if it is named, even in passing or by a short name, or if it is what the "
            "sentence is describing - including 'it' or 'these' when the review is about it."
        )
    )


def build_questions(catalogue: list[dict[str, str]]) -> dict:
    """Every question for one sentence, keyed by the name its answer comes back under.

    `catalogue` is the master product list as `{"product_name", "product_category"}`
    rows. Products get one Noul each rather than a single Choice, because a
    sentence can mention more than one ("fits my FireMaster perfectly").
    """
    questions = {
        # --- asked for ---
        "frustration": Score(
            instructions="How frustrated is the customer in this sentence? Judge the sentence, using the review only to resolve what it refers to.",
            criteria=FRUSTRATION_LEVELS,
        ),
        "problem_category": Choice(
            instructions="What kind of problem does this sentence raise? Choose 'none' if it raises no problem.",
            criteria=PROBLEM_CATEGORIES,
        ),
        "language": Choice(
            instructions="What language is the sentence written in?",
            criteria=LANGUAGES,
        ),
        # --- added: signals worth acting on ---
        "sentiment": Choice(
            instructions="What is the sentiment of this sentence?",
            criteria=SENTIMENT,
        ),
        "recommendation": Choice(
            instructions="Does this sentence itself tell others to buy the product, or tell them not to? A complaint on its own is not a warning.",
            criteria=RECOMMENDATION,
        ),
        "churn_risk": Noul(
            instructions=(
                "Does this sentence itself say the customer is returning it, wants a refund, is "
                "switching to another brand or going back to their old one, or will not buy again? "
                "Answer from this sentence alone, not from the rest of the review."
            ),
        ),
        "safety_concern": Noul(
            instructions="Does this sentence describe a safety risk: injury, burns, fire, a gas leak, or food left unsafe to eat?",
        ),
        "suggestion": Noul(
            instructions=(
                "Does this sentence explicitly suggest or wish for a specific change to the product "
                "(e.g. 'the cord could be longer', 'wish it came in more colours')? "
                "Describing a fault is not a suggestion."
            ),
        ),
        "competitor_mention": Noul(
            instructions=(
                "Does this sentence mention a product that is not in our range - a competitor, "
                "another brand, or a product the customer used before? Our range is: "
                + "; ".join(p["product_name"] for p in catalogue) + "."
            ),
        ),
    }
    for product in catalogue:
        questions[f"mentions__{product_slug(product['product_name'])}"] = mention_question(
            product["product_name"], product["product_category"]
        )
    return questions


def build_state(sentence: dict) -> dict:
    """The JSON state for one sentence: the sentence, and the review around it.

    Two things are left out on purpose:

    - Demographics. Country would be a strong prior for the language question,
      and nothing about who wrote a sentence should change what it says.
    - The star rating. It would anchor the frustration score, and keeping it
      out is what makes "does frustration track the rating?" a fair check of
      the model rather than a test of whether it read the number.
    """
    return {
        "task": "Classify one sentence from a customer review of a barbecue product.",
        "note": "Judge what the sentence itself says. The full review is context for resolving what 'it' or 'this' refers to.",
        "sentence": sentence["sentence"],
        "sentence_position": f"{sentence['sentence_index'] + 1} of {sentence['sentence_count']}",
        "review": {
            "product_reviewed": sentence["product_name"],
            "product_category": sentence["product_category"],
            "full_text": sentence["review_text"],
        },
    }
