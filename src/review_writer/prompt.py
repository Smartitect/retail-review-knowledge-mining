"""
The brief for one review, and the prompt it becomes.

A `ReviewBrief` is everything the generator decided about a review: the
product, the stars, how satisfied the customer really was, what to talk about,
and which language to write in. The prompt is rendered from it deterministically,
and `prompt_hash` identifies the (prompt, model) pair so the text cache knows
when it can reuse an answer and when something has changed.
"""

import hashlib
from dataclasses import dataclass

SYSTEM = (
    "You write realistic, varied customer reviews for products sold by a fictional barbecue retailer. "
    "Write only the review text as the customer would type it: no title, no star rating, no quotation "
    "marks, no preamble or explanation. Invent concrete, plausible details. Never mention that the "
    "review is generated or fictional."
)

LANGUAGE_NAMES = {
    "english": "English", "german": "German", "portuguese": "Brazilian Portuguese", "spanish": "Spanish",
    "japanese": "Japanese", "chinese": "Simplified Chinese", "swahili": "Swahili", "zulu": "Zulu",
}


@dataclass(frozen=True)
class ReviewBrief:
    review_id: str
    product_name: str
    product_category: str
    product_description: str
    rating: int
    satisfaction: float
    aspect: str
    language: str = "english"

    @property
    def sentences(self) -> int:
        """2-7 sentences, fixed per review so the prompt does not change between runs."""
        return 2 + stable_int(self.review_id) % 6


def stable_int(text: str) -> int:
    """A hash that is the same in every process, unlike `hash()`."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def feeling(satisfaction: float) -> str:
    if satisfaction < 0.15:
        return "furious - the purchase was a real letdown and they feel cheated"
    if satisfaction < 0.35:
        return "clearly disappointed and frustrated"
    if satisfaction < 0.5:
        return "a bit let down; it has real flaws"
    if satisfaction < 0.65:
        return "mostly satisfied, with reservations"
    if satisfaction < 0.85:
        return "pleased; it does the job well"
    return "delighted - they love it and would recommend it"


def render(brief: ReviewBrief) -> str:
    """The user message for this brief."""
    return (
        f"Product: {brief.product_name} ({brief.product_category}). {brief.product_description}\n"
        f"Stars the customer gave: {brief.rating} of 5.\n"
        f"How the customer feels: {feeling(brief.satisfaction)}.\n"
        f"Focus mainly on: {brief.aspect.replace('_', ' ')}.\n"
        f"Write it in {LANGUAGE_NAMES.get(brief.language, brief.language)}.\n"
        f"Length: about {brief.sentences} sentences."
    )


def prompt_hash(brief: ReviewBrief, model: str) -> str:
    """Identifies what was asked of which model: change either and the cached text no longer applies."""
    return hashlib.sha256(f"{model}\x1f{SYSTEM}\x1f{render(brief)}".encode()).hexdigest()[:16]
