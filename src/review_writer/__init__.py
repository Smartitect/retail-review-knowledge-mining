"""Review text for generated reviews: Azure AI Foundry, or hand-written templates offline, cached by prompt."""

from .prompt import ReviewBrief, prompt_hash, render, stable_int
from .text_cache import TEXT_SCHEMA, write_review_texts
from .writers import FoundryReviewWriter, ReviewWriter, TemplateReviewWriter

__all__ = [
    "TEXT_SCHEMA", "FoundryReviewWriter", "ReviewBrief", "ReviewWriter", "TemplateReviewWriter", "prompt_hash",
    "render", "stable_int", "write_review_texts",
]
