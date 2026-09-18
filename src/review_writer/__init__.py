"""Review text for generated reviews: Azure AI Foundry, or hand-written templates offline, cached by prompt."""

from .model_secrets import ModelSettings, model_settings
from .prompt import ReviewBrief, prompt_hash, render, stable_int
from .text_cache import TEXT_SCHEMA, write_review_texts
from .writers import FoundryReviewWriter, ReviewWriter, TemplateReviewWriter

__all__ = [
    "TEXT_SCHEMA", "FoundryReviewWriter", "ModelSettings", "ReviewBrief", "ReviewWriter", "TemplateReviewWriter",
    "model_settings", "prompt_hash", "render", "stable_int", "write_review_texts",
]
