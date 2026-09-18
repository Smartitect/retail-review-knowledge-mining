"""
Two ways to turn a brief into review text, behind one interface.

- `FoundryReviewWriter` asks a chat model deployed on Azure AI Foundry. This is
  the intended source. It reads its endpoint, key, API version and deployment
  from Key Vault (`model_secrets`), and talks to Foundry's OpenAI-compatible
  API, so any chat deployment in the Foundry catalogue works.
- `TemplateReviewWriter` needs no service: it picks one of the hand-written
  reviews in `data/input/product_reviews.json` for the same product and
  (nearest) star rating. It is for running offline and in tests, is only used
  when chosen explicitly, and labels its output `template` so it can never be
  mistaken for generated text. It writes English only, whatever the brief asks.
"""

import json
from pathlib import Path
from typing import Protocol

from .model_secrets import ModelSettings, model_settings
from .prompt import LANGUAGE_NAMES, SYSTEM, ReviewBrief, render, stable_int

LEGACY_REVIEWS = Path(__file__).resolve().parents[2] / "data" / "input" / "product_reviews.json"


class ReviewWriter(Protocol):
    source: str  # "foundry" | "template"
    model: str   # what produced the text, recorded with it

    def language_written(self, brief: ReviewBrief) -> str: ...

    async def write(self, brief: ReviewBrief) -> str: ...

    async def aclose(self) -> None: ...


class FoundryReviewWriter:
    source = "foundry"

    def __init__(self, *, endpoint: str, api_key: str, deployment: str, api_version: str = "v1",
                 temperature: float = 0.9, max_tokens: int = 400, http_client=None):
        """`endpoint` is the resource root. API version "v1" is the version-less `/openai/v1/` API;
        a dated version goes through the classic Azure OpenAI deployment routes."""
        from openai import AsyncAzureOpenAI, AsyncOpenAI

        self.model = deployment
        self.temperature = temperature
        self.max_tokens = max_tokens
        options = {"api_key": api_key, "max_retries": 3, "timeout": 60.0, "http_client": http_client}
        if api_version == "v1":
            self._client = AsyncOpenAI(base_url=f"{endpoint.rstrip('/')}/openai/v1/", **options)
        else:
            self._client = AsyncAzureOpenAI(azure_endpoint=endpoint, api_version=api_version, **options)

    @classmethod
    def from_settings(cls, settings: ModelSettings, **kwargs) -> "FoundryReviewWriter":
        return cls(endpoint=settings.endpoint, api_key=settings.api_key, deployment=settings.deployment,
                   api_version=settings.api_version, **kwargs)

    @classmethod
    def from_env(cls, **kwargs) -> "FoundryReviewWriter":
        """The reflection model named in `.env`, with its settings read from Key Vault."""
        return cls.from_settings(model_settings(), **kwargs)

    def language_written(self, brief: ReviewBrief) -> str:
        return brief.language

    async def write(self, brief: ReviewBrief) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(brief)}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=stable_int(brief.review_id) % 2**31,  # best effort: not every model honours it
        )
        text = (response.choices[0].message.content or "").strip().strip('"').strip()
        if not text:
            raise ValueError(f"empty completion for {brief.review_id}")
        return text

    async def aclose(self) -> None:
        await self._client.close()


class TemplateReviewWriter:
    source = "template"

    def __init__(self, path: Path | str = LEGACY_REVIEWS):
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        self.model = f"template:{Path(path).name}"
        self._texts: dict[str, dict[int, list[str]]] = {}
        for r in records:
            by_rating = self._texts.setdefault(r["product_name"], {})
            texts = by_rating.setdefault(r["rating"], [])
            if r["review_text"] not in texts:
                texts.append(r["review_text"])

    def language_written(self, brief: ReviewBrief) -> str:
        return "english"

    async def write(self, brief: ReviewBrief) -> str:
        by_rating = self._texts.get(brief.product_name)
        if not by_rating:
            raise KeyError(f"no template reviews for {brief.product_name!r}")
        nearest = min(by_rating, key=lambda r: (abs(r - brief.rating), r))
        texts = by_rating[nearest]
        return texts[stable_int(brief.review_id) % len(texts)]

    async def aclose(self) -> None:
        return None


__all__ = ["LANGUAGE_NAMES", "FoundryReviewWriter", "ReviewWriter", "TemplateReviewWriter"]
