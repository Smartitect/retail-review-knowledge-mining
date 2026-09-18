"""
Two ways to turn a brief into review text, behind one interface.

- `FoundryReviewWriter` asks a chat model deployed on Azure AI Foundry. This is
  the intended source. It reads its endpoint, key and deployment from the
  environment (`.env`), through Foundry's OpenAI-compatible `/openai/v1` API,
  so any chat deployment in the Foundry catalogue works.
- `TemplateReviewWriter` needs no service: it picks one of the hand-written
  reviews in `data/input/product_reviews.json` for the same product and
  (nearest) star rating. It is for running offline and in tests, is only used
  when chosen explicitly, and labels its output `template` so it can never be
  mistaken for generated text. It writes English only, whatever the brief asks.
"""

import json
import os
from pathlib import Path
from typing import Protocol

from .prompt import LANGUAGE_NAMES, SYSTEM, ReviewBrief, render, stable_int

LEGACY_REVIEWS = Path(__file__).resolve().parents[2] / "data" / "input" / "product_reviews.json"

ENDPOINT_ENV = "AZURE_FOUNDRY_ENDPOINT"
KEY_ENV = "AZURE_FOUNDRY_API_KEY"
DEPLOYMENT_ENV = "AZURE_FOUNDRY_DEPLOYMENT"


class ReviewWriter(Protocol):
    source: str  # "foundry" | "template"
    model: str   # what produced the text, recorded with it

    def language_written(self, brief: ReviewBrief) -> str: ...

    async def write(self, brief: ReviewBrief) -> str: ...

    async def aclose(self) -> None: ...


class FoundryReviewWriter:
    source = "foundry"

    def __init__(self, *, endpoint: str, api_key: str, deployment: str, temperature: float = 0.9,
                 max_tokens: int = 400, http_client=None):
        from openai import AsyncOpenAI

        self.model = deployment
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(base_url=endpoint, api_key=api_key, max_retries=3, timeout=60.0,
                                   http_client=http_client)

    @classmethod
    def from_env(cls, **kwargs) -> "FoundryReviewWriter":
        missing = [name for name in (ENDPOINT_ENV, KEY_ENV, DEPLOYMENT_ENV) if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                f"Azure AI Foundry is not configured: set {', '.join(missing)} in .env "
                "(the endpoint is the resource's OpenAI-compatible base URL, ending /openai/v1/)."
            )
        return cls(endpoint=os.environ[ENDPOINT_ENV], api_key=os.environ[KEY_ENV],
                   deployment=os.environ[DEPLOYMENT_ENV], **kwargs)

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
