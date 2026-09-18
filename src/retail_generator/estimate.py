"""
What a batch of reviews will cost to write and to classify, before doing either.

Jev figures are measured on this project's question set (26 questions, the
sentence plus its whole review as state): about 2,750 input tokens and 300 ms
per sentence, at $0.042 per million input tokens (output is free).

Sentence counts come from the briefs: each asks for a fixed number of
sentences, so the count is exact for Foundry and an upper bound for templates.
Template text repeats, and Jev's cache answers a repeated sentence for free.

Foundry pricing depends on the deployment, so tokens are always reported and
cost only when prices are given.
"""

from dataclasses import dataclass

from review_writer import ReviewBrief

JEV_TOKENS_PER_SENTENCE = 2_750
JEV_SECONDS_PER_SENTENCE = 0.30
JEV_USD_PER_MILLION_INPUT = 0.042
JEV_CONCURRENCY = 8

FOUNDRY_PROMPT_TOKENS = 260       # system prompt + brief
FOUNDRY_TOKENS_PER_SENTENCE = 28  # generated review text
FOUNDRY_CONCURRENCY = 8
FOUNDRY_SECONDS_PER_REVIEW = 2.5


@dataclass(frozen=True)
class Estimate:
    reviews: int
    sentences: int
    jev_tokens: int
    jev_usd: float
    jev_minutes: float
    foundry_input_tokens: int
    foundry_output_tokens: int
    foundry_usd: float | None
    foundry_minutes: float

    def lines(self) -> list[str]:
        foundry_cost = f"${self.foundry_usd:,.2f}" if self.foundry_usd is not None else "set prices to cost it"
        return [
            f"{self.reviews:,} new reviews, up to {self.sentences:,} sentences",
            (f"Azure AI Foundry text: {self.foundry_input_tokens:,} input + {self.foundry_output_tokens:,} output tokens, "
            f"{foundry_cost}, ~{self.foundry_minutes:,.1f} min"),
            (f"Jev classification:    {self.jev_tokens:,} input tokens, ${self.jev_usd:,.2f}, ~{self.jev_minutes:,.1f} min "
            "(upper bound: cached sentences are free)"),
        ]


def estimate(briefs: list[ReviewBrief], *, foundry_usd_per_million_input: float | None = None,
             foundry_usd_per_million_output: float | None = None) -> Estimate:
    sentences = sum(b.sentences for b in briefs)
    jev_tokens = sentences * JEV_TOKENS_PER_SENTENCE
    f_in = len(briefs) * FOUNDRY_PROMPT_TOKENS
    f_out = sentences * FOUNDRY_TOKENS_PER_SENTENCE
    priced = foundry_usd_per_million_input is not None and foundry_usd_per_million_output is not None
    return Estimate(
        reviews=len(briefs),
        sentences=sentences,
        jev_tokens=jev_tokens,
        jev_usd=jev_tokens / 1e6 * JEV_USD_PER_MILLION_INPUT,
        jev_minutes=sentences * JEV_SECONDS_PER_SENTENCE / JEV_CONCURRENCY / 60,
        foundry_input_tokens=f_in,
        foundry_output_tokens=f_out,
        foundry_usd=(f_in / 1e6 * foundry_usd_per_million_input + f_out / 1e6 * foundry_usd_per_million_output)
        if priced else None,
        foundry_minutes=len(briefs) * FOUNDRY_SECONDS_PER_REVIEW / FOUNDRY_CONCURRENCY / 60,
    )
