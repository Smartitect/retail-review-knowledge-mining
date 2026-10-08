"""
The DSPy language model for an Azure AI Foundry deployment.

Settings come from Key Vault exactly as for review writing (`review_writer.model_settings`),
so the classifier runs on the same reflection deployment unless told otherwise.

Two details matter:

- **Engine.** DSPy's `lm15` engine is pinned. Its `litellm` engine fails to import
  alongside `openai` 3.x (a circular import inside `openai`), and `engine="auto"`
  could fall back to it; pinning makes that a clear error rather than a silent switch.
  It is also why only Foundry's `v1` API is supported: dated Azure OpenAI versions
  need the `litellm` engine.
- **Reasoning models.** DSPy treats every gpt-5 deployment as a reasoning model,
  which needs `temperature=None` and at least 16,000 output tokens (the hidden
  reasoning counts against them). `reasoning_effort` trades accuracy for speed;
  it is recorded with the results, so a timing is never quoted without it.
"""

import dspy

from review_writer import ModelSettings, model_settings

MAX_TOKENS = 16_000


def foundry_lm(settings: ModelSettings, *, reasoning_effort: str | None = None, num_retries: int = 3) -> dspy.LM:
    """A DSPy LM for this Foundry deployment. DSPy's own cache is off: results are cached by the classifier."""
    if settings.api_version != "v1":
        raise ValueError(f"the DSPy classifier needs Foundry's v1 API, not {settings.api_version!r}: dated Azure "
                         "OpenAI versions route through DSPy's litellm engine, which does not import alongside openai 3.x")
    extra = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
    return dspy.LM(f"openai/{settings.deployment}", api_base=f"{settings.endpoint.rstrip('/')}/openai/v1/",
                   api_key=settings.api_key, temperature=None, max_tokens=MAX_TOKENS, cache=False,
                   num_retries=num_retries, engine="lm15", **extra)


def foundry_lm_from_env(**kwargs) -> dspy.LM:
    """The reflection deployment named in `.env`, with its settings read from Key Vault."""
    return foundry_lm(model_settings(), **kwargs)
