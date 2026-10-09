"""
Connection settings for a model deployed on Azure AI Foundry, read from `.env`.

Each model role has four variables sharing a prefix: `<prefix>_ENDPOINT`,
`_KEY`, `_DEPLOYMENT` and `_API_VERSION`. They live in `.env` rather than in a
key vault so that the demo runs for someone with no Azure identity of their own:
an endpoint and key are all Foundry needs. `.env` is gitignored, and
`ModelSettings` never prints the key.

The API version defaults to `v1`, the version-less OpenAI API, which is also the
only one the DSPy classifier supports.
"""

import os
from dataclasses import dataclass

REFLECTION_MODEL = "KG_REFLECTION_MODEL"  # the chat model that writes review text, and the DSPy classifier's default

REQUIRED = ("ENDPOINT", "KEY", "DEPLOYMENT")
DEFAULT_API_VERSION = "v1"


@dataclass(frozen=True)
class ModelSettings:
    endpoint: str     # resource root, https://<resource>.services.ai.azure.com/
    api_key: str
    api_version: str  # "v1" for the version-less OpenAI API, or a dated Azure OpenAI version
    deployment: str

    def __repr__(self) -> str:  # never print the key
        return (f"ModelSettings(endpoint={self.endpoint!r}, api_version={self.api_version!r}, "
                f"deployment={self.deployment!r})")


def model_settings(prefix: str = REFLECTION_MODEL) -> ModelSettings:
    """Read the role named by `prefix` from the environment, which `load_dotenv` fills from `.env`."""
    def read(part: str) -> str:
        return os.environ.get(f"{prefix}_{part}", "").strip()

    missing = [f"{prefix}_{part}" for part in REQUIRED if not read(part)]
    if missing:
        raise RuntimeError(
            f"Azure AI Foundry is not configured: set {', '.join(missing)} in .env "
            f"(the resource endpoint, its API key and the deployment name; see .env.example)."
        )
    return ModelSettings(endpoint=read("ENDPOINT"), api_key=read("KEY"),
                         api_version=read("API_VERSION") or DEFAULT_API_VERSION, deployment=read("DEPLOYMENT"))
