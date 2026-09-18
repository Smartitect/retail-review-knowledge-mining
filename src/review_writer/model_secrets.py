"""
Connection settings for a model deployed on Azure AI Foundry, read from Key Vault.

Nothing but the vault's address and a secret prefix lives in `.env`. Each model
role has a bundle of four secrets, `<prefix>-endpoint`, `-key`, `-api-version`
and `-deployment`, read with whatever identity `DefaultAzureCredential` finds,
which in the dev container is your `az login`.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

VAULT_ENV = "KG_KEY_VAULT_URI"
REFLECTION_MODEL_ENV = "KG_REFLECTION_MODEL_SECRETS"  # the chat model that writes review text

PARTS = ("endpoint", "key", "api-version", "deployment")


@dataclass(frozen=True)
class ModelSettings:
    endpoint: str     # resource root, https://<resource>.services.ai.azure.com/
    api_key: str
    api_version: str  # "v1" for the version-less OpenAI API, or a dated Azure OpenAI version
    deployment: str

    def __repr__(self) -> str:  # never print the key
        return (f"ModelSettings(endpoint={self.endpoint!r}, api_version={self.api_version!r}, "
                f"deployment={self.deployment!r})")


def model_settings(prefix_env: str = REFLECTION_MODEL_ENV,
                   get_secret: Callable[[str], str] | None = None) -> ModelSettings:
    """Read the bundle named by `prefix_env`. `get_secret` replaces Key Vault, for tests."""
    missing = [name for name in (VAULT_ENV, prefix_env) if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Azure AI Foundry is not configured: set {', '.join(missing)} in .env "
            f"(the Key Vault URI, and the prefix of its -endpoint, -key, -api-version and -deployment secrets)."
        )
    prefix = os.environ[prefix_env]
    get_secret = get_secret or _key_vault_reader(os.environ[VAULT_ENV])
    values = {part: get_secret(f"{prefix}-{part}").strip() for part in PARTS}
    return ModelSettings(endpoint=values["endpoint"], api_key=values["key"], api_version=values["api-version"],
                         deployment=values["deployment"])


def _key_vault_reader(vault_uri: str) -> Callable[[str], str]:
    from azure.core.exceptions import AzureError, ResourceNotFoundError
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    client = SecretClient(vault_url=vault_uri, credential=DefaultAzureCredential())

    def get_secret(name: str) -> str:
        try:
            value = client.get_secret(name).value
        except ResourceNotFoundError as exc:
            raise RuntimeError(f"secret {name!r} is not in {vault_uri}") from exc
        except AzureError as exc:
            raise RuntimeError(f"could not read {name!r} from {vault_uri} (have you run `az login`?): {exc}") from exc
        if not value:
            raise RuntimeError(f"secret {name!r} in {vault_uri} is empty")
        return value

    return get_secret
