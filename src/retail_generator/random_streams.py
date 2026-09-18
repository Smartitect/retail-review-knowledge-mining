"""
Independent, reproducible random streams.

Every entity draws from its own stream, keyed on the dataset seed, a stream
name and the entity's number: `rng(seed, "customer", 1234)`. So customer 1,234
comes out the same whether it was generated alone, in a batch of ten thousand,
or in the third of several incremental batches. Adding a draw to one stream
never shifts another.
"""

import hashlib

import numpy as np


def _word(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


def rng(seed: int, stream: str, *keys: int) -> np.random.Generator:
    return np.random.default_rng([seed, _word(stream), *keys])


def child_seed(generator: np.random.Generator) -> int:
    """A seed drawn from a stream, for libraries such as Faker that want an int."""
    return int(generator.integers(0, 2**31 - 1))
