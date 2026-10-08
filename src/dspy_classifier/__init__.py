"""Jev's questions answered by a foundation model on Azure AI Foundry, through DSPy, for comparison with Jev."""

from .comparison import (
    Run,
    accuracy,
    agreement,
    cost,
    disagreements,
    speed,
    stratified_sample,
)
from .foundry import foundry_lm, foundry_lm_from_env
from .sentence_classifier import DSPY_RESULT_SCHEMA, classify_sentences, flatten
from .signature import build_signature, output_fields

__all__ = [
    "DSPY_RESULT_SCHEMA", "Run", "accuracy", "agreement", "build_signature", "classify_sentences", "cost",
    "disagreements", "flatten", "foundry_lm", "foundry_lm_from_env", "output_fields", "speed",
    "stratified_sample",
]
