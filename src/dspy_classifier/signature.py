"""
Jev's question set as a DSPy signature, so a foundation model can answer it.

The signature is built from `jev_classifier.build_questions`, never written
out by hand, so the two classifiers always ask the same questions with the same
labels and wording. Each question becomes typed output fields:

| Jev question | DSPy output fields |
|---|---|
| `Score` | `<name>: float` on the rubric's 0..N-1 scale, and `<name>_confidence: float` |
| `Choice` | `<name>: Literal[labels]`, and `<name>_confidence: float` |
| `Noul` | `<name>: float`, the probability (0-1) that the answer is yes |

The input is the same JSON state Jev is sent (`jev_classifier.build_state`): the
sentence and its whole review, never the star rating or the customer.

Jev measures its confidences; a foundation model can only report them. They
fill the same columns, so the insights code works on either, but they are the
model's own estimate and are not calibrated.
"""

from typing import Literal

import dspy
from typesafe_sdk import Choice, Noul, Score

from jev_classifier import build_questions

INSTRUCTIONS = (
    "Classify one sentence from a customer review of a barbecue product by answering every output field. "
    "Judge what the sentence itself says; the full review in the state is only context for resolving what "
    "'it' or 'this' refers to. Each *_confidence field is how sure you are of that answer, from 0 (a guess) "
    "to 1 (certain). Each yes/no field is the probability, from 0 to 1, that the answer is yes."
)


def _options(criteria: dict) -> str:
    return " Options: " + "; ".join(f"'{label}': {meaning}" for label, meaning in criteria.items())


def output_fields(questions: dict) -> dict[str, tuple[type, dspy.OutputField]]:
    """Typed output fields for each Jev question, keyed by field name."""
    fields = {}
    for name, q in questions.items():
        if isinstance(q, Score):
            rubric = " ".join(f"{i}: {level}" for i, level in enumerate(q.criteria))
            top = len(q.criteria) - 1
            fields[name] = (float, dspy.OutputField(
                desc=f"{q.instructions} A number from 0 to {top}; fractions are allowed. {rubric}"))
            fields[f"{name}_confidence"] = (float, dspy.OutputField(desc=f"Confidence in {name}, 0-1."))
        elif isinstance(q, Choice):
            labels = tuple(q.criteria)
            fields[name] = (Literal[labels], dspy.OutputField(desc=q.instructions + _options(q.criteria)))
            fields[f"{name}_confidence"] = (float, dspy.OutputField(desc=f"Confidence in {name}, 0-1."))
        elif isinstance(q, Noul):
            fields[name] = (float, dspy.OutputField(desc=f"{q.instructions} Probability that the answer is yes, 0-1."))
        else:
            raise TypeError(f"no DSPy field for question {name!r} of type {type(q).__name__}")
    return fields


def build_signature(catalogue: list[dict[str, str]]) -> type[dspy.Signature]:
    """The signature for one sentence, from the same catalogue rows Jev's questions are built from."""
    fields = {"state": (dict, dspy.InputField(desc="The sentence to classify and the review around it."))}
    fields |= output_fields(build_questions(catalogue))
    return dspy.make_signature(fields, INSTRUCTIONS, signature_name="ClassifyReviewSentence")
