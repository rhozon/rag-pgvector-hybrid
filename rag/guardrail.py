"""Numeric guardrail: every number in the answer must appear in the retrieved context.

In member-facing answers (limits, co-pays, deadlines) a wrong number is the costly error,
and it is cheap to catch deterministically. Citation markers like [2] are ignored.
"""
import re
from dataclasses import dataclass, field
from typing import Iterable

CITATION = re.compile(r"\[\d+\]")
NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def normalise(num: str) -> str:
    value = num.replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def numbers_in(text: str) -> list[str]:
    return [normalise(m) for m in NUMBER.findall(CITATION.sub(" ", text))]


@dataclass(frozen=True)
class GuardrailResult:
    ok: bool
    unsupported: list[str] = field(default_factory=list)


def check_numbers(answer: str, contexts: Iterable[str]) -> GuardrailResult:
    allowed = {n for ctx in contexts for n in numbers_in(ctx)}
    unsupported = [n for n in dict.fromkeys(numbers_in(answer)) if n not in allowed]
    return GuardrailResult(ok=not unsupported, unsupported=unsupported)
