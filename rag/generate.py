"""Answer generation over retrieved chunks, with the numeric guardrail applied last."""
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .embeddings import tokenize
from .guardrail import check_numbers
from .retrieve import Retriever
from .store import Hit

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM = (
    "You answer questions about a health plan handbook using only the numbered context passages. "
    "Cite every claim with the passage number in brackets, like [1]. Copy numbers exactly as written. "
    f'If the passages do not contain the answer, reply exactly: "{NO_ANSWER}"'
)

STOP = {"the", "a", "an", "is", "are", "do", "does", "how", "what", "when", "which", "of", "for",
        "to", "in", "on", "my", "i", "can", "if", "it", "and", "or", "be", "with", "there", "much", "many"}


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    passages = "\n\n".join(f"[{i}] ({h.doc_id}) {h.content}" for i, h in enumerate(hits, start=1))
    return f"Context passages:\n\n{passages}\n\nQuestion: {question}"


class LLM(Protocol):
    def generate(self, question: str, hits: Sequence[Hit]) -> str: ...


class ExtractiveLLM:
    """No-API baseline: returns the context sentence with the largest overlap of content words."""

    def generate(self, question, hits):
        q = set(tokenize(question)) - STOP
        best, best_score, best_ref = None, 0, 0
        for ref, hit in enumerate(hits, start=1):
            for sentence in re.split(r"(?<=[.!?])\s+", hit.content):
                score = len(q & set(tokenize(sentence)))
                if score > best_score:
                    best, best_score, best_ref = sentence.strip(), score, ref
        return f"{best} [{best_ref}]" if best else NO_ANSWER


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 400):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question, hits):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(question, hits)}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


def get_llm(name: str, model: str) -> LLM:
    if name == "extractive":
        return ExtractiveLLM()
    if name == "anthropic":
        return AnthropicLLM(model)
    raise ValueError(f"unknown llm: {name}")


@dataclass
class Answer:
    text: str
    blocked: bool
    unsupported_numbers: list[str] = field(default_factory=list)
    sources: list[Hit] = field(default_factory=list)


class RAG:
    def __init__(self, retriever: Retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str, k: int = 5, mode: str = "hybrid") -> Answer:
        hits = self.retriever.search(question, k=k, mode=mode)
        if not hits:
            return Answer(NO_ANSWER, blocked=False)
        draft = self.llm.generate(question, hits)
        check = check_numbers(draft, [h.content for h in hits])
        if not check.ok:
            # Fail closed: a wrong limit or deadline is worse than no answer.
            text = ("The draft answer contained numbers not found in the sources "
                    f"({', '.join(check.unsupported)}), so it was withheld.")
            return Answer(text, blocked=True, unsupported_numbers=check.unsupported, sources=hits)
        return Answer(draft, blocked=False, sources=hits)
