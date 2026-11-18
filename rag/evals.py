"""Retrieval evals over a small golden set: hit@k and MRR per retrieval mode."""
import json
from pathlib import Path

from .retrieve import MODES, Retriever


def load_golden(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(retriever: Retriever, golden: list[dict], k: int = 3, mode: str = "hybrid") -> dict:
    hits, rr = 0, 0.0
    misses = []
    for item in golden:
        docs = [h.doc_id for h in retriever.search(item["question"], k=k, mode=mode)]
        if item["doc_id"] in docs:
            hits += 1
            rr += 1.0 / (docs.index(item["doc_id"]) + 1)
        else:
            misses.append(item["question"])
    n = len(golden)
    return {"mode": mode, "k": k, "n": n, "hit_at_k": hits / n, "mrr": rr / n, "misses": misses}


def evaluate_all(retriever: Retriever, golden: list[dict], k: int = 3) -> list[dict]:
    return [evaluate(retriever, golden, k, mode) for mode in MODES]
