from pathlib import Path

import pytest

from rag.embeddings import HashEmbedder
from rag.retrieve import Retriever
from rag.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = ROOT / "data" / "handbook"
GOLDEN = ROOT / "data" / "golden.jsonl"


def load_handbook(retriever: Retriever) -> int:
    return sum(retriever.ingest(p.stem, p.read_text(encoding="utf-8")) for p in sorted(HANDBOOK.glob("*.md")))


@pytest.fixture
def memory_retriever() -> Retriever:
    r = Retriever(MemoryStore(), HashEmbedder(384))
    load_handbook(r)
    return r
