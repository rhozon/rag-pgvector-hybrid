"""Integration tests against a real Postgres with pgvector. Skipped unless DATABASE_URL is set
(CI runs them against the pgvector/pgvector:pg16 service container)."""
import os

import pytest

from rag.config import Settings
from rag.evals import evaluate_all, load_golden
from rag.generate import RAG

from .conftest import GOLDEN, load_handbook

pytestmark = [
    pytest.mark.pg,
    pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set"),
]


@pytest.fixture(scope="module")
def pg_rag():
    from rag.service import build

    rag, store = build(Settings(embedder="hash", llm="extractive"), reset=True)
    assert load_handbook(rag.retriever) > 0
    yield rag
    store.conn.close()


def test_indexes_exist(pg_rag):
    names = {r[0] for r in pg_rag.retriever.store.conn.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'").fetchall()}
    assert {"chunks_embedding_hnsw", "chunks_tsv_gin"} <= names


def test_reingest_replaces_chunks(pg_rag):
    store = pg_rag.retriever.store
    before = store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    pg_rag.retriever.ingest("dental", (GOLDEN.parent / "handbook" / "dental.md").read_text(encoding="utf-8"))
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == before


def test_vector_and_keyword_search_hit(pg_rag):
    r = pg_rag.retriever
    assert r.search("orthodontics lifetime maximum", k=3, mode="keyword")[0].doc_id == "dental"
    assert "dental" in [h.doc_id for h in r.search("orthodontics lifetime maximum", k=3, mode="vector")]


def test_hybrid_eval_on_postgres(pg_rag):
    results = {r["mode"]: r for r in evaluate_all(pg_rag.retriever, load_golden(GOLDEN), k=3)}
    assert results["hybrid"]["hit_at_k"] >= 0.8, results["hybrid"]["misses"]


def test_api_end_to_end(pg_rag):
    from fastapi.testclient import TestClient

    from rag.api import app, get_rag

    app.dependency_overrides[get_rag] = lambda: pg_rag
    try:
        client = TestClient(app)
        assert client.get("/health").json() == {"status": "ok", "db": True}
        out = client.post("/ask", json={"question": "How long do I have to appeal a denied claim?"}).json()
        assert out["blocked"] is False
        assert "180 days" in out["answer"]
        assert out["sources"][0]["doc_id"] == "appeals"
        assert client.post("/ask", json={"question": "hi", "mode": "bogus"}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_guardrail_through_rag_on_postgres(pg_rag):
    class Liar:
        def generate(self, question, hits):
            return "You have 90 days to appeal [1]."

    ans = RAG(pg_rag.retriever, Liar()).ask("How long do I have to appeal a denied claim?")
    assert ans.blocked and ans.unsupported_numbers == ["90"]
