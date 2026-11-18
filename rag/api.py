"""FastAPI service: POST /ingest, POST /ask, GET /health.

Run: uvicorn rag.api:app --reload
"""
from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from .config import Settings
from .generate import RAG
from .service import build

app = FastAPI(title="rag-pgvector-hybrid", version="0.1.0")


@lru_cache
def get_rag() -> RAG:
    rag, _ = build()
    return rag


class IngestIn(BaseModel):
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class AskIn(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(vector|keyword|hybrid)$")


class Source(BaseModel):
    doc_id: str
    chunk: int
    score: float


class AskOut(BaseModel):
    answer: str
    blocked: bool
    unsupported_numbers: list[str]
    sources: list[Source]


@app.get("/health")
def health(rag: RAG = Depends(get_rag)) -> dict:
    return {"status": "ok", "db": rag.retriever.store.ping()}


@app.post("/ingest")
def ingest(body: IngestIn, rag: RAG = Depends(get_rag)) -> dict:
    s = Settings()
    n = rag.retriever.ingest(body.doc_id, body.text, s.chunk_words, s.chunk_overlap)
    return {"doc_id": body.doc_id, "chunks": n}


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, rag: RAG = Depends(get_rag)) -> AskOut:
    ans = rag.ask(body.question, k=body.k, mode=body.mode)
    return AskOut(
        answer=ans.text,
        blocked=ans.blocked,
        unsupported_numbers=ans.unsupported_numbers,
        sources=[Source(doc_id=h.doc_id, chunk=h.idx, score=round(float(h.score), 4)) for h in ans.sources],
    )
