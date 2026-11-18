# rag-pgvector-hybrid

Hybrid RAG on **Postgres + pgvector**: vector search (HNSW, cosine) and Postgres full-text
search fused with **Reciprocal Rank Fusion**, a FastAPI service, a **numeric guardrail** that
withholds answers containing numbers not found in the sources, and **retrieval evals that run in CI**
against a real pgvector container.

The corpus is a small fictional health plan handbook (`data/handbook`), the kind of document
where a wrong co-pay or deadline is the expensive mistake.

RESULTS_PLACEHOLDER
