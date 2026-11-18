# rag-pgvector-hybrid

[![ci](https://github.com/rhozon/rag-pgvector-hybrid/actions/workflows/ci.yml/badge.svg)](https://github.com/rhozon/rag-pgvector-hybrid/actions/workflows/ci.yml)

Hybrid RAG on **Postgres + pgvector**: vector search (HNSW, cosine) and Postgres full-text search
fused with **Reciprocal Rank Fusion**, a **FastAPI** service, a **numeric guardrail** that withholds
answers containing numbers not found in the sources, and **retrieval evals that run in CI** against a
real `pgvector/pgvector:pg16` container.

The corpus is a small fictional health plan handbook (`data/handbook`), the kind of document where a
wrong co-pay or deadline is the expensive mistake.

## Quickstart

```bash
docker compose up -d                      # Postgres 16 + pgvector on localhost:5432
pip install -e ".[dev,embed,llm]"
export EMBEDDER=fastembed                 # or "hash": offline, deterministic, lexical
python -m rag.cli ingest data/handbook --reset
python -m rag.cli ask "How long do I have to appeal a denied claim?"
python -m rag.cli eval --k 1
pytest -v                                 # Postgres tests run when DATABASE_URL is set
```

API:

```bash
uvicorn rag.api:app --reload
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How many telehealth visits are free each year?", "mode": "hybrid"}'
```

The response has `answer`, `blocked`, `unsupported_numbers` and `sources` (doc id, chunk, fused score).

To answer with Claude instead of the extractive baseline: `LLM=anthropic`, `ANTHROPIC_API_KEY=...`,
optionally `ANTHROPIC_MODEL=...`.

## How it works

```
question ──► embed ──► pgvector HNSW (cosine) ──┐
        └──► to_tsquery (OR of terms) ──► GIN ──┴─► RRF (k=60) ─► top-k chunks
                                                                     │
                                         LLM (Claude, or extractive) ◄┘
                                                    │
                          numeric guardrail: every number in the draft must
                          appear in the retrieved chunks, else the answer is withheld
```

| module | what it does |
|---|---|
| `rag/chunking.py` | paragraph-aware chunks, 60 words with 10 words of overlap |
| `rag/embeddings.py` | `HashEmbedder` (offline, for tests) and `FastEmbedEmbedder` (BAAI/bge-small-en-v1.5, 384 d, ONNX) |
| `rag/store.py` | one `chunks` table: text, `vector(384)` with HNSW, generated `tsvector` with GIN |
| `rag/retrieve.py` | vector, keyword and hybrid (RRF) retrieval |
| `rag/guardrail.py` | number extraction and normalisation (`$1,500.00` = `1500`), citation markers ignored |
| `rag/generate.py` | prompt with numbered passages, Claude or extractive answerer, guardrail applied last |
| `rag/evals.py` | hit@k and MRR on `data/golden.jsonl` (16 questions, written in different words than the handbook) |
| `rag/api.py` | FastAPI: `POST /ingest`, `POST /ask`, `GET /health` |

## Results

Retrieval on Postgres + pgvector, measured in CI (GitHub Actions, 18 Nov 2026), 16 golden questions,
8 documents, 21 chunks.

| embedder | mode | hit@1 | hit@3 | MRR@3 |
|---|---|---|---|---|
| hash (lexical, offline) | vector | 0.75 | 0.81 | 0.78 |
| | keyword | 0.81 | 0.94 | 0.88 |
| | **hybrid** | **0.88** | 0.88 | 0.88 |
| bge-small-en-v1.5 | vector | 1.00 | 1.00 | 1.00 |
| | keyword | 0.81 | 0.94 | 0.88 |
| | **hybrid** | **1.00** | **1.00** | **1.00** |

What I read from it:

- With a weak embedder, fusion beats both retrievers at hit@1: each one fixes different misses.
- With bge-small the vector side already saturates this small set, and hybrid keeps that result. Keyword
  alone misses paraphrases ("ER" vs "emergency room", "therapy" vs "therapist").
- 16 questions is a smoke test, not a benchmark. The CI gate (`--min-hit 0.8` at k=3) exists to catch
  regressions when chunking, indexes or the query change, not to claim a score.

## Decisions

- **Postgres instead of a dedicated vector DB.** Text, embeddings and the full-text index live in one
  row, in one transaction. Re-ingesting a document is `DELETE` + `INSERT` inside a transaction, so the
  two retrievers never see different versions. For a corpus up to millions of chunks this is one less
  system to run.
- **HNSW with cosine** (`vector_cosine_ops`): no training step and good recall at small scale; IVFFlat
  would need a representative sample to build its lists.
- **Generated `tsvector` column** instead of computing it at query time, with a GIN index.
- **OR-ed `to_tsquery`.** Natural questions rarely contain every term of the answer, and
  `plainto_tsquery` requires all of them (AND). Terms are OR-ed and `ts_rank_cd` orders the matches.
  Tokens are reduced to `[a-z0-9]+` before building the query, which also keeps it safe.
- **RRF instead of weighted scores.** Cosine similarity and `ts_rank_cd` live on different scales;
  ranks avoid normalising and tuning a weight on a 16-question set.
- **Guardrail fails closed.** If the draft contains a number not present in the retrieved text, the API
  returns `blocked: true` and the offending numbers instead of the answer. For limits, co-pays and
  deadlines a missing answer is cheaper than a wrong one. It does not check numbers written as words.
- **Two embedders on purpose.** The hash embedder makes tests deterministic and lets CI run without a
  model download; the evals then run again with bge-small to show the real numbers.
- **Extractive baseline** so the whole pipeline and its tests run with no API key. It picks the
  sentence with most word overlap, with a bonus for sentences carrying a number when the question is
  quantitative. It answers 12 of the 16 golden questions correctly; the LLM path is the real answerer.

## Limitations and next steps

- No reranker yet (a cross-encoder on the fused top 20 is the next step).
- Evals cover retrieval only. Answer quality evals (faithfulness, LLM-as-judge calibrated on a few
  human labels) are not here.
- English full-text configuration only.

## Layout

```
rag/        library, API and CLI
data/       handbook corpus and golden questions
tests/      unit tests (in-memory store) and Postgres integration tests
.github/    CI: tests + evals against pgvector/pgvector:pg16
```

MIT License.
