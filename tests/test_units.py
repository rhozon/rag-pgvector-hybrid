import numpy as np
import pytest

from rag.chunking import chunk_text
from rag.embeddings import HashEmbedder
from rag.generate import NO_ANSWER, RAG, ExtractiveLLM
from rag.guardrail import check_numbers, numbers_in
from rag.retrieve import rrf
from rag.store import Hit, or_tsquery


# chunking -----------------------------------------------------------------

def test_short_document_is_one_chunk():
    chunks = chunk_text("d", "First paragraph.\n\nSecond paragraph.", max_words=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0].text == "First paragraph. Second paragraph."


def test_chunks_respect_limit_and_overlap():
    paras = "\n\n".join(" ".join(f"p{i}w{j}" for j in range(30)) for i in range(5))
    chunks = chunk_text("d", paras, max_words=70, overlap=10)
    assert all(len(c.text.split()) <= 70 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.text.split()[-10:] == nxt.text.split()[:10]


def test_long_paragraph_is_windowed():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text("d", text, max_words=100, overlap=20)
    words = [w for c in chunks for w in c.text.split()]
    assert set(words) == {f"w{i}" for i in range(250)}
    assert all(len(c.text.split()) <= 100 for c in chunks)


def test_bad_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("d", "x", max_words=10, overlap=10)


# embeddings ---------------------------------------------------------------

def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(64)
    a, b = e.embed(["dental cleaning twice a year", "dental cleaning twice a year"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_hash_embedder_prefers_lexical_neighbours():
    e = HashEmbedder(384)
    q, near, far = e.embed(["emergency room co-pay", "the emergency room co-pay is $150", "orthodontics for children"])
    assert float(q @ near) > float(q @ far)


# fusion -------------------------------------------------------------------

def _h(cid):
    return Hit(cid, f"d{cid}", 0, "", 0.0)


def test_rrf_rewards_agreement_between_retrievers():
    vector = [_h(1), _h(2), _h(3)]
    keyword = [_h(3), _h(4), _h(1)]
    fused = [h.chunk_id for h in rrf([vector, keyword], top_n=4)]
    assert fused[:2] == [1, 3]  # both appear in both lists
    assert set(fused) == {1, 2, 3, 4}


def test_or_tsquery_is_sanitised():
    assert or_tsquery("What's the co-pay? DROP TABLE x;--") == "what | s | the | co | pay | drop | table | x"


# guardrail ----------------------------------------------------------------

def test_numbers_are_normalised():
    assert numbers_in("Pay $1,500.00 or 20 percent [3].") == ["1500", "20"]


def test_guardrail_passes_supported_numbers():
    assert check_numbers("The co-pay is $150 [1].", ["The emergency room co-pay is $150 per visit."]).ok


def test_guardrail_flags_invented_numbers():
    res = check_numbers("The co-pay is $175 within 24 hours [1].", ["The co-pay is $150, waived within 24 hours."])
    assert not res.ok and res.unsupported == ["175"]


# generation ---------------------------------------------------------------

class FakeLLM:
    def __init__(self, text):
        self.text = text

    def generate(self, question, hits):
        return self.text


def test_extractive_answer_cites_the_right_sentence(memory_retriever):
    ans = RAG(memory_retriever, ExtractiveLLM()).ask("How many telehealth visits are free each year?")
    assert not ans.blocked
    assert "6 telehealth visits" in ans.text


def test_rag_blocks_answers_with_unsupported_numbers(memory_retriever):
    ans = RAG(memory_retriever, FakeLLM("The ER co-pay is $999 [1].")).ask("ER co-pay?")
    assert ans.blocked and ans.unsupported_numbers == ["999"]
    assert "999" in ans.text and "withheld" in ans.text


def test_extractive_says_it_does_not_know():
    assert ExtractiveLLM().generate("zzz qqq", [Hit(1, "d", 0, "Unrelated text.", 0.0)]) == NO_ANSWER
