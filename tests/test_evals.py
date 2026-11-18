from rag.evals import evaluate_all, load_golden

from .conftest import GOLDEN


def test_hybrid_retrieval_meets_the_bar_in_memory(memory_retriever):
    results = {r["mode"]: r for r in evaluate_all(memory_retriever, load_golden(GOLDEN), k=3)}
    assert results["hybrid"]["hit_at_k"] >= 0.8, results["hybrid"]["misses"]
