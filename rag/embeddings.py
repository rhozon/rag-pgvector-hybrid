"""Embedders.

HashEmbedder: hashing trick over unigrams and bigrams, L2-normalised. Deterministic and
offline, so tests and CI never download a model. It is lexical, not semantic.

FastEmbedEmbedder: BAAI/bge-small-en-v1.5 (384 dims) through fastembed (ONNX, no torch).
"""
import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]: ...


class HashEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        features = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for feat in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            sign = 1.0 if (h >> 63) == 0 else -1.0
            vec[h % self.dim] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._vector(t) for t in texts]


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model)
        self.dim = 384

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(list(texts))]


def get_embedder(name: str, dim: int) -> Embedder:
    if name == "hash":
        return HashEmbedder(dim)
    if name == "fastembed":
        emb = FastEmbedEmbedder()
        if emb.dim != dim:
            raise ValueError(f"EMBED_DIM={dim} but the fastembed model has {emb.dim} dims")
        return emb
    raise ValueError(f"unknown embedder: {name}")
