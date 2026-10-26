"""Sparse lexical retrieval with rank_bm25."""

import pickle
import re

from rank_bm25 import BM25Okapi

from config import DATA_DIR
from schemas import Chunk

BM25_PATH = DATA_DIR / "bm25.pkl"

_TOKEN = re.compile(r"\w+", re.UNICODE)

_bm25: BM25Okapi | None = None
_chunk_ids: list[str] = []


def tokenize(text: str) -> list[str]:
    """Tokenizer used for BOTH corpus and queries."""
    return _TOKEN.findall(text.lower())


def build_bm25(chunks: list[Chunk]) -> None:
    """Build the BM25 index over chunks and persist it under data/."""
    global _bm25, _chunk_ids
    if not chunks:
        raise ValueError("cannot build BM25 over zero chunks")
    # Index title + section + text so heading terms are searchable.
    corpus = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
    _bm25 = BM25Okapi(corpus)
    _chunk_ids = [c.chunk_id for c in chunks]
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": _bm25, "chunk_ids": _chunk_ids}, f)


def load_bm25() -> None:
    """Load the persisted BM25 index (local trusted file; pickle)."""
    global _bm25, _chunk_ids
    with BM25_PATH.open("rb") as f:
        blob = pickle.load(f)
    _bm25, _chunk_ids = blob["bm25"], blob["chunk_ids"]


def bm25_search(query: str, k: int) -> list[tuple[str, float]]:
    """Return top-k (chunk_id, score), best first."""
    if _bm25 is None:
        load_bm25()
    assert _bm25 is not None
    tokens = tokenize(query)
    if not tokens or k <= 0:
        return []
    scores = _bm25.get_scores(tokens)
    # Stable sort: ties keep corpus order -> deterministic.
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(_chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]
