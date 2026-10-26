"""Cohere Rerank over fused candidates."""

import cohere

from config import COHERE_RERANK_MODEL, cohere_api_key
from schemas import Chunk


def _doc(c: Chunk) -> str:
    return f"{c.title} | {c.section}\n{c.text}"


def rerank(query: str, candidates: list[Chunk], top_n: int) -> list[tuple[Chunk, float]]:
    """Rerank candidates against query. Return (chunk, relevance_score), best first."""
    if not candidates or top_n <= 0:
        return []
    co = cohere.ClientV2(api_key=cohere_api_key())
    resp = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=[_doc(c) for c in candidates],
        top_n=min(top_n, len(candidates)),
    )
    # r.index points back into `candidates`; results already sorted by relevance.
    return [(candidates[r.index], float(r.relevance_score)) for r in resp.results]
