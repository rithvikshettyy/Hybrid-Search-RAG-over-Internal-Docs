"""Glue: chunk store + the four retrieval systems compared in evals."""

from functools import lru_cache
from pathlib import Path

from config import CHUNKS_PATH
from index.bm25 import bm25_search
from index.dense import dense_search
from retrieve.fusion import reciprocal_rank_fusion
from retrieve.rerank import rerank
from schemas import Chunk

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
CANDIDATES = 30  # per retriever, before fusion


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    load_chunk_store.cache_clear()


@lru_cache(maxsize=1)
def load_chunk_store(path: Path = CHUNKS_PATH) -> dict[str, Chunk]:
    with path.open(encoding="utf-8") as f:
        return {c.chunk_id: c for c in (Chunk.model_validate_json(line) for line in f if line.strip())}


def retrieve(query: str, mode: str = "hybrid_rerank", top_n: int = 5) -> list[Chunk]:
    """Return top_n chunks, best first, using the given system."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    store = load_chunk_store()
    if mode == "dense":
        ids = [i for i, _ in dense_search(query, top_n)]
    elif mode == "bm25":
        ids = [i for i, _ in bm25_search(query, top_n)]
    else:
        dense_ids = [i for i, _ in dense_search(query, CANDIDATES)]
        bm25_ids = [i for i, _ in bm25_search(query, CANDIDATES)]
        fused = reciprocal_rank_fusion(
            [dense_ids, bm25_ids], top_n=CANDIDATES if mode == "hybrid_rerank" else top_n
        )
        ids = [i for i, _ in fused]
        if mode == "hybrid_rerank":
            cands = [store[i] for i in ids if i in store]
            return [c for c, _ in rerank(query, cands, top_n)]
    return [store[i] for i in ids if i in store]
