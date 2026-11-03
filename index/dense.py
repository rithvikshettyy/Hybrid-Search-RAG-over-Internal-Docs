"""Dense retrieval: local Ollama embeddings (EmbeddingGemma) + Pinecone."""

import time

import httpx
from pinecone import Pinecone

from config import (
    EMBED_DIM,
    OLLAMA_EMBED_MODEL,
    OLLAMA_HOST,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    pinecone_api_key,
)
from schemas import Chunk

EMBED_BATCH = 32  # texts per Ollama /api/embed request (small GPU; bigger batches were no faster)
UPSERT_BATCH = 96  # Pinecone upsert / resume-check granularity


def _pinecone() -> Pinecone:
    return Pinecone(api_key=pinecone_api_key())


def _is_transient(e: Exception) -> bool:
    if isinstance(e, (httpx.TransportError, OSError, TimeoutError)):  # resets, DNS blips, Ollama restarting
        return True
    name = type(e).__name__
    return name in {"ServiceUnavailableError", "GatewayTimeoutError", "InternalServerError"} or any(
        s in name for s in ("Connection", "Timeout", "Unavailable")
    )


def _with_retry(fn, tries: int = 8):
    """Back off on network errors (flaky connections, Pinecone blips)."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1 or not _is_transient(e):
                raise
            time.sleep(min(60, 3 * 2**attempt))


# EmbeddingGemma prompt formats (model card): queries and documents are embedded differently.
def _format(text: str, input_type: str) -> str:
    if input_type == "search_query":
        return f"task: search result | query: {text}"
    return text  # documents arrive pre-formatted by upsert_chunks as "title: ... | text: ..."


def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed texts via local Ollama. input_type: 'search_document' for corpus, 'search_query' for queries."""
    out: list[list[float]] = []
    with httpx.Client(base_url=OLLAMA_HOST, timeout=600) as http:
        for i in range(0, len(texts), EMBED_BATCH):
            batch = [_format(t, input_type) for t in texts[i : i + EMBED_BATCH]]

            def call():
                r = http.post("/api/embed", json={"model": OLLAMA_EMBED_MODEL, "input": batch})
                r.raise_for_status()
                return r.json()["embeddings"]

            vecs = _with_retry(call)
            if len(vecs) != len(batch) or any(len(v) != EMBED_DIM for v in vecs):
                raise RuntimeError(f"Ollama returned unexpected embeddings (want dim {EMBED_DIM})")
            out.extend(vecs)
    return out


def ensure_index() -> None:
    """Create the serverless index if missing and wait until ready."""
    pc = _pinecone()
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.indexes.create(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            deployment={
                "deployment_type": "managed",
                "cloud": PINECONE_CLOUD,
                "region": PINECONE_REGION,
            },
        )
    for _ in range(60):
        if pc.describe_index(PINECONE_INDEX_NAME).status.ready:
            return
        time.sleep(2)
    raise RuntimeError("Pinecone index not ready after 120s")


def _index():
    pc = _pinecone()
    host = pc.describe_index(PINECONE_INDEX_NAME).host
    return pc.Index(host=host)


def upsert_chunks(chunks: list[Chunk]) -> None:
    """Embed and upsert chunks into Pinecone (batched). Ids are deterministic -> idempotent."""
    ensure_index()
    index = _index()
    for i in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[i : i + UPSERT_BATCH]
        # Resume: skip chunks already indexed, so restarts don't re-spend embed calls.
        have = _with_retry(lambda: index.fetch(ids=[c.chunk_id for c in batch])).vectors
        batch = [c for c in batch if c.chunk_id not in have]
        if not batch:
            continue
        # Title/section go in with the text so heading context is in the vector.
        vecs = embed_texts(
            [f"title: {c.title} | text: {c.section}\n{c.text}" for c in batch], "search_document"
        )
        _with_retry(
            lambda: index.upsert(
                vectors=[
                    {"id": c.chunk_id, "values": v, "metadata": {"source": c.source}}
                    for c, v in zip(batch, vecs)
                ],
                show_progress=False,
            )
        )
        print(f"upserted {i + len(batch)}/{len(chunks)}", flush=True)


def dense_search(query: str, k: int) -> list[tuple[str, float]]:
    """Return top-k (chunk_id, score), best first."""
    if k <= 0 or not query.strip():
        return []
    [qvec] = embed_texts([query], "search_query")
    index = _index()
    res = _with_retry(lambda: index.query(vector=qvec, top_k=k))
    return [(m["id"], float(m["score"])) for m in res["matches"]]
