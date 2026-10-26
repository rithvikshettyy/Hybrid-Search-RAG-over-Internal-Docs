# Hybrid-Search-RAG-over-Internal-Docs

Retrieval-augmented QA over a document corpus using hybrid retrieval.

**Pipeline**
1. Ingest: load + chunk docs, assign stable `chunk_id`s, persist to `chunks.jsonl`
2. Dense retrieval: embeddings stored in Pinecone, top-k by cosine similarity
3. Sparse retrieval: BM25 (`rank_bm25`) over the same chunks
4. Fusion: Reciprocal Rank Fusion, `score = Σ 1/(60 + rank)`
5. Rerank: Cohere Rerank on fused candidates, keep top-n
6. Generate: Claude answers from context only, cites chunks inline `[n]`
7. Verify: each claim checked against its cited chunk; unsupported claims flagged

**API**: `POST /ingest`, `POST /query` → `{answer, citations, verified, timings}`, `GET /health`

**Evals**: labeled question set, recall@5 / MRR across dense, BM25, hybrid, hybrid+rerank. Results: see `evals/`.

**Stack**: Python, LangChain (loaders/splitters), Pinecone, rank_bm25, Cohere Rerank, Claude API, FastAPI
