# Hybrid-Search-RAG-over-Internal-Docs

Retrieval-augmented QA over a document corpus using hybrid retrieval. Learning project; corpus is the GitLab public handbook (markdown).

**Pipeline**
1. Ingest: load + chunk docs, assign stable `chunk_id`s, persist to `chunks.jsonl`
2. Dense retrieval: local Ollama EmbeddingGemma (768-d) embeddings stored in Pinecone, top-k by cosine similarity
3. Sparse retrieval: BM25 (`rank_bm25`) over the same chunks
4. Fusion: Reciprocal Rank Fusion, `score = Σ 1/(60 + rank)`
5. Rerank: Cohere Rerank on fused candidates, keep top-n
6. Generate: Gemini answers from context only, cites chunks inline `[n]`
7. Verify: each claim checked against its cited chunk; unsupported claims flagged

**API**: `POST /ask` → `{answer, citations, verdicts}`, `GET /health`

**Evals**: labeled question set, recall@5 / MRR across dense, BM25, hybrid, hybrid+rerank. Results: see `evals/`.

**Stack**: Python, LangChain (loaders/splitters), Ollama (EmbeddingGemma embeddings), Pinecone, rank_bm25, Cohere (rerank), Gemini API, FastAPI

## Setup

```
uv sync
ollama pull embeddinggemma   # embeddings run locally; Ollama must be running (default http://localhost:11434)
cp .env.example .env         # fill GEMINI_API_KEY, COHERE_API_KEY, PINECONE_API_KEY
uv run pytest
```

Optional env overrides: `OLLAMA_HOST`, `OLLAMA_EMBED_MODEL`, `EMBED_DIM` (768), `PINECONE_INDEX_NAME` (`handbook-hybrid-gemma`), `PINECONE_CLOUD` / `PINECONE_REGION`, `COHERE_RERANK_MODEL`, `GEMINI_MODEL`, `HANDBOOK_DIR`.

## Run

```
git clone --depth 1 --filter=blob:none --sparse https://gitlab.com/gitlab-com/content-sites/handbook.git data/handbook_repo
git -C data/handbook_repo sparse-checkout set content/handbook
uv run python -m ingest.run --skip-dense   # chunks.jsonl + BM25 only, no keys needed
uv run python -m ingest.run                # + Ollama embed + Pinecone upsert (~54k chunks, slow on small GPU; resumable)
uv run python -m evals.make_dataset --n 50 # synthetic Q set via Gemini
uv run python -m evals.run_evals           # recall@5 / MRR per system
uv run uvicorn api.main:app --reload       # POST /ask {"question": "..."}
```

## Layout

| Path | Role |
|---|---|
| `schemas.py` | `Chunk` model |
| `config.py` | env loading |
| `ingest/` | load, chunk, run |
| `index/` | dense (Ollama embeddings + Pinecone), BM25 |
| `retrieve/` | RRF, rerank, pipeline |
| `generate/` | LLM wrapper, answer, verify |
| `api/` | FastAPI |
| `evals/` | dataset + metrics |
