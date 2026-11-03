# Hybrid Search RAG over Internal Docs

Purpose: user is building this to LEARN RAG. User writes the core logic. Claude's role: scaffold, explain, review. Not implement.

## Pipeline
ingest + chunk -> dense retrieval (Pinecone) + BM25 (rank_bm25) -> Reciprocal Rank Fusion -> LLM listwise rerank (Gemini) -> Claude answers with inline [n] citations -> per-claim verification against cited chunk.

## Stack
Python 3.11+, uv, LangChain (loaders + text splitters ONLY), Pinecone, rank_bm25, Ollama EmbeddingGemma (embed), Gemini LLM rerank, Anthropic SDK, FastAPI, Pydantic, python-dotenv, pytest.
Corpus: GitLab public handbook (markdown).

## Rules for every session
- NEVER write implementation for: `index/dense.py`, `index/bm25.py`, `retrieve/fusion.py`, `retrieve/rerank.py`, `generate/answer.py`, `generate/verify.py`. User writes these.
- MAY write: boilerplate, config, FastAPI wiring, Pydantic models, tests, eval harness plumbing.
- When user is stuck: explain concept, point to docs, give hints or pseudocode. No full solutions unless user says "show me the code".
- When user asks for review: point out bugs, edge cases, and why. Let user fix them.
- Before suggesting any library API, verify against current docs. Don't assume versions.
- Keep explanations short and technical.

## Build order
0 concepts -> 1 ingest -> 2 dense -> 3 BM25 -> 4 RRF -> 5 rerank -> 6 generation + verify -> 7 API -> 8 evals (recall@5, MRR: dense vs BM25 vs hybrid vs hybrid+rerank).

## Layout
- `schemas.py` — `Chunk` (chunk_id, source, title, section, text)
- `config.py` — env loading (keys read lazily via functions)
- `ingest/{load,chunk}.py`, `index/{dense,bm25}.py`, `retrieve/{fusion,rerank}.py`, `generate/{answer,verify}.py`
- `api/main.py` — FastAPI
- `evals/{run_evals.py,dataset.jsonl}`
- `data/` — gitignored (handbook clone, BM25 index)
