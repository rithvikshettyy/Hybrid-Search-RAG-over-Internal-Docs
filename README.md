# Hybrid-Search-RAG-over-Internal-Docs
Hybrid search RAG over internal docs. Dense (Pinecone) + BM25 retrieval fused with Reciprocal Rank Fusion, reranked by Cohere, answered by Claude with inline citations and per-claim source verification. FastAPI service with retrieval evals (recall@k, MRR) comparing dense, BM25, hybrid, hybrid+rerank.
