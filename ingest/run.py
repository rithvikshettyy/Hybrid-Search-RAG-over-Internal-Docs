"""One-shot ingest: load -> chunk -> save chunks.jsonl -> build BM25 -> upsert to Pinecone.

Usage: uv run python -m ingest.run [--skip-dense]
"""

import argparse

from config import HANDBOOK_DIR
from index.bm25 import build_bm25
from index.dense import upsert_chunks
from ingest.chunk import chunk_documents
from ingest.load import load_documents
from retrieve.pipeline import save_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dense", action="store_true", help="skip embedding + Pinecone upsert")
    args = ap.parse_args()

    docs = load_documents(HANDBOOK_DIR)
    chunks = chunk_documents(docs)
    print(f"{len(docs)} docs -> {len(chunks)} chunks")
    save_chunks(chunks)
    build_bm25(chunks)
    print("bm25 built")
    if not args.skip_dense:
        upsert_chunks(chunks)
        print("dense upserted")


if __name__ == "__main__":
    main()
