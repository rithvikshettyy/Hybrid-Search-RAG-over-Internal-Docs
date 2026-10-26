"""Env-driven settings. Loads .env once at import."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


# Required at call time, not import time, so tests/ingest can run without every key.
def gemini_api_key() -> str:
    return _require("GEMINI_API_KEY")


def cohere_api_key() -> str:
    return _require("COHERE_API_KEY")


def pinecone_api_key() -> str:
    return _require("PINECONE_API_KEY")


PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "handbook-hybrid")
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
HANDBOOK_DIR = Path(os.getenv("HANDBOOK_DIR", str(DATA_DIR / "handbook_repo" / "content" / "handbook")))
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
