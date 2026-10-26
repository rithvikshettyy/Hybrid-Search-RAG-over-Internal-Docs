"""Split Documents into Chunks (structure-aware: headers first, size second)."""

import hashlib
import re
from pathlib import PurePosixPath

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from schemas import Chunk

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    """Split docs into Chunks with stable chunk_id, source, title, section, text."""
    # strip_headers=False keeps heading words in chunk text (helps BM25 + embeddings).
    header_splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks: list[Chunk] = []
    for doc in docs:
        source = doc.metadata["source"]
        title = doc.metadata.get("title") or _fallback_title(doc.page_content, source)
        index = 0
        for sec in header_splitter.split_text(doc.page_content):
            section = " > ".join(sec.metadata[k] for _, k in _HEADERS if k in sec.metadata)
            for piece in size_splitter.split_text(sec.page_content):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(source, index),
                        source=source,
                        title=title,
                        section=section or title,
                        text=piece,
                    )
                )
                index += 1
    return chunks


def _fallback_title(body: str, source: str) -> str:
    m = _H1.search(body)
    return m.group(1) if m else PurePosixPath(source).stem


def make_chunk_id(source: str, index: int) -> str:
    """Deterministic, ASCII-safe id (Pinecone ids must be ASCII) so re-ingest is idempotent."""
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{index}"
