"""Ingest contract tests. Red until ingest/load.py and ingest/chunk.py are implemented."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingest.chunk import chunk_documents, make_chunk_id
from ingest.load import load_documents
from schemas import Chunk

# Edit to match the chunk size you choose (chars). Loose upper bound on any single chunk.
MAX_CHUNK_CHARS = 4000

MD = """---
title: Remote Work
---

# Remote Work

Intro paragraph about remote work at the company.

## Equipment

Employees may expense a laptop, monitor and chair. Receipts are required.

## Time Zones

Teams are async-first. Overlap hours are optional.
"""

LONG_SECTION = "# Big\n\n## Wall of text\n\n" + ("This sentence repeats for length. " * 400)


@pytest.fixture
def handbook(tmp_path: Path) -> Path:
    (tmp_path / "people").mkdir()
    (tmp_path / "people" / "remote.md").write_text(MD, encoding="utf-8")
    (tmp_path / "big.md").write_text(LONG_SECTION, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not markdown", encoding="utf-8")
    return tmp_path


# ---- load ----------------------------------------------------------------


def test_load_only_markdown(handbook: Path):
    docs = load_documents(handbook)
    assert len(docs) == 2
    assert all(d.page_content.strip() for d in docs)


def test_load_sets_source_metadata(handbook: Path):
    docs = load_documents(handbook)
    sources = {Path(d.metadata["source"]).name for d in docs}
    assert sources == {"remote.md", "big.md"}


# ---- chunk ---------------------------------------------------------------


@pytest.fixture
def chunks(handbook: Path) -> list[Chunk]:
    return chunk_documents(load_documents(handbook))


def test_returns_chunk_models(chunks: list[Chunk]):
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_ids_unique(chunks: list[Chunk]):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_stable_across_runs(handbook: Path):
    a = [c.chunk_id for c in chunk_documents(load_documents(handbook))]
    b = [c.chunk_id for c in chunk_documents(load_documents(handbook))]
    assert sorted(a) == sorted(b)


def test_make_chunk_id_deterministic():
    assert make_chunk_id("a/b.md", 3) == make_chunk_id("a/b.md", 3)
    assert make_chunk_id("a/b.md", 3) != make_chunk_id("a/b.md", 4)
    assert make_chunk_id("a/b.md", 3) != make_chunk_id("a/c.md", 3)


def test_no_empty_chunks(chunks: list[Chunk]):
    assert all(c.text.strip() for c in chunks)


def test_all_fields_populated(chunks: list[Chunk]):
    for c in chunks:
        assert c.chunk_id and c.source and c.title


def test_title_from_front_matter(chunks: list[Chunk]):
    remote = [c for c in chunks if c.source.endswith("remote.md")]
    assert remote and all(c.title == "Remote Work" for c in remote)


def test_front_matter_not_in_text(chunks: list[Chunk]):
    assert not any("title: Remote Work" in c.text for c in chunks)


def test_sections_tracked(chunks: list[Chunk]):
    remote = [c for c in chunks if c.source.endswith("remote.md")]
    equipment = [c for c in remote if "laptop" in c.text]
    assert equipment and all("Equipment" in c.section for c in equipment)
    tz = [c for c in remote if "async-first" in c.text]
    assert tz and all("Time Zones" in c.section for c in tz)


def test_sections_not_merged(chunks: list[Chunk]):
    """A chunk should not straddle two ## sections."""
    assert not any("laptop" in c.text and "async-first" in c.text for c in chunks)


def test_oversized_section_is_split(chunks: list[Chunk]):
    big = [c for c in chunks if c.source.endswith("big.md")]
    assert len(big) > 1
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
