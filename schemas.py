"""Shared Pydantic models."""

from pydantic import BaseModel


class Chunk(BaseModel):
    """One retrievable unit of a source document."""

    chunk_id: str
    source: str
    title: str
    section: str
    text: str
