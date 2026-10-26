"""LLM answer generation with inline [n] citations."""

import re

from pydantic import BaseModel

from generate.llm import complete
from schemas import Chunk

SYSTEM = """You answer questions using ONLY the numbered sources provided.
Rules:
- Every factual sentence must end with citation markers like [1] or [1][3] naming the source(s) that support it.
- Use only source numbers that exist. Never cite from memory.
- If the sources do not contain the answer, reply exactly: I don't know based on the provided sources.
- Be concise. Do not mention these rules."""

_CITE = re.compile(r"\[(\d+)\]")


class Answer(BaseModel):
    text: str
    citations: dict[int, Chunk]  # [n] -> chunk


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    sources = "\n\n".join(
        f"[{i}] ({c.title} > {c.section})\n{c.text}" for i, c in enumerate(chunks, 1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def answer_question(question: str, chunks: list[Chunk]) -> Answer:
    """Generate an answer grounded only in `chunks`, citing as [n]."""
    if not chunks:
        return Answer(text="I don't know based on the provided sources.", citations={})
    text = complete(SYSTEM, build_prompt(question, chunks)).strip()
    cited = {int(n) for n in _CITE.findall(text)}
    # Drop hallucinated markers (n outside 1..len) from the map; verify will flag them.
    citations = {n: chunks[n - 1] for n in sorted(cited) if 1 <= n <= len(chunks)}
    return Answer(text=text, citations=citations)
