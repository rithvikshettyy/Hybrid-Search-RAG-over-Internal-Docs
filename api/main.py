"""FastAPI app: /ask endpoint wiring the full pipeline."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from generate.answer import answer_question
from generate.verify import ClaimVerdict, verify_answer
from retrieve.pipeline import retrieve
from schemas import Chunk

app = FastAPI(title="Hybrid Search RAG")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: dict[int, Chunk]
    verdicts: list[ClaimVerdict]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """dense + bm25 -> RRF -> rerank -> answer -> verify."""
    chunks = retrieve(req.question, mode="hybrid_rerank", top_n=req.top_k)
    answer = answer_question(req.question, chunks)
    verdicts = verify_answer(answer)
    return AskResponse(answer=answer.text, citations=answer.citations, verdicts=verdicts)
