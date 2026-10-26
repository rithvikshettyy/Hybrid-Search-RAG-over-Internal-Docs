"""Offline tests: LLM / Cohere / Pinecone calls are faked."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import generate.answer as answer_mod
import generate.verify as verify_mod
import retrieve.rerank as rerank_mod
from evals.run_evals import recall_at_k, reciprocal_rank, run_evals
from generate.answer import Answer, answer_question
from generate.verify import split_claims, verify_answer
from schemas import Chunk


def _c(i, text="t"):
    return Chunk(chunk_id=f"c{i}", source="a.md", title="T", section="S", text=text)


# ---- answer ----


def test_answer_maps_citations_and_drops_bad(monkeypatch):
    monkeypatch.setattr(answer_mod, "complete", lambda s, u, **k: "Fact one [1]. Fact two [2][9].")
    ans = answer_question("q", [_c(0), _c(1)])
    assert set(ans.citations) == {1, 2}
    assert ans.citations[1].chunk_id == "c0"


def test_answer_no_chunks_skips_llm(monkeypatch):
    monkeypatch.setattr(answer_mod, "complete", lambda *a, **k: pytest.fail("called"))
    assert answer_question("q", []).citations == {}


# ---- verify ----


def test_split_claims_keeps_markers():
    claims = split_claims("Laptops are reimbursed [1]. Monitors too [1][2]. No marker here.")
    assert claims == ["Laptops are reimbursed [1].", "Monitors too [1][2].", "No marker here."]


def test_verify_flags_uncited_and_phantom(monkeypatch):
    monkeypatch.setattr(verify_mod, "complete", lambda *a, **k: '{"supported": true, "reason": "ok"}')
    ans = Answer(text="Good [1]. Uncited. Ghost [5].", citations={1: _c(0)})
    v = verify_answer(ans)
    assert [x.supported for x in v] == [True, False, False]
    assert v[1].reason == "no citation"
    assert "nonexistent" in v[2].reason


def test_verify_unparseable_is_unsupported(monkeypatch):
    monkeypatch.setattr(verify_mod, "complete", lambda *a, **k: "dunno")
    v = verify_answer(Answer(text="Claim [1].", citations={1: _c(0)}))
    assert v[0].supported is False


# ---- rerank ----


def test_rerank_maps_indices(monkeypatch):
    class FakeCo:
        def __init__(self, **k): ...
        def rerank(self, **k):
            assert len(k["documents"]) == 3
            res = [SimpleNamespace(index=2, relevance_score=0.9), SimpleNamespace(index=0, relevance_score=0.1)]
            return SimpleNamespace(results=res)

    monkeypatch.setattr(rerank_mod.cohere, "ClientV2", FakeCo)
    monkeypatch.setattr(rerank_mod, "cohere_api_key", lambda: "k")
    out = rerank_mod.rerank("q", [_c(0), _c(1), _c(2)], top_n=2)
    assert [(c.chunk_id, s) for c, s in out] == [("c2", 0.9), ("c0", 0.1)]


# ---- api ----


def test_api_ask(monkeypatch):
    monkeypatch.setattr(api_main, "retrieve", lambda q, mode, top_n: [_c(0, "laptop")])
    monkeypatch.setattr(answer_mod, "complete", lambda *a, **k: "Laptops allowed [1].")
    monkeypatch.setattr(verify_mod, "complete", lambda *a, **k: '{"supported": true, "reason": "ok"}')
    client = TestClient(api_main.app)
    r = client.post("/ask", json={"question": "laptop?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"]["1"]["chunk_id"] == "c0"
    assert body["verdicts"][0]["supported"] is True
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}


# ---- evals ----


def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"c", "z"}, 2) == 0.0
    assert recall_at_k(["a", "b", "c"], {"c", "a"}, 3) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a"], {"z"}) == 0.0


def test_run_evals_all_systems(monkeypatch):
    import evals.run_evals as ev

    monkeypatch.setattr(ev, "retrieve", lambda q, mode, top_n: [_c(0), _c(1)])
    res = run_evals([{"question": "q", "relevant_chunk_ids": ["c1"]}])
    assert set(res) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    assert res["dense"] == {"recall@5": 1.0, "mrr": 0.5}
