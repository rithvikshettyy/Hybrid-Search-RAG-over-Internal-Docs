import pytest

import index.bm25 as bm25
from retrieve.fusion import reciprocal_rank_fusion
from schemas import Chunk


def _c(i, text):
    return Chunk(chunk_id=f"c{i}", source="a.md", title="T", section="S", text=text)


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(bm25, "BM25_PATH", tmp_path / "bm25.pkl")
    bm25.build_bm25(
        [_c(0, "laptop expense policy"), _c(1, "async communication norms"), _c(2, "vacation days")]
    )


def test_bm25_ranks_exact_term(built):
    assert bm25.bm25_search("laptop", 3)[0][0] == "c0"


def test_bm25_drops_zero_scores_and_empty_query(built):
    assert bm25.bm25_search("zzzz", 3) == []
    assert bm25.bm25_search("   ", 3) == []


def test_bm25_roundtrip(built):
    before = bm25.bm25_search("vacation", 3)
    bm25._bm25 = None
    assert bm25.bm25_search("vacation", 3) == before


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert [c for c, _ in fused][:2] == ["a", "b"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_top_n_and_dupes():
    fused = reciprocal_rank_fusion([["a", "a", "b"]], top_n=1)
    assert fused == [("a", pytest.approx(1 / 61))]


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
