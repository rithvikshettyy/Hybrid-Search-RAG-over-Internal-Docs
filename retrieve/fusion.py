"""Reciprocal Rank Fusion over multiple ranked lists."""


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, top_n: int | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked lists of chunk_ids. Return (chunk_id, fused_score), best first.

    score(d) = sum over lists of 1 / (k + rank_in_list), rank 1-based.
    Only ranks matter, so BM25 and cosine scores need no normalization.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for cid in ranking:
            if cid in seen:  # duplicate inside one list: count best rank only
                continue
            seen.add(cid)
            rank += 1
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    # dict preserves first-seen order; sorted is stable -> deterministic ties.
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return fused if top_n is None else fused[:top_n]
