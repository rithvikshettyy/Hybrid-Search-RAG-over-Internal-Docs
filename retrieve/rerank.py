"""LLM listwise rerank over fused candidates: one call ranks all candidates."""

import json
import re

from generate.llm import complete
from schemas import Chunk

MAX_CHARS = 700  # per-candidate text cap keeps the prompt small

SYSTEM = """You rank passages by how well they answer a query.
Return ONLY a JSON object: {"ranking": [<passage ids, most relevant first>]}.
Use only the given integer ids, each at most once. Omit passages that are irrelevant."""


def _doc(i: int, c: Chunk) -> str:
    return f"[{i}] {c.title} | {c.section}\n{c.text[:MAX_CHARS]}"


def _parse_ranking(raw: str, n: int) -> list[int]:
    """Extract valid, de-duplicated candidate indices in the order the LLM gave them."""
    m = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    items = obj.get("ranking", []) if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return []
    seen: set[int] = set()
    order: list[int] = []
    for x in items:
        if isinstance(x, int) and not isinstance(x, bool) and 0 <= x < n and x not in seen:
            seen.add(x)
            order.append(x)
    return order


def rerank(query: str, candidates: list[Chunk], top_n: int) -> list[tuple[Chunk, float]]:
    """Rerank candidates against query. Return (chunk, relevance_score), best first.

    Score is rank-derived (1.0 for best, decreasing), not a calibrated probability.
    If the LLM output is unparseable, falls back to the input (RRF) order.
    """
    if not candidates or top_n <= 0:
        return []
    n = len(candidates)
    prompt = f"Query: {query}\n\nPassages:\n\n" + "\n\n".join(_doc(i, c) for i, c in enumerate(candidates))
    # API/key errors propagate on purpose: a silent fallback would make evals measure plain RRF.
    order = _parse_ranking(complete(SYSTEM, prompt, temperature=0.0, json_mode=True), n)
    # Candidates the LLM omitted keep their original relative order, after the ranked ones.
    ranked = set(order)
    order += [i for i in range(n) if i not in ranked]
    return [(candidates[i], 1.0 - pos / n) for pos, i in enumerate(order[: min(top_n, n)])]
