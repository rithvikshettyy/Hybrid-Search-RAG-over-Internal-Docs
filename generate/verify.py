"""Per-claim verification of an answer against its cited chunks."""

import json
import re

from pydantic import BaseModel

from generate.answer import Answer
from generate.llm import complete

_CITE = re.compile(r"\[(\d+)\]")
# Sentence end: . ! ? optionally followed by citation markers, then whitespace.
_SENT_END = re.compile(r"(?<=[.!?])((?:\s*\[\d+\])*)\s+(?=[A-Z0-9\"'(\[])")

SYSTEM = """You are a strict fact checker. Given one CLAIM and the SOURCE text(s) it cites, decide
whether the sources directly support the claim. Partial or implied support is NOT supported.
Reply with ONLY a JSON object: {"supported": true|false, "reason": "<one short sentence>"}"""


class ClaimVerdict(BaseModel):
    claim: str
    citation_ids: list[int]
    supported: bool
    reason: str


def split_claims(answer_text: str) -> list[str]:
    """Split answer into atomic claims (one per sentence), each keeping its [n] markers."""
    text = answer_text.strip()
    if not text:
        return []
    # Re-attach trailing citation markers to the sentence they belong to.
    parts = _SENT_END.sub(lambda m: m.group(1) + "\x00", text).split("\x00")
    return [p.strip() for p in parts if p.strip()]


def _parse_verdict(raw: str) -> tuple[bool, str]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else {}
        return bool(obj["supported"]), str(obj.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        return False, f"unparseable verifier output: {raw[:80]!r}"


def verify_answer(answer: Answer) -> list[ClaimVerdict]:
    """Check each claim against the chunk(s) it cites."""
    verdicts: list[ClaimVerdict] = []
    for claim in split_claims(answer.text):
        ids = sorted({int(n) for n in _CITE.findall(claim)})
        if not ids:
            verdicts.append(
                ClaimVerdict(claim=claim, citation_ids=[], supported=False, reason="no citation")
            )
            continue
        missing = [n for n in ids if n not in answer.citations]
        if missing:
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    citation_ids=ids,
                    supported=False,
                    reason=f"cites nonexistent source(s) {missing}",
                )
            )
            continue
        sources = "\n\n".join(f"SOURCE [{n}]:\n{answer.citations[n].text}" for n in ids)
        raw = complete(SYSTEM, f"CLAIM: {_CITE.sub('', claim).strip()}\n\n{sources}", max_tokens=200)
        supported, reason = _parse_verdict(raw)
        verdicts.append(
            ClaimVerdict(claim=claim, citation_ids=ids, supported=supported, reason=reason)
        )
    return verdicts
