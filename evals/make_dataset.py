"""Build a synthetic eval set: the LLM writes one question per sampled chunk.

Usage: uv run python -m evals.make_dataset [--n 50] [--seed 0]
Caveat: questions generated from a chunk share its wording -> favors BM25; hand-curate for rigor.
"""

import argparse
import json
import random

from evals.run_evals import DATASET_PATH
from generate.llm import complete
from retrieve.pipeline import load_chunk_store

SYSTEM = (
    "Write ONE natural question that a GitLab employee might ask, answerable only from the passage. "
    "Do not quote the passage verbatim and do not mention 'the passage'. Output only the question."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks = [c for c in load_chunk_store().values() if len(c.text) >= 400]
    random.Random(args.seed).shuffle(chunks)
    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for c in chunks[: args.n]:
            q = complete(SYSTEM, f"{c.title} > {c.section}\n\n{c.text}", max_tokens=100).strip()
            f.write(json.dumps({"question": q, "relevant_chunk_ids": [c.chunk_id]}) + "\n")
    print(f"wrote {min(args.n, len(chunks))} examples to {DATASET_PATH}")


if __name__ == "__main__":
    main()
