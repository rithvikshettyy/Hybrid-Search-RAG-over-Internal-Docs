"""Retrieval evals: recall@5 and MRR for dense, BM25, hybrid, hybrid+rerank."""

import json
from pathlib import Path

from retrieve.pipeline import MODES, retrieve

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"  # {"question": str, "relevant_chunk_ids": [str]}
DEPTH = 10  # ranking depth fed to metrics; recall is cut at 5, MRR uses full depth


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant ids found in top-k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of first relevant hit, 0 if none."""
    for rank, cid in enumerate(retrieved, 1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def run_evals(dataset: list[dict] | None = None) -> dict[str, dict[str, float]]:
    """Return {system_name: {"recall@5": x, "mrr": y}}."""
    dataset = dataset if dataset is not None else load_dataset()
    if not dataset:
        raise ValueError("empty eval dataset; run `python -m evals.make_dataset` first")
    results: dict[str, dict[str, float]] = {}
    for mode in MODES:
        recalls, rrs = [], []
        for ex in dataset:
            relevant = set(ex["relevant_chunk_ids"])
            ids = [c.chunk_id for c in retrieve(ex["question"], mode=mode, top_n=DEPTH)]
            recalls.append(recall_at_k(ids, relevant, 5))
            rrs.append(reciprocal_rank(ids, relevant))
        results[mode] = {
            "recall@5": sum(recalls) / len(recalls),
            "mrr": sum(rrs) / len(rrs),
        }
    return results


if __name__ == "__main__":
    for name, m in run_evals().items():
        print(f"{name:15s} recall@5={m['recall@5']:.3f}  mrr={m['mrr']:.3f}")
