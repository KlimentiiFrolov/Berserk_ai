from typing import List, Set


def recall_at_k(retrieved_idxs: List[str], gold_set: Set[str], k: int = 5):
    if not gold_set:
        return 0.0
    top_k = retrieved_idxs[:k]
    hits = len(set(top_k) & gold_set)
    return hits / len(gold_set)

def precision_at_k(retrieved_idxs: List[str], gold_set: Set[str], k: int = 5):
    top_k = retrieved_idxs[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & gold_set)
    return hits / len(top_k)

def mrr(retrieved_idxs: List[str], gold_set: Set[str]):
    for rank, idx in enumerate(retrieved_idxs, start=1):
        if idx in gold_set:
            return 1.0 / rank
    return 0.0
