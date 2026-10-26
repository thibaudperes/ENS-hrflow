from collections.abc import Sequence

import numpy as np

K = 10


def mrr_at_k(recs: Sequence[Sequence[int]], targets: Sequence[int], k: int = K) -> float:
    ranks = []
    for rec, target in zip(recs, targets, strict=True):
        top = list(rec[:k])
        ranks.append(1 / (top.index(target) + 1) if target in top else 0.0)
    return float(np.mean(ranks))


def action_accuracy(pred: Sequence[str], true: Sequence[str]) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(true)))


def final_score(mrr: float, acc: float) -> float:
    return 0.7 * mrr + 0.3 * acc
