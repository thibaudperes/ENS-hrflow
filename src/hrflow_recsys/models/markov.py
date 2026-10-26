from collections import Counter, defaultdict
from collections.abc import Sequence

import pandas as pd

from hrflow_recsys.metrics import K
from hrflow_recsys.models.baselines import IdNeighbours

WEIGHTS = (1.0, 0.5, 0.25)


class Markov:
    def __init__(self, weights: Sequence[float] = WEIGHTS, normalize: bool = False):
        self.weights = weights
        self.normalize = normalize

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "Markov":
        sessions = x.merge(y, on="session_id", validate="one_to_one")
        self.transitions = defaultdict(Counter)
        for job_ids, target in zip(sessions["job_ids"], sessions["job_id"], strict=True):
            path = list(job_ids) + [target]
            for a, b in zip(path[:-1], path[1:], strict=True):
                self.transitions[a][b] += 1
        self.out_counts = {a: sum(nexts.values()) for a, nexts in self.transitions.items()}
        self.neighbours = IdNeighbours().fit(x, y)
        return self

    def recommend(self, job_ids: Sequence[int]) -> list[int]:
        seen = set(job_ids)
        votes = Counter()
        for weight, job in zip(self.weights, reversed(job_ids), strict=False):
            nexts = self.transitions.get(job, {})
            norm = self.out_counts[job] if self.normalize and nexts else 1
            for b, count in nexts.items():
                if b not in seen:
                    votes[b] += weight * count / norm
        recs = [int(b) for b, _ in votes.most_common(K)]
        # Markov picks go first so they count as seen; the last job stays last for IdNeighbours.
        fill = self.neighbours.recommend(recs + list(job_ids))
        return recs + fill[: K - len(recs)]
