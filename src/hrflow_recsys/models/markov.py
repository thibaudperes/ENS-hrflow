from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

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
        self.out_counts = Counter()
        for job_ids, target in zip(sessions["job_ids"], sessions["job_id"], strict=True):
            self._add(list(job_ids) + [target])
        self.neighbours = IdNeighbours().fit(x, y)
        return self

    def update(self, sequences: Iterable[Sequence[int]]) -> "Markov":
        """Add the transitions inside new sessions, without their targets."""
        for job_ids in sequences:
            self._add(job_ids)
        return self

    def _add(self, path: Sequence[int]) -> None:
        for a, b in zip(path[:-1], path[1:], strict=True):
            self.transitions[a][b] += 1
            self.out_counts[a] += 1

    def votes(self, job_ids: Sequence[int]) -> Counter:
        seen = set(job_ids)
        votes = Counter()
        for weight, job in zip(self.weights, reversed(job_ids), strict=False):
            nexts = self.transitions.get(job, {})
            norm = self.out_counts[job] if self.normalize and nexts else 1
            for b, count in nexts.items():
                if b not in seen:
                    votes[b] += weight * count / norm
        return votes

    def recommend(self, job_ids: Sequence[int]) -> list[int]:
        recs = [int(b) for b, _ in self.votes(job_ids).most_common(K)]
        # Markov picks go first so they count as seen; the last job stays last for IdNeighbours.
        fill = self.neighbours.recommend(recs + list(job_ids))
        return recs + fill[: K - len(recs)]
