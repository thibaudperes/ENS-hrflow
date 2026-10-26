from collections.abc import Sequence

import numpy as np
import pandas as pd

from hrflow_recsys.metrics import K
from hrflow_recsys.models.baselines import IdNeighbours
from hrflow_recsys.models.markov import Markov
from hrflow_recsys.models.text import TextModel

N_CANDIDATES = 50
CANDIDATE_GAP = 50
# Tuned on half A of the val split with scripts/tune_hybrid.py.
W_MARKOV, W_ID, W_TEXT = 1.0, 0.5, 0.5


def rank(candidates: np.ndarray, features: np.ndarray, weights: Sequence[float]) -> list[int]:
    scores = features @ np.asarray(weights)
    return candidates[np.argsort(-scores, kind="stable")[:K]].tolist()


class Hybrid:
    def __init__(self, weights: Sequence[float] = (W_MARKOV, W_ID, W_TEXT)):
        self.weights = weights

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, jobs: pd.Series) -> "Hybrid":
        self.markov = Markov().fit(x, y)
        self.neighbours = IdNeighbours(CANDIDATE_GAP).fit(x, y)
        self.text = TextModel().fit(x, y, jobs)
        return self

    def candidate_scores(
        self, sessions: Sequence[Sequence[int]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per session: unseen candidates and their (markov, id, text) scores, max-normalised."""
        text_scores = self.text.scores(sessions)
        text_top = np.argpartition(-text_scores, N_CANDIDATES, axis=1)[:, :N_CANDIDATES]
        job_index = self.text.job_index.to_numpy()
        result = []
        for job_ids, row, top in zip(sessions, text_scores, text_top, strict=True):
            votes = self.markov.votes(job_ids)
            pool = {int(b) for b, _ in votes.most_common(N_CANDIDATES)}
            pool |= set(self.neighbours.neighbours(job_ids))
            pool |= {int(j) for j in job_index[top]}
            candidates = np.array(sorted(pool))
            features = np.column_stack(
                [
                    [votes.get(c, 0.0) for c in candidates],
                    1 / (1 + np.abs(candidates - job_ids[-1])),
                    row[self.text.job_index.get_indexer(candidates)],
                ]
            )
            top_score = features.max(axis=0)
            features = np.divide(
                features, top_score, out=np.zeros_like(features), where=top_score > 0
            )
            result.append((candidates, features))
        return result

    def recommend_batch(self, sessions: Sequence[Sequence[int]]) -> list[list[int]]:
        return [rank(c, f, self.weights) for c, f in self.candidate_scores(sessions)]
