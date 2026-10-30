from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from hrflow_recsys.fragments import build_chains
from hrflow_recsys.metrics import K
from hrflow_recsys.models.baselines import _click_counts
from hrflow_recsys.models.hybrid import N_CANDIDATES, Hybrid
from hrflow_recsys.split import grouped_split

FEATURES = [
    "markov_norm",
    "id_norm",
    "text_norm",
    "markov_log",
    "gap_log",
    "gap_sign",
    "text_raw",
    "popularity_log",
    "session_length",
]
RANK_FRAC = 0.3


class Ranker:
    """Learns how to combine the Hybrid signals instead of a fixed weighted sum.

    The training sessions are split by chain: the Hybrid is fitted on one part and builds
    candidates for the other, so the classifier never sees a candidate scored by a model
    that already counted the target transition.
    """

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, jobs: pd.Series) -> "Ranker":
        chains = build_chains(x, y)
        x_fit, x_rank, y_fit, y_rank = grouped_split(x, y, chains, val_frac=RANK_FRAC)
        self._set_base(x_fit, y_fit, jobs)
        sessions = x_rank.merge(y_rank, on="session_id", validate="one_to_one")
        features, labels = [], []
        for (candidates, feats), target in zip(
            self.candidate_features(sessions["job_ids"].tolist()), sessions["job_id"], strict=True
        ):
            hit = candidates == target
            if hit.any():  # sessions whose target is not a candidate teach nothing about order
                features.append(feats)
                labels.append(hit)
        self.classifier = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=0
        ).fit(np.vstack(features), np.concatenate(labels))
        # The final candidates come from a Hybrid fitted on all the training sessions.
        self._set_base(x, y, jobs)
        return self

    def _set_base(self, x: pd.DataFrame, y: pd.DataFrame, jobs: pd.Series) -> None:
        self.hybrid = Hybrid().fit(x, y, jobs)
        self.markov = self.hybrid.markov
        self.popularity = _click_counts(x, y)

    def candidate_features(
        self, sessions: Sequence[Sequence[int]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per session: Hybrid candidates and one row of FEATURES per candidate."""
        hybrid = self.hybrid
        text_scores = hybrid.text.scores(sessions)
        text_top = np.argpartition(-text_scores, N_CANDIDATES, axis=1)[:, :N_CANDIDATES]
        job_index = hybrid.text.job_index.to_numpy()
        result = []
        for job_ids, row, top in zip(sessions, text_scores, text_top, strict=True):
            votes = hybrid.markov.votes(job_ids)
            pool = {int(b) for b, _ in votes.most_common(N_CANDIDATES)}
            pool |= set(hybrid.neighbours.neighbours(job_ids))
            pool |= {int(j) for j in job_index[top]}
            candidates = np.array(sorted(pool))
            markov = np.array([votes.get(c, 0.0) for c in candidates])
            gap = candidates - job_ids[-1]
            proximity = 1 / (1 + np.abs(gap))
            text = row[hybrid.text.job_index.get_indexer(candidates)]
            text = np.where(np.isfinite(text), text, 0.0)
            popularity = self.popularity.reindex(candidates, fill_value=0).to_numpy()
            feats = np.column_stack(
                [
                    _norm(markov),
                    _norm(proximity),
                    _norm(text),
                    np.log1p(markov),
                    np.log1p(np.abs(gap)),
                    np.sign(gap),
                    text,
                    np.log1p(popularity),
                    np.full(len(candidates), len(job_ids)),
                ]
            )
            result.append((candidates, feats))
        return result

    def recommend_batch(self, sessions: Sequence[Sequence[int]]) -> list[list[int]]:
        recs = []
        for candidates, feats in self.candidate_features(sessions):
            scores = self.classifier.predict_proba(feats)[:, 1]
            recs.append(candidates[np.argsort(-scores, kind="stable")[:K]].tolist())
        return recs


def _norm(values: np.ndarray) -> np.ndarray:
    top = values.max()
    return values / top if top > 0 else np.zeros_like(values, dtype=float)
