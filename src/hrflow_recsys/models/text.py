from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from hrflow_recsys.metrics import K


class TextModel:
    def fit(self, x: pd.DataFrame, y: pd.DataFrame, jobs: pd.Series) -> "TextModel":
        # Unsupervised: only the job texts are used, x and y keep the common fit signature.
        vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=2,
            max_df=0.5,
            max_features=50_000,
            dtype=np.float32,
        )
        self.vectors = vectorizer.fit_transform(jobs.to_numpy())
        self.job_index = jobs.index
        return self

    def recommend(self, job_ids: Sequence[int]) -> list[int]:
        return self.recommend_batch([job_ids])[0]

    def scores(self, sessions: Sequence[Sequence[int]]) -> np.ndarray:
        """Similarity of each session to every job (sessions x jobs), -inf on seen jobs."""
        rows, cols, weights = [], [], []
        for i, job_ids in enumerate(sessions):
            positions = np.arange(1, len(job_ids) + 1)
            rows += [i] * len(job_ids)
            cols += list(self.job_index.get_indexer(job_ids))
            weights += list(positions / positions.sum())
        session_weights = sparse.csr_matrix(
            (weights, (rows, cols)), shape=(len(sessions), len(self.job_index)), dtype=np.float32
        )
        profiles = session_weights @ self.vectors
        # Job vectors are L2-normalised, so the dot product ranks jobs like the cosine.
        scores = (profiles @ self.vectors.T).toarray()
        scores[session_weights.nonzero()] = -np.inf
        return scores

    def recommend_batch(self, sessions: Sequence[Sequence[int]]) -> list[list[int]]:
        scores = self.scores(sessions)
        top = np.argpartition(-scores, K, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)
        return self.job_index.to_numpy()[top].tolist()
