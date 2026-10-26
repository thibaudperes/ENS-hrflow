from collections.abc import Sequence

import numpy as np
import pandas as pd

from hrflow_recsys.metrics import K

MAX_GAP = 10


def _click_counts(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    clicks = pd.concat([x["job_ids"].explode(), y["job_id"]]).astype(int)
    return clicks.value_counts()


class Popularity:
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "Popularity":
        self.ranking = _click_counts(x, y).index.to_numpy()
        return self

    def recommend(self, job_ids: Sequence[int]) -> list[int]:
        seen = set(job_ids)
        # The K most popular plus len(seen) extra are always enough after removing seen jobs.
        top = self.ranking[: K + len(seen)]
        return [int(j) for j in top if j not in seen][:K]


class IdNeighbours:
    def __init__(self, max_gap: int = MAX_GAP):
        self.max_gap = max_gap

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "IdNeighbours":
        counts = _click_counts(x, y)
        self.popularity = Popularity().fit(x, y)
        self.ids = np.sort(counts.index.to_numpy())
        self.clicks = counts.reindex(self.ids).to_numpy()
        return self

    def neighbours(self, job_ids: Sequence[int]) -> list[int]:
        """Unseen train jobs within max_gap of the last job, closest first."""
        last = job_ids[-1]
        lo, hi = np.searchsorted(self.ids, [last - self.max_gap, last + self.max_gap + 1])
        ids, clicks = self.ids[lo:hi], self.clicks[lo:hi]
        # Ties broken by popularity.
        order = np.lexsort((-clicks, np.abs(ids - last)))
        seen = set(job_ids)
        return [int(j) for j in ids[order] if j not in seen]

    def recommend(self, job_ids: Sequence[int]) -> list[int]:
        recs = self.neighbours(job_ids)[:K]
        fill = self.popularity.recommend(list(job_ids) + recs)
        return recs + fill[: K - len(recs)]


def always_view(actions: Sequence[str]) -> str:
    return "view"


def last_action(actions: Sequence[str]) -> str:
    return actions[-1]
