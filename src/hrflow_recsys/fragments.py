import numpy as np
import pandas as pd

from hrflow_recsys.metrics import K


def first_items(x: pd.DataFrame) -> pd.DataFrame:
    """First job and action of each session, indexed by session_id."""
    return pd.DataFrame(
        {"job_id": x["job_ids"].str[0].to_numpy(), "action": x["actions"].str[0].to_numpy()},
        index=x["session_id"].to_numpy(),
    )


def build_chains(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    """chain_id per session_id: s and s+1 share a chain when y[s] is the first job of s+1."""
    sessions = x.merge(y, on="session_id", validate="one_to_one").sort_values("session_id")
    ids = sessions["session_id"].to_numpy()
    first = sessions["job_ids"].str[0].to_numpy()
    linked = (ids[1:] == ids[:-1] + 1) & (first[1:] == sessions["job_id"].to_numpy()[:-1])
    # A new chain starts at every session not linked to the previous one.
    chain_id = np.concatenate([[0], np.cumsum(~linked)])
    return pd.Series(chain_id, index=ids, name="chain_id")


def next_first(x: pd.DataFrame) -> pd.DataFrame:
    """First job and action of session s+1, indexed by s, for every s whose s+1 is in x."""
    first = first_items(x)
    first.index = first.index - 1
    return first[first.index.isin(x["session_id"])]


def next_gap(x: pd.DataFrame) -> pd.Series:
    """|first job of s+1 - last job of s|, indexed by s, for every s whose s+1 is in x."""
    last = pd.Series(x["job_ids"].str[-1].to_numpy(), index=x["session_id"].to_numpy())
    following = next_first(x)["job_id"]
    return (following - last.reindex(following.index)).abs()


def with_next_first(recs: list[list[int]], x: pd.DataFrame, x_all: pd.DataFrame) -> list[list[int]]:
    """Put the first job of s+1, searched in x_all, at rank 1 when it exists and is unseen in s."""
    following = next_first(x_all)["job_id"].reindex(x["session_id"])
    result = []
    for rec, job_ids, job in zip(recs, x["job_ids"], following, strict=True):
        if pd.isna(job) or job in job_ids:
            result.append(rec)
        else:
            result.append([int(job)] + [r for r in rec if r != job][: K - 1])
    return result
