import numpy as np
import pandas as pd


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
