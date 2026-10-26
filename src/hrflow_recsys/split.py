import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def train_val_split(
    x: pd.DataFrame, y: pd.DataFrame, val_frac: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Random split by session; notebook 02 compares it with grouped_split on session chains.
    rng = np.random.default_rng(seed)
    session_ids = x["session_id"].to_numpy()
    val_ids = rng.choice(session_ids, size=int(len(session_ids) * val_frac), replace=False)
    x_val_mask = x["session_id"].isin(val_ids)
    y_val_mask = y["session_id"].isin(val_ids)
    return x[~x_val_mask], x[x_val_mask], y[~y_val_mask], y[y_val_mask]


def grouped_split(
    x: pd.DataFrame, y: pd.DataFrame, groups: pd.Series, val_frac: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split where all sessions of a group (indexed by session_id) land on the same side."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    _, val_rows = next(splitter.split(x, groups=groups.loc[x["session_id"]].to_numpy()))
    val_ids = x["session_id"].iloc[val_rows]
    x_val_mask = x["session_id"].isin(val_ids)
    y_val_mask = y["session_id"].isin(val_ids)
    return x[~x_val_mask], x[x_val_mask], y[~y_val_mask], y[y_val_mask]
