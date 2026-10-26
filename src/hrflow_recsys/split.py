import numpy as np
import pandas as pd

SEED = 42


def train_val_split(
    x: pd.DataFrame, y: pd.DataFrame, val_frac: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Random split by session; notebook 02 compares it with a split grouped by candidate.
    rng = np.random.default_rng(seed)
    session_ids = x["session_id"].to_numpy()
    val_ids = rng.choice(session_ids, size=int(len(session_ids) * val_frac), replace=False)
    x_val_mask = x["session_id"].isin(val_ids)
    y_val_mask = y["session_id"].isin(val_ids)
    return x[~x_val_mask], x[x_val_mask], y[~y_val_mask], y[y_val_mask]
