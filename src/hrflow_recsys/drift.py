import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from hrflow_recsys.action import action_features

ALPHA = 0.01


def session_profile(x: pd.DataFrame, known_jobs: set[int]) -> pd.DataFrame:
    """One row per session: action features plus where its jobs sit in the catalogue."""
    profile = action_features(x)
    profile["median_job_id"] = x["job_ids"].map(np.median).to_numpy()
    profile["unseen_share"] = (
        x["job_ids"].map(lambda jobs: np.mean([j not in known_jobs for j in jobs])).to_numpy()
    )
    return profile


def drift_table(
    reference: pd.DataFrame, current: pd.DataFrame, alpha: float = ALPHA
) -> pd.DataFrame:
    """Kolmogorov-Smirnov test per column: does current differ from reference?"""
    rows = []
    for col in reference.columns:
        result = ks_2samp(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "reference_mean": reference[col].mean(),
                "current_mean": current[col].mean(),
                "ks_statistic": result.statistic,
                "p_value": result.pvalue,
                "drift": result.pvalue < alpha,
            }
        )
    return pd.DataFrame(rows)
