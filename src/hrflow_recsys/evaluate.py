from collections.abc import Callable, Sequence

import pandas as pd

from hrflow_recsys.metrics import action_accuracy, mrr_at_k


def evaluate_recommender(model, x: pd.DataFrame, y: pd.DataFrame) -> float:
    sessions = x.merge(y, on="session_id", validate="one_to_one")
    recs = [model.recommend(job_ids) for job_ids in sessions["job_ids"]]
    return mrr_at_k(recs, sessions["job_id"].tolist())


def evaluate_action_rule(
    rule: Callable[[Sequence[str]], str], x: pd.DataFrame, y: pd.DataFrame
) -> float:
    sessions = x.merge(y, on="session_id", validate="one_to_one")
    pred = [rule(actions) for actions in sessions["actions"]]
    return action_accuracy(pred, sessions["action"].tolist())
