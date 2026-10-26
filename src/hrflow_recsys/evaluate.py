from collections.abc import Callable, Sequence

import pandas as pd

from hrflow_recsys.metrics import action_accuracy, mrr_at_k


def evaluate_recommender(model, x: pd.DataFrame, y: pd.DataFrame) -> float:
    recs = [model.recommend(job_ids) for job_ids in x["job_ids"]]
    return mrr_at_k(recs, y["job_id"].tolist())


def evaluate_action_rule(
    rule: Callable[[Sequence[str]], str], x: pd.DataFrame, y: pd.DataFrame
) -> float:
    pred = [rule(actions) for actions in x["actions"]]
    return action_accuracy(pred, y["action"].tolist())
