from collections.abc import Callable, Sequence

import pandas as pd

from hrflow_recsys.metrics import action_accuracy, mrr_at_k

BATCH_SIZE = 1000


def recommend_all(model, sessions: list) -> list[list[int]]:
    if not hasattr(model, "recommend_batch"):
        return [model.recommend(job_ids) for job_ids in sessions]
    recs = []
    for start in range(0, len(sessions), BATCH_SIZE):
        recs += model.recommend_batch(sessions[start : start + BATCH_SIZE])
    return recs


def evaluate_recommender(model, x: pd.DataFrame, y: pd.DataFrame) -> float:
    sessions = x.merge(y, on="session_id", validate="one_to_one")
    recs = recommend_all(model, sessions["job_ids"].tolist())
    return mrr_at_k(recs, sessions["job_id"].tolist())


def evaluate_action_rule(
    rule: Callable[[Sequence[str]], str], x: pd.DataFrame, y: pd.DataFrame
) -> float:
    sessions = x.merge(y, on="session_id", validate="one_to_one")
    pred = [rule(actions) for actions in sessions["actions"]]
    return action_accuracy(pred, sessions["action"].tolist())


def evaluate_action_model(model, x: pd.DataFrame, y: pd.DataFrame) -> float:
    sessions = x.merge(y, on="session_id", validate="one_to_one")
    return action_accuracy(model.predict(sessions), sessions["action"].tolist())


def markdown_table(header: tuple[str, ...], rows: dict[str, tuple]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|---" + "|---:" * (len(header) - 1) + "|"]
    for name, values in rows.items():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]
        lines.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(lines)
