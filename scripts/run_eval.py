import time

import mlflow
import pandas as pd

from hrflow_recsys.action import ActionModel
from hrflow_recsys.data import load_data
from hrflow_recsys.evaluate import (
    evaluate_action_model,
    evaluate_action_rule,
    evaluate_recommender,
    markdown_table,
)
from hrflow_recsys.fragments import build_chains
from hrflow_recsys.metrics import final_score
from hrflow_recsys.models.baselines import IdNeighbours, Popularity, always_view, last_action
from hrflow_recsys.models.hybrid import Hybrid
from hrflow_recsys.models.markov import Markov
from hrflow_recsys.models.text import TextModel
from hrflow_recsys.split import grouped_split

EXPERIMENT = "hrflow-recsys"


def online(model: Hybrid, x_new: pd.DataFrame) -> Hybrid:
    model.markov.update(x_new["job_ids"])
    return model


def main() -> None:
    x, y, _, jobs = load_data()
    chains = build_chains(x, y)
    x_train, x_val, y_train, y_val = grouped_split(x, y, chains)
    seen = set(x_train["job_ids"].explode()) | set(y_train["job_id"])
    y_cold = y_val[~y_val["job_id"].isin(seen)]
    x_cold = x_val[x_val["session_id"].isin(y_cold["session_id"])]

    recommenders = {
        "Popularity": lambda: Popularity().fit(x_train, y_train),
        "IdNeighbours": lambda: IdNeighbours().fit(x_train, y_train),
        "Markov": lambda: Markov().fit(x_train, y_train),
        "TextModel": lambda: TextModel().fit(x_train, y_train, jobs),
        "Hybrid": lambda: Hybrid().fit(x_train, y_train, jobs),
        "Hybrid + online": lambda: online(Hybrid().fit(x_train, y_train, jobs), x_val),
    }
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="run_eval"):
        mlflow.log_params(
            {"split": "grouped_by_chain", "n_train": len(x_train), "n_val": len(x_val)}
        )
        mrr = {}
        for name, fit in recommenders.items():
            with mlflow.start_run(run_name=name, nested=True):
                start = time.perf_counter()
                model = fit()
                val = evaluate_recommender(model, x_val, y_val)
                cold = evaluate_recommender(model, x_cold, y_cold)
                seconds = time.perf_counter() - start
                mlflow.log_param("recommender", name)
                if hasattr(model, "weights"):
                    mlflow.log_param("weights", list(model.weights))
                mlflow.log_metrics({"mrr_at_10": val, "mrr_at_10_cold": cold, "seconds": seconds})
            mrr[name] = (val, cold, f"{seconds:.1f}")

        rules = {"always_view": always_view, "last_action": last_action}
        acc = {name: (evaluate_action_rule(rule, x_val, y_val),) for name, rule in rules.items()}
        action_model = ActionModel().fit(x_train, y_train)
        acc["ActionModel"] = (evaluate_action_model(action_model, x_val, y_val),)
        for name, (accuracy,) in acc.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("action_rule", name)
                mlflow.log_metric("accuracy", accuracy)

        best_rec = max(mrr, key=lambda name: mrr[name][0])
        best_rule = max(acc, key=lambda name: acc[name][0])
        score = final_score(mrr[best_rec][0], acc[best_rule][0])
        mlflow.log_params({"best_recommender": best_rec, "best_action_rule": best_rule})
        mlflow.log_metric("final_score", score)

    cold_header = f"MRR@10 cold start (n={len(y_cold)})"
    print(markdown_table(("recommender", "MRR@10", cold_header, "seconds"), mrr))
    print()
    print(markdown_table(("action rule", "accuracy"), acc))
    print(f"\nBest final score: {score:.4f} ({best_rec} + {best_rule})")


if __name__ == "__main__":
    main()
