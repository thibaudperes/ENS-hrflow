import time

from hrflow_recsys.data import load_data
from hrflow_recsys.evaluate import evaluate_action_rule, evaluate_recommender
from hrflow_recsys.metrics import final_score
from hrflow_recsys.models.baselines import IdNeighbours, Popularity, always_view, last_action
from hrflow_recsys.models.markov import Markov
from hrflow_recsys.models.text import TextModel
from hrflow_recsys.split import train_val_split


def markdown_table(header: tuple[str, ...], rows: dict[str, tuple]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|---" + "|---:" * (len(header) - 1) + "|"]
    for name, values in rows.items():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in values]
        lines.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(lines)


def main() -> None:
    x, y, _, jobs = load_data()
    x_train, x_val, y_train, y_val = train_val_split(x, y)
    seen = set(x_train["job_ids"].explode()) | set(y_train["job_id"])
    y_cold = y_val[~y_val["job_id"].isin(seen)]
    x_cold = x_val[x_val["session_id"].isin(y_cold["session_id"])]

    recommenders = {
        "Popularity": lambda: Popularity().fit(x_train, y_train),
        "IdNeighbours": lambda: IdNeighbours().fit(x_train, y_train),
        "Markov": lambda: Markov().fit(x_train, y_train),
        "TextModel": lambda: TextModel().fit(x_train, y_train, jobs),
    }
    mrr = {}
    for name, fit in recommenders.items():
        start = time.perf_counter()
        model = fit()
        mrr[name] = (
            evaluate_recommender(model, x_val, y_val),
            evaluate_recommender(model, x_cold, y_cold),
            f"{time.perf_counter() - start:.1f}",
        )
    rules = {"always_view": always_view, "last_action": last_action}
    acc = {name: (evaluate_action_rule(rule, x_val, y_val),) for name, rule in rules.items()}

    cold_header = f"MRR@10 cold start (n={len(y_cold)})"
    print(markdown_table(("recommender", "MRR@10", cold_header, "seconds"), mrr))
    print()
    print(markdown_table(("action rule", "accuracy"), acc))
    best_rec = max(mrr, key=lambda name: mrr[name][0])
    best_rule = max(acc, key=lambda name: acc[name][0])
    score = final_score(mrr[best_rec][0], acc[best_rule][0])
    print(f"\nBest final score: {score:.4f} ({best_rec} + {best_rule})")


if __name__ == "__main__":
    main()
