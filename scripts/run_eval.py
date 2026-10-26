from hrflow_recsys.data import load_data
from hrflow_recsys.evaluate import evaluate_action_rule, evaluate_recommender
from hrflow_recsys.metrics import final_score
from hrflow_recsys.models.baselines import IdNeighbours, Popularity, always_view, last_action
from hrflow_recsys.split import train_val_split


def markdown_table(header: tuple[str, str], scores: dict[str, float]) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "|---|---:|"]
    lines += [f"| {name} | {score:.4f} |" for name, score in scores.items()]
    return "\n".join(lines)


def main() -> None:
    x, y, _, _ = load_data()
    x_train, x_val, y_train, y_val = train_val_split(x, y)

    recommenders = {"Popularity": Popularity(), "IdNeighbours": IdNeighbours()}
    mrr = {
        name: evaluate_recommender(model.fit(x_train, y_train), x_val, y_val)
        for name, model in recommenders.items()
    }
    rules = {"always_view": always_view, "last_action": last_action}
    acc = {name: evaluate_action_rule(rule, x_val, y_val) for name, rule in rules.items()}

    print(markdown_table(("recommender", "MRR@10"), mrr))
    print()
    print(markdown_table(("action rule", "accuracy"), acc))
    best_rec, best_rule = max(mrr, key=mrr.get), max(acc, key=acc.get)
    score = final_score(mrr[best_rec], acc[best_rule])
    print(f"\nBest final score: {score:.4f} ({best_rec} + {best_rule})")


if __name__ == "__main__":
    main()
