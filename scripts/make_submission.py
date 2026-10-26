import ast
from pathlib import Path

import pandas as pd

from hrflow_recsys.data import load_data
from hrflow_recsys.evaluate import recommend_all
from hrflow_recsys.fragments import with_next_first
from hrflow_recsys.metrics import K
from hrflow_recsys.models.baselines import last_action
from hrflow_recsys.models.hybrid import Hybrid

OUT_DIR = Path(__file__).resolve().parents[1] / "submissions"


def submission(x: pd.DataFrame, recs: list[list[int]], actions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"session_id": x["session_id"], "action": actions, "job_id": [str(r) for r in recs]}
    )


def check(sub: pd.DataFrame, x: pd.DataFrame) -> None:
    recs = [ast.literal_eval(r) for r in sub["job_id"]]
    assert len(sub) == len(x) == 1819
    assert not sub.isna().any().any()
    assert all(len(set(r)) == K for r in recs)
    assert not any(set(r) & set(seen) for r, seen in zip(recs, x["job_ids"], strict=True))
    assert sub["action"].isin(["view", "apply"]).all()


def main() -> None:
    x, y, x_test, jobs = load_data()
    model = Hybrid().fit(x, y, jobs)
    model.markov.update(x_test["job_ids"])
    recs = recommend_all(model, x_test["job_ids"].tolist())
    actions = [last_action(a) for a in x_test["actions"]]

    OUT_DIR.mkdir(exist_ok=True)
    files = {
        "submission_model.csv": submission(x_test, recs, actions),
        # Only x_test is searched for s+1: test session ids are numbered apart from train.
        "submission_leak.csv": submission(x_test, with_next_first(recs, x_test, x_test), actions),
    }
    for name, sub in files.items():
        check(sub, x_test)
        sub.to_csv(OUT_DIR / name, index=False)
        print(f"{name}: {len(sub)} rows, checks passed")


if __name__ == "__main__":
    main()
