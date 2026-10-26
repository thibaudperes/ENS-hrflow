import time

import pandas as pd

from hrflow_recsys.data import load_data
from hrflow_recsys.evaluate import BATCH_SIZE, markdown_table
from hrflow_recsys.metrics import mrr_at_k
from hrflow_recsys.models.hybrid import Hybrid, rank
from hrflow_recsys.split import train_val_split

# Only ratios matter, so the Markov weight is fixed to 1.
ID_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
TEXT_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def candidate_scores(model: Hybrid, x: pd.DataFrame, y: pd.DataFrame) -> tuple[list, list]:
    sessions = x.merge(y, on="session_id", validate="one_to_one")
    job_ids = sessions["job_ids"].tolist()
    scores = []
    for start in range(0, len(job_ids), BATCH_SIZE):
        scores += model.candidate_scores(job_ids[start : start + BATCH_SIZE])
    return scores, sessions["job_id"].tolist()


def mrr(scores: list, targets: list, weights: tuple) -> float:
    return mrr_at_k([rank(c, f, weights) for c, f in scores], targets)


def main() -> None:
    start = time.perf_counter()
    x, y, _, jobs = load_data()
    x_train, x_val, y_train, y_val = train_val_split(x, y)
    x_a, x_b, y_a, y_b = train_val_split(x_val, y_val, val_frac=0.5)
    model = Hybrid().fit(x_train, y_train, jobs)
    half_a, half_b = candidate_scores(model, x_a, y_a), candidate_scores(model, x_b, y_b)

    grid = [(1.0, w_id, w_text) for w_id in ID_WEIGHTS for w_text in TEXT_WEIGHTS]
    best = sorted(grid, key=lambda w: mrr(*half_a, w), reverse=True)[:5]
    rows = {str(w): (mrr(*half_a, w), mrr(*half_b, w)) for w in best}
    header = ("(markov, id, text)", f"MRR@10 A (n={len(y_a)})", f"MRR@10 B (n={len(y_b)})")
    print(f"{len(grid)} combinations tested on A\n")
    print(markdown_table(header, rows))
    print(f"\n{time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
