"""Compute the aggregates behind the README figures from data/raw and write results/*.json.

Only counts and rates are written, never a session or an offer.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hrflow_recsys.data import load_data
from hrflow_recsys.evaluate import evaluate_recommender
from hrflow_recsys.fragments import build_chains, first_items
from hrflow_recsys.models.hybrid import Hybrid
from hrflow_recsys.models.markov import Markov
from hrflow_recsys.split import grouped_split, train_val_split

RESULTS = Path("results")
SEED = 42
GAP_RANGE = 10
THRESHOLDS = [0, 1, 5, 10, 50, 100]


def write(name: str, content: dict) -> None:
    (RESULTS / f"{name}.json").write_text(json.dumps(content, indent=2) + "\n")


def fragments(x_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    train = x_train.merge(y_train, on="session_id")
    first = first_items(x_train)
    rates = {}
    for offset in [-1, 1, 2]:
        shifted = first.reindex(train["session_id"] + offset)
        rates[f"s{offset:+d}"] = float((shifted["job_id"] == train["job_id"].to_numpy()).mean())
    chain_len = build_chains(x_train, y_train).value_counts()
    return {
        "n_sessions": len(train),
        "target_is_first_job_of": rates,
        "n_chains": len(chain_len),
        "median_chain_length": float(chain_len.median()),
        "max_chain_length": int(chain_len.max()),
    }


def split_comparison(x_train: pd.DataFrame, y_train: pd.DataFrame, jobs: pd.Series) -> dict:
    chains = build_chains(x_train, y_train)
    splits = {
        "random": train_val_split(x_train, y_train),
        "grouped by chain": grouped_split(x_train, y_train, chains),
    }
    result = {}
    for split, (x_fit, x_val, y_fit, y_val) in splits.items():
        result[split] = {
            "Markov": evaluate_recommender(Markov().fit(x_fit, y_fit), x_val, y_val),
            "Hybrid": evaluate_recommender(Hybrid().fit(x_fit, y_fit, jobs), x_val, y_val),
        }
    return result


def id_gaps(x_train: pd.DataFrame, y_train: pd.DataFrame, jobs: pd.Series) -> dict:
    # Same analysis as notebook 01, section 6, with the gap kept signed.
    last_job = x_train["job_ids"].map(lambda ids: ids[-1])
    signed = y_train["job_id"] - last_job
    gap = signed.abs()
    rng = np.random.default_rng(SEED)
    random_gap = np.abs(rng.choice(jobs.index.to_numpy(), size=len(last_job)) - last_job)
    counts = signed[gap <= GAP_RANGE].value_counts().reindex(range(-GAP_RANGE, GAP_RANGE + 1))
    return {
        "n_sessions": len(signed),
        "signed_gap_counts": {str(d): int(n) for d, n in counts.fillna(0).items()},
        "share_within": {
            str(d): {"target": float((gap <= d).mean()), "random": float((random_gap <= d).mean())}
            for d in THRESHOLDS
        },
    }


def main() -> None:
    x_train, y_train, _, jobs = load_data()
    RESULTS.mkdir(exist_ok=True)
    write("fragments", fragments(x_train, y_train))
    write("id_gaps", id_gaps(x_train, y_train, jobs))
    write("split_comparison", split_comparison(x_train, y_train, jobs))


if __name__ == "__main__":
    main()
