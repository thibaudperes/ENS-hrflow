import numpy as np
import pandas as pd

from hrflow_recsys.drift import session_profile
from hrflow_recsys.monitoring import MIN_SESSIONS, check, sessions_from_logs


def payload(job_ids, actions, source="api"):
    recommended = [j + 100 for j in range(10)]
    return {
        "message": "recommendation",
        "source": source,
        "job_ids": job_ids,
        "actions": actions,
        "recommended": recommended,
        "action": actions[-1],
    }


def random_sessions(rng, n, apply_rate):
    rows = []
    for _ in range(n):
        length = int(rng.integers(2, 10))
        actions = np.where(rng.random(length) < apply_rate, "apply", "view").tolist()
        rows.append({"job_ids": rng.integers(1, 500, length).tolist(), "actions": actions})
    return pd.DataFrame(rows)


def test_sessions_from_logs_keeps_recommendations_only():
    payloads = [
        payload([1, 2], ["view", "apply"], source="demo"),
        {"message": "rejected", "source": "api", "n_jobs": 2, "n_unknown": 1},
    ]
    sessions = sessions_from_logs(payloads)
    assert sessions.to_dict(orient="records") == [
        {"job_ids": [1, 2], "actions": ["view", "apply"], "source": "demo"}
    ]


def test_check_waits_for_enough_sessions():
    rng = np.random.default_rng(0)
    known = set(range(1, 500))
    reference = session_profile(random_sessions(rng, 500, 0.3), known)
    report = check(reference, known, random_sessions(rng, MIN_SESSIONS - 1, 0.3))
    assert report["status"] == "not_enough_sessions"


def test_check_flags_a_shift_in_actions_and_not_noise():
    rng = np.random.default_rng(0)
    known = set(range(1, 500))
    reference = session_profile(random_sessions(rng, 2000, 0.3), known)
    same = check(reference, known, random_sessions(rng, 300, 0.3))
    shifted = check(reference, known, random_sessions(rng, 300, 0.8))
    assert same["status"] == "ok"
    assert shifted["status"] == "drift"
    assert "apply_share" in shifted["drifted"]


def test_check_flags_unseen_jobs():
    rng = np.random.default_rng(0)
    known = set(range(1, 250))
    reference = session_profile(random_sessions(rng, 2000, 0.3), set(range(1, 500)))
    report = check(reference, known, random_sessions(rng, 300, 0.3))
    assert "unseen_share" in report["drifted"]
