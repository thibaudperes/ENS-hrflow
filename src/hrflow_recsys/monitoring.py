"""Scheduled drift check on the sessions the API received.

Run as a Cloud Run job: `python -m hrflow_recsys.monitoring`. It reads the `recommendation`
entries that the API wrote to Cloud Logging, profiles those sessions and compares them with
the training sessions saved by `scripts/train_final.py`.
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from hrflow_recsys.api import MODELS_DIR
from hrflow_recsys.drift import drift_table, session_profile

SERVICE = "hrflow-api"
DAYS = 7
MAX_ENTRIES = 5000
MIN_SESSIONS = 30
UNSEEN_LIMIT = 0.2  # share of jobs never seen in training; 0.067 on the challenge test set


def sessions_from_logs(payloads: list[dict]) -> pd.DataFrame:
    """Sessions from the JSON payloads of the API `recommendation` log entries."""
    rows = [
        {"job_ids": p["job_ids"], "actions": p["actions"], "source": p.get("source", "")}
        for p in payloads
        if "job_ids" in p and "recommended" in p
    ]
    return pd.DataFrame(rows, columns=["job_ids", "actions", "source"])


def check(reference: pd.DataFrame, known_jobs: set[int], sessions: pd.DataFrame) -> dict:
    """Drift report: KS test on the session features, a plain threshold on unseen jobs."""
    report = {"n_sessions": len(sessions)}
    if len(sessions) < MIN_SESSIONS:
        return report | {"status": "not_enough_sessions", "drifted": []}
    current = session_profile(sessions, known_jobs)
    features = [c for c in reference.columns if c != "unseen_share"]
    table = drift_table(reference[features], current[features])
    drifted = table.loc[table["drift"], "feature"].tolist()
    unseen = float(current["unseen_share"].mean())
    if unseen > UNSEEN_LIMIT:
        drifted.append("unseen_share")
    return report | {
        "status": "drift" if drifted else "ok",
        "drifted": drifted,
        "unseen_share": round(unseen, 4),
        "p_values": {
            row.feature: round(float(row.p_value), 6) for row in table.itertuples(index=False)
        },
    }


def load_reference(path: str | Path | None = None) -> tuple[pd.DataFrame, set[int]]:
    path = Path(path or os.environ.get("REFERENCE_PATH", MODELS_DIR / "reference.json"))
    data = json.loads(path.read_text())
    return pd.DataFrame(data["profile"]), set(data["known_jobs"])


def fetch_payloads(days: int = DAYS) -> list[dict]:
    from google.cloud import logging  # only needed on Google Cloud, not in the tests

    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    log_filter = (
        'resource.type="cloud_run_revision" '
        f'resource.labels.service_name="{SERVICE}" '
        "jsonPayload.recommended:* "
        f'timestamp>="{since}"'
    )
    entries = logging.Client().list_entries(
        filter_=log_filter, order_by=logging.DESCENDING, max_results=MAX_ENTRIES
    )
    return [entry.payload for entry in entries if isinstance(entry.payload, dict)]


def main() -> None:
    reference, known_jobs = load_reference()
    report = check(reference, known_jobs, sessions_from_logs(fetch_payloads()))
    severity = "WARNING" if report["status"] == "drift" else "INFO"
    print(json.dumps({"severity": severity, "message": "drift_check", **report}), flush=True)


if __name__ == "__main__":
    main()
