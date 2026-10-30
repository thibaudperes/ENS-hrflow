import json
import re
from pathlib import Path

import joblib

from hrflow_recsys.data import load_data
from hrflow_recsys.drift import session_profile
from hrflow_recsys.models.hybrid import Hybrid

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
N_EXAMPLES = 50
SNIPPET_CHARS = 140


def shorten(text: str, n: int = SNIPPET_CHARS) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def snippet(text: str) -> str:
    """Offer title and the start of its summary, separated by a newline."""
    text = re.sub(r"\s+", " ", text).strip()
    match = re.match(r"TITLE (.*?) SUMMARY (.*)", text)
    title, summary = (match.group(1), match.group(2)) if match else (text, "")
    return shorten(title.strip(), 90) + "\n" + shorten(summary.strip())


def main() -> None:
    x, y, x_test, jobs = load_data()
    model = Hybrid().fit(x, y, jobs)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "hybrid.joblib")

    examples = x_test.sample(N_EXAMPLES, random_state=0)
    demo = {
        "catalog": {str(job_id): snippet(text) for job_id, text in jobs.items()},
        "examples": [
            {"job_ids": [int(j) for j in job_ids], "actions": list(actions)}
            for job_ids, actions in zip(examples["job_ids"], examples["actions"], strict=True)
        ],
    }
    (MODELS_DIR / "demo.json").write_text(json.dumps(demo, ensure_ascii=False))

    # Reference for the scheduled drift check: training session profiles, no raw sessions.
    known_jobs = sorted(int(j) for j in set(x["job_ids"].explode()) | set(y["job_id"]))
    reference = {
        "known_jobs": known_jobs,
        "profile": session_profile(x, set(known_jobs)).round(4).to_dict(orient="list"),
    }
    (MODELS_DIR / "reference.json").write_text(json.dumps(reference))

    for name in ("hybrid.joblib", "demo.json", "reference.json"):
        size_mb = (MODELS_DIR / name).stat().st_size / 1e6
        print(f"{name}: {size_mb:.1f} MB")
    print(f"Hybrid fitted on {len(x)} sessions, demo with {N_EXAMPLES} test sessions")


if __name__ == "__main__":
    main()
