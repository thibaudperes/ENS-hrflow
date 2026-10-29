from pathlib import Path

import joblib

from hrflow_recsys.data import load_data
from hrflow_recsys.models.hybrid import Hybrid

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hybrid.joblib"


def main() -> None:
    x, y, _, jobs = load_data()
    model = Hybrid().fit(x, y, jobs)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Hybrid fitted on {len(x)} sessions, saved to {MODEL_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
