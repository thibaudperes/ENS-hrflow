import ast
import json
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _read_sessions(path: Path) -> pd.DataFrame:
    sessions = pd.read_csv(path)
    sessions["job_ids"] = sessions["job_ids"].map(ast.literal_eval)
    sessions["actions"] = sessions["actions"].map(ast.literal_eval)
    return sessions


def load_data(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    data_dir = Path(data_dir)
    x_train = _read_sessions(data_dir / "x_train.csv")
    x_test = _read_sessions(data_dir / "x_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    with open(data_dir / "job_listings.json", encoding="utf-8") as f:
        listings = json.load(f)
    # Accents are stored decomposed (letter + combining mark): NFC makes "é" a single character.
    jobs = pd.Series(
        {int(job_id): unicodedata.normalize("NFC", text) for job_id, text in listings.items()}
    ).sort_index()
    return x_train, y_train, x_test, jobs
