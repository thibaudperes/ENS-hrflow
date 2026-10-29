import os
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hrflow_recsys.models.baselines import last_action
from hrflow_recsys.models.hybrid import Hybrid

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hybrid.joblib"


class Session(BaseModel):
    job_ids: list[int] = Field(min_length=1, max_length=100)
    actions: list[Literal["view", "apply"]] = Field(min_length=1, max_length=100)


class Recommendation(BaseModel):
    job_ids: list[int]
    action: Literal["view", "apply"]


def load_model(path: str | Path | None = None) -> Hybrid:
    return joblib.load(path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


def create_app(model: Hybrid) -> FastAPI:
    app = FastAPI(title="HrFlow session recommender")
    known_jobs = set(model.text.job_index)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/recommend")
    def recommend(session: Session) -> Recommendation:
        if len(session.job_ids) != len(session.actions):
            raise HTTPException(422, "job_ids and actions must have the same length")
        unknown = sorted(set(session.job_ids) - known_jobs)
        if unknown:
            raise HTTPException(422, f"unknown job ids: {unknown[:10]}")
        job_ids = model.recommend_batch([session.job_ids])[0]
        return Recommendation(job_ids=job_ids, action=last_action(session.actions))

    return app
