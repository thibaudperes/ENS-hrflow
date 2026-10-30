import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from hrflow_recsys.demo_page import DEMO_HTML
from hrflow_recsys.models.baselines import last_action
from hrflow_recsys.models.hybrid import Hybrid

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def log_event(event: str, **fields) -> None:
    """One JSON line on stdout: Cloud Run sends it to Cloud Logging as a structured entry."""
    print(json.dumps({"severity": "INFO", "message": event, **fields}), file=sys.stdout, flush=True)


class Session(BaseModel):
    job_ids: list[int] = Field(min_length=1, max_length=100)
    actions: list[Literal["view", "apply"]] = Field(min_length=1, max_length=100)


class Recommendation(BaseModel):
    job_ids: list[int]
    action: Literal["view", "apply"]


class Job(BaseModel):
    job_id: int
    text: str


class DemoResult(BaseModel):
    session: list[Job]
    actions: list[Literal["view", "apply"]]
    recommendations: list[Job]
    action: Literal["view", "apply"]


def load_model(path: str | Path | None = None) -> Hybrid:
    return joblib.load(path or os.environ.get("MODEL_PATH", MODELS_DIR / "hybrid.joblib"))


def load_demo(path: str | Path | None = None) -> dict | None:
    path = Path(path or os.environ.get("DEMO_PATH", MODELS_DIR / "demo.json"))
    return json.loads(path.read_text()) if path.exists() else None


def create_app(model: Hybrid, demo: dict | None = None) -> FastAPI:
    app = FastAPI(title="HrFlow session recommender")
    known_jobs = set(model.text.job_index)

    def recommend_session(session: Session, source: str) -> Recommendation:
        if len(session.job_ids) != len(session.actions):
            raise HTTPException(422, "job_ids and actions must have the same length")
        unknown = sorted(set(session.job_ids) - known_jobs)
        if unknown:
            log_event(
                "rejected",
                source=source,
                n_jobs=len(session.job_ids),
                n_unknown=len(unknown),
            )
            raise HTTPException(422, f"unknown job ids: {unknown[:10]}")
        start = time.perf_counter()
        job_ids = model.recommend_batch([session.job_ids])[0]
        rec = Recommendation(job_ids=job_ids, action=last_action(session.actions))
        log_event(
            "recommendation",
            source=source,
            job_ids=session.job_ids,
            actions=session.actions,
            recommended=rec.job_ids,
            action=rec.action,
            latency_ms=round(1000 * (time.perf_counter() - start), 1),
        )
        return rec

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return DEMO_HTML

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/recommend")
    def recommend(session: Session) -> Recommendation:
        return recommend_session(session, source="api")

    @app.get("/demo/random")
    def demo_random() -> DemoResult:
        if demo is None:
            raise HTTPException(404, "demo data not loaded")
        session = Session(**random.choice(demo["examples"]))
        rec = recommend_session(session, source="demo")

        def jobs(ids: list[int]) -> list[Job]:
            return [Job(job_id=j, text=demo["catalog"].get(str(j), "")) for j in ids]

        return DemoResult(
            session=jobs(session.job_ids),
            actions=session.actions,
            recommendations=jobs(rec.job_ids),
            action=rec.action,
        )

    return app
