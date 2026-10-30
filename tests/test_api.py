import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hrflow_recsys.api import create_app
from hrflow_recsys.models.hybrid import Hybrid


def fitted_model() -> tuple[Hybrid, pd.Series]:
    job_ids = list(range(1, 81))  # Hybrid keeps the top 50 text matches
    words = ["data", "sales", "python", "logistics", "finance", "design"]
    jobs = pd.Series({j: f"{words[j % 6]} {words[(j + 1) % 6]} offer" for j in job_ids})
    x = pd.DataFrame({"session_id": [0, 1, 2], "job_ids": [[1, 2], [2, 3], [5, 6]]})
    y = pd.DataFrame({"session_id": [0, 1, 2], "job_id": [3, 4, 7]})
    return Hybrid().fit(x, y, jobs), jobs


@pytest.fixture(scope="module")
def client() -> TestClient:
    model, _ = fitted_model()
    return TestClient(create_app(model))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_recommend_returns_ten_unseen_jobs(client):
    response = client.post("/recommend", json={"job_ids": [1, 2], "actions": ["view", "apply"]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["job_ids"]) == 10
    assert not {1, 2} & set(body["job_ids"])
    assert body["job_ids"][0] == 3
    assert body["action"] == "apply"


def test_rejects_unknown_job(client):
    response = client.post("/recommend", json={"job_ids": [999], "actions": ["view"]})
    assert response.status_code == 422


def test_rejects_length_mismatch(client):
    response = client.post("/recommend", json={"job_ids": [1, 2], "actions": ["view"]})
    assert response.status_code == 422


def test_demo_needs_demo_data(client):
    assert client.get("/demo/random").status_code == 404
    assert "Another session" in client.get("/").text


def test_demo_returns_texts():
    model, jobs = fitted_model()
    demo = {
        "catalog": {str(j): f"Title {j}\nSummary of offer {j}" for j in jobs.index},
        "examples": [{"job_ids": [1, 2], "actions": ["view", "apply"]}],
    }
    body = TestClient(create_app(model, demo)).get("/demo/random").json()
    assert [job["job_id"] for job in body["session"]] == [1, 2]
    assert body["session"][0]["text"] == "Title 1\nSummary of offer 1"
    assert len(body["recommendations"]) == 10
    assert body["action"] == "apply"


def test_recommendation_is_logged_as_json(client, capsys):
    client.post("/recommend", json={"job_ids": [1, 2], "actions": ["view", "apply"]})
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    entry = lines[-1]
    assert entry["message"] == "recommendation"
    assert entry["source"] == "api"
    assert entry["job_ids"] == [1, 2]
    assert len(entry["recommended"]) == 10
    assert entry["latency_ms"] >= 0


def test_rejection_is_logged(client, capsys):
    client.post("/recommend", json={"job_ids": [1, 999], "actions": ["view", "view"]})
    entry = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert entry == {
        "severity": "INFO",
        "message": "rejected",
        "source": "api",
        "n_jobs": 2,
        "n_unknown": 1,
    }
