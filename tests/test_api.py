import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hrflow_recsys.api import create_app
from hrflow_recsys.models.hybrid import Hybrid


@pytest.fixture(scope="module")
def client() -> TestClient:
    job_ids = list(range(1, 81))  # Hybrid keeps the top 50 text matches
    words = ["data", "sales", "python", "logistics", "finance", "design"]
    jobs = pd.Series({j: f"{words[j % 6]} {words[(j + 1) % 6]} offer" for j in job_ids})
    x = pd.DataFrame({"session_id": [0, 1, 2], "job_ids": [[1, 2], [2, 3], [5, 6]]})
    y = pd.DataFrame({"session_id": [0, 1, 2], "job_id": [3, 4, 7]})
    return TestClient(create_app(Hybrid().fit(x, y, jobs)))


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
