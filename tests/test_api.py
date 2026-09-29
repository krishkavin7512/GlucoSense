"""Smoke tests for the HTTP API (needs trained artifacts: run `python -m backend.train`)."""
from fastapi.testclient import TestClient

from backend.api import app
from backend import config

client = TestClient(app)


def test_overview_and_graphs():
    o = client.get("/api/overview").json()
    assert o["rows"] == 253680
    assert 0.75 < o["test"]["roc_auc"] < 0.9
    g = client.get("/api/graphs").json()["graphs"]
    assert len(g) >= 25
    assert all({"what", "how", "read", "where"} <= set(x["explain"]) for x in g)


def test_predict_high_risk_beats_low_risk():
    low = client.post("/api/predict", json={"answers": config.DEFAULT_ANSWERS}).json()
    high = client.post("/api/predict", json={"answers": {
        **config.DEFAULT_ANSWERS, "HighBP": 1, "HighChol": 1, "BMI": 38, "GenHlth": 4, "Age": 11,
        "DiffWalk": 1, "HeartDiseaseorAttack": 1}}).json()
    assert 0 < low["risk"] < high["risk"] < 1
    assert high["flag"] and not low["flag"]
    assert len(high["contributions"]) == len(config.FEATURES)


def test_predict_rejects_bad_input():
    assert client.post("/api/predict", json={"answers": {"HighBP": 3}}).status_code == 400
    assert client.post("/api/predict", json={"answers": {"Nope": 1}}).status_code == 400


def test_polyfit_overfits_at_high_degree():
    lo = client.get("/api/polyfit", params={"degree": 2}).json()
    hi = client.get("/api/polyfit", params={"degree": 11}).json()
    assert hi["train_rms"] < lo["train_rms"]
    assert hi["test_rms"] > lo["test_rms"]


def test_data_preview_and_pages():
    p = client.get("/api/data/preview", params={"offset": 10, "limit": 5}).json()
    assert len(p["rows"]) == 5 and p["total"] == 253680
    assert client.get("/").status_code == 200
    assert client.get("/graphs").status_code == 200
