"""FastAPI backend: serves the trained models, the statistics and the frontend.

Start with `python run.py` (trains first if artifacts are missing).
Interactive API docs: http://127.0.0.1:8001/docs
"""
import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .calibration import BinningCalibrator
from .data import load_raw, preview
from .naive_bayes import MixedNaiveBayes
from .polyfit import PolynomialCurveFit

app = FastAPI(title="GlucoSense API", version="1.0",
              description="Diabetes risk screening with a from-scratch Naive Bayes model.")


def _load(name):
    path = config.ARTIFACTS / name
    if not path.exists():
        raise RuntimeError(f"{path} is missing. Run `python -m backend.train` first.")
    return json.loads(path.read_text())


MODEL = _load("model.json")
REPORT = _load("report.json")
GRAPHS = _load("graphs.json")
POLY = _load("poly_data.json")
NB = MixedNaiveBayes.from_dict(MODEL["naive_bayes"])
CAL = BinningCalibrator.from_dict(MODEL["calibrator"])
OPS = MODEL["operating_points"]
POLY_MODEL = PolynomialCurveFit(MODEL["polynomial"]["degree"], math.exp(MODEL["polynomial"]["log_lambda"]))
POLY_MODEL.w = np.array(MODEL["polynomial"]["weights"])


@lru_cache(maxsize=1)
def dataset() -> pd.DataFrame:
    return load_raw()


# --------------------------------------------------------------------------- #
@app.get("/api/overview")
def overview():
    s, nb = REPORT["stats"], REPORT["nb"]
    return {
        "rows": s["n_rows"], "features": s["n_features"], "positives": s["positives"],
        "base_rate": s["base_rate"], "missing": s["missing"], "duplicates": s["duplicates"],
        "sizes": nb["sizes"], "test": nb["test"], "val": nb["val"], "cv": nb["cv"],
        "alpha": nb["alpha"], "threshold": nb["threshold"], "threshold_calibrated": nb["threshold_calibrated"],
        "sklearn": {k: v for k, v in nb["sklearn"].items() if k != "agreement_sample"},
        "bmi_density_choice": nb["bmi_density_choice"],
        "polynomial": {k: v for k, v in REPORT["poly"]["selected"].items() if k != "curve"},
        "trained_at": REPORT["trained_at"],
    }


@app.get("/api/features")
def features():
    return {"features": config.FEATURES, "defaults": config.DEFAULT_ANSWERS,
            "day_bins": config.DAY_BIN_LABELS, "operating_points": OPS,
            "default_recall": MODEL["default_recall"]}


@app.get("/api/data/preview")
def data_preview(offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=200)):
    return preview(dataset(), offset, limit)


@app.get("/api/data/summary")
def data_summary():
    return {"summary": REPORT["stats"]["summary"]}


@app.get("/api/stats/probability")
def probability():
    s = REPORT["stats"]
    return {"base_rate": s["base_rate"], "bayes": s["bayes"], "joint_highbp": s["joint_highbp"],
            "target_independence": s["target_independence"], "top_dependent_pairs": s["top_dependent_pairs"],
            "rate_by": s["rate_by"]}


@app.get("/api/stats/bmi")
def bmi_stats():
    b = REPORT["stats"]["bmi"]
    return {k: b[k] for k in ("overall", "by_class", "quantiles", "fits", "histogram", "total_expectation")}


# --------------------------------------------------------------------------- #
@app.get("/api/polyfit")
def polyfit(degree: int = Query(3, ge=0, le=12), log_lambda: float | None = Query(None, ge=-30, le=5),
            size: str = Query("1000")):
    if size not in POLY["samples"]:
        raise HTTPException(400, f"size must be one of {list(POLY['samples'])}")
    train = POLY["samples"][size]
    test = POLY["test"]
    lam = 0.0 if log_lambda is None else math.exp(log_lambda)
    m = PolynomialCurveFit(degree, lam).fit(train[0], train[1])
    grid = np.linspace(15, 60, 181)
    curve = m.predict(grid)
    return {
        "degree": degree, "lambda": lam, "size": size,
        "train": {"x": train[0], "t": train[1], "n": train[2]},
        "test": {"x": test[0], "t": test[1]},
        "grid": grid.tolist(), "curve": np.clip(curve, -1, 2).tolist(),
        "train_rms": m.erms(train[0], train[1]), "test_rms": m.erms(test[0], test[1]),
        "weights": m.w.tolist(), "max_weight": float(np.abs(m.w).max()),
    }


# --------------------------------------------------------------------------- #
class Answers(BaseModel):
    answers: dict[str, float] = Field(..., description="Feature key -> raw survey answer")
    target_recall: float = Field(config.TARGET_RECALL, ge=0.5, le=0.95)


def _validate(answers: dict) -> dict:
    row = dict(config.DEFAULT_ANSWERS)
    for k, v in answers.items():
        f = config.FEATURE_BY_KEY.get(k)
        if f is None:
            raise HTTPException(400, f"unknown feature {k}")
        if f["kind"] == "binary" and v not in (0, 1):
            raise HTTPException(400, f"{k} must be 0 or 1")
        if f["kind"] == "ordinal" and v not in f["levels"]:
            raise HTTPException(400, f"{k} must be one of {f['levels']}")
        if f["kind"] == "days" and not 0 <= v <= 30:
            raise HTTPException(400, f"{k} must be between 0 and 30")
        if f["kind"] == "continuous" and not f["min"] <= v <= f["max"]:
            raise HTTPException(400, f"{k} must be between {f['min']} and {f['max']}")
        row[k] = v
    return row


def _operating_point(target_recall: float) -> dict:
    return min(OPS, key=lambda o: abs(o["target_recall"] - target_recall))


@app.post("/api/predict")
def predict(body: Answers):
    row = _validate(body.answers)
    X = pd.DataFrame([row])
    raw = float(NB.predict_proba(X)[0])
    risk = float(CAL.transform([raw])[0])
    op = _operating_point(body.target_recall)
    ex = NB.explain(X)
    contributions = sorted(
        [{"key": k, "label": config.LABEL[k], "value": row[k], "log_lr": t, "odds_multiplier": math.exp(t)}
         for k, t in ex["terms"].items()], key=lambda c: -abs(c["log_lr"]))
    base = REPORT["stats"]["base_rate"]
    return {
        "raw_score": raw, "risk": risk, "base_rate": base, "relative_risk": risk / base,
        "flag": raw >= op["threshold"], "operating_point": op,
        "prior_log_odds": ex["prior_log_odds"], "posterior_log_odds": ex["posterior_log_odds"],
        "contributions": contributions,
        "population_rate_at_bmi": float(np.clip(POLY_MODEL.predict([row["BMI"]])[0], 0, 1)),
    }


# --------------------------------------------------------------------------- #
@app.get("/api/graphs")
def graphs():
    return {"graphs": GRAPHS}


@app.get("/api/graphs/{gid}")
def graph(gid: str):
    for g in GRAPHS:
        if g["id"] == gid:
            return g
    raise HTTPException(404, "no such graph")


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def revalidate_assets(request, call_next):
    """Make the browser re-check front-end files so edits show up without a hard refresh."""
    response = await call_next(request)
    if request.url.path.startswith("/assets") or request.url.path in ("/", "/graphs"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/assets", StaticFiles(directory=config.FRONTEND / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND / "index.html")


@app.get("/graphs", include_in_schema=False)
def graphs_page():
    return FileResponse(config.FRONTEND / "graphs.html")
