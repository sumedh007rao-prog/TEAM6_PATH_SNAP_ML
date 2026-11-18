"""PathSnap API (runs on Laptop 1).

Start:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
        (run from the project root)

Endpoints
  GET  /health            liveness + loaded model name
  GET  /trips             ids of the evaluation trips (ids only, no data)
  POST /predict           {"trip_id": "00000001"} -> predicted route + metrics

The reference route is read server-side only to compute the metrics; it is
never passed to the prediction pipeline and never returned to the client.
Set PATHSNAP_PARTITION=student_testing to rehearse with the development trips.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader  # noqa: E402
from src.pipeline import Predictor  # noqa: E402

PARTITION = os.environ.get("PATHSNAP_PARTITION", "evaluator_testing")
log = logging.getLogger("uvicorn.error")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor()
    log.info("loaded model '%s'; serving partition '%s' (%d trips)",
             state["predictor"].model_name, PARTITION, len(data_loader.list_trips(PARTITION)))
    yield
    state.clear()


app = FastAPI(title="PathSnap API", lifespan=lifespan)


class PredictRequest(BaseModel):
    trip_id: str


def _round(a, nd=6):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


@app.get("/health")
def health():
    return {"status": "ok", "model": state["predictor"].model_name, "partition": PARTITION}


@app.get("/trips")
def trips():
    return {"partition": PARTITION, "trips": data_loader.list_trips(PARTITION)}


@app.post("/predict")
def predict(req: PredictRequest):
    log.info("POST /predict trip_id=%s", req.trip_id)
    if req.trip_id not in data_loader.list_trips(PARTITION):
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id!r}")

    # 1) the prediction pipeline sees only track + road network (no route)
    trip = data_loader.load_trip(PARTITION, req.trip_id, with_route=False)
    result = state["predictor"].predict_trip(trip)

    # 2) scoring uses the reference route, loaded separately, server-side
    reference = data_loader.load_reference_route(PARTITION, req.trip_id)
    metrics = state["predictor"].score(result, reference) if reference is not None else {}
    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}

    net, obs = result["network"], result["obs"]
    response = {
        "trip_id": req.trip_id,
        "n_observations": int(len(obs)),
        "model": state["predictor"].model_name,
        "gps": {"lat": _round(obs["lat"]), "lon": _round(obs["lon"])},
        "predicted_route_hmm": net.arcs_to_polylines(result["hmm_arcs"].tolist()),
        "predicted_route_classifier": net.arcs_to_polylines(result["classifier_arcs"].tolist()),
        "hmm_arc_ids": [int(a) for a in result["hmm_arcs"]],
        "metrics": metrics,
        "timing_s": {k: round(v, 3) for k, v in result["timing_s"].items()},
    }
    log.info("trip %s done: hmm_top1=%s classifier_top1=%s total=%.2fs", req.trip_id,
             metrics.get("hmm_top1_acc"), metrics.get("classifier_top1_acc"),
             result["timing_s"]["total"])
    return response
