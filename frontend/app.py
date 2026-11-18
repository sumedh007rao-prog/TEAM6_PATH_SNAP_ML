"""PathSnap evaluator UI (runs on Laptop 2).

Start:  PATHSNAP_API_URL=http://<laptop1-ip>:8000 python frontend/app.py
Open:   http://localhost:5000

The UI lists the evaluation trips offered by the API, sends the chosen trip id
to POST /predict and draws the returned route (Folium) with the metrics.
It never reads any dataset file.
"""
import os
import sys
from pathlib import Path

import requests
from flask import Flask, abort, render_template

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualisation import build_trip_map  # noqa: E402

API_URL = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")
app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))
_maps: dict[str, str] = {}   # trip_id -> rendered folium html

METRIC_ROWS = [
    ("Top-1 accuracy (observation level)", "classifier_top1_acc", "hmm_top1_acc"),
    ("Route continuity", "classifier_route_continuity", "hmm_route_continuity"),
    ("Route F1 (arc overlap)", "classifier_route_f1", "hmm_route_f1"),
    ("Route precision", "classifier_route_precision", "hmm_route_precision"),
    ("Route recall", "classifier_route_recall", "hmm_route_recall"),
    ("Route breaks (count)", "classifier_n_breaks", "hmm_n_breaks"),
]


def _api(method, path, **kw):
    try:
        r = requests.request(method, API_URL + path, timeout=300, **kw)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        abort(502, description=f"API at {API_URL} not reachable or returned an error: {e}")


@app.route("/")
def index():
    data = _api("GET", "/trips")
    return render_template("index.html", trips=data["trips"], partition=data["partition"],
                           api_url=API_URL)


@app.route("/trip/<trip_id>")
def trip(trip_id):
    res = _api("POST", "/predict", json={"trip_id": trip_id})
    gps = list(zip(res["gps"]["lat"], res["gps"]["lon"]))
    m = build_trip_map([list(p) for p in gps],
                       classifier_polylines=res["predicted_route_classifier"],
                       hmm_polylines=res["predicted_route_hmm"],
                       title=f"Trip {trip_id}")
    _maps[trip_id] = m.get_root().render()
    rows = [(label, res["metrics"].get(a), res["metrics"].get(b)) for label, a, b in METRIC_ROWS]
    return render_template("trip.html", trip_id=trip_id, res=res, rows=rows)


@app.route("/map/<trip_id>")
def map_view(trip_id):
    if trip_id not in _maps:
        abort(404)
    return _maps[trip_id]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
