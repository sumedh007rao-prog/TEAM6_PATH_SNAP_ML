"""Inference pipeline shared by notebooks and the FastAPI backend.

``Predictor`` loads the saved scaler, model and configuration, then turns a raw
trip (track + network, NO labels) into classifier-only and HMM-decoded arcs.
Scoring against the reference route is a separate step (``score``) that only
the server calls, so labels never enter the prediction path.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from .candidate_generation import assign_reference_arcs
from .data_loader import PROJECT_ROOT, Trip
from .evaluation import predicted_arc_per_obs, route_continuity, route_overlap
from .feature_engineering import FEATURE_COLUMNS, FeatureConfig, build_trip_features
from .route_reconstruction.hmm import HMMConfig, RouteHMM

MODELS_DIR = PROJECT_ROOT / "models"


class Predictor:
    def __init__(self, models_dir: Path | str = MODELS_DIR):
        models_dir = Path(models_dir)
        self.model = joblib.load(models_dir / "final_model.pkl")
        self.scaler = joblib.load(models_dir / "scaler.pkl")
        with open(models_dir / "configurations" / "pipeline_config.json") as f:
            cfg = json.load(f)
        self.model_name = cfg["model_name"]
        self.feature_columns = cfg["feature_columns"]
        assert self.feature_columns == FEATURE_COLUMNS, "feature list changed since training"
        self.feature_config = FeatureConfig.from_dict(cfg["feature_config"])
        self.hmm_config = HMMConfig.from_dict(cfg["hmm_config"])

    def predict_trip(self, trip: Trip) -> dict:
        """Run the full pipeline on a trip. ``trip.route`` is ignored."""
        t0 = time.perf_counter()
        tf = build_trip_features(trip, self.feature_config, with_labels=False, keep_network=True)
        t_feat = time.perf_counter() - t0

        c, obs, net = tf.cands, tf.obs, tf.network
        c = c.assign(obs_key=c["obs_idx"].to_numpy())
        obs = obs.assign(obs_key=obs["obs_idx"].to_numpy())

        t1 = time.perf_counter()
        if len(c):
            X = self.scaler.transform(c[self.feature_columns])
            proba = self.model.predict_proba(X)[:, 1]
            clf_arcs = predicted_arc_per_obs(c, proba, obs)
        else:
            proba, clf_arcs = np.zeros(0), np.full(len(obs), -1, dtype=np.int64)
        t_model = time.perf_counter() - t1

        t2 = time.perf_counter()
        hmm_arcs = RouteHMM(net, self.hmm_config).decode(c, proba, obs["t"].to_numpy())
        t_hmm = time.perf_counter() - t2

        return {"trip_id": trip.trip_id, "cands": c, "obs": obs, "network": net,
                "proba": proba, "classifier_arcs": clf_arcs, "hmm_arcs": hmm_arcs,
                "timing_s": {"features": t_feat, "model": t_model, "hmm": t_hmm,
                             "total": time.perf_counter() - t0}}

    @staticmethod
    def score(result: dict, reference_route: np.ndarray) -> dict:
        """Compare predictions with the reference route (server-side only)."""
        c, obs, net = result["cands"], result["obs"], result["network"]
        ref = assign_reference_arcs(c, reference_route, len(obs))
        has_ref = ref >= 0
        out = {"n_obs": int(len(obs)), "candidate_miss_rate": float(1 - has_ref.mean())}
        for name, arcs in (("classifier", result["classifier_arcs"]), ("hmm", result["hmm_arcs"])):
            out[f"{name}_top1_acc"] = float(np.mean((arcs == ref) & has_ref))
            out[f"{name}_top1_acc_given_ref"] = (float(np.mean(arcs[has_ref] == ref[has_ref]))
                                                 if has_ref.any() else float("nan"))
            out.update({f"{name}_{k}": v for k, v in route_continuity(arcs, net).items()})
            out.update({f"{name}_{k}": v for k, v in route_overlap(arcs, reference_route).items()})
        return out
