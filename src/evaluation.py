"""Metrics, group-aware cross-validation and timing helpers.

Definitions (used consistently in every table of the report)
------------------------------------------------------------
* A classifier scores every candidate row. The *top-1 decision* of an
  observation is its highest-scoring candidate.
* Observation-level Top-1 accuracy: share of ALL observations whose top-1
  candidate equals the reference arc. Observations whose reference arc is not
  in the candidate set (candidate miss) count as wrong.
  ``top1_acc_given_ref`` restricts the denominator to observations whose
  reference arc is among the candidates.
* Macro F1 / precision / recall: candidate-level, labels y in {0,1}, predicted
  label = 1 only for the top-1 candidate of each observation; computed on
  observations whose reference arc exists in the candidate set. Macro F1 is the
  mean of the F1 of class 0 and class 1.
* Log loss: binary log loss of the predicted probabilities on the same rows.
* Candidate miss rate: share of observations with no reference arc among the
  candidates.
* Route continuity: share of consecutive (distinct) predicted arcs that are
  physically connected in the road network.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold


# ------------------------------------------------------------------ decisions
def top1_decision(cands: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    """Positional row indices of the highest-scoring candidate of each observation."""
    s = pd.DataFrame({"k": cands["obs_key"].to_numpy(), "p": np.asarray(proba)})
    return s.groupby("k")["p"].idxmax().to_numpy()


def predicted_arc_per_obs(cands: pd.DataFrame, proba: np.ndarray, obs: pd.DataFrame) -> np.ndarray:
    """Top-1 arc for every row of ``obs`` (-1 when the observation has no candidate)."""
    rows = top1_decision(cands, proba)
    chosen = pd.Series(cands["arc_id"].to_numpy()[rows], index=cands["obs_key"].to_numpy()[rows])
    return chosen.reindex(obs["obs_key"].to_numpy()).fillna(-1).astype(np.int64).to_numpy()


# ------------------------------------------------------------------- metrics
def observation_metrics(cands: pd.DataFrame, obs: pd.DataFrame, proba: np.ndarray) -> dict:
    """Candidate- and observation-level metrics for scored candidates."""
    proba = np.asarray(proba, dtype=float)
    pred = predicted_arc_per_obs(cands, proba, obs)
    ref = obs["ref_arc"].to_numpy()
    has_ref = ref >= 0

    out = {
        "n_obs": int(len(obs)),
        "top1_acc": float(np.mean((pred == ref) & has_ref)),
        "top1_acc_given_ref": float(np.mean(pred[has_ref] == ref[has_ref])) if has_ref.any() else np.nan,
        "candidate_miss_rate": float(1.0 - has_ref.mean()),
        "no_candidate_rate": float((obs["n_cands"].to_numpy() == 0).mean()),
    }

    rows = top1_decision(cands, proba)
    dec = np.zeros(len(cands), dtype=np.int8)
    dec[rows] = 1
    lab = cands["has_ref"].to_numpy()
    y = cands["y"].to_numpy()[lab]
    if len(np.unique(y)) == 2:
        out["macro_f1"] = float(f1_score(y, dec[lab], average="macro"))
        out["precision_pos"] = float(precision_score(y, dec[lab], zero_division=0))
        out["recall_pos"] = float(recall_score(y, dec[lab], zero_division=0))
        out["log_loss"] = float(log_loss(y, np.clip(proba[lab], 1e-15, 1 - 1e-15), labels=[0, 1]))
    else:
        out.update(macro_f1=np.nan, precision_pos=np.nan, recall_pos=np.nan, log_loss=np.nan)
    return out


def route_continuity(pred_arcs, network) -> dict:
    """Connectivity of a predicted arc sequence (consecutive duplicates and -1 removed)."""
    seq = np.asarray([a for a in pred_arcs if a >= 0], dtype=np.int64)
    if len(seq):
        seq = seq[np.concatenate([[True], seq[1:] != seq[:-1]])]
    if len(seq) < 2:
        return {"route_continuity": 1.0, "n_breaks": 0, "n_transitions": 0}
    ok = network.dst[seq[:-1]] == network.src[seq[1:]]
    return {"route_continuity": float(ok.mean()), "n_breaks": int((~ok).sum()),
            "n_transitions": int(len(ok))}


def route_overlap(pred_arcs, ref_route) -> dict:
    """Sequence-level agreement between predicted and reference arc sets."""
    p = set(int(a) for a in pred_arcs if a >= 0)
    r = set(int(a) for a in ref_route)
    inter = len(p & r)
    prec = inter / len(p) if p else 0.0
    rec = inter / len(r) if r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


# ---------------------------------------------------------------- CV / timing
def subsample_by_obs(cands: pd.DataFrame, mask: np.ndarray, max_rows: int | None,
                     seed: int = 42) -> np.ndarray:
    """Row indices of ``cands[mask]``, thinned by whole observations to ~max_rows."""
    idx = np.flatnonzero(mask)
    if max_rows is None or len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    keys = cands["obs_key"].to_numpy()[idx]
    uniq = np.unique(keys)
    keep = rng.choice(uniq, size=max(1, int(len(uniq) * max_rows / len(idx))), replace=False)
    return idx[np.isin(keys, keep)]


def fit_model(build_fn, X: np.ndarray, cands: pd.DataFrame, train_rows: np.ndarray):
    """Fit a fresh model on labelled rows; returns (model, fit_seconds)."""
    model = build_fn()
    t0 = time.perf_counter()
    model.fit(X[train_rows], cands["y"].to_numpy()[train_rows])
    return model, time.perf_counter() - t0


def predict_proba_1(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def group_cv(build_fn, X: np.ndarray, cands: pd.DataFrame, obs: pd.DataFrame,
             n_splits: int = 5, seed: int = 42, max_train_rows: int | None = None) -> pd.DataFrame:
    """Group-aware K-fold CV (groups = trips): one row of metrics per fold."""
    groups = cands["group"].to_numpy()
    labelled = cands["has_ref"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, groups=groups)):
        tr_rows = subsample_by_obs(cands, np.isin(np.arange(len(cands)), tr) & labelled,
                                   max_train_rows, seed + fold)
        model, fit_s = fit_model(build_fn, X, cands, tr_rows)
        c_te = cands.iloc[te]
        t0 = time.perf_counter()
        p = predict_proba_1(model, X[te])
        pred_s = time.perf_counter() - t0
        o_te = obs[obs["group"].isin(np.unique(groups[te]))]
        m = observation_metrics(c_te, o_te, p)
        m.update(fold=fold, fit_seconds=fit_s,
                 predict_ms_per_1k_obs=1000 * pred_s / max(len(o_te), 1) * 1000)
        rows.append(m)
    return pd.DataFrame(rows)


def oof_proba(build_fn, X: np.ndarray, cands: pd.DataFrame, n_splits: int = 5,
              seed: int = 42, max_train_rows: int | None = None) -> np.ndarray:
    """Out-of-fold probabilities for every candidate row (group-aware).

    Every row is scored by a model that never saw its trip. Used to tune the
    HMM decoder on training trips without in-sample (over-confident) scores.
    """
    groups = cands["group"].to_numpy()
    labelled = cands["has_ref"].to_numpy()
    out = np.zeros(len(cands))
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(splitter.split(X, groups=groups)):
        mask = np.zeros(len(cands), dtype=bool)
        mask[tr] = True
        rows = subsample_by_obs(cands, mask & labelled, max_train_rows, seed + fold)
        model, _ = fit_model(build_fn, X, cands, rows)
        out[te] = predict_proba_1(model, X[te])
    return out


def pooled_route_continuity(arcs_per_obs: np.ndarray, obs: pd.DataFrame, networks: dict) -> dict:
    """Route continuity pooled over trips.

    arcs_per_obs : predicted arc for each row of ``obs`` (same order);
    networks     : {int trip id (``group``): RoadNetwork}.
    """
    breaks = trans = 0
    groups = obs["group"].to_numpy()
    for g in np.unique(groups):
        r = route_continuity(arcs_per_obs[groups == g], networks[int(g)])
        breaks += r["n_breaks"]
        trans += r["n_transitions"]
    return {"route_continuity": 1.0 - breaks / trans if trans else 1.0,
            "n_breaks": breaks, "n_transitions": trans}
