"""Bayesian optimisation (Optuna TPE) of XGBoost on the GPU.
Falls back to the CPU automatically if no CUDA device is found."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
from xgboost import XGBClassifier

from ..evaluation import group_cv


def _detect_device() -> str:
    try:
        m = XGBClassifier(device="cuda", n_estimators=2, tree_method="hist")
        m.fit(np.random.rand(50, 3), np.random.randint(0, 2, 50))
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = _detect_device()


def build_xgb(**params) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", device=DEVICE, random_state=42, n_jobs=-1, **params)


def suggest_xgb(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
    }


def optimise_gpu(X, cands, obs, n_trials: int = 30, n_splits: int = 3, seed: int = 42,
                 max_train_rows: int | None = 400_000,
                 out_dir: str | Path = "outputs",
                 model_dir: str | Path = "models/configurations"):
    """Search XGBoost hyperparameters by group-aware CV macro F1 (training trips only).
    Returns (best_params, best_value, trials_dataframe)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = suggest_xgb(trial)
        t0 = time.perf_counter()
        res = group_cv(lambda: build_xgb(**params), X, cands, obs,
                       n_splits=n_splits, seed=seed, max_train_rows=max_train_rows)
        secs = time.perf_counter() - t0
        score = float(res["macro_f1"].mean())
        trial.set_user_attr("seconds", secs)
        print(f"trial {trial.number:3d} | device={DEVICE} | cv macro F1={score:.4f} "
              f"| {secs:6.1f}s | {params}", flush=True)
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                study_name="xgboost_gpu_tpe")
    study.optimize(objective, n_trials=n_trials)

    trials = study.trials_dataframe()
    out_dir, model_dir = Path(out_dir), Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out_dir / "optimisation_trials.csv", index=False)
    with open(model_dir / "best_params.json", "w") as f:
        json.dump({"model": "xgboost", "device": DEVICE, "best_cv_macro_f1": study.best_value,
                   "n_trials": n_trials, "n_splits": n_splits, "seed": seed,
                   "params": study.best_params}, f, indent=2)
    return study.best_params, study.best_value, trials