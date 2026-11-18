"""Random forest: bagged trees, stable probability estimates."""
from sklearn.ensemble import RandomForestClassifier

NAME = "random_forest"
FAMILY = "Ensemble (bagging)"
DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": 14, "min_samples_leaf": 20,
                  "max_features": "sqrt"}
MAX_TRAIN_ROWS = 600_000


def build(**overrides):
    p = {**DEFAULT_PARAMS, **overrides}
    return RandomForestClassifier(random_state=42, n_jobs=-1, **p)


def suggest(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0)}
