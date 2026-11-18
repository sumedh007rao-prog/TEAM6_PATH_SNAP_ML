"""Logistic regression: linear, probabilistic baseline."""
from sklearn.linear_model import LogisticRegression

NAME = "logistic_regression"
FAMILY = "Linear / probabilistic"
DEFAULT_PARAMS = {"C": 1.0, "max_iter": 300, "solver": "lbfgs", "class_weight": None}
MAX_TRAIN_ROWS = None


def build(**overrides):
    p = {**DEFAULT_PARAMS, **overrides}
    return LogisticRegression(random_state=42, **p)


def suggest(trial):
    return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
