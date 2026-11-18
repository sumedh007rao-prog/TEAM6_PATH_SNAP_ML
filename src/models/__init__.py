"""Registry of the models compared in the experiments."""
from . import decision_tree, knn, logistic_regression, random_forest, xgboost_model

MODELS = {m.NAME: m for m in
          (logistic_regression, knn, decision_tree, random_forest, xgboost_model)}


def get_model(name):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name]
