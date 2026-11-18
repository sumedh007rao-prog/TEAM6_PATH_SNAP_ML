"""XGBoost: gradient-boosted trees (requires the ``xgboost`` package)."""

NAME = "xgboost"
FAMILY = "Boosting"
DEFAULT_PARAMS = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 1.0, "reg_lambda": 1.0}
MAX_TRAIN_ROWS = None


def build(**overrides):
    from xgboost import XGBClassifier
    from ..optimisation.bayes_gpu import DEVICE   # "cuda" if a GPU is found, otherwise "cpu"
    p = {**DEFAULT_PARAMS, **overrides}
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         tree_method="hist", device=DEVICE,
                         random_state=42, n_jobs=-1, **p)


def suggest(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True)}
