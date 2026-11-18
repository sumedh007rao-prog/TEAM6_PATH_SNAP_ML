"""Single decision tree: non-linear, interpretable, prone to overfitting."""
from sklearn.tree import DecisionTreeClassifier

NAME = "decision_tree"
FAMILY = "Tree-based"
DEFAULT_PARAMS = {"max_depth": 10, "min_samples_leaf": 50, "criterion": "gini"}
MAX_TRAIN_ROWS = None


def build(**overrides):
    p = {**DEFAULT_PARAMS, **overrides}
    return DecisionTreeClassifier(random_state=42, **p)


def suggest(trial):
    return {"max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 500, log=True)}
