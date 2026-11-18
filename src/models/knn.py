"""K-nearest neighbours: instance-based, needs scaled features."""
from sklearn.neighbors import KNeighborsClassifier

NAME = "knn"
FAMILY = "Instance-based"
DEFAULT_PARAMS = {"n_neighbors": 25, "weights": "uniform", "p": 2}
# KNN stores every training row, so the training set is thinned (whole observations).
MAX_TRAIN_ROWS = 150_000


def build(**overrides):
    p = {**DEFAULT_PARAMS, **overrides}
    return KNeighborsClassifier(n_jobs=-1, **p)


def suggest(trial):
    return {"n_neighbors": trial.suggest_int("n_neighbors", 5, 100),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"])}
