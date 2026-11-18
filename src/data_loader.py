"""Raw data access for PathSnap.

Each trip lives in ``dataset/<partition>/<trip_id>/`` and has four
whitespace-separated, header-less files:

* ``<id>.track`` : longitude, latitude, timestamp (s)
* ``<id>.nodes`` : longitude, latitude         (row index = node id)
* ``<id>.arcs``  : from_node_id, to_node_id    (row index = arc id)
* ``<id>.route`` : reference arc ids in travel order (may be absent / private)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
PARTITIONS = ("training", "student_testing", "evaluator_testing")
FILE_EXTS = ("track", "nodes", "arcs", "route")


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # columns: lon, lat, t
    nodes: np.ndarray            # (n_nodes, 2) lon, lat
    arcs: np.ndarray             # (n_arcs, 2)  from_node, to_node
    route: np.ndarray | None     # reference arc ids or None


def partition_dir(partition: str, dataset_dir: Path | str | None = None) -> Path:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}; expected one of {PARTITIONS}")
    return Path(dataset_dir or DATASET_DIR) / partition


def list_trips(partition: str, dataset_dir: Path | str | None = None) -> list[str]:
    """Sorted trip ids (sub-folder names) present in a partition."""
    root = partition_dir(partition, dataset_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def trip_file(partition: str, trip_id: str, ext: str,
              dataset_dir: Path | str | None = None) -> Path:
    return partition_dir(partition, dataset_dir) / trip_id / f"{trip_id}.{ext}"


def file_status(partition: str, trip_id: str,
                dataset_dir: Path | str | None = None) -> dict:
    """Existence and size (bytes) of the four raw files of a trip."""
    out = {}
    for ext in FILE_EXTS:
        p = trip_file(partition, trip_id, ext, dataset_dir)
        out[f"has_{ext}"] = p.exists()
        out[f"bytes_{ext}"] = p.stat().st_size if p.exists() else 0
    return out


def _read_table(path: Path, names: list[str], dtype) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame({n: pd.Series(dtype=dtype) for n in names})
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_reference_route(partition: str, trip_id: str,
                         dataset_dir: Path | str | None = None) -> np.ndarray | None:
    """Reference arc sequence. Only the server-side scorer should call this
    for the evaluator partition."""
    path = trip_file(partition, trip_id, "route", dataset_dir)
    if not path.exists():
        return None
    df = _read_table(path, ["arc"], np.int64)
    return df["arc"].to_numpy()


def load_trip(partition: str, trip_id: str, with_route: bool = True,
              dataset_dir: Path | str | None = None) -> Trip:
    track = _read_table(trip_file(partition, trip_id, "track", dataset_dir),
                        ["lon", "lat", "t"], np.float64)
    nodes = _read_table(trip_file(partition, trip_id, "nodes", dataset_dir),
                        ["lon", "lat"], np.float64).to_numpy()
    arcs = _read_table(trip_file(partition, trip_id, "arcs", dataset_dir),
                       ["src", "dst"], np.int64).to_numpy()
    route = load_reference_route(partition, trip_id, dataset_dir) if with_route else None
    return Trip(trip_id, partition, track, nodes, arcs, route)
