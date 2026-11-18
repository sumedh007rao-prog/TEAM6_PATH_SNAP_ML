"""Cleaning, projection and per-trip statistics.

Projection choice
-----------------
Longitude/latitude are angular, so Euclidean distance in degrees is wrong.
Every trip covers at most a few tens of kilometres, so a local equirectangular
(plane) projection centred on the trip gives metre-accurate distances
(see notebook 02 for the measured error against the haversine formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_loader

EARTH_RADIUS_M = 6_371_008.8


class LocalProjection:
    """Equirectangular projection around (lon0, lat0); output in metres."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self._cos = np.cos(np.radians(self.lat0))

    def forward(self, lon, lat):
        x = np.radians(np.asarray(lon) - self.lon0) * EARTH_RADIUS_M * self._cos
        y = np.radians(np.asarray(lat) - self.lat0) * EARTH_RADIUS_M
        return x, y

    def inverse(self, x, y):
        lon = self.lon0 + np.degrees(np.asarray(x) / (EARTH_RADIUS_M * self._cos))
        lat = self.lat0 + np.degrees(np.asarray(y) / EARTH_RADIUS_M)
        return lon, lat


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------- cleaning
def clean_track(track: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove unusable GPS rows. Returns (clean_track, report).

    Steps: drop NaN/inf, drop impossible coordinates, sort by time, drop
    repeated timestamps. ``orig_idx`` keeps the row number in the raw file.
    """
    df = track.copy()
    df["orig_idx"] = np.arange(len(df))
    n_raw = len(df)

    finite = np.isfinite(df[["lon", "lat", "t"]]).all(axis=1)
    valid_range = df["lon"].between(-180, 180) & df["lat"].between(-90, 90)
    n_missing = int((~finite).sum())
    n_bad_coord = int((finite & ~valid_range).sum())
    df = df[finite & valid_range]

    was_sorted = bool(df["t"].is_monotonic_increasing)
    df = df.sort_values("t", kind="stable")
    n_dup_t = int(df["t"].duplicated().sum())
    df = df.drop_duplicates("t", keep="first").reset_index(drop=True)

    report = {"n_raw": n_raw, "n_missing": n_missing, "n_bad_coord": n_bad_coord,
              "was_time_sorted": was_sorted, "n_duplicate_time": n_dup_t,
              "n_clean": len(df)}
    return df, report


def add_motion_columns(track: pd.DataFrame, projection: LocalProjection,
                       window: int = 2, min_move_m: float = 3.0) -> pd.DataFrame:
    """Add projected x/y, time step, step length, speed and heading.

    Speed and heading use a centred window (index i-window .. i+window) which
    is far less noisy than point-to-point values on 1 Hz GPS. Heading is only
    trusted when the vehicle moved at least ``min_move_m`` in the window,
    otherwise it is carried over from the last reliable value.
    """
    df = track.copy()
    x, y = projection.forward(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["y"] = x, y
    n = len(df)
    t = df["t"].to_numpy()

    dt = np.diff(t, prepend=np.nan)
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    df["dt_prev"] = np.where(np.isnan(dt), 0.0, dt)
    df["step_prev_m"] = np.where(np.isnan(step), 0.0, step)

    idx = np.arange(n)
    lo, hi = np.clip(idx - window, 0, n - 1), np.clip(idx + window, 0, n - 1)
    dx, dy = x[hi] - x[lo], y[hi] - y[lo]
    dts = np.maximum(t[hi] - t[lo], 1e-6)
    disp = np.hypot(dx, dy)
    df["speed_mps"] = disp / dts

    heading = np.degrees(np.arctan2(dx, dy)) % 360.0
    reliable = disp >= min_move_m
    heading = pd.Series(np.where(reliable, heading, np.nan)).ffill().bfill()
    df["heading_deg"] = heading.fillna(0.0).to_numpy()
    df["heading_reliable"] = reliable.astype(np.int8)
    return df


# ------------------------------------------------------------------- statistics
def track_statistics(track_clean: pd.DataFrame, gap_s: float = 5.0) -> dict:
    """Sampling and geometry statistics of one cleaned trajectory."""
    if len(track_clean) < 2:
        return {"n_gps": len(track_clean)}
    dt = np.diff(track_clean["t"].to_numpy())
    lon, lat = track_clean["lon"].to_numpy(), track_clean["lat"].to_numpy()
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    speed = step / np.maximum(dt, 1e-6)
    extent_x = haversine_m(lon.min(), lat.mean(), lon.max(), lat.mean())
    extent_y = haversine_m(lon.mean(), lat.min(), lon.mean(), lat.max())
    return {
        "n_gps": len(track_clean),
        "duration_s": float(track_clean["t"].iloc[-1] - track_clean["t"].iloc[0]),
        "share_dt_1s": float(np.mean(dt == 1.0)),
        "median_dt_s": float(np.median(dt)),
        "max_gap_s": float(dt.max()),
        "n_gaps_gt_5s": int((dt > gap_s).sum()),
        "path_length_km": float(step.sum() / 1000),
        "mean_speed_mps": float(speed.mean()),
        "max_speed_mps": float(speed.max()),
        "extent_x_km": float(extent_x / 1000),
        "extent_y_km": float(extent_y / 1000),
    }


def build_inventory(partitions=data_loader.PARTITIONS,
                    dataset_dir=None) -> pd.DataFrame:
    """One row per trip: files, GPS quality, network size/connectivity, route."""
    from .road_network import RoadNetwork  # local import avoids a cycle

    rows = []
    for part in partitions:
        for tid in data_loader.list_trips(part, dataset_dir):
            row = {"partition": part, "trip_id": tid}
            row.update(data_loader.file_status(part, tid, dataset_dir))
            trip = data_loader.load_trip(part, tid, with_route=True, dataset_dir=dataset_dir)
            clean, rep = clean_track(trip.track)
            row.update(rep)
            row.update(track_statistics(clean))
            net = RoadNetwork(trip.nodes, trip.arcs)
            row.update(net.summary())
            row["route_len_arcs"] = 0 if trip.route is None else len(trip.route)
            if trip.route is not None and len(trip.route):
                row["route_ids_valid"] = bool(trip.route.min() >= 0 and trip.route.max() < net.n_arcs)
                row["route_connected_frac"] = (net.route_connected_fraction(trip.route)
                                               if row["route_ids_valid"] else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)
