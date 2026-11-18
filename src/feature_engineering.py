"""Per-(observation, candidate) feature extraction and dataset assembly.

One row = one candidate arc for one GPS observation. The label ``y`` is 1 when
the candidate is the aligned reference arc of that observation. Features use
only GPS, timing and network geometry, never the reference route, so the same
code runs on evaluator trips.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import data_loader
from .candidate_generation import CandidateGenerator, assign_reference_arcs
from .data_loader import Trip
from .preprocessing import add_motion_columns, clean_track
from .road_network import RoadNetwork

FEATURE_COLUMNS = [
    "dist_m", "dist_rank", "dist_gap_to_best", "dist_ratio_to_best",
    "proj_frac", "dist_to_arc_end_m", "arc_len_m", "n_cands",
    "heading_diff_deg", "heading_cos", "speed_mps", "heading_reliable",
    "step_prev_m", "dt_prev",
    "to_node_out_deg", "from_node_in_deg", "has_reverse_arc",
    "neigh_presence", "neigh_mean_dist",
]


@dataclass
class FeatureConfig:
    radius_m: float = 60.0
    max_candidates: int = 10
    spacing_m: float = 10.0
    heading_window: int = 2
    min_move_m: float = 3.0
    neighbour_offsets: tuple = (-3, -2, -1, 1, 2, 3)

    def to_dict(self):
        d = asdict(self)
        d["neighbour_offsets"] = list(self.neighbour_offsets)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "neighbour_offsets" in d:
            d["neighbour_offsets"] = tuple(d["neighbour_offsets"])
        return cls(**d)


@dataclass
class TripFeatures:
    trip_id: str
    cands: pd.DataFrame           # one row per candidate with features (+ y if labelled)
    obs: pd.DataFrame             # one row per cleaned GPS observation
    network: RoadNetwork | None
    report: dict = field(default_factory=dict)


def _neighbour_features(cands: pd.DataFrame, n_arcs: int, offsets, radius_m: float):
    """How persistently the same arc is a candidate in neighbouring observations.

    neigh_presence : share of the neighbouring observations (offsets) that also
                     have this arc among their candidates.
    neigh_mean_dist: mean distance to this arc over those neighbours (radius if none).
    """
    key = cands["obs_idx"].to_numpy() * n_arcs + cands["arc_id"].to_numpy()
    order = np.argsort(key)
    skey, sdist = key[order], cands["dist_m"].to_numpy()[order]
    obs, arc = cands["obs_idx"].to_numpy(), cands["arc_id"].to_numpy()
    found_cnt = np.zeros(len(cands))
    dist_sum = np.zeros(len(cands))
    for o in offsets:
        nkey = (obs + o) * n_arcs + arc
        pos = np.clip(np.searchsorted(skey, nkey), 0, len(skey) - 1)
        hit = (skey[pos] == nkey) & (obs + o >= 0)
        found_cnt += hit
        dist_sum += np.where(hit, sdist[pos], 0.0)
    presence = found_cnt / len(offsets)
    mean_dist = np.where(found_cnt > 0, dist_sum / np.maximum(found_cnt, 1), radius_m)
    return presence, mean_dist


def build_trip_features(trip: Trip, config: FeatureConfig | None = None,
                        with_labels: bool = True, keep_network: bool = False) -> TripFeatures:
    """Full per-trip pipeline: clean -> project -> candidates -> features -> labels."""
    cfg = config or FeatureConfig()
    track, clean_report = clean_track(trip.track)
    net = RoadNetwork(trip.nodes, trip.arcs)
    obs = add_motion_columns(track, net.projection, cfg.heading_window, cfg.min_move_m)
    obs.insert(0, "obs_idx", np.arange(len(obs)))
    obs.insert(0, "trip_id", trip.trip_id)

    gen = CandidateGenerator(net, cfg.radius_m, cfg.max_candidates, cfg.spacing_m)
    c = gen.generate(obs["x"].to_numpy(), obs["y"].to_numpy())

    n_cands = c.groupby("obs_idx")["arc_id"].transform("size").to_numpy()
    best = c.groupby("obs_idx")["dist_m"].transform("min").to_numpy()
    arc = c["arc_id"].to_numpy()
    oi = c["obs_idx"].to_numpy()

    c["dist_rank"] = c.groupby("obs_idx").cumcount().to_numpy()
    c["dist_gap_to_best"] = c["dist_m"] - best
    c["dist_ratio_to_best"] = c["dist_m"] / (best + 1.0)
    c["arc_len_m"] = net.arc_len[arc]
    c["dist_to_arc_end_m"] = net.arc_len[arc] * (1.0 - c["proj_frac"].to_numpy())
    c["n_cands"] = n_cands

    hd = np.abs(obs["heading_deg"].to_numpy()[oi] - net.arc_bearing[arc]) % 360.0
    hd = np.where(hd > 180.0, 360.0 - hd, hd)
    c["heading_diff_deg"] = hd
    c["heading_cos"] = np.cos(np.radians(hd))
    for col in ("speed_mps", "heading_reliable", "step_prev_m", "dt_prev"):
        c[col] = obs[col].to_numpy()[oi]
    # outlier guards (a few trips contain hour-long timestamp jumps): clip, keep raw values in ``obs``
    c["dt_prev"] = c["dt_prev"].clip(upper=60.0)
    c["step_prev_m"] = c["step_prev_m"].clip(upper=300.0)
    c["speed_mps"] = c["speed_mps"].clip(upper=60.0)

    c["to_node_out_deg"] = net.out_deg[net.dst[arc]]
    c["from_node_in_deg"] = net.in_deg[net.src[arc]]
    c["has_reverse_arc"] = (net.reverse_arc[arc] >= 0).astype(np.int8)

    if len(c):
        pres, md = _neighbour_features(c, net.n_arcs, cfg.neighbour_offsets, cfg.radius_m)
    else:
        pres, md = np.zeros(0), np.zeros(0)
    c["neigh_presence"], c["neigh_mean_dist"] = pres, md

    obs["n_cands"] = c.groupby("obs_idx").size().reindex(obs["obs_idx"]).fillna(0).astype(int).to_numpy()

    if with_labels and trip.route is not None:
        ref = assign_reference_arcs(c, trip.route, len(obs))
        obs["ref_arc"] = ref
        c["y"] = (c["arc_id"].to_numpy() == ref[oi]).astype(np.int8)
        c["has_ref"] = (ref[oi] >= 0)
    c.insert(0, "trip_id", trip.trip_id)

    return TripFeatures(trip.trip_id, c, obs, net if keep_network else None, clean_report)


def build_dataset(partition: str, trip_ids: list[str] | None = None,
                  config: FeatureConfig | None = None, n_jobs: int = 1,
                  verbose: bool = True, dataset_dir=None):
    """Features for many trips of one partition.

    Returns ``(cands, obs)``. ``obs_key`` is a global observation id (row
    position in ``obs``); ``group`` is the integer trip id used for group-aware
    cross-validation (so a trip never spans two folds).
    """
    cfg = config or FeatureConfig()
    ids = trip_ids or data_loader.list_trips(partition, dataset_dir)

    def one(tid):
        trip = data_loader.load_trip(partition, tid, with_route=True, dataset_dir=dataset_dir)
        tf = build_trip_features(trip, cfg, with_labels=True)
        return tf.cands, tf.obs

    t0 = time.time()
    if n_jobs == 1:
        results = []
        for i, tid in enumerate(ids):
            results.append(one(tid))
            if verbose and (i + 1) % 10 == 0:
                print(f"  {partition}: {i + 1}/{len(ids)} trips ({time.time() - t0:.0f}s)")
    else:
        results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(one)(t) for t in ids)

    cands_parts, obs_parts, offset = [], [], 0
    for tid, (c, o) in zip(ids, results):
        o = o.copy()
        o["obs_key"] = offset + o["obs_idx"].to_numpy()
        c = c.copy()
        c["obs_key"] = offset + c["obs_idx"].to_numpy()
        o["group"] = c["group"] = int(tid)
        offset += len(o)
        cands_parts.append(c)
        obs_parts.append(o)
    cands = pd.concat(cands_parts, ignore_index=True)
    obs = pd.concat(obs_parts, ignore_index=True)
    if verbose:
        print(f"  {partition}: {len(ids)} trips, {len(obs):,} observations, "
              f"{len(cands):,} candidate rows in {time.time() - t0:.0f}s")
    return cands, obs


def save_dataset(path, cands, obs, config: FeatureConfig):
    joblib.dump({"cands": cands, "obs": obs, "config": config.to_dict()}, path, compress=3)


def load_dataset(path):
    d = joblib.load(path)
    return d["cands"], d["obs"], FeatureConfig.from_dict(d["config"])
