"""Candidate road segments for every GPS observation, and reference labelling.

Problem framing
---------------
For each GPS point we collect nearby arcs (candidates). A classifier then
scores every (observation, candidate) pair: "is this the arc the vehicle was on?"

Candidate search: every arc is sampled every ``spacing_m`` metres and the
samples go in a KD-tree. A radius query finds arcs close to a GPS point; the
exact point-to-segment distance is then computed and used to filter / rank.

Reference labelling: the ``.route`` file gives the ordered arcs of the trip
but no timestamps, so the arc a vehicle was on at each GPS point is not given.
``assign_reference_arcs`` aligns observations to the route with a monotone
dynamic programme (see function doc). This is an approximation and is treated
as reference information, not independently verified truth.
"""
from __future__ import annotations

from itertools import chain

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .road_network import RoadNetwork, point_segment_distance


class CandidateGenerator:
    def __init__(self, network: RoadNetwork, radius_m: float = 60.0,
                 max_candidates: int = 10, spacing_m: float = 10.0,
                 chunk: int = 4000):
        self.net = network
        self.radius_m, self.max_candidates = radius_m, max_candidates
        self.spacing_m, self.chunk = spacing_m, chunk

        n_samp = np.maximum(2, np.ceil(network.arc_len / spacing_m).astype(np.int64) + 1)
        owner = np.repeat(np.arange(network.n_arcs), n_samp)
        starts = np.cumsum(n_samp) - n_samp
        within = np.arange(owner.size) - np.repeat(starts, n_samp)
        frac = within / np.repeat(n_samp - 1, n_samp)
        a, b = network.arc_a[owner], network.arc_b[owner]
        pts = a + frac[:, None] * (b - a)
        self.owner = owner
        self.tree = cKDTree(pts)

    def generate(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Candidates for projected GPS points.

        Returns columns obs_idx, arc_id, dist_m, proj_frac, sorted by
        (obs_idx, dist_m) and truncated to ``max_candidates`` per observation.
        Observations with no arc within the radius simply have no rows.
        """
        net, n_arcs = self.net, max(self.net.n_arcs, 1)
        pts = np.column_stack([x, y])
        reach = self.radius_m + self.spacing_m / 2.0   # sample-vs-segment slack
        parts = []
        for s in range(0, len(pts), self.chunk):
            e = min(len(pts), s + self.chunk)
            lists = self.tree.query_ball_point(pts[s:e], r=reach, workers=-1,
                                               return_sorted=False)
            lens = np.fromiter((len(l) for l in lists), dtype=np.int64, count=e - s)
            total = int(lens.sum())
            if total == 0:
                continue
            idx = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=total)
            obs = np.repeat(np.arange(s, e), lens)
            key = np.unique(obs * n_arcs + self.owner[idx])
            obs, arc = key // n_arcs, key % n_arcs
            d, t = point_segment_distance(
                x[obs], y[obs], net.arc_a[arc, 0], net.arc_a[arc, 1],
                net.arc_b[arc, 0], net.arc_b[arc, 1])
            keep = d <= self.radius_m
            parts.append(pd.DataFrame({"obs_idx": obs[keep], "arc_id": arc[keep],
                                       "dist_m": d[keep], "proj_frac": t[keep]}))
        if not parts:
            return pd.DataFrame({"obs_idx": pd.Series(dtype=np.int64),
                                 "arc_id": pd.Series(dtype=np.int64),
                                 "dist_m": pd.Series(dtype=float),
                                 "proj_frac": pd.Series(dtype=float)})
        df = pd.concat(parts, ignore_index=True)
        df = df.sort_values(["obs_idx", "dist_m", "arc_id"], kind="stable").reset_index(drop=True)
        rank = df.groupby("obs_idx").cumcount()
        return df[rank < self.max_candidates].reset_index(drop=True)


def assign_reference_arcs(cands: pd.DataFrame, route: np.ndarray, n_obs: int,
                          skip_cost: float = 2.0, backward_cost: float = 1000.0) -> np.ndarray:
    """Align GPS observations to the ordered reference route.

    Each observation may only take an arc that is (a) one of its candidates and
    (b) on the reference route. A dynamic programme chooses one route position
    per observation, minimising the sum of GPS-to-arc distances, with a
    near-hard penalty for moving backwards along the route and a small penalty
    (``skip_cost`` m per arc) for skipping route arcs. This enforces that the
    vehicle moves forward along the route.

    Returns ``ref_arc`` (length ``n_obs``): the aligned arc id, or -1 when none
    of the observation's candidates lies on the route (a candidate miss).
    """
    ref = np.full(n_obs, -1, dtype=np.int64)
    if len(cands) == 0 or route is None or len(route) == 0:
        return ref

    rt = pd.DataFrame({"pos": np.arange(len(route)), "arc_id": np.asarray(route, dtype=np.int64)})
    m = cands[["obs_idx", "arc_id", "dist_m"]].merge(rt, on="arc_id", how="inner")
    if m.empty:
        return ref
    m = m.sort_values(["obs_idx", "pos"]).reset_index(drop=True)
    obs_ids = m["obs_idx"].to_numpy()
    bounds = np.flatnonzero(np.diff(obs_ids)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(m)]])
    pos_all, dist_all, arc_all = m["pos"].to_numpy(), m["dist_m"].to_numpy(), m["arc_id"].to_numpy()

    steps = []                      # (obs_idx, pos[], arc[], cost[], back[])
    prev_pos = prev_cost = None
    for s, e in zip(starts, ends):
        pos, dist, arc = pos_all[s:e], dist_all[s:e], arc_all[s:e]
        if prev_pos is None:
            cost, back = dist.copy(), np.full(len(pos), -1)
        else:
            jump = pos[None, :] - prev_pos[:, None]
            trans = np.where(jump < 0, backward_cost, skip_cost * np.maximum(jump - 1, 0))
            total = prev_cost[:, None] + trans
            back = total.argmin(axis=0)
            cost = total[back, np.arange(len(pos))] + dist
        steps.append((obs_ids[s], arc, cost, back))
        prev_pos, prev_cost = pos, cost

    k = int(np.argmin(steps[-1][2]))
    for obs, arc, _, back in reversed(steps):
        ref[obs] = arc[k]
        k = back[k] if back[k] >= 0 else 0
    return ref
