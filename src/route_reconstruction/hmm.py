"""HMM route reconstruction on top of classifier probabilities.

Hidden states   : the candidate arcs of each GPS observation (one state set per
                  observation; state = arc id, so the candidate-to-arc mapping
                  is preserved).
Emission        : log of the classifier probability of the candidate, first
                  normalised within the observation, times ``emission_weight``.
Transition      : depends on how the previous arc connects to the next one in
                  the road network (forward BFS up to ``max_hops`` arcs):
                    same arc               cost 0
                    reachable in h hops    cost hop_cost * h  (+ uturn_cost if
                                           the next arc is the reverse of the previous)
                    not reachable          cost far_cost
                  log transition = -cost (an unnormalised score; it is only
                  compared across candidates of the same step).
Missing candidates: observations with no candidates are skipped by the decoder
                  and (``fill_empty``) receive the nearest decoded arc.
GPS gaps        : if the time gap between two decoded observations exceeds
                  ``gap_seconds`` the transition matrix is flat (zeros), so the
                  decoder restarts from the emissions instead of forcing
                  connectivity across an unobserved stretch.
Disconnected    : consecutive observations that only fit disconnected arcs pay
                  ``far_cost`` instead of being forbidden, so a route is always
                  returned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..road_network import RoadNetwork
from .viterbi import viterbi


@dataclass
class HMMConfig:
    emission_weight: float = 1.0
    hop_cost: float = 1.0
    far_cost: float = 8.0
    uturn_cost: float = 3.0
    max_hops: int = 3
    gap_seconds: float = 10.0
    prob_floor: float = 1e-6
    fill_empty: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class RouteHMM:
    def __init__(self, network: RoadNetwork, config: HMMConfig | None = None):
        self.net = network
        self.cfg = config or HMMConfig()
        self._reach_cache: dict[int, dict[int, int]] = {}

    def _reach(self, arc: int) -> dict[int, int]:
        """arc -> hops (0 = itself) for arcs reachable going forward."""
        hit = self._reach_cache.get(arc)
        if hit is not None:
            return hit
        hops = {arc: 0}
        frontier = [arc]
        for h in range(1, self.cfg.max_hops + 1):
            nxt = []
            for a in frontier:
                for b in self.net.successors(a):
                    b = int(b)
                    if b not in hops:
                        hops[b] = h
                        nxt.append(b)
            frontier = nxt
        self._reach_cache[arc] = hops
        return hops

    def _transition(self, prev_arcs, cur_arcs, dt: float) -> np.ndarray:
        cfg = self.cfg
        if dt > cfg.gap_seconds:
            return np.zeros((len(prev_arcs), len(cur_arcs)))
        cost = np.full((len(prev_arcs), len(cur_arcs)), cfg.far_cost)
        for i, a in enumerate(prev_arcs):
            reach = self._reach(int(a))
            rev = self.net.reverse_arc[a]
            for j, b in enumerate(cur_arcs):
                h = reach.get(int(b))
                if h is not None:
                    cost[i, j] = cfg.hop_cost * h + (cfg.uturn_cost if b == rev else 0.0)
        return -cost

    def decode(self, cands: pd.DataFrame, proba: np.ndarray, obs_t: np.ndarray) -> np.ndarray:
        """Decode one trip.

        cands : rows with ``obs_idx`` and ``arc_id`` (one trip);
        proba : classifier probability per row; obs_t : timestamp per observation.
        Returns the decoded arc id for every observation (-1 where undecidable).
        """
        n_obs = len(obs_t)
        out = np.full(n_obs, -1, dtype=np.int64)
        if len(cands) == 0:
            return out
        cfg = self.cfg
        df = pd.DataFrame({"o": cands["obs_idx"].to_numpy(), "a": cands["arc_id"].to_numpy(),
                           "p": np.maximum(np.asarray(proba, dtype=float), cfg.prob_floor)})
        df["p"] = df["p"] / df.groupby("o")["p"].transform("sum")
        df = df.sort_values(["o"], kind="stable")
        steps, arcs_l, em_l = [], [], []
        for o, g in df.groupby("o", sort=True):
            steps.append(int(o))
            arcs_l.append(g["a"].to_numpy())
            em_l.append(cfg.emission_weight * np.log(g["p"].to_numpy()))

        def trans(k):
            dt = obs_t[steps[k]] - obs_t[steps[k - 1]]
            return self._transition(arcs_l[k - 1], arcs_l[k], dt)

        path = viterbi(em_l, trans)
        for o, arcs, s in zip(steps, arcs_l, path):
            out[o] = arcs[s]

        if cfg.fill_empty:
            s = pd.Series(np.where(out >= 0, out, np.nan))
            out = s.ffill().bfill().fillna(-1).astype(np.int64).to_numpy()
        return out


def tune_hmm(trip_inputs, n_iter: int = 30, seed: int = 42, base: HMMConfig | None = None):
    """Random search of decoder parameters (use training / student-test trips only).

    trip_inputs: list of dicts with keys network, cands, proba, obs (obs has
    ``t`` and ``ref_arc``). Objective: mean observation-level top-1 accuracy
    (reference misses count as wrong). Returns (best_config, trials DataFrame).
    """
    rng = np.random.default_rng(seed)
    base = base or HMMConfig()

    def score(cfg):
        accs = []
        for ti in trip_inputs:
            pred = RouteHMM(ti["network"], cfg).decode(ti["cands"], ti["proba"], ti["obs"]["t"].to_numpy())
            ref = ti["obs"]["ref_arc"].to_numpy()
            accs.append(np.mean((pred == ref) & (ref >= 0)))
        return float(np.mean(accs))

    rows = []
    for i in range(n_iter):
        cfg = base if i == 0 else HMMConfig(
            emission_weight=float(rng.choice([0.5, 1.0, 2.0, 3.0])),
            hop_cost=float(rng.choice([0.25, 0.5, 1.0, 2.0])),
            far_cost=float(rng.choice([3.0, 5.0, 8.0, 12.0, 20.0])),
            uturn_cost=float(rng.choice([0.0, 1.0, 3.0, 6.0])),
            max_hops=int(rng.choice([2, 3, 4])),
            gap_seconds=float(rng.choice([5.0, 10.0, 20.0])))
        rows.append({**cfg.to_dict(), "top1_acc": score(cfg), "trial": i})
    trials = pd.DataFrame(rows)
    best = trials.sort_values("top1_acc", ascending=False).iloc[0]
    best_cfg = HMMConfig(**{k: (int(best[k]) if k == "max_hops" else (bool(best[k]) if k == "fill_empty" else float(best[k])))
                            for k in HMMConfig().to_dict()})
    return best_cfg, trials
