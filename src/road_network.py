"""Road network built from a trip's ``.nodes`` and ``.arcs`` files.

Arcs are directed straight segments between nodes. Geometry is kept in a local
metric (projected) coordinate system so distances are in metres.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .preprocessing import LocalProjection


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from points to segments A->B (all arrays, metres) and the
    fractional position (0..1) of the closest point along the segment."""
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    t = ((px - ax) * abx + (py - ay) * aby) / np.where(denom > 0, denom, 1.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * abx, ay + t * aby
    return np.hypot(px - qx, py - qy), t


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray,
                 projection: LocalProjection | None = None):
        nodes_lonlat = np.asarray(nodes_lonlat, dtype=np.float64)
        arcs = np.asarray(arcs, dtype=np.int64)
        self.n_nodes, self.n_arcs = len(nodes_lonlat), len(arcs)
        if self.n_arcs and (arcs.min() < 0 or arcs.max() >= self.n_nodes):
            raise ValueError("arc file references node ids outside the node table")

        self.nodes_lonlat = nodes_lonlat
        self.projection = projection or LocalProjection(
            nodes_lonlat[:, 0].mean(), nodes_lonlat[:, 1].mean())
        x, y = self.projection.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.node_xy = np.column_stack([x, y])

        self.src, self.dst = arcs[:, 0], arcs[:, 1]
        self.arc_a = self.node_xy[self.src]
        self.arc_b = self.node_xy[self.dst]
        d = self.arc_b - self.arc_a
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        self.arc_bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0  # from north

        self.out_deg = np.bincount(self.src, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.dst, minlength=self.n_nodes)
        self._out_order = np.argsort(self.src, kind="stable")
        self._out_ptr = np.concatenate([[0], np.cumsum(self.out_deg)])
        self.reverse_arc = self._find_reverse_arcs()

    # ------------------------------------------------------------- topology
    def _find_reverse_arcs(self) -> np.ndarray:
        """For each arc (u->v) the id of an arc (v->u), or -1."""
        if self.n_arcs == 0:
            return np.zeros(0, dtype=np.int64)
        n = self.n_nodes
        keys = self.src * n + self.dst
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        q = self.dst * n + self.src
        pos = np.clip(np.searchsorted(sorted_keys, q), 0, len(keys) - 1)
        found = sorted_keys[pos] == q
        return np.where(found, order[pos], -1)

    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_order[self._out_ptr[node]:self._out_ptr[node + 1]]

    def successors(self, arc: int) -> np.ndarray:
        """Arcs that can be entered right after leaving ``arc``."""
        return self.out_arcs(self.dst[arc])

    def is_connected_pair(self, a: int, b: int) -> bool:
        return bool(self.dst[a] == self.src[b])

    def components(self) -> tuple[int, np.ndarray]:
        """Weakly connected components over nodes."""
        g = coo_matrix((np.ones(self.n_arcs), (self.src, self.dst)),
                       shape=(self.n_nodes, self.n_nodes))
        return connected_components(g, directed=True, connection="weak")

    def isolated_nodes(self) -> np.ndarray:
        return np.where((self.in_deg + self.out_deg) == 0)[0]

    def summary(self) -> dict:
        n_comp, labels = self.components()
        sizes = np.bincount(labels)
        return {
            "n_nodes": self.n_nodes, "n_arcs": self.n_arcs,
            "n_components": int(n_comp),
            "largest_component_share": float(sizes.max() / self.n_nodes) if self.n_nodes else 0.0,
            "n_isolated_nodes": int(len(self.isolated_nodes())),
            "share_arcs_with_reverse": float((self.reverse_arc >= 0).mean()) if self.n_arcs else 0.0,
            "arc_len_median_m": float(np.median(self.arc_len)) if self.n_arcs else 0.0,
            "arc_len_max_m": float(self.arc_len.max()) if self.n_arcs else 0.0,
        }

    def route_connected_fraction(self, route: np.ndarray) -> float:
        """Share of consecutive route arcs that are physically connected."""
        route = np.asarray(route)
        if len(route) < 2:
            return 1.0
        return float(np.mean(self.dst[route[:-1]] == self.src[route[1:]]))

    # ------------------------------------------------------------- geometry
    def arcs_to_polylines(self, arc_ids) -> list[list[list[float]]]:
        """Arc sequence -> list of [lat, lon] polylines (split where the
        sequence is not physically connected). For drawing maps."""
        arc_ids = [int(a) for a in arc_ids if a >= 0]
        polylines, current, prev = [], [], None
        for a in arc_ids:
            if prev is not None and a == prev:
                continue
            s, d = self.nodes_lonlat[self.src[a]], self.nodes_lonlat[self.dst[a]]
            if prev is None or self.dst[prev] != self.src[a]:
                if current:
                    polylines.append(current)
                current = [[float(s[1]), float(s[0])]]
            current.append([float(d[1]), float(d[0])])
            prev = a
        if current:
            polylines.append(current)
        return polylines

    def arc_segments_latlon(self, arc_ids) -> list[list[list[float]]]:
        """One [[lat,lon],[lat,lon]] segment per arc (for drawing context)."""
        out = []
        for a in arc_ids:
            s, d = self.nodes_lonlat[self.src[a]], self.nodes_lonlat[self.dst[a]]
            out.append([[float(s[1]), float(s[0])], [float(d[1]), float(d[0])]])
        return out
