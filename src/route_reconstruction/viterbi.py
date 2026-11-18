"""Generic Viterbi decoder (log domain).

``log_emissions[t]`` is a 1-D array with one score per candidate state at step
``t``. ``log_transition(t)`` returns the (K_{t-1} x K_t) matrix of log
transition scores from the states of step t-1 to the states of step t.

    delta_0(j) = e_0(j)
    delta_t(j) = e_t(j) + max_i [ delta_{t-1}(i) + A_t(i, j) ]
    path is recovered by back-pointers from argmax_j delta_{T-1}(j)
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def viterbi(log_emissions: Sequence[np.ndarray],
            log_transition: Callable[[int], np.ndarray]) -> list[int]:
    T = len(log_emissions)
    if T == 0:
        return []
    delta = np.asarray(log_emissions[0], dtype=float)
    back = []
    for t in range(1, T):
        scores = delta[:, None] + log_transition(t)
        bp = scores.argmax(axis=0)
        delta = scores[bp, np.arange(scores.shape[1])] + log_emissions[t]
        back.append(bp)
    path = [int(delta.argmax())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    return path[::-1]
