# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exact fault-count stratification for unequal mechanism probabilities.

With independent mechanisms ``x_j ~ Bernoulli(p_j)`` the fault count
``K = sum_j x_j`` is Poisson-binomial.  The logical error rate decomposes as

    LER = sum_k P(K = k) * f(k),      f(k) = P(fail | K = k).

IBM's technique I assumes equal probabilities so ``P(K=k)`` is binomial and
``K=k`` subsets are uniform.  Here both pieces are exact for unequal ``p_j``:
``P(K=k)`` comes from the standard dynamic programme and conditional subsets
are drawn position by position by inverse transform on cumulative tables
built from the suffix tables of the same programme, so a batch costs
``k`` vectorised searches instead of one Python step per mechanism.

The conditional law of the *subset* given ``K=k`` depends on the mechanism
odds ``o_j = p_j / (1 - p_j)`` only through their ratios, so for a
single-parameter noise family (every ``p_j`` roughly proportional to ``p``)
``f(k)`` barely changes with ``p``.  :func:`cross_rate_log_weights` gives the
exact importance weight that transports samples drawn at one rate to another,
so the approximation never has to be trusted blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Stop extending the Poisson-binomial table once the remaining mass is below
# this fraction; the omitted tail is reported, never silently dropped.
DEFAULT_TAIL_MASS = 1e-15


def poisson_binomial_pmf(probabilities: np.ndarray,
                         max_count: int | None = None) -> np.ndarray:
    """``P(K = k)`` for ``k = 0..max_count`` by the exact dynamic programme.

    ``max_count`` defaults to the full support ``len(probabilities)``.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("probabilities must be a vector")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    kmax = p.size if max_count is None else int(max_count)
    if kmax < 0:
        raise ValueError("max_count must be non-negative")
    kmax = min(kmax, p.size)
    pmf = np.zeros(kmax + 1, dtype=np.float64)
    pmf[0] = 1.0
    for pj in p:
        # pmf[k] <- pmf[k] (1 - pj) + pmf[k-1] pj, computed in place from the
        # top so pmf[k-1] is still the previous value.
        pmf[1:] = pmf[1:] * (1.0 - pj) + pmf[:-1] * pj
        pmf[0] *= 1.0 - pj
    return pmf


def suffix_count_table(probabilities: np.ndarray, max_count: int) -> np.ndarray:
    """``T[j, k] = P(exactly k of mechanisms j..N-1 fire)`` for ``k <= max_count``.

    Shape ``(N + 1, max_count + 1)``; row ``N`` is the empty suffix.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    n = p.size
    table = np.zeros((n + 1, max_count + 1), dtype=np.float64)
    table[n, 0] = 1.0
    for j in range(n - 1, -1, -1):
        nxt = table[j + 1]
        table[j, 0] = nxt[0] * (1.0 - p[j])
        table[j, 1:] = nxt[1:] * (1.0 - p[j]) + nxt[:-1] * p[j]
    return table


@dataclass
class ConditionalSampler:
    """Exact sampler of mechanism subsets conditioned on the fault count.

    Positions are drawn in increasing order.  With ``r`` faults still to
    place and every mechanism before ``i`` decided, the next fault sits at
    ``j >= i`` with probability

        prod_{i <= m < j} (1 - p_m) * p_j * T[j+1, r-1] / T[i, r],

    where ``T`` is the suffix count table.  Multiplying numerator and
    denominator by ``prod_{m < i} (1 - p_m)`` turns this into a difference
    of one cumulative table ``C_r`` per remaining count, so each step is a
    vectorised ``searchsorted`` over the batch: ``k`` numpy calls per
    batch instead of a Python step per mechanism.  The law is identical to
    the sequential (Chen-Liu) scheme; only the order of the random draws
    differs.
    """

    probabilities: np.ndarray
    max_count: int

    def __post_init__(self) -> None:
        self.probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if self.max_count < 0 or self.max_count > self.probabilities.size:
            raise ValueError("max_count outside the mechanism count")
        p = self.probabilities
        n = p.size
        self._table = suffix_count_table(p, self.max_count)
        # clear[j] = P(no fault among mechanisms 0..j-1).
        clear = np.concatenate(([1.0], np.cumprod(1.0 - p)))
        # cdf[r, j+1] = P(r faults in total and the first one at position
        # <= j); cdf[r, n] = P(K = r).
        self._cdf = np.zeros((self.max_count + 1, n + 1), dtype=np.float64)
        for r in range(1, self.max_count + 1):
            weights = clear[:-1] * p * self._table[1:, r - 1]
            self._cdf[r, 1:] = np.cumsum(weights)

    @property
    def pmf(self) -> np.ndarray:
        """``P(K = k)`` for ``k = 0..max_count`` (row 0 of the suffix table)."""
        return self._table[0].copy()

    def sample(self, count: int, shots: int,
               rng: np.random.Generator) -> np.ndarray:
        """Draw ``shots`` fault rows, each with exactly ``count`` faults."""
        if not 0 <= count <= self.max_count:
            raise ValueError("count must lie in [0, max_count]")
        if shots < 0:
            raise ValueError("shots must be non-negative")
        n = self.probabilities.size
        faults = np.zeros((shots, n), dtype=np.uint8)
        if shots == 0 or count == 0:
            return faults
        rows = np.arange(shots)
        start = np.zeros(shots, dtype=np.int64)
        uniforms = rng.random((count, shots))
        for step in range(count):
            cdf = self._cdf[count - step]
            low = cdf[start]
            target = low + uniforms[step] * (cdf[n] - low)
            position = np.searchsorted(cdf, target, side="right") - 1
            position = np.minimum(np.maximum(position, start), n - 1)
            faults[rows, position] = 1
            start = position + 1
        if np.any(faults.sum(axis=1) != count):
            raise RuntimeError("conditional sampler did not place all faults")
        return faults


def log_odds(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("probabilities must lie strictly inside (0, 1)")
    return np.log(p) - np.log1p(-p)


def cross_rate_log_weights(faults: np.ndarray, source_priors: np.ndarray,
                           target_priors: np.ndarray, count: int,
                           source_pmf: np.ndarray,
                           target_pmf: np.ndarray) -> np.ndarray:
    """``log P_target(E | K=k) - log P_source(E | K=k)`` for fault rows ``E``.

    Exact for independent mechanisms:
    ``P(E | K=k) = prod_{j in E} o_j * prod_j (1 - p_j) / P(K=k)``.
    """
    faults = np.asarray(faults)
    delta_log_odds = log_odds(target_priors) - log_odds(source_priors)
    constant = (float(np.log1p(-np.asarray(target_priors)).sum()) -
                float(np.log1p(-np.asarray(source_priors)).sum()) +
                np.log(source_pmf[count]) - np.log(target_pmf[count]))
    return faults.astype(np.float64) @ delta_log_odds + constant


def truncation_count(pmf: np.ndarray,
                     tail_mass: float = DEFAULT_TAIL_MASS) -> int:
    """Smallest ``k`` such that ``P(K > k) <= tail_mass``."""
    cumulative = np.cumsum(pmf)
    above = np.flatnonzero(1.0 - cumulative <= tail_mass)
    return int(above[0]) if above.size else int(pmf.size - 1)
