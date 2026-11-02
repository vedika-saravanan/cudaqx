# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Minimal failing sets by descent from observed failures.

IBM's technique II counts *all* minimum-weight failing sets exactly, which
needs a logical-operator enumeration we do not reproduce here.  This module
provides the cheap, decoder-driven half of that idea: shrink each observed
failing configuration until no single fault can be removed without the
decoder succeeding.  The smallest weight found is an upper bound on the
failure onset ``w0`` (the true minimum failing weight is at most this), and
the reduced sets expose the structure of failures and make good chain seeds.
"""

from __future__ import annotations

import numpy as np

from .decoders import FailureOracle


def reduce_failing_set(oracle: FailureOracle, faults: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """Greedy single-removal descent to a locally minimal failing set.

    Every round decodes all single-fault removals in one batch and keeps a
    random one that still fails, so the result does not depend on mechanism
    index order.
    """
    current = np.asarray(faults, dtype=np.uint8).copy()
    if current.ndim != 1:
        raise ValueError("faults must be a single fault row")
    if not bool(oracle.failures(current[None, :])[0]):
        raise ValueError("the supplied configuration does not fail")
    while True:
        active = np.flatnonzero(current)
        if active.size == 0:
            return current
        candidates = np.repeat(current[None, :], active.size, axis=0)
        candidates[np.arange(active.size), active] = 0
        still_failing = np.flatnonzero(oracle.failures(candidates))
        if still_failing.size == 0:
            return current
        current = candidates[rng.choice(still_failing)]


def minimal_failing_sets(oracle: FailureOracle, failing: np.ndarray,
                         rng: np.random.Generator) -> dict[str, object]:
    """Reduce every row of ``failing`` and summarise the weights found."""
    reduced = np.stack(
        [reduce_failing_set(oracle, row, rng) for row in failing])
    weights = reduced.sum(axis=1)
    unique = np.unique(reduced, axis=0)
    return {
        "reduced": reduced,
        "weights": weights,
        "onset_upper_bound": int(weights.min()),
        "distinct_minimal_sets": int(unique.shape[0]),
        "weight_histogram": {
            int(w): int(c)
            for w, c in zip(*np.unique(weights, return_counts=True))
        },
    }
