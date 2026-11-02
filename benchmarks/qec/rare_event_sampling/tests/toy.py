# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tiny exactly-solvable model shared by the tests.

The model is small enough (``N`` mechanisms, ``N <= 14``) to enumerate all
``2**N`` fault configurations, so the exact logical error rate, exact
``f(k)`` and the exact set of failing configurations are available to check
every estimator against.
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

from rare_event.decoders import DecodeBatch
from rare_event.dem import DetectorModel


def toy_model(num_mechanisms: int = 10,
              num_detectors: int = 5,
              num_observables: int = 1,
              rate: float = 0.05,
              seed: int = 7) -> DetectorModel:
    """Random sparse DEM with unequal priors proportional to ``rate``."""
    rng = np.random.default_rng(seed)
    H = np.zeros((num_detectors, num_mechanisms), dtype=np.uint8)
    O = np.zeros((num_observables, num_mechanisms), dtype=np.uint8)
    for j in range(num_mechanisms):
        H[rng.choice(num_detectors, size=2, replace=False), j] = 1
        if rng.random() < 0.5:
            O[rng.integers(num_observables), j] = 1
    scales = rng.uniform(0.2, 1.0, size=num_mechanisms)
    return DetectorModel(H=sp.csr_matrix(H),
                         O=sp.csr_matrix(O),
                         priors=rate * scales)


def rescaled(model: DetectorModel, factor: float) -> DetectorModel:
    return DetectorModel(H=model.H, O=model.O, priors=model.priors * factor)


class MaxLikelihoodDecoder:
    """Exact most-likely-error decoder by enumeration (test oracle only)."""

    def __init__(self, model: DetectorModel):
        n = model.num_mechanisms
        self.all_faults = np.array(list(itertools.product([0, 1], repeat=n)),
                                   dtype=np.uint8)
        log_prob = (self.all_faults @ np.log(model.priors) +
                    (1 - self.all_faults) @ np.log1p(-model.priors))
        syndromes = model.syndromes(self.all_faults)
        observables = model.observables(self.all_faults)
        self.table: dict[bytes, np.ndarray] = {}
        best: dict[bytes, float] = {}
        for s, o, lp in zip(syndromes, observables, log_prob):
            key = s.tobytes()
            if lp > best.get(key, -np.inf):
                best[key] = lp
                self.table[key] = o
        self.num_observables = model.num_observables

    def decode_batch(self, syndromes: np.ndarray) -> DecodeBatch:
        predicted = np.stack([self.table[s.tobytes()] for s in syndromes])
        return DecodeBatch(observables=predicted,
                           converged=np.ones(syndromes.shape[0], dtype=bool))


def exact_failure_table(model: DetectorModel,
                        decoder: MaxLikelihoodDecoder) -> np.ndarray:
    """Boolean failure flag for every enumerated configuration."""
    faults = decoder.all_faults
    truth = model.observables(faults)
    predicted = decoder.decode_batch(model.syndromes(faults)).observables
    return np.any(predicted != truth, axis=1)


def exact_logical_error(model: DetectorModel, decoder: MaxLikelihoodDecoder,
                        failing: np.ndarray) -> float:
    faults = decoder.all_faults
    log_prob = (faults @ np.log(model.priors) +
                (1 - faults) @ np.log1p(-model.priors))
    return float(np.exp(log_prob[failing]).sum())


def exact_spectrum(model: DetectorModel, decoder: MaxLikelihoodDecoder,
                   failing: np.ndarray) -> dict[int, float]:
    """Exact ``f(k) = P(fail | K = k)`` at the model's priors."""
    faults = decoder.all_faults
    log_prob = (faults @ np.log(model.priors) +
                (1 - faults) @ np.log1p(-model.priors))
    weights = faults.sum(axis=1)
    result = {}
    for k in np.unique(weights):
        mask = weights == k
        total = np.exp(log_prob[mask]).sum()
        result[int(k)] = float(np.exp(log_prob[mask & failing]).sum() / total)
    return result
