# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import itertools

import numpy as np
import pytest

from rare_event import strata


def _brute_force_pmf(p: np.ndarray) -> np.ndarray:
    pmf = np.zeros(p.size + 1)
    for bits in itertools.product([0, 1], repeat=p.size):
        x = np.asarray(bits)
        pmf[x.sum()] += np.prod(np.where(x == 1, p, 1 - p))
    return pmf


def test_poisson_binomial_matches_brute_force():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.4, size=8)
    assert np.allclose(strata.poisson_binomial_pmf(p), _brute_force_pmf(p))
    assert np.allclose(strata.poisson_binomial_pmf(p, max_count=3),
                       _brute_force_pmf(p)[:4])


def test_suffix_table_row_zero_is_pmf():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.3, size=20)
    table = strata.suffix_count_table(p, max_count=6)
    assert np.allclose(table[0], strata.poisson_binomial_pmf(p, 6))
    assert table[-1, 0] == 1.0 and np.all(table[-1, 1:] == 0.0)


def test_conditional_sampler_matches_exact_conditional_law():
    rng = np.random.default_rng(3)
    p = np.array([0.02, 0.3, 0.1, 0.25, 0.05])
    k = 2
    sampler = strata.ConditionalSampler(p, max_count=3)
    shots = 60_000
    faults = sampler.sample(k, shots, rng)
    assert np.all(faults.sum(axis=1) == k)
    # Exact conditional probability of each 2-subset.
    exact = {}
    for subset in itertools.combinations(range(p.size), k):
        x = np.zeros(p.size)
        x[list(subset)] = 1
        exact[subset] = np.prod(np.where(x == 1, p, 1 - p))
    norm = sum(exact.values())
    for subset, mass in exact.items():
        empirical = np.mean(np.all(faults[:, list(subset)] == 1, axis=1))
        expected = mass / norm
        tolerance = 4 * np.sqrt(expected * (1 - expected) / shots) + 1e-3
        assert abs(empirical - expected) < tolerance, (subset, empirical,
                                                       expected)


def test_conditional_sampler_edge_counts():
    p = np.full(6, 0.1)
    sampler = strata.ConditionalSampler(p, max_count=6)
    rng = np.random.default_rng(0)
    assert sampler.sample(0, 5, rng).sum() == 0
    assert np.all(sampler.sample(6, 5, rng) == 1)
    with pytest.raises(ValueError):
        sampler.sample(7, 1, rng)


def test_cross_rate_weights_are_exact_likelihood_ratios():
    rng = np.random.default_rng(4)
    source = rng.uniform(0.01, 0.2, size=7)
    target = source * 0.37
    k = 3
    source_pmf = strata.poisson_binomial_pmf(source)
    target_pmf = strata.poisson_binomial_pmf(target)
    sampler = strata.ConditionalSampler(source, max_count=7)
    faults = sampler.sample(k, 50, rng)
    logw = strata.cross_rate_log_weights(faults, source, target, k, source_pmf,
                                         target_pmf)

    def conditional(priors, x, pmf):
        return np.prod(np.where(x == 1, priors, 1 - priors)) / pmf[k]

    expected = np.log([
        conditional(target, x, target_pmf) / conditional(source, x, source_pmf)
        for x in faults
    ])
    assert np.allclose(logw, expected)


def test_reweighted_mean_is_unbiased_in_expectation():
    """E_source[w * fail] equals E_target[fail] for any fixed indicator."""
    rng = np.random.default_rng(5)
    source = np.array([0.05, 0.2, 0.1, 0.15, 0.08, 0.3])
    target = source * 0.5
    k = 2
    indicator = lambda x: x[0] == 1 or x[5] == 1
    source_pmf = strata.poisson_binomial_pmf(source)
    target_pmf = strata.poisson_binomial_pmf(target)
    sampler = strata.ConditionalSampler(source, max_count=6)
    faults = sampler.sample(k, 200_000, rng)
    w = np.exp(
        strata.cross_rate_log_weights(faults, source, target, k, source_pmf,
                                      target_pmf))
    hits = np.array([indicator(x) for x in faults])
    estimate = np.mean(w * hits)
    exact = 0.0
    for subset in itertools.combinations(range(6), k):
        x = np.zeros(6, dtype=np.uint8)
        x[list(subset)] = 1
        if indicator(x):
            exact += np.prod(np.where(x == 1, target, 1 - target))
    exact /= target_pmf[k]
    assert abs(estimate - exact) < 0.01


def test_truncation_count_covers_mass():
    pmf = strata.poisson_binomial_pmf(np.full(50, 0.02))
    k = strata.truncation_count(pmf, tail_mass=1e-6)
    assert 1.0 - pmf[:k + 1].sum() <= 1e-6


def _sequential_reference(p: np.ndarray, count: int, shots: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Chen-Liu sequential sampler (one decision per mechanism), kept as
    the reference law for the inverse-transform sampler."""
    table = strata.suffix_count_table(p, count)
    faults = np.zeros((shots, p.size), dtype=np.uint8)
    remaining = np.full(shots, count)
    for j in range(p.size):
        active = remaining > 0
        denominator = table[j, remaining]
        numerator = p[j] * table[j + 1, np.maximum(remaining - 1, 0)]
        with np.errstate(divide="ignore", invalid="ignore"):
            include = np.where(active & (denominator > 0),
                               numerator / denominator, 0.0)
        hit = rng.random(shots) < include
        faults[hit, j] = 1
        remaining[hit] -= 1
    assert np.all(remaining == 0)
    return faults


def test_conditional_sampler_matches_sequential_reference():
    rng = np.random.default_rng(11)
    # Unequal odds, a zero-probability mechanism and a near-certain one.
    p = np.array([0.001, 0.0, 0.2, 0.05, 0.9, 0.01, 0.3, 0.002])
    k = 3
    shots = 100_000
    sampler = strata.ConditionalSampler(p, max_count=k)
    fast = sampler.sample(k, shots, rng)
    reference = _sequential_reference(p, k, shots, rng)
    assert np.all(fast.sum(axis=1) == k)
    assert fast[:, 1].sum() == 0
    # Per-mechanism inclusion frequencies agree within sampling error.
    f_fast = fast.mean(axis=0)
    f_ref = reference.mean(axis=0)
    tolerance = 5 * np.sqrt(f_ref * (1 - f_ref) / shots) + 1e-9
    assert np.all(np.abs(f_fast - f_ref) < tolerance)
    # The cumulative tables end at the count masses.
    assert np.allclose(sampler._cdf[1:, -1], sampler.pmf[1:])
