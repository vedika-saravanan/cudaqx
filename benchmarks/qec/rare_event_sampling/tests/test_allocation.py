# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import numpy as np
import pytest

from rare_event import allocation


def test_empty_pilot_stratum_shrinks_to_pooled_rate_not_pseudo_failure():
    # Two strata: one with a measured rate, one large and empty.  The empty
    # stratum's sigma must track the pooled rate, otherwise a whole
    # pseudo-failure would hand it most of a Neyman allocation.
    failures = np.array([20.0, 0.0])
    trials = np.array([5000.0, 5000.0])
    weights = np.array([0.1, 0.9])
    f = allocation.shrunk_rates(failures, trials, weights=weights)
    pooled = 0.1 * 20.0 / 5000.0
    m = allocation.PILOT_PSEUDO_TRIALS
    assert f[1] == pytest.approx(pooled * m / (trials[1] + m), rel=1e-12)
    assert f[1] < 1.0 / 5000.0
    sigmas = allocation.stratum_sigmas(failures, trials, weights=weights)
    assert sigmas[1] < sigmas[0] / 5


def test_fixed_weight_estimate_is_unbiased_under_adaptive_rounds():
    # Rare-failure regime like the BB72 onset stratum: a heavy never-failing
    # family and a few failing groups.  The fixed-weight estimate must be
    # unbiased and its per-round plug-in must match the empirical variance;
    # a pooled-rate plug-in sits well below it here.
    rng = np.random.default_rng(11)
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    rates = np.array([0.0, 0.0, 3e-4, 3e-3])
    families = np.array(["spread", "spread", "compact", "compact"])
    truth = float(weights @ rates)
    pilot, round_budget, rounds, floor = 3000, 30000, 3, 100
    estimates, variances, pooled_plugin = [], [], []
    for _ in range(300):
        labels = [np.full(4, pilot)]
        failures = [rng.binomial(pilot, rates)]
        for _ in range(rounds):
            n = allocation.round_labels(round_budget,
                                        weights,
                                        np.sum(failures, axis=0),
                                        np.sum(labels, axis=0),
                                        floor,
                                        pseudo=0.2 * pilot,
                                        families=families,
                                        floor_share=0.2)
            labels.append(n)
            failures.append(rng.binomial(n, rates))
        est, var = allocation.fixed_weight_estimate(weights, labels[1:],
                                                    failures[1:])
        estimates.append(est)
        variances.append(var)
        L = np.array(labels[1:], float)
        F = np.array(failures[1:], float)
        pooled = F.sum(0) / L.sum(0)
        pooled_plugin.append(
            float(
                np.sum(weights**2 * (pooled * (1 - pooled) / L).sum(0) /
                       rounds**2)))
    estimates = np.array(estimates)
    se = estimates.std(ddof=1) / np.sqrt(estimates.size)
    assert abs(estimates.mean() - truth) < 3 * se
    empirical = estimates.var(ddof=1)
    assert np.mean(variances) == pytest.approx(empirical, rel=0.2)
    assert np.mean(pooled_plugin) < np.mean(variances)
    # Budget concentrates on the failing groups.
    assert labels[-1][0] + labels[-1][1] < 0.5 * round_budget


def test_fixed_weight_variance_bound_exceeds_plugin_for_empty_strata():
    weights = np.array([0.9, 0.1])
    labels = [np.array([1000.0, 1000.0]), np.array([5000.0, 1000.0])]
    failures = [np.array([0.0, 10.0]), np.array([0.0, 12.0])]
    _, var = allocation.fixed_weight_estimate(weights, labels, failures)
    bound = allocation.fixed_weight_variance_bound(weights, labels, failures)
    assert bound > var
    # Stratum 0 at rule-of-three rate 3/6000 in both rounds.
    b = 3.0 / 6000.0
    extra = 0.9**2 * (b * (1 - b) / 1000.0 + b * (1 - b) / 5000.0) / 4
    assert bound - var == pytest.approx(extra, rel=1e-9)


def test_family_shrinkage_pools_evidence_of_a_failure_free_family():
    # Four strata: two 'compact' with failures, two large 'spread' with
    # none.  Flat shrinkage reads each empty stratum at the global rate
    # times m / (n + m); family shrinkage reads it at the family's pooled
    # rate, which all the family's trials have driven down.
    failures = np.array([30.0, 20.0, 0.0, 0.0])
    trials = np.array([5000.0, 5000.0, 50000.0, 50000.0])
    weights = np.array([0.1, 0.1, 0.4, 0.4])
    families = np.array(["compact", "compact", "spread", "spread"])
    flat = allocation.shrunk_rates(failures, trials, weights=weights)
    hier = allocation.shrunk_rates(failures,
                                   trials,
                                   weights=weights,
                                   families=families)
    assert hier[2] < flat[2] / 10
    # Failing strata are barely affected.
    assert hier[0] == pytest.approx(flat[0], rel=0.15)
    # A family that fails keeps its rate.
    families2 = np.array(["a", "b", "a", "b"])
    hier2 = allocation.shrunk_rates(failures,
                                    trials,
                                    weights=weights,
                                    families=families2)
    assert hier2[2] > hier[2]


def test_round_labels_proportional_floor_bounds_starvation():
    weights = np.array([0.5, 0.5])
    # Stratum 0 looks empty after the pilot, stratum 1 fails often.
    failures = np.array([0.0, 50.0])
    trials = np.array([1000.0, 1000.0])
    bare = allocation.round_labels(10000,
                                   weights,
                                   failures,
                                   trials,
                                   10,
                                   pseudo=50.0)
    floored = allocation.round_labels(10000,
                                      weights,
                                      failures,
                                      trials,
                                      10,
                                      pseudo=50.0,
                                      floor_share=0.3)
    assert bare[0] < 0.3 * 0.5 * 10000
    assert floored[0] == int(0.3 * 0.5 * 10000)
    assert floored[1] == bare[1]


def test_variance_bound_pools_family_trials():
    weights = np.array([0.45, 0.45, 0.1])
    labels = [np.array([1000.0, 1000.0, 1000.0])]
    failures = [np.array([0.0, 0.0, 10.0])]
    per_group = allocation.fixed_weight_variance_bound(weights, labels,
                                                       failures)
    pooled = allocation.fixed_weight_variance_bound(weights,
                                                    labels,
                                                    failures,
                                                    families=np.array(
                                                        ["s", "s", "c"]))
    assert pooled < per_group
    # Empty strata at 3/2000 instead of 3/1000.
    b1, b2 = 3.0 / 1000.0, 3.0 / 2000.0
    diff = 2 * 0.45**2 * (b1 * (1 - b1) - b2 * (1 - b2)) / 1000.0
    assert per_group - pooled == pytest.approx(diff, rel=1e-9)


def test_zero_group_break_even_rate():
    weights = np.array([0.8, 0.2])
    labels = [np.array([2000.0, 8000.0]), np.array([2000.0, 8000.0])]
    failures = [np.array([0.0, 40.0]), np.array([0.0, 40.0])]
    _, var = allocation.fixed_weight_estimate(weights, labels, failures)
    plain = 4.0 * var
    b = allocation.zero_group_break_even(weights, labels, failures, plain)
    slope = 0.8**2 * (1 / 2000.0 + 1 / 2000.0) / 4
    assert b == pytest.approx((plain - var) / slope, rel=1e-9)
    assert allocation.zero_group_break_even(weights, labels, failures,
                                            var / 2) == 0.0
    no_empty = [np.array([5.0, 40.0]), np.array([3.0, 40.0])]
    assert allocation.zero_group_break_even(weights, labels, no_empty,
                                            plain) == float("inf")


def test_proportional_pilot_respects_floor_and_mass():
    weights = np.array([0.6, 0.39, 0.01])
    pilot = allocation.proportional_pilot(weights, 10000, 200)
    assert pilot[0] == 6000 and pilot[1] == 3900 and pilot[2] == 200


def test_certified_variance_charges_empty_strata_and_uses_certification():
    weights = np.array([0.5, 0.3, 0.2])
    families = np.array(["s", "s", "c"])
    labels = [np.array([1000.0, 1000.0, 2000.0])]
    failures = [np.array([0.0, 0.0, 20.0])]
    _, plugin = allocation.fixed_weight_estimate(weights, labels, failures)
    out = allocation.certified_variance(weights, labels, failures, families)
    # Family 's': 2000 failure-free labels -> bound 3/2000 on both strata.
    b = 3.0 / 2000.0
    expected = plugin + (0.5**2 + 0.3**2) * b * (1 - b) / 1000.0
    assert out["mixture"] == pytest.approx(expected, rel=1e-9)
    assert out["worst_case"] >= out["mixture"]
    assert out["empty_mass"] == pytest.approx(0.8)
    # Certification labels tighten the bound; a certification failure turns
    # the stratum into a measured one at its certification rate.
    tighter = allocation.certified_variance(weights, labels, failures, families,
                                            np.array([6000, 0, 0]),
                                            np.array([0, 0, 0]))
    assert tighter["mixture"] < out["mixture"]
    measured = allocation.certified_variance(weights, labels, failures,
                                             families, np.array([6000, 0, 0]),
                                             np.array([3, 0, 0]))
    r = 3.0 / 6000.0
    b2 = 3.0 / 1000.0  # only stratum 1 is still empty in family 's'
    expected2 = (plugin + 0.5**2 * r * (1 - r) / 1000.0 + 0.3**2 * b2 *
                 (1 - b2) / 1000.0)
    assert measured["mixture"] == pytest.approx(expected2, rel=1e-9)


def test_certification_floor_targets_family_bound_term():
    weights = np.array([0.5, 0.3, 0.15, 0.05])
    families = np.array(["s", "s", "c", "d"])
    empty = np.array([True, True, True, False])
    f_hat, N, eps, rounds = 1e-4, 600000, 0.1, 3
    # Allowance split over the two empty families by W^(2/3).
    W_s, W_c = 0.8, 0.15
    shares = np.array([W_s, W_c])**(2 / 3)
    shares /= shares.sum()
    floors = allocation.certification_floor(weights, families, empty, f_hat, N,
                                            eps, rounds, 1, np.zeros(4))
    target_s = W_s * np.sqrt(3.0 * N / (eps * shares[0] * f_hat)) / 3
    target_c = W_c * np.sqrt(3.0 * N / (eps * shares[1] * f_hat)) / 3
    assert floors[3] == 0
    assert floors[0] == int(np.floor(target_s * 0.5 / W_s))
    assert floors[1] == int(np.floor(target_s * 0.3 / W_s))
    assert floors[2] == int(np.floor(target_c))
    # The sum of the family bound terms at the full targets is eps * f / N.
    N_s, N_c = 3 * target_s, 3 * target_c
    total = 3.0 * W_s**2 / N_s**2 + 3.0 * W_c**2 / N_c**2
    assert total == pytest.approx(eps * f_hat / N, rel=1e-9)
    # Labels already present reduce the shortfall; a met target gives zero.
    met = allocation.certification_floor(weights, families, empty, f_hat, N,
                                         eps, rounds, 1,
                                         np.array([1e9, 1e9, 1e9, 0.0]))
    assert met.sum() == 0
    # The floor enters round_labels as a minimum.
    n = allocation.round_labels(10000,
                                weights,
                                np.array([0.0, 0.0, 0.0, 5.0]),
                                np.array([1000.0, 1000.0, 1000.0, 1000.0]),
                                10,
                                min_labels=floors)
    assert np.all(n >= floors)


def test_optimal_certify_fraction_balances_bound_and_plug_in():
    weights = np.array([0.5, 0.3, 0.2])
    families = np.array(["a", "a", "b"])
    sigmas = np.sqrt(np.array([1e-3, 1e-4, 1e-4]))
    # Nothing to certify.
    assert allocation.optimal_certify_fraction(weights, families,
                                               np.zeros(3, bool), sigmas, 5e-4,
                                               100_000, 10_000) is None
    empty = np.array([False, False, True])
    N = 100_000
    f = 5e-4
    eps = allocation.optimal_certify_fraction(weights, families, empty, sigmas,
                                              f, N, 10_000)
    assert eps is not None and 1e-3 <= eps <= 10
    # The chosen point is within a grid step of the continuous optimum.
    A = float(np.dot(weights, sigmas))**2
    W = weights[empty].sum()
    cost = W * np.sqrt(allocation.ZERO_FAILURE_BOUND_EVENTS * N / f)

    def projected(e):
        return A / (N - cost / np.sqrt(e)) + e * f / N

    fine = np.logspace(-3, 1, 4001)
    fine = fine[N - cost / np.sqrt(fine) >= 0.25 * N]
    best = fine[np.argmin([projected(e) for e in fine])]
    assert abs(np.log(eps / best)) < np.log(10) / 40 + 1e-9
    # A heavier never-failing family asks for a larger allowance (it is
    # dearer to certify), a cheaper plug-in side for a smaller one.
    heavier = allocation.optimal_certify_fraction(np.array([0.2, 0.3,
                                                            0.5]), families,
                                                  empty, sigmas, f, N, 10_000)
    assert heavier >= eps
    # A budget too small to afford any allowance up to 10 still returns the
    # loosest affordable one instead of giving up.
    tiny_rate = 5e-6
    tiny = allocation.optimal_certify_fraction(weights, families, empty, sigmas,
                                               tiny_rate, 200, 1_000_000)
    cost_tiny = W * np.sqrt(
        allocation.ZERO_FAILURE_BOUND_EVENTS * 200 / tiny_rate)
    assert cost_tiny / np.sqrt(10.0) > 0.75 * 200  # 10 is unaffordable
    assert tiny is not None and tiny > 10
    assert 200 - cost_tiny / np.sqrt(tiny) >= 0.25 * 200 - 1e-6
    # No failure yet: the rate floor keeps the rule defined.
    assert allocation.optimal_certify_fraction(weights, families, empty, sigmas,
                                               0.0, N, 10_000) > 0
