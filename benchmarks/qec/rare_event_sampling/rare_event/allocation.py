# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Neyman allocation of decoder budget across fault-count strata.

For target rates ``p`` with stratum masses ``P_k(p)`` and conditional
failure rates ``f_k`` (independent of ``p`` to first order), the variance
of ``LER(p) = sum_k P_k f_hat_k`` at labels ``n_k`` is
``sum_k P_k^2 f_k (1 - f_k) / n_k``.  Minimising the total labels subject to
a relative standard error ``eps`` at every target rate is a small convex
programme; for one target rate the solution is Neyman's
``n_k proportional to P_k sqrt(f_k (1 - f_k))``.

The two-stage estimator (pilot, then Neyman) stays unbiased when the second
stage's allocation is fixed before its outcomes are observed, which is how
:func:`two_stage_plan` is meant to be used.  With several adaptive rounds
the pooled ratio ``sum F / sum n`` is biased (the labels of a stratum depend
on its earlier failures); :func:`fixed_weight_estimate` combines the
per-round means with weights fixed in advance and stays exactly unbiased.
"""

from __future__ import annotations

import math

import numpy as np

# Pseudo-trials at the pooled failure rate mixed into every stratum's pilot
# counts: an unlucky zero in a stratum of n pilot shots is read as rate
# pooled * m / (n + m) instead of zero, so the stratum keeps a share of the
# next allocation without a whole pseudo-failure handing it the budget.
PILOT_PSEUDO_TRIALS = 1000.0
# Smallest share of the budget any measured stratum keeps (guards the
# allocation against a badly estimated pilot sigma).
EXPLORATION_FLOOR = 0.01


def shrunk_rates(pilot_failures: np.ndarray,
                 pilot_trials: np.ndarray,
                 pseudo: float = PILOT_PSEUDO_TRIALS,
                 weights: np.ndarray | None = None,
                 families: np.ndarray | None = None) -> np.ndarray:
    """Pilot failure rates shrunk toward a pooled rate.

    A stratum with no pilot failure gets ``pseudo`` pseudo-observations at
    the pooled failure rate (mass-weighted when ``weights`` are given),
    not a whole pseudo-failure: with rare failures a full pseudo-failure
    would hand large, empty strata most of the budget.

    With ``families`` (one label per stratum) the shrinkage is
    hierarchical: a stratum shrinks toward its family's pooled rate, which
    itself shrinks toward the global one.  Strata of a family that never
    fails then share the evidence of all the family's trials instead of
    each being read at the global rate, while a family that does fail
    keeps its share.
    """
    failures = np.asarray(pilot_failures, dtype=np.float64)
    trials = np.asarray(pilot_trials, dtype=np.float64)
    w = (np.ones_like(failures)
         if weights is None else np.asarray(weights, dtype=np.float64))

    def pooled_rate(members: np.ndarray) -> float:
        rates = np.divide(failures[members],
                          trials[members],
                          out=np.zeros(members.size),
                          where=trials[members] > 0)
        rate = float(np.dot(w[members], rates) / w[members].sum())
        return max(rate, 1.0 / max(trials[members].sum(), 1.0))

    everyone = np.arange(failures.size)
    prior = np.full(failures.size, pooled_rate(everyone))
    if families is not None:
        families = np.asarray(families)
        for family in np.unique(families):
            members = np.flatnonzero(families == family)
            family_rate = (failures[members].sum() + pseudo *
                           prior[members[0]]) / (trials[members].sum() + pseudo)
            prior[members] = family_rate
    return (failures + pseudo * prior) / (trials + pseudo)


def stratum_sigmas(pilot_failures: np.ndarray,
                   pilot_trials: np.ndarray,
                   pseudo: float = PILOT_PSEUDO_TRIALS,
                   weights: np.ndarray | None = None,
                   families: np.ndarray | None = None) -> np.ndarray:
    """``sqrt(f (1 - f))`` from :func:`shrunk_rates`."""
    f = shrunk_rates(pilot_failures, pilot_trials, pseudo, weights, families)
    return np.sqrt(f * (1.0 - f))


def round_labels(budget: int,
                 weights: np.ndarray,
                 failures: np.ndarray,
                 trials: np.ndarray,
                 floor_labels: int,
                 pseudo: float = PILOT_PSEUDO_TRIALS,
                 families: np.ndarray | None = None,
                 floor_share: float = 0.0,
                 min_labels: np.ndarray | None = None) -> np.ndarray:
    """Neyman labels for one round from the counts accumulated so far.

    ``n_h`` is proportional to ``weights_h * sigma_h`` with the sigmas
    shrunk toward the pooled rate (per family when ``families`` are
    given).  Every stratum receives at least ``floor_labels`` and at least
    ``floor_share`` of its mass-proportional share ``weights_h * budget``;
    the latter bounds the variance a stratum can contribute at
    ``1 / floor_share`` times its plain-sampling contribution however badly
    its pilot underestimated it.  ``min_labels`` adds per-stratum minima
    (see :func:`certification_floor`).
    """
    w = np.asarray(weights, dtype=np.float64)
    sigmas = stratum_sigmas(failures, trials, pseudo, w, families)
    shares = w * sigmas
    labels = np.floor(budget * shares / shares.sum()).astype(int)
    proportional = np.floor(floor_share * w * budget).astype(int)
    labels = np.maximum(np.maximum(labels, floor_labels), proportional)
    if min_labels is not None:
        labels = np.maximum(labels, np.asarray(min_labels, dtype=int))
    return labels


# Rule-of-three upper bound on the rate of a stratum with no failure in n
# trials (95% level), used for the conservative variance of the stratified
# estimator: ZERO_FAILURE_BOUND_EVENTS / n.
ZERO_FAILURE_BOUND_EVENTS = 3.0


def fixed_weight_estimate(
        weights: np.ndarray, round_labels: list[np.ndarray],
        round_failures: list[np.ndarray]) -> tuple[float, float]:
    """Stratified estimate and variance from several allocation rounds.

    Round ``r``'s labels were fixed from earlier rounds, so each per-round,
    per-stratum mean is unbiased given the past; averaging the rounds with
    equal (pre-specified) weights keeps the estimator exactly unbiased under
    adaptive allocation.  When the rounds' allocations agree the variance
    equals that of the pooled estimator.

    The variance plug-in is the per-round unbiased binomial variance
    ``F_r (L_r - F_r) / (L_r^2 (L_r - 1))`` of each round's mean, which is
    conditionally unbiased given that round's labels.  A rate pooled over
    rounds must not be used here: later labels grow with earlier failures,
    so a pooled rate is negatively correlated with ``sum_r 1/L_r`` and
    underestimates the variance (by 20-40% in the BB72 onset regime).
    Strata with no failure in any round contribute zero; see
    :func:`certified_variance` for the conservative companion.
    """
    L, F, w = _rounds(weights, round_labels, round_failures)
    rounds = L.shape[0]
    stratum_rates = (F / L).mean(axis=0)
    per_round = F * (L - F) / (L**2 * np.maximum(L - 1.0, 1.0))
    stratum_variance = per_round.sum(axis=0) / rounds**2
    estimate = float(np.dot(w, stratum_rates))
    variance = float(np.sum(w**2 * stratum_variance))
    return estimate, variance


def zero_group_break_even(weights: np.ndarray, round_labels: list[np.ndarray],
                          round_failures: list[np.ndarray],
                          plain_variance: float) -> float:
    """Failure rate of the never-failing strata at which the stratified
    variance would equal ``plain_variance``.

    The plug-in gives those strata zero variance; if their common true
    rate were ``b`` the variance would grow by ``b * sum_g w_g^2 sum_r
    1/(R^2 L_{g,r})`` to first order.  The gain over plain sampling is
    therefore certified only while their rate is below this break-even
    value, which the labels bound through the family rule of three.
    Returns ``inf`` when no stratum is failure-free.
    """
    L, F, w = _rounds(weights, round_labels, round_failures)
    rounds = L.shape[0]
    _, variance = fixed_weight_estimate(weights, round_labels, round_failures)
    empty = F.sum(axis=0) == 0
    if not np.any(empty) or plain_variance <= variance:
        return float("inf") if not np.any(empty) else 0.0
    slope = float(
        np.sum(w[empty]**2 * (1.0 / L[:, empty]).sum(axis=0)) / rounds**2)
    return (plain_variance - variance) / slope


def proportional_pilot(weights: np.ndarray, total: int,
                       floor_labels: int) -> np.ndarray:
    """Pilot labels proportional to stratum mass, at least ``floor_labels``
    each.  A uniform pilot spends the same on a 0.1%-mass stratum as on a
    10% one and becomes most of the budget once there are tens of strata;
    the proportional pilot keeps the pilot share fixed and lets the family
    shrinkage pool the small strata's evidence."""
    w = np.asarray(weights, dtype=np.float64)
    labels = np.floor(total * w / w.sum()).astype(int)
    return np.maximum(labels, floor_labels)


def round_inverse_labels(round_labels: list[np.ndarray]) -> np.ndarray:
    """``sum_r 1 / (R^2 L_{g,r})`` per stratum: the factor multiplying a
    stratum's rate in the fixed-weight variance."""
    L = np.asarray(round_labels, dtype=np.float64)
    return (1.0 / L).sum(axis=0) / L.shape[0]**2


def certified_variance(
        weights: np.ndarray,
        round_labels: list[np.ndarray],
        round_failures: list[np.ndarray],
        bound_events: float = ZERO_FAILURE_BOUND_EVENTS) -> dict[str, object]:
    """Variance of :func:`fixed_weight_estimate` with the never-failing
    strata charged for what the data cannot exclude (exposure-aware).

    Strata that failed keep the per-round plug-in.  For the set ``E`` of
    strata with no failure in any round, with ``n_g`` labels each and
    unknown true rates ``f_g``, the probability of the observation is
    ``prod_g (1 - f_g)^{n_g} <= exp(-sum_g n_g f_g)``, so the event
    ``sum_{g in E} n_g f_g > bound_events`` has probability at most
    ``exp(-bound_events)`` (5% for 3).  Because the inequality holds path
    by path for whatever labels were drawn, it also holds when the labels
    were chosen adaptively from earlier rounds.  On the complementary
    event the variance those strata add,
    ``sum_g w_g^2 f_g (1 - f_g) I_g <= sum_g w_g^2 I_g f_g`` with
    ``I_g = sum_r 1 / (R^2 L_{g,r})``, is a linear function of the rates
    under one linear constraint, so its maximum puts the whole allowance
    on the stratum with the largest ``w_g^2 I_g / n_g``:

        certified = plug-in + max_{g in E} w_g^2 I_g min(1, bound_events / n_g).

    A pooled family rate (``bound_events`` over the family's total
    labels applied to every member) is *not* a bound: a stratum that
    received fewer labels than its share of the mass can hide a higher
    rate.  The bound is one-sided at level ``1 - exp(-bound_events)`` per
    count stratum; combining several count strata needs a union bound
    (``bound_events = ln(K / alpha)`` for ``K`` strata).
    """
    L, F, w = _rounds(weights, round_labels, round_failures)
    inverse = round_inverse_labels(round_labels)
    per_round = (F * (L - F) /
                 (L**2 * np.maximum(L - 1.0, 1.0))).sum(axis=0) / L.shape[0]**2
    failed = F.sum(axis=0) > 0
    empty = ~failed
    plug_in = float(np.sum(w[failed]**2 * per_round[failed]))
    labels = L.sum(axis=0)
    result = {
        "plug_in": plug_in,
        "certified": plug_in,
        "empty_mass": float(w[empty].sum()),
        "bound_group": None,
        "bound_rate": None,
    }
    if np.any(empty):
        rates = np.minimum(1.0, bound_events / np.maximum(labels, 1.0))
        terms = np.where(empty, w**2 * inverse * rates, 0.0)
        g = int(np.argmax(terms))
        result["certified"] = plug_in + float(terms[g])
        result["bound_group"] = g
        result["bound_rate"] = float(rates[g])
    return result


def certification_floor(weights: np.ndarray, empty: np.ndarray,
                        rate_estimate: float, plain_decodes: int,
                        epsilon: float, rounds: int, round_index: int,
                        current_labels: np.ndarray) -> np.ndarray:
    """Per-stratum minimum labels for round ``round_index`` (1-based) that
    certify the never-failing strata inside the estimate.

    With labels spread evenly over the rounds the bound term of an empty
    stratum in :func:`certified_variance` is ``bound_events w_g^2 / n_g^2``.
    Requiring the largest of them to be at most ``epsilon`` times the
    plain-sampling variance ``f / N`` at the planned decode count gives
    every empty stratum the same target ``n_g = w_g sqrt(bound_events N /
    (epsilon f))``, reached linearly over the rounds.  Because these
    labels sit in the rounds they also shrink the variance, so the bound
    falls as ``1 / n_g^2``; the total certification labels scale as
    ``sqrt(N)`` and their share of the budget falls as ``1 / sqrt(N)``.
    """
    w = np.asarray(weights, dtype=np.float64)
    current = np.asarray(current_labels, dtype=np.float64)
    out = np.zeros(w.size, dtype=int)
    if rate_estimate <= 0 or not np.any(empty) or epsilon <= 0:
        return out
    scale = math.sqrt(ZERO_FAILURE_BOUND_EVENTS * plain_decodes /
                      (epsilon * rate_estimate)) * round_index / rounds
    target = w * scale
    shortfall = np.maximum(0.0, target - current)
    out[empty] = np.floor(shortfall[empty]).astype(int)
    return out


# Search range for the certification allowance (log-spaced points per
# decade), widened upwards when the budget cannot afford the range, and the
# smallest share of the budget the failing strata must keep whatever the
# allowance.
CERTIFY_FRACTION_MIN = 1e-3
CERTIFY_FRACTION_MAX = 10.0
CERTIFY_FRACTION_POINTS_PER_DECADE = 40
MIN_ESTIMATE_SHARE = 0.25


def optimal_certify_fraction(weights: np.ndarray, empty: np.ndarray,
                             sigmas: np.ndarray, rate_estimate: float,
                             plain_decodes: int,
                             labels_so_far: float) -> float | None:
    """Allowance ``epsilon`` for :func:`certification_floor` that minimises
    the projected certified variance from the data so far.

    Certifying the never-failing strata to a bound of ``epsilon f / N``
    costs ``N_cert(epsilon) = W_E sqrt(bound_events N / (epsilon f))``
    labels (the floor rule's total, ``W_E`` the empty mass), which the
    failing strata lose.  With Neyman labels on the rest the plug-in
    variance is about ``(sum_g w_g sigma_g)^2 / (N - N_cert)``, so the
    projected certified variance is

        A / (N - N_cert(epsilon)) + epsilon f / N,

    and the search takes the log-grid point that minimises it while leaving
    the failing strata at least ``MIN_ESTIMATE_SHARE`` of the budget.  When
    the budget cannot afford any allowance up to ``CERTIFY_FRACTION_MAX``
    the grid is extended upwards to the loosest affordable one, so a small
    budget still certifies as far as it can instead of not at all.  With no
    failure yet the rate is read as one failure in the labels so far, the
    same floor :func:`shrunk_rates` uses.  Returns ``None`` only when no
    stratum is empty.  The choice only steers the allocation; the estimator
    stays unbiased for any value.  Measured against a fixed allowance this
    rule realises more plug-in gain and certifies less at equal decodes
    (it does not credit the failures its labels would discover), so it is
    not the default.
    """
    w = np.asarray(weights, dtype=np.float64)
    empty = np.asarray(empty, dtype=bool)
    if not np.any(empty):
        return None
    rate = max(float(rate_estimate), 1.0 / max(float(labels_so_far), 1.0))
    neyman = float(np.dot(w, np.asarray(sigmas, dtype=np.float64)))**2
    labels_per_sqrt = float(w[empty].sum()) * math.sqrt(
        ZERO_FAILURE_BOUND_EVENTS * plain_decodes / rate)
    # Smallest allowance that leaves the failing strata their share.
    affordable = (labels_per_sqrt /
                  ((1.0 - MIN_ESTIMATE_SHARE) * plain_decodes))**2
    low = max(CERTIFY_FRACTION_MIN, affordable)
    high = max(CERTIFY_FRACTION_MAX, 100.0 * low)
    points = int(
        CERTIFY_FRACTION_POINTS_PER_DECADE * math.log10(high / low)) + 1
    grid = np.logspace(math.log10(low), math.log10(high), points)
    remaining = plain_decodes - labels_per_sqrt / np.sqrt(grid)
    value = neyman / remaining + grid * rate / plain_decodes
    return float(grid[int(np.argmin(value))])


def _rounds(weights, round_labels, round_failures):
    L = np.asarray(round_labels, dtype=np.float64)
    F = np.asarray(round_failures, dtype=np.float64)
    if L.ndim != 2 or L.shape != F.shape or np.any(L <= 0):
        raise ValueError("round_labels/round_failures must be rounds x "
                         "strata with positive labels")
    return L, F, np.asarray(weights, dtype=np.float64)
