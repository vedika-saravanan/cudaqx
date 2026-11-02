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
    :func:`fixed_weight_variance_bound` for the conservative companion.
    """
    L, F, w = _rounds(weights, round_labels, round_failures)
    rounds = L.shape[0]
    stratum_rates = (F / L).mean(axis=0)
    per_round = F * (L - F) / (L**2 * np.maximum(L - 1.0, 1.0))
    stratum_variance = per_round.sum(axis=0) / rounds**2
    estimate = float(np.dot(w, stratum_rates))
    variance = float(np.sum(w**2 * stratum_variance))
    return estimate, variance


def fixed_weight_variance_bound(weights: np.ndarray,
                                round_labels: list[np.ndarray],
                                round_failures: list[np.ndarray],
                                bound_events: float = ZERO_FAILURE_BOUND_EVENTS,
                                families: np.ndarray | None = None) -> float:
    """Variance of :func:`fixed_weight_estimate` with every stratum that
    never failed set to a rule-of-three rate instead of zero: ``bound_events``
    over the stratum's own labels, or over the labels of all never-failing
    strata of its family when ``families`` are given (the family rate is
    what the labels actually bound; per-stratum bounds are far looser).
    Quoting the gain as a range between this and the plug-in makes the
    asymmetry with plain sampling (whose variance contains whatever those
    strata contribute) explicit."""
    L, F, w = _rounds(weights, round_labels, round_failures)
    rounds = L.shape[0]
    per_round = F * (L - F) / (L**2 * np.maximum(L - 1.0, 1.0))
    empty = F.sum(axis=0) == 0
    trials = L.sum(axis=0).copy()
    if families is not None:
        families = np.asarray(families)
        for family in np.unique(families[empty]):
            members = empty & (families == family)
            trials[members] = trials[members].sum()
    bound = bound_events / trials
    per_round[:, empty] = (bound * (1.0 - bound) / L)[:, empty]
    return float(np.sum(w**2 * per_round.sum(axis=0) / rounds**2))


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
        families: np.ndarray,
        certification_labels: np.ndarray | None = None,
        certification_failures: np.ndarray | None = None,
        bound_events: float = ZERO_FAILURE_BOUND_EVENTS) -> dict[str, object]:
    """Variance of :func:`fixed_weight_estimate` with the never-failing
    strata charged for what the data cannot exclude.

    Strata that failed in the rounds keep the per-round plug-in.  A stratum
    that failed only in the certification stage uses its certification
    rate (independent data, so a valid plug-in for the variance; the point
    estimate never uses certification samples).  Strata with no failure
    anywhere are set, per family, to the rule-of-three rate of all the
    family's failure-free labels, rounds plus certification
    (``mixture``), or to the worst distribution of that family rate over
    its strata (``worst_case``: every allowed failure in the stratum where
    it costs the most variance, rates capped at 1).
    """
    L, F, w = _rounds(weights, round_labels, round_failures)
    families = np.asarray(families)
    G = w.size
    cert_L = (np.zeros(G) if certification_labels is None else np.asarray(
        certification_labels, dtype=np.float64))
    cert_F = (np.zeros(G) if certification_failures is None else np.asarray(
        certification_failures, dtype=np.float64))
    inverse = round_inverse_labels(round_labels)
    per_round = (F * (L - F) /
                 (L**2 * np.maximum(L - 1.0, 1.0))).sum(axis=0) / L.shape[0]**2
    failed = F.sum(axis=0) > 0
    cert_only = ~failed & (cert_F > 0)
    empty = ~failed & ~cert_only
    terms = np.where(failed, per_round, 0.0)
    rate = np.divide(cert_F, cert_L, out=np.zeros(G), where=cert_L > 0)
    terms = np.where(cert_only, rate * (1.0 - rate) * inverse, terms)
    worst = terms.copy()
    worst_family = 0.0
    bound_rates = {}
    for family in np.unique(families[empty]):
        members = np.flatnonzero(empty & (families == family))
        trials = float((L.sum(axis=0) + cert_L)[members].sum())
        b = bound_events / trials
        bound_rates[str(family)] = b
        terms[members] = b * (1.0 - b) * inverse[members]
        # Worst case: the family's allowed failures all in the stratum
        # where they cost the most variance (rate b W_f / w_g, capped at 1).
        W = w[members].sum()
        concentrated = np.minimum(1.0, b * W / w[members])
        worst[members] = 0.0
        worst_family += float(
            np.max(w[members]**2 * concentrated * (1.0 - concentrated) *
                   inverse[members]))
    return {
        "mixture": float(np.sum(w**2 * terms)),
        "worst_case": float(np.sum(w**2 * worst) + worst_family),
        "empty_mass": float(w[empty].sum()),
        "bound_rates": bound_rates,
    }


def certification_floor(weights: np.ndarray, families: np.ndarray,
                        empty: np.ndarray, rate_estimate: float,
                        plain_decodes: int, epsilon: float, rounds: int,
                        round_index: int,
                        current_labels: np.ndarray) -> np.ndarray:
    """Per-stratum minimum labels for round ``round_index`` (1-based) that
    certify the never-failing families inside the estimate.

    With labels split by mass inside a family, the family's bound term in
    :func:`certified_variance` is ``3 W_f^2 / N_f^2`` where ``N_f`` is its
    total labels over the rounds.  The *sum* of those terms over the
    never-failing families is required to be at most ``epsilon`` times the
    plain-sampling variance ``f / N`` at the planned decode count.  The
    cheapest split of that allowance (minimum total labels for a given
    sum) gives family ``f`` the share ``W_f^(2/3) / sum W^(2/3)``, hence
    ``N_f = W_f sqrt(3 N / (epsilon_f f))``, reached linearly over the
    rounds.  Because these labels sit in the rounds they also shrink the
    variance, so the bound falls as ``1/N_f^2`` rather than the ``1/N_f``
    of samples kept outside the estimate.  The total certification labels
    scale as ``sqrt(N)``, so their share of the budget falls as
    ``1/sqrt(N)``.
    """
    w = np.asarray(weights, dtype=np.float64)
    families = np.asarray(families)
    current = np.asarray(current_labels, dtype=np.float64)
    out = np.zeros(w.size, dtype=int)
    if rate_estimate <= 0 or not np.any(empty) or epsilon <= 0:
        return out
    fams = np.unique(families[empty])
    masses = np.array([w[empty & (families == f)].sum() for f in fams])
    shares = masses**(2.0 / 3.0)
    shares /= shares.sum()
    for family, W, share in zip(fams, masses, shares):
        members = np.flatnonzero(empty & (families == family))
        target = W * math.sqrt(ZERO_FAILURE_BOUND_EVENTS * plain_decodes /
                               (epsilon * share * rate_estimate))
        target *= round_index / rounds
        shortfall = max(0.0, target - current[members].sum())
        out[members] = np.floor(shortfall * w[members] / W).astype(int)
    return out


# Search range for the certification allowance (log-spaced points per
# decade), widened upwards when the budget cannot afford the range, and the
# smallest share of the budget the failing strata must keep whatever the
# allowance.
CERTIFY_FRACTION_MIN = 1e-3
CERTIFY_FRACTION_MAX = 10.0
CERTIFY_FRACTION_POINTS_PER_DECADE = 40
MIN_ESTIMATE_SHARE = 0.25


def optimal_certify_fraction(weights: np.ndarray, families: np.ndarray,
                             empty: np.ndarray, sigmas: np.ndarray,
                             rate_estimate: float, plain_decodes: int,
                             labels_so_far: float) -> float | None:
    """Allowance ``epsilon`` for :func:`certification_floor` that minimises
    the projected certified variance from the data so far.

    Certifying the never-failing families to a bound of ``epsilon f / N``
    costs ``N_cert(epsilon) = sum_f W_f sqrt(3 N / (epsilon s_f f))``
    labels (the floor rule's total), which the failing strata lose.  With
    Neyman labels on the rest the plug-in variance is about
    ``(sum_g w_g sigma_g)^2 / (N - N_cert)``, so the projected certified
    variance is

        A / (N - N_cert(epsilon)) + epsilon f / N,

    and the search takes the log-grid point that minimises it while leaving
    the failing strata at least ``MIN_ESTIMATE_SHARE`` of the budget.  When
    the budget cannot afford any allowance up to ``CERTIFY_FRACTION_MAX``
    the grid is extended upwards to the loosest affordable one, so a small
    budget still certifies as far as it can instead of not at all.  With no
    failure yet the rate is read as one failure in the labels so far, the
    same floor :func:`shrunk_rates` uses.  Returns ``None`` only when no
    stratum is empty.  The choice only steers the allocation; the estimator
    stays unbiased for any value.
    """
    w = np.asarray(weights, dtype=np.float64)
    families = np.asarray(families)
    empty = np.asarray(empty, dtype=bool)
    if not np.any(empty):
        return None
    rate = max(float(rate_estimate), 1.0 / max(float(labels_so_far), 1.0))
    neyman = float(np.dot(w, np.asarray(sigmas, dtype=np.float64)))**2
    fams = np.unique(families[empty])
    masses = np.array([w[empty & (families == f)].sum() for f in fams])
    shares = masses**(2.0 / 3.0)
    shares /= shares.sum()
    labels_per_sqrt = float(
        np.sum(masses * np.sqrt(ZERO_FAILURE_BOUND_EVENTS * plain_decodes /
                                (shares * rate))))
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
