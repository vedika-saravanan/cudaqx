# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Technique I of IBM Fail Fast: the failure spectrum ``f(k)``.

``f(k) = P(fail | exactly k faults)`` is measured by decoding fault sets drawn
exactly from the conditional law of the DEM given ``K = k`` (see
:mod:`rare_event.strata`).  The logical error rate at any physical rate is
then ``LER(p) = sum_k P_p(K = k) f_p(k)``.  Two routes give ``f_p(k)`` at a
rate other than the one sampled:

* exact importance reweighting of the stored failing configurations
  (unbiased, variance grows with the rate gap), or
* the IBM low-parameter ansatz fitted to the measured spectrum (model
  dependent; reported separately and never mixed into the exact bounds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from . import stats
from .decoders import FailureOracle
from .strata import ConditionalSampler, cross_rate_log_weights

# Crossover sharpness ``c`` of the five-parameter ansatz, fixed in the paper.
ANSATZ_CROSSOVER_SHARPNESS = 2.0
# Clamp on the ansatz exponent so exp() never overflows.
_MAX_LOG_ARGUMENT = 50.0
# Number of crossover starting points for the multi-start f5 fit.
ANSATZ_CROSSOVER_STARTS = 8


@dataclass
class Stratum:
    """Monte Carlo record for one fault count ``k``."""

    count: int
    trials: int = 0
    failures: int = 0
    failing_faults: list[np.ndarray] = field(default_factory=list)

    @property
    def failure_fraction(self) -> float:
        return self.failures / self.trials if self.trials else math.nan

    def interval(self, confidence: float) -> tuple[float, float]:
        if self.trials == 0:
            return 0.0, 1.0
        return stats.clopper_pearson(self.failures, self.trials, confidence)


@dataclass
class FailureSpectrum:
    """Measured spectrum together with the sampling rate's priors."""

    source_priors: np.ndarray
    source_pmf: np.ndarray
    strata: dict[int, Stratum] = field(default_factory=dict)

    def measured_counts(self) -> list[int]:
        return sorted(k for k, s in self.strata.items() if s.trials > 0)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = np.asarray(self.measured_counts(), dtype=np.int64)
        trials = np.asarray([self.strata[k].trials for k in counts],
                            dtype=np.int64)
        failures = np.asarray([self.strata[k].failures for k in counts],
                              dtype=np.int64)
        return counts, trials, failures


def measure_spectrum(oracle: FailureOracle,
                     sampler: ConditionalSampler,
                     shots_per_count: dict[int, int],
                     rng: np.random.Generator,
                     spectrum: FailureSpectrum | None = None,
                     batch_size: int = 8192) -> FailureSpectrum:
    """Decode exact conditional samples for each requested fault count.

    Passing an existing ``spectrum`` extends it (adaptive allocation).
    """
    if spectrum is None:
        spectrum = FailureSpectrum(source_priors=sampler.probabilities.copy(),
                                   source_pmf=sampler.pmf)
    for count, shots in shots_per_count.items():
        stratum = spectrum.strata.setdefault(count, Stratum(count=count))
        remaining = int(shots)
        while remaining > 0:
            chunk = min(remaining, batch_size)
            faults = sampler.sample(count, chunk, rng)
            failed = oracle.failures(faults)
            stratum.trials += chunk
            stratum.failures += int(failed.sum())
            if np.any(failed):
                stratum.failing_faults.append(faults[failed].copy())
            remaining -= chunk
    return spectrum


def logical_error_bounds(
        spectrum: FailureSpectrum,
        pmf: np.ndarray,
        confidence: float = stats.DEFAULT_CONFIDENCE) -> dict[str, float]:
    """Point estimate and simultaneous conservative bounds of ``LER``.

    Measured strata get Bonferroni-corrected exact intervals; strata that
    were never sampled contribute ``[0, P(K=k)]``; mass beyond the table
    contributes ``[0, tail]``.  The result is a valid (conservative)
    confidence interval, not a fitted projection.
    """
    counts = spectrum.measured_counts()
    per_interval = stats.bonferroni_confidence(confidence, max(1, len(counts)))
    point = lower = upper = 0.0
    unmeasured = 0.0
    for k in range(pmf.size):
        weight = float(pmf[k])
        stratum = spectrum.strata.get(k)
        if stratum is None or stratum.trials == 0:
            unmeasured += weight
            upper += weight
            continue
        lo, hi = stratum.interval(per_interval)
        point += weight * stratum.failure_fraction
        lower += weight * lo
        upper += weight * hi
    tail = max(0.0, 1.0 - float(pmf.sum()))
    upper += tail
    return {
        "point": point,
        "lower": lower,
        "upper": min(1.0, upper),
        "unmeasured_mass": unmeasured + tail,
        "relative_half_width": stats.relative_half_width(lower, upper),
    }


def reweighted_spectrum(
        spectrum: FailureSpectrum, target_priors: np.ndarray,
        target_pmf: np.ndarray) -> dict[int, tuple[float, float]]:
    """``f_target(k)`` and its standard error by exact importance reweighting
    of the stored failing configurations.

    The estimator is the ordinary (not self-normalised) mean
    ``(1/T) sum_i w_i fail_i`` and is unbiased for every ``k``.
    """
    result: dict[int, tuple[float, float]] = {}
    for k, stratum in spectrum.strata.items():
        if stratum.trials == 0:
            continue
        if not stratum.failing_faults:
            result[k] = (0.0, 0.0)
            continue
        faults = np.concatenate(stratum.failing_faults, axis=0)
        weights = np.exp(
            cross_rate_log_weights(faults, spectrum.source_priors,
                                   target_priors, k, spectrum.source_pmf,
                                   target_pmf))
        total = stratum.trials
        mean = float(weights.sum() / total)
        second_moment = float(np.square(weights).sum() / total)
        variance = max(0.0, second_moment - mean * mean) / total
        result[k] = (mean, math.sqrt(variance))
    return result


def logical_error_from_reweighting(spectrum: FailureSpectrum,
                                   target_priors: np.ndarray,
                                   target_pmf: np.ndarray) -> dict[str, float]:
    """Unbiased LER estimate at a target rate from a spectrum sampled at a
    different rate.  Unmeasured strata are reported, not guessed."""
    transported = reweighted_spectrum(spectrum, target_priors, target_pmf)
    point = variance = unmeasured = 0.0
    for k in range(target_pmf.size):
        weight = float(target_pmf[k])
        if k not in transported:
            unmeasured += weight
            continue
        mean, standard_error = transported[k]
        point += weight * mean
        variance += (weight * standard_error)**2
    tail = max(0.0, 1.0 - float(target_pmf.sum()))
    return {
        "point": point,
        "standard_error": math.sqrt(variance),
        "unmeasured_mass": unmeasured + tail,
    }


# --------------------------------------------------------------------------- #
# IBM ansatz fits (model dependent)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnsatzFit:
    """Fitted low-parameter failure-spectrum model.

    ``model`` is ``f2`` (onset exponent fixed to ``w0``), ``f3`` (free power
    ``gamma1``) or ``f5`` (power-law crossover from ``gamma1`` to ``gamma2``
    at ``crossover``).  ``saturation`` is ``a = 1 - 2**(-K)``.
    """

    model: str
    onset: int
    saturation: float
    amplitude: float
    gamma1: float
    gamma2: float
    crossover: float
    log_likelihood: float
    converged: bool

    @property
    def parameter_count(self) -> int:
        return {"f2": 1, "f3": 2, "f5": 4}[self.model]


def ansatz(counts: np.ndarray, fit: AnsatzFit) -> np.ndarray:
    """Evaluate the fitted ansatz at integer fault counts."""
    w = np.asarray(counts, dtype=np.float64)
    out = np.zeros_like(w)
    above = w >= fit.onset
    if not np.any(above):
        return out
    ratio = w[above] / fit.onset
    if fit.model == "f2":
        log_shape = fit.onset * np.log(ratio)
    elif fit.model == "f3":
        log_shape = fit.gamma1 * np.log(ratio)
    elif fit.model == "f5":
        c = ANSATZ_CROSSOVER_SHARPNESS
        transition = (np.log1p((w[above] / fit.crossover)**c) - math.log1p(
            (fit.onset / fit.crossover)**c))
        log_shape = (fit.gamma1 * np.log(ratio) +
                     (fit.gamma2 - fit.gamma1) / c * transition)
    else:
        raise ValueError(f"unknown ansatz model '{fit.model}'")
    log_argument = math.log(fit.amplitude / fit.saturation) + log_shape
    argument = np.exp(np.minimum(log_argument, _MAX_LOG_ARGUMENT))
    out[above] = fit.saturation * (-np.expm1(-argument))
    return out


def _binomial_log_likelihood(predicted: np.ndarray, failures: np.ndarray,
                             trials: np.ndarray) -> float:
    eps = 1e-300
    clipped = np.clip(predicted, eps, 1.0 - 1e-12)
    return float(
        np.sum(failures * np.log(clipped) +
               (trials - failures) * np.log1p(-clipped)))


def fit_ansatz(spectrum: FailureSpectrum,
               onset: int,
               num_observables: int,
               model: str = "f3") -> AnsatzFit:
    """Maximum-likelihood fit of the IBM ansatz to the measured spectrum.

    Only strata with ``k >= onset`` enter the fit; an observed failure below
    ``onset`` is a contradiction and raises.
    """
    counts, trials, failures = spectrum.arrays()
    if np.any(failures[counts < onset] > 0):
        raise ValueError("observed failure below the supplied onset")
    keep = counts >= onset
    counts, trials, failures = counts[keep], trials[keep], failures[keep]
    if counts.size == 0 or failures.sum() == 0:
        raise ValueError("need at least one failure at or above the onset")
    saturation = 1.0 - 2.0**(-num_observables)

    def unpack(theta: np.ndarray) -> AnsatzFit:
        amplitude = saturation / (1.0 + math.exp(-theta[0]))
        gamma1 = math.exp(theta[1]) if model != "f2" else float(onset)
        gamma2 = math.exp(theta[2]) if model == "f5" else gamma1
        crossover = math.exp(theta[3]) if model == "f5" else float(onset)
        return AnsatzFit(model, onset, saturation, amplitude, gamma1, gamma2,
                         crossover, math.nan, False)

    def negative_log_likelihood(theta: np.ndarray) -> float:
        fit = unpack(theta)
        return -_binomial_log_likelihood(ansatz(counts, fit), failures, trials)

    first = max(failures[0] / trials[0], 1.0 / (10.0 * trials[0]))
    first = min(first, 0.999 * saturation)
    theta0 = [
        math.log(first / (saturation - first)),
        math.log(float(onset)),
        math.log(float(onset)),
        math.log(2.0 * onset),
    ]
    sizes = {"f2": 1, "f3": 2, "f5": 4}
    # The f5 likelihood is very flat in the crossover and has local optima
    # that collapse onto f3 (crossover -> 0); start from several crossovers
    # spread over the measured range and keep the best.
    if model == "f5":
        crossover_starts = np.geomspace(onset,
                                        max(2 * onset, counts.max()),
                                        num=ANSATZ_CROSSOVER_STARTS)
    else:
        crossover_starts = np.asarray([2.0 * onset])
    result = None
    for crossover in crossover_starts:
        start = list(theta0[:sizes[model]])
        if model == "f5":
            start[3] = math.log(float(crossover))
        candidate = optimize.minimize(negative_log_likelihood,
                                      np.asarray(start),
                                      method="Nelder-Mead",
                                      options={
                                          "maxiter": 20000,
                                          "xatol": 1e-8,
                                          "fatol": 1e-10
                                      })
        if result is None or candidate.fun < result.fun:
            result = candidate
    fit = unpack(np.concatenate([result.x, theta0[sizes[model]:]]))
    return AnsatzFit(fit.model, fit.onset, fit.saturation, fit.amplitude,
                     fit.gamma1, fit.gamma2, fit.crossover, -float(result.fun),
                     bool(result.success))


def logical_error_from_fit(fit: AnsatzFit, pmf: np.ndarray) -> float:
    """``sum_k P(K=k) f_fit(k)`` over the table; model dependent."""
    counts = np.arange(pmf.size)
    return float(np.dot(pmf, ansatz(counts, fit)))


def bootstrap_fit_projection(
        spectrum: FailureSpectrum,
        pmf: np.ndarray,
        onset: int,
        num_observables: int,
        model: str,
        replicates: int,
        rng: np.random.Generator,
        confidence: float = stats.DEFAULT_CONFIDENCE) -> dict[str, float]:
    """Parametric bootstrap of the fitted LER, conditional on the ansatz."""
    counts, trials, failures = spectrum.arrays()
    projections = []
    for _ in range(replicates):
        resampled = FailureSpectrum(spectrum.source_priors, spectrum.source_pmf)
        for k, t, f in zip(counts, trials, failures):
            drawn = int(rng.binomial(t, f / t)) if t else 0
            resampled.strata[int(k)] = Stratum(int(k), int(t), drawn)
        try:
            fit = fit_ansatz(resampled, onset, num_observables, model)
        except ValueError:
            continue
        projections.append(logical_error_from_fit(fit, pmf))
    if len(projections) < 2:
        return {"replicates": len(projections)}
    values = np.asarray(projections)
    alpha = 1.0 - confidence
    return {
        "replicates": len(projections),
        "median": float(np.median(values)),
        "lower": float(np.quantile(values, alpha / 2.0)),
        "upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
    }


def suggest_allocation(pmf: np.ndarray, onset: int, pilot: FailureSpectrum,
                       target_failures: int, max_count: int) -> dict[int, int]:
    """Shots per fault count so each measured stratum collects roughly
    ``target_failures`` failures, following the paper's allocation rule.

    Counts are spaced approximately uniformly in ``log k`` from the onset
    to ``max_count``; the pilot's failure fractions (or a conservative
    extrapolation) set the budget.
    """
    if max_count < onset:
        raise ValueError("max_count must be at least the onset")
    grid = np.unique(
        np.rint(
            np.geomspace(onset, max_count, num=min(max_count - onset + 1,
                                                   12))).astype(int))
    allocation: dict[int, int] = {}
    pilot_fraction = {
        k: s.failure_fraction
        for k, s in pilot.strata.items()
        if s.trials and s.failures
    }
    for k in grid:
        fraction = pilot_fraction.get(int(k))
        if fraction is None:
            nearest = min(pilot_fraction,
                          key=lambda j: abs(j - k),
                          default=None)
            fraction = pilot_fraction[nearest] if nearest is not None else 0.01
        allocation[int(k)] = int(
            math.ceil(target_failures / max(fraction, 1e-6)))
    return allocation
