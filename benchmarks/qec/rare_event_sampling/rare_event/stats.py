# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Interval estimates for Bernoulli rates and products of ratios."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as _stats

DEFAULT_CONFIDENCE = 0.95


def clopper_pearson(
        failures: int,
        trials: int,
        confidence: float = DEFAULT_CONFIDENCE) -> tuple[float, float]:
    """Exact two-sided binomial interval; valid at zero and at all failures."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= failures <= trials:
        raise ValueError("failures must lie in [0, trials]")
    alpha = 1.0 - confidence
    lower = 0.0 if failures == 0 else float(
        _stats.beta.ppf(alpha / 2.0, failures, trials - failures + 1))
    upper = 1.0 if failures == trials else float(
        _stats.beta.ppf(1.0 - alpha / 2.0, failures + 1, trials - failures))
    return lower, upper


def wilson(failures: int,
           trials: int,
           confidence: float = DEFAULT_CONFIDENCE) -> tuple[float, float]:
    """Wilson score interval (used only for allocation heuristics)."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    z = float(_stats.norm.ppf(0.5 + confidence / 2.0))
    estimate = failures / trials
    denominator = 1.0 + z * z / trials
    centre = (estimate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(estimate * (1.0 - estimate) / trials + z * z /
                           (4.0 * trials * trials)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def bonferroni_confidence(confidence: float, comparisons: int) -> float:
    """Per-interval confidence giving simultaneous ``confidence`` overall."""
    if comparisons < 1:
        raise ValueError("comparisons must be positive")
    return 1.0 - (1.0 - confidence) / comparisons


def weighted_mean_interval(
        values: np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE) -> tuple[float, float, float]:
    """Mean, standard error and half-width of a sample mean (normal approx)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("at least two values are required")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    z = float(_stats.norm.ppf(0.5 + confidence / 2.0))
    return mean, standard_error, z * standard_error


def log_product_interval(
        log_values: np.ndarray,
        log_standard_errors: np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE) -> tuple[float, float, float]:
    """Interval for ``prod exp(log_values)`` with independent log errors.

    Returns ``(point, lower, upper)``; relative errors add in quadrature on
    the log scale.
    """
    log_values = np.asarray(log_values, dtype=np.float64)
    log_standard_errors = np.asarray(log_standard_errors, dtype=np.float64)
    total = float(log_values.sum())
    spread = float(math.sqrt(np.square(log_standard_errors).sum()))
    z = float(_stats.norm.ppf(0.5 + confidence / 2.0))
    return math.exp(total), math.exp(total - z * spread), math.exp(total +
                                                                   z * spread)


def relative_half_width(lower: float, upper: float) -> float:
    """Half-width of ``[lower, upper]`` relative to its lower endpoint.

    Conservative on purpose: a zero lower endpoint never counts as precise.
    """
    if lower <= 0.0:
        return math.inf
    return 0.5 * (upper - lower) / lower
