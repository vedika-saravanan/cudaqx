# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Rare-event logical-error-rate estimation for a fixed, black-box decoder.

The package implements exact fault-count stratification (the unbiased core
of IBM's "Fail fast", arXiv:2511.15177) and, inside each count stratum, the
exact layer-composition stratification with certified allocation that is
the method under test, on top of a Stim detector error model (DEM) and any
decoder exposing a batch ``decode`` contract:

* :mod:`rare_event.dem` -- circuit -> DEM -> sparse ``H``/``O``/priors, noise
  rescaling, and fault propagation.
* :mod:`rare_event.strata` -- exact fault-count stratification for unequal
  mechanism probabilities (Poisson-binomial weights, exact conditional
  sampling by inverse transform, exact cross-rate reweighting).
* :mod:`rare_event.spectrum` -- failure spectrum ``f(k)`` per count and the
  IBM ansatz fits, kept as the baseline.
* :mod:`rare_event.layers` -- exact layer-composition strata of a count
  stratum, position grouping and exact group sampling.
* :mod:`rare_event.allocation` -- shrinkage, Neyman rounds with fixed-weight
  combination, certified variance and the certification allowance.
* :mod:`rare_event.onset` -- minimal failing sets found by descent from
  observed failures (an upper bound on the failure onset weight).
* :mod:`rare_event.decoders` -- adapters for PyMatching, ``ldpc`` BP-OSD and
  the CUDA-Q QEC ``nv-qldpc-decoder`` (Relay-BP).
* :mod:`rare_event.stats` -- exact binomial intervals and uncertainty
  propagation helpers.
"""

from . import allocation, dem, decoders, layers, onset, spectrum, stats, strata

__all__ = [
    "allocation", "dem", "decoders", "layers", "onset", "spectrum", "stats",
    "strata"
]
