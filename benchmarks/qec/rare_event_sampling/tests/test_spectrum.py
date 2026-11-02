# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import numpy as np
import pytest

from rare_event import spectrum, strata
from rare_event.decoders import FailureOracle

from toy import (MaxLikelihoodDecoder, exact_failure_table, exact_logical_error,
                 exact_spectrum, rescaled, toy_model)


@pytest.fixture(scope="module")
def toy():
    model = toy_model(num_mechanisms=10, rate=0.08)
    decoder = MaxLikelihoodDecoder(model)
    failing = exact_failure_table(model, decoder)
    return model, decoder, failing


def test_measured_spectrum_matches_exact_conditional_failure(toy):
    model, decoder, failing = toy
    oracle = FailureOracle(model, decoder)
    sampler = strata.ConditionalSampler(model.priors, max_count=6)
    rng = np.random.default_rng(11)
    measured = spectrum.measure_spectrum(oracle, sampler,
                                         {k: 20_000 for k in range(0, 7)}, rng)
    exact = exact_spectrum(model, decoder, failing)
    for k, stratum in measured.strata.items():
        lo, hi = stratum.interval(0.999)
        assert lo <= exact[k] <= hi, (k, exact[k], lo, hi)
    assert oracle.decoder_calls == 7 * 20_000


def test_bounds_contain_exact_logical_error(toy):
    model, decoder, failing = toy
    oracle = FailureOracle(model, decoder)
    pmf = strata.poisson_binomial_pmf(model.priors)
    sampler = strata.ConditionalSampler(model.priors, max_count=10)
    rng = np.random.default_rng(12)
    measured = spectrum.measure_spectrum(oracle, sampler,
                                         {k: 5_000 for k in range(0, 8)}, rng)
    bounds = spectrum.logical_error_bounds(measured, pmf)
    exact = exact_logical_error(model, decoder, failing)
    assert bounds["lower"] <= exact <= bounds["upper"]
    assert abs(bounds["point"] - exact) < 0.05 * exact + 1e-4
    # Strata 8..10 were never measured and are charged as worst case.
    assert bounds["unmeasured_mass"] == pytest.approx(pmf[8:].sum())


def test_reweighting_transports_spectrum_to_lower_rate(toy):
    model, decoder, failing = toy
    oracle = FailureOracle(model, decoder)
    sampler = strata.ConditionalSampler(model.priors, max_count=10)
    rng = np.random.default_rng(13)
    measured = spectrum.measure_spectrum(oracle, sampler,
                                         {k: 40_000 for k in range(0, 8)}, rng)
    target = rescaled(model, 0.25)
    target_pmf = strata.poisson_binomial_pmf(target.priors)
    exact_target = exact_logical_error(target, decoder, failing)
    transported = spectrum.logical_error_from_reweighting(
        measured, target.priors, target_pmf)
    assert transported["unmeasured_mass"] == pytest.approx(target_pmf[8:].sum())
    assert abs(transported["point"] -
               exact_target) < (4 * transported["standard_error"] +
                                transported["unmeasured_mass"] + 1e-9)


def test_ansatz_fit_recovers_generated_spectrum():
    rng = np.random.default_rng(14)
    truth = spectrum.AnsatzFit("f3", 3, 0.5, 0.02, 2.5, 2.5, 3.0, 0.0, True)
    counts = np.arange(3, 16)
    f = spectrum.ansatz(counts, truth)
    measured = spectrum.FailureSpectrum(np.zeros(1), np.zeros(1))
    for k, fk in zip(counts, f):
        trials = 200_000
        measured.strata[int(k)] = spectrum.Stratum(
            int(k), trials, int(rng.binomial(trials, fk)))
    fit = spectrum.fit_ansatz(measured, onset=3, num_observables=1, model="f3")
    assert fit.converged
    assert fit.amplitude == pytest.approx(truth.amplitude, rel=0.1)
    assert fit.gamma1 == pytest.approx(truth.gamma1, rel=0.1)
    assert np.allclose(spectrum.ansatz(counts, fit), f, rtol=0.05, atol=1e-4)


def test_fit_rejects_failures_below_onset():
    measured = spectrum.FailureSpectrum(np.zeros(1), np.zeros(1))
    measured.strata[1] = spectrum.Stratum(1, 100, 1)
    measured.strata[3] = spectrum.Stratum(3, 100, 10)
    with pytest.raises(ValueError):
        spectrum.fit_ansatz(measured, onset=2, num_observables=1)
