# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import numpy as np
import pytest

stim = pytest.importorskip("stim")

from rare_event import dem, strata
from rare_event.decoders import FailureOracle, PyMatchingDecoder


def test_detector_model_matches_stim_sampler():
    circuit = dem.repetition_code_circuit(distance=5,
                                          rounds=3,
                                          physical_rate=0.01)
    model = dem.detector_model_from_circuit(circuit)
    assert model.num_detectors == circuit.num_detectors
    assert model.num_observables == circuit.num_observables
    assert np.all((model.priors > 0) & (model.priors < 0.5))
    # Propagating stim's own DEM samples through H/O reproduces its
    # detector and observable outputs exactly.
    sampler = circuit.detector_error_model(flatten_loops=True).compile_sampler(
        seed=1)
    dets, obs, errs = sampler.sample(200, return_errors=True)
    # stim returns errors in its own (undecomposed) mechanism order, which
    # is the order detector_model_from_dem walks.
    assert errs.shape[1] == model.num_mechanisms
    assert np.array_equal(model.syndromes(errs.astype(np.uint8)),
                          dets.astype(np.uint8))
    assert np.array_equal(model.observables(errs.astype(np.uint8)),
                          obs.astype(np.uint8))


def test_rescaling_scales_every_prior_linearly_to_first_order():
    circuit = dem.surface_code_circuit(distance=3,
                                       rounds=2,
                                       physical_rate=0.004)
    assert dem.circuit_noise_strength(circuit) == pytest.approx(0.004)
    priors = dem.priors_at_rates(circuit, 0.004, [0.004, 0.001])
    ratio = priors[0.001] / priors[0.004]
    # Mechanisms combine several gate faults, so the ratio is 0.25 up to
    # O(p) corrections.
    assert np.all(np.abs(ratio - 0.25) < 0.02)


def test_expected_fault_count_and_pmf_consistency():
    circuit = dem.surface_code_circuit(distance=3,
                                       rounds=3,
                                       physical_rate=0.002)
    model = dem.detector_model_from_circuit(circuit)
    pmf = strata.poisson_binomial_pmf(model.priors, max_count=40)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    mean = np.dot(np.arange(pmf.size), pmf)
    assert mean == pytest.approx(model.expected_fault_count, rel=1e-6)


def test_pymatching_oracle_on_surface_code_agrees_with_direct_sampling():
    pytest.importorskip("pymatching")
    circuit = dem.surface_code_circuit(distance=3, rounds=3, physical_rate=0.01)
    model = dem.detector_model_from_circuit(circuit)
    oracle = FailureOracle(model, PyMatchingDecoder(circuit))
    rng = np.random.default_rng(2)
    shots = 20_000
    faults = (rng.random((shots, model.num_mechanisms))
              < model.priors).astype(np.uint8)
    rate_a = oracle.failures(faults).mean()
    # Compare with stim's circuit sampler decoded by the same matcher.
    sampler = circuit.compile_detector_sampler(seed=3)
    dets, obs = sampler.sample(shots, separate_observables=True)
    predicted = PyMatchingDecoder(circuit).decode_batch(dets.astype(
        np.uint8)).observables
    rate_b = np.any(predicted != obs.astype(np.uint8), axis=1).mean()
    assert abs(rate_a - rate_b) < 4 * np.sqrt(rate_b *
                                              (1 - rate_b) / shots) + 2e-3


def test_propagation_gather_matches_sparse_product():
    circuit = dem.surface_code_circuit(distance=3,
                                       rounds=3,
                                       physical_rate=0.003)
    model = dem.detector_model_from_circuit(circuit)
    rng = np.random.default_rng(5)
    # Direct-noise rows (variable fault counts, some empty), a conditional
    # batch and an all-zero batch must agree with the sparse product.
    direct = (rng.random((500, model.num_mechanisms)) < 0.02).astype(np.uint8)
    direct[:7] = 0
    conditional = strata.ConditionalSampler(model.priors, 4).sample(4, 300, rng)
    for faults in (direct, conditional,
                   np.zeros((3, model.num_mechanisms), np.uint8)):
        for matrix, name in ((model.H, "H"), (model.O, "O")):
            reference = dem._propagate(matrix, None, faults)
            fast = dem._propagate(matrix, model._dense_transpose(name), faults)
            assert fast.dtype == np.uint8
            assert np.array_equal(fast, reference)
