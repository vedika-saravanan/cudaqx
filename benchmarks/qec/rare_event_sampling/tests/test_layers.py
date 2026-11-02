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

from rare_event import layers, strata
from rare_event.dem import DetectorModel

import scipy.sparse as sp


def _toy_layers():
    rng = np.random.default_rng(5)
    n = 12
    layer_of = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, -1, -1])
    priors = rng.uniform(0.02, 0.2, size=n)
    return layer_of, priors


def test_composition_masses_sum_to_count_stratum():
    layer_of, priors = _toy_layers()
    for k in (1, 2, 3):
        part = layers.LayerPartition(layer_of, priors, k)
        comp = layers.CompositionStrata(part, k, lambda c: 0)
        total = strata.poisson_binomial_pmf(priors, k)[k]
        assert comp.total_mass == pytest.approx(float(total), rel=1e-12)


def test_composition_mass_matches_brute_force():
    layer_of, priors = _toy_layers()
    k = 2
    part = layers.LayerPartition(layer_of, priors, k)
    comp = layers.CompositionStrata(part, k, lambda c: c)
    # Brute force over all weight-2 configurations.
    brute = {}
    for pair in itertools.combinations(range(priors.size), k):
        x = np.zeros(priors.size)
        x[list(pair)] = 1
        mass = float(np.prod(np.where(x == 1, priors, 1 - priors)))
        key = tuple(int((layer_of[list(pair)] == t).sum()) for t in part.layers)
        brute[key] = brute.get(key, 0.0) + mass
    for c, m in zip(comp.compositions, comp.masses):
        assert m == pytest.approx(brute[c], rel=1e-12)


def test_group_sampling_has_exact_composition_and_frequencies():
    layer_of, priors = _toy_layers()
    k = 3
    rng = np.random.default_rng(1)
    part = layers.LayerPartition(layer_of, priors, k)
    stat = lambda c: layers.densest_window(c, part.layers, 2)
    comp = layers.CompositionStrata(part, k, stat)
    group = max(comp.groups, key=lambda g: comp.masses[comp.groups[g]].sum())
    draws = comp.sample_group(group, 20000, rng)
    assert np.all(draws.sum(axis=1) == k)
    # Every draw's composition belongs to the group and the compositions
    # appear with their mass proportions.
    counts = {}
    for row in draws:
        c = tuple(
            int((layer_of[np.flatnonzero(row)] == t).sum())
            for t in part.layers)
        assert stat(c) == group
        counts[c] = counts.get(c, 0) + 1
    idx = comp.groups[group]
    expected = comp.masses[idx] / comp.masses[idx].sum()
    for c, e in zip([comp.compositions[i] for i in idx], expected):
        if e > 0.02:
            assert counts.get(c, 0) / 20000 == pytest.approx(e, abs=0.015)


def test_densest_window_statistic():
    layer_list = [-1, 0, 1, 2, 3]
    assert layers.densest_window((2, 1, 0, 0, 0), layer_list, 1) == 1
    assert layers.densest_window((0, 1, 2, 0, 1), layer_list, 1) == 2
    assert layers.densest_window((0, 1, 2, 0, 1), layer_list, 2) == 3
    assert layers.densest_window((5, 0, 0, 0, 0), layer_list, 2) == 0


def test_position_family_drops_position_only():
    assert layers.position_family(("compact", 2, 4)) == ("compact", 2)
    assert layers.position_family(("compact", 5, 5)) == ("compact", 0)
    assert layers.position_family(("spread", 1, 3)) == ("spread", 3)
    assert layers.position_family(("rest",)) == ("rest",)


def test_composition_helpers():
    layer_list = [-1, 0, 1, 2, 3]
    c = (1, 2, 0, 0, 1)  # one detector-free fault, two in layer 0, one in 3
    assert layers.span(c, layer_list) == 3
    assert layers.occupied_layers(c, layer_list) == 2
    assert layers.touches_boundary(c, layer_list) == 3
    assert layers.densest_window_start(c, layer_list, 2) == 0
    assert layers.densest_window_start((0, 0, 1, 2, 0), layer_list, 2) == 1
    assert layers.span((3, 0, 0, 0, 0), layer_list) == -1
    assert layers.touches_boundary((0, 0, 1, 0, 0), layer_list) == 0


def test_position_core_group_and_family():
    layer_list = [-1, 0, 1, 2, 3]
    g = layers.position_core_group((0, 2, 1, 0, 0), layer_list, 2, 2)
    assert g == ("compact", 0, 1, 2)
    assert layers.position_family(g) == ("compact", 1, 2)
    g = layers.position_core_group((0, 1, 0, 0, 2), layer_list, 2, 2)
    assert g == ("spread", 3, 2, 2)
    assert layers.position_family(g) == ("spread", 2, 2)


def test_mechanism_layers_from_coordinates_are_dense_time_ranks():
    stim = pytest.importorskip("stim")
    from rare_event import dem
    circuit = dem.surface_code_circuit(distance=3, rounds=4, physical_rate=0.01)
    model = dem.detector_model_from_circuit(circuit)
    layer_of = layers.mechanism_layers(model, circuit=circuit)
    real = sorted(set(int(t) for t in layer_of if t >= 0))
    assert real == list(range(len(real)))
    H = sp.csc_matrix(model.H)
    for j in range(model.num_mechanisms):
        has_detector = H.indptr[j + 1] > H.indptr[j]
        assert (layer_of[j] >= 0) == has_detector


def test_mechanism_layers_rejects_single_valued_coordinates():
    stim = pytest.importorskip("stim")
    from rare_event import dem
    circuit = stim.Circuit("""
        X_ERROR(0.1) 0
        M 0
        DETECTOR(5) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    model = dem.detector_model_from_circuit(circuit)
    with pytest.raises(ValueError):
        layers.mechanism_layers(model, circuit=circuit)
