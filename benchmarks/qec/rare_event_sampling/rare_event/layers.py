# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exact layer-composition stratification of a fault-count stratum.

Mechanisms are partitioned by the detection layer they first touch.  For a
fixed fault count ``k`` the layer occupancy pattern ``n = (n_0, ..., n_{L-1})``
(``sum n_t = k``) has probability

    Mass(n) = prod_t PB_t(n_t),

the product of the per-layer Poisson-binomial masses, because the layers
are disjoint mechanism sets and mechanisms are independent.  Summing over
all compositions recovers ``P(K = k)``.  A configuration from stratum ``n``
is drawn exactly by sampling ``n_t`` faults from each layer's exact
conditional subset sampler and concatenating.

Compositions are grouped by a time-structure statistic (the *densest
window*: the largest number of faults within ``w`` adjacent layers), which
is what separates decoder failures (a compact core, possibly plus stray
faults) from typical fault sets spread over the circuit.  Group masses and
group sampling are exact, so the stratified estimator is unbiased for any
decoder; only the gain depends on how compact the decoder's failures are.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .dem import DetectorModel
from .strata import ConditionalSampler, poisson_binomial_pmf

# Mechanisms that flip no detector are kept in their own pseudo-layer so the
# composition masses still sum to P(K = k); they never form a dense window.
NO_DETECTOR_LAYER = -1


@dataclass
class LayerPartition:
    """Mechanism indices and priors per layer, plus per-layer samplers."""

    layer_of_mechanism: np.ndarray
    priors: np.ndarray
    max_count: int
    layers: list[int] = field(init=False)
    members: dict[int, np.ndarray] = field(init=False)
    pmf: dict[int, np.ndarray] = field(init=False)
    samplers: dict[int, ConditionalSampler] = field(init=False)

    def __post_init__(self) -> None:
        self.layer_of_mechanism = np.asarray(self.layer_of_mechanism)
        self.priors = np.asarray(self.priors, dtype=np.float64)
        self.layers = sorted(int(t) for t in np.unique(self.layer_of_mechanism))
        self.members = {
            t: np.flatnonzero(self.layer_of_mechanism == t) for t in self.layers
        }
        self.pmf = {
            t:
                poisson_binomial_pmf(self.priors[self.members[t]],
                                     min(self.max_count, self.members[t].size))
            for t in self.layers
        }
        self.samplers = {}

    def sampler(self, layer: int) -> ConditionalSampler:
        if layer not in self.samplers:
            m = self.members[layer]
            self.samplers[layer] = ConditionalSampler(
                self.priors[m], min(self.max_count, m.size))
        return self.samplers[layer]

    def layer_mass(self, layer: int, count: int) -> float:
        pmf = self.pmf[layer]
        return float(pmf[count]) if count < pmf.size else 0.0


def position_group(composition: tuple[int, ...], layers: list[int],
                   max_span: int, width: int) -> tuple:
    """Position-resolved time statistic.

    Compact sets (span <= ``max_span`` over real layers) are grouped by
    their exact (first, last) layers; spread sets by the start layer and
    size of their densest ``width``-layer window.  Boundary layers behave
    differently from bulk layers, so position carries information that a
    position-free statistic discards.
    """
    real = [(t, c)
            for t, c in zip(layers, composition)
            if t != NO_DETECTOR_LAYER and c > 0]
    if not real:
        return ("none",)
    first, last = real[0][0], real[-1][0]
    if last - first <= max_span:
        return ("compact", first, last)
    counts = dict(real)
    best, start = -1, first
    for t, _ in real:
        total = sum(counts.get(t + d, 0) for d in range(width))
        if total > best:
            best, start = total, t
    return ("spread", start, best)


def position_core_group(composition: tuple[int, ...], layers: list[int],
                        max_span: int, width: int) -> tuple:
    """:func:`position_group` refined by the largest single-layer count, the
    time statistic with the highest measured Neyman ceiling above onset."""
    return (*position_group(composition, layers, max_span, width),
            densest_window(composition, layers, 1))


def position_family(group: tuple) -> tuple:
    """Position-free coarsening of a :func:`position_group` key: compact
    sets by their span, spread sets by their densest-window count.  Groups
    of one family share a failure-rate prior when allocating labels."""
    extra = tuple(group[3:])  # refinements such as the single-layer count
    if group and group[0] == "compact":
        return ("compact", group[2] - group[1]) + extra
    if group and group[0] == "spread":
        return ("spread", group[2]) + extra
    return tuple(group)


def _real(composition: tuple[int, ...],
          layers: list[int]) -> list[tuple[int, int]]:
    return [(t, c)
            for t, c in zip(layers, composition)
            if t != NO_DETECTOR_LAYER and c > 0]


def span(composition: tuple[int, ...], layers: list[int]) -> int:
    """Last minus first occupied real layer (-1 if none)."""
    real = _real(composition, layers)
    return real[-1][0] - real[0][0] if real else -1


def occupied_layers(composition: tuple[int, ...], layers: list[int]) -> int:
    """Number of real layers holding at least one fault."""
    return len(_real(composition, layers))


def touches_boundary(composition: tuple[int, ...], layers: list[int]) -> int:
    """1 if the first real layer is occupied, plus 2 if the last one is."""
    real = _real(composition, layers)
    if not real:
        return 0
    bounds = [t for t in layers if t != NO_DETECTOR_LAYER]
    return int(real[0][0] == bounds[0]) + 2 * int(real[-1][0] == bounds[-1])


def densest_window_start(composition: tuple[int, ...], layers: list[int],
                         width: int) -> int:
    """First layer of the earliest densest ``width``-layer window (-1 if
    the composition has no real fault)."""
    real = _real(composition, layers)
    if not real:
        return -1
    counts = dict(real)
    best, start = -1, real[0][0]
    for t, _ in real:
        total = sum(counts.get(t + d, 0) for d in range(width))
        if total > best:
            best, start = total, t
    return start


def densest_window(composition: tuple[int, ...], layers: list[int],
                   width: int) -> int:
    """Largest number of faults inside ``width`` consecutive real layers."""
    real = [
        (t, c) for t, c in zip(layers, composition) if t != NO_DETECTOR_LAYER
    ]
    if not real:
        return 0
    counts = dict(real)
    best = 0
    for t, _ in real:
        best = max(best, sum(counts.get(t + d, 0) for d in range(width)))
    return best


@dataclass
class CompositionStrata:
    """All compositions of ``count`` faults over the layers, with exact
    masses, grouped by a statistic of the composition."""

    partition: LayerPartition
    count: int
    group_of: object  # callable composition -> hashable group key
    compositions: list[tuple[int, ...]] = field(init=False)
    masses: np.ndarray = field(init=False)
    groups: dict[object, list[int]] = field(init=False)

    def __post_init__(self) -> None:
        layers = self.partition.layers
        self.compositions, masses = [], []
        for comp in _compositions(self.count, len(layers)):
            mass = 1.0
            for t, c in zip(layers, comp):
                mass *= self.partition.layer_mass(t, c)
                if mass == 0.0:
                    break
            if mass > 0.0:
                self.compositions.append(tuple(comp))
                masses.append(mass)
        self.masses = np.asarray(masses, dtype=np.float64)
        self.groups = {}
        for index, comp in enumerate(self.compositions):
            self.groups.setdefault(self.group_of(comp), []).append(index)

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())

    def group_masses(self) -> dict[object, float]:
        return {
            g: float(self.masses[idx].sum()) for g, idx in self.groups.items()
        }

    def sample_group(self, group: object, shots: int,
                     rng: np.random.Generator) -> np.ndarray:
        """Exact samples from the union of compositions in ``group``.

        A composition is drawn per shot with probability proportional to
        its mass; then, layer by layer, the shots are grouped by their
        count in that layer and each count's rows are filled from the
        layer's exact conditional sampler in one call.  Layer draws are
        independent given the composition, so this equals sampling each
        composition separately, at ``layers x counts`` sampler calls
        instead of one per distinct composition.
        """
        indices = np.asarray(self.groups[group])
        weights = self.masses[indices] / self.masses[indices].sum()
        chosen = rng.choice(indices, size=shots, p=weights)
        counts = np.asarray(self.compositions, dtype=np.int64)[chosen]
        out = np.zeros((shots, self.partition.priors.size), dtype=np.uint8)
        for column, layer in enumerate(self.partition.layers):
            members = self.partition.members[layer]
            for count in np.unique(counts[:, column]):
                if count == 0:
                    continue
                rows = np.flatnonzero(counts[:, column] == count)
                draw = self.partition.sampler(layer).sample(
                    int(count), rows.size, rng)
                out[np.ix_(rows, members)] = draw
        return out


def _compositions(total: int, parts: int):
    """All tuples of ``parts`` non-negative integers summing to ``total``."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def mechanism_layers(model: DetectorModel,
                     detectors_per_layer: int | None = None,
                     circuit=None) -> np.ndarray:
    """Earliest detection layer per mechanism, from a fixed detectors-per-
    layer count or from Stim detector time coordinates; -1 if no detector."""
    import scipy.sparse as sp

    H = sp.csc_matrix(model.H)
    if circuit is not None:
        # Stim's generated circuits give every detector (x, y, t)
        # coordinates with time last.  A circuit whose detectors carry a
        # single value (the vendored bivariate-bicycle assets) has no time
        # coordinate and must use detectors_per_layer; reading its one
        # value as time would turn spatial labels into layers.
        coordinates = circuit.get_detector_coordinates()
        raw = np.zeros(model.num_detectors, dtype=np.float64)
        for det in range(model.num_detectors):
            coords = coordinates.get(det)
            if not coords or len(coords) < 2:
                raise ValueError(
                    "detector coordinates lack a time component; give "
                    "detectors_per_layer instead of the circuit")
            raw[det] = coords[-1]
        # Dense ranks, so window arithmetic runs over consecutive layers.
        _, times = np.unique(np.round(raw).astype(np.int64),
                             return_inverse=True)
        times = times.ravel().astype(np.int64)
    elif detectors_per_layer:
        times = np.arange(model.num_detectors) // detectors_per_layer
    else:
        raise ValueError("give detectors_per_layer or circuit")
    layers = np.full(model.num_mechanisms, NO_DETECTOR_LAYER, dtype=np.int64)
    for j in range(model.num_mechanisms):
        rows = H.indices[H.indptr[j]:H.indptr[j + 1]]
        if rows.size:
            layers[j] = int(times[rows].min())
    return layers
