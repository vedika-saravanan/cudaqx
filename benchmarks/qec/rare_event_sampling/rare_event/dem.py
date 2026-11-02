# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Detector error models as sparse matrices, with noise rescaling.

A Stim DEM is a list of independent error mechanisms.  Mechanism ``j`` fires
with probability ``p_j`` and flips the detectors in column ``j`` of ``H`` and
the observables in column ``j`` of ``O``.  Everything in this package works on
that compact representation; nothing is expanded into equal-probability
copies, so there is no multiplicity-inference approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import stim

# Stim instructions whose parenthesised arguments are error probabilities.
# Measurement-style instructions take an optional flip probability.
_NOISE_INSTRUCTIONS = frozenset({
    "X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E", "CORRELATED_ERROR",
    "ELSE_CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1", "M",
    "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP", "MPAD"
})

# Largest total probability a DEPOLARIZE2 channel may carry (15 of 16 Paulis).
_MAX_DEPOLARIZE2 = 15.0 / 16.0

# Dense uint8 copies of H and O are kept for the gather-based propagation
# up to this many bytes each; larger models use the sparse product.
DENSE_GATHER_MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class DetectorModel:
    """Sparse detector error model: ``H`` (detectors x mechanisms), ``O``
    (observables x mechanisms) and the per-mechanism prior ``priors``."""

    H: sp.csr_matrix
    O: sp.csr_matrix
    priors: np.ndarray

    @property
    def num_detectors(self) -> int:
        return self.H.shape[0]

    @property
    def num_mechanisms(self) -> int:
        return self.H.shape[1]

    @property
    def num_observables(self) -> int:
        return self.O.shape[0]

    @property
    def expected_fault_count(self) -> float:
        return float(self.priors.sum())

    def syndromes(self, faults: np.ndarray) -> np.ndarray:
        """Return ``H x mod 2`` for a batch of fault rows ``faults``."""
        return _propagate(self.H, self._dense_transpose("H"), faults)

    def observables(self, faults: np.ndarray) -> np.ndarray:
        """Return ``O x mod 2`` for a batch of fault rows ``faults``."""
        return _propagate(self.O, self._dense_transpose("O"), faults)

    def _dense_transpose(self, name: str) -> np.ndarray | None:
        """Cached ``(mechanisms x rows)`` uint8 copy of ``H`` or ``O`` for
        the gather path of :func:`_propagate`; ``None`` above the size cap."""
        matrix = getattr(self, name)
        if matrix.shape[0] * matrix.shape[1] > DENSE_GATHER_MAX_BYTES:
            return None
        cache = self.__dict__.get("_dense_cache")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_dense_cache", cache)
        if name not in cache:
            cache[name] = np.ascontiguousarray(matrix.toarray().T,
                                               dtype=np.uint8)
        return cache[name]

    def dense_H(self) -> np.ndarray:
        return self.H.toarray()

    def dense_O(self) -> np.ndarray:
        return self.O.toarray()


def _propagate(matrix: sp.csr_matrix, dense_transpose: np.ndarray | None,
               faults: np.ndarray) -> np.ndarray:
    faults = np.asarray(faults)
    if faults.ndim != 2 or faults.shape[1] != matrix.shape[1]:
        raise ValueError("faults must have shape (shots, num_mechanisms)")
    shots = faults.shape[0]
    if shots == 0:
        return np.empty((0, matrix.shape[0]), dtype=np.uint8)
    if dense_transpose is None:
        # int32 accumulation then parity; sparse x dense keeps memory
        # bounded for models too large for the gather path.
        product = matrix.astype(np.int32) @ faults.astype(np.int32).T
        return np.asarray(product.T & 1, dtype=np.uint8)
    # Fault rows are sparse (a handful of faults per shot), so gather the
    # fired mechanisms' columns and XOR them per row: work proportional to
    # the number of faults instead of the number of mechanisms.
    rows, mechanisms = np.nonzero(faults)
    out = np.zeros((shots, matrix.shape[0]), dtype=np.uint8)
    if rows.size == 0:
        return out
    # np.nonzero returns rows in increasing order; segment at row changes.
    starts = np.flatnonzero(np.concatenate(([True], rows[1:] != rows[:-1])))
    out[rows[starts]] = np.bitwise_xor.reduceat(dense_transpose[mechanisms],
                                                starts,
                                                axis=0)
    return out


def detector_model_from_dem(dem: stim.DetectorErrorModel) -> DetectorModel:
    """Convert a (possibly undecomposed) Stim DEM into sparse matrices.

    Error mechanisms separated by ``^`` (decomposed hyperedges) are merged
    back into a single column so the sampled fault vector is exactly the
    DEM's independent-mechanism model.
    """
    flat = dem.flattened()
    h_rows, h_cols, o_rows, o_cols, priors = [], [], [], [], []
    column = 0
    for instruction in flat:
        if instruction.type != "error":
            continue
        probability = float(instruction.args_copy()[0])
        detectors: set[int] = set()
        observables: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= {int(target.val)}
            elif target.is_logical_observable_id():
                observables ^= {int(target.val)}
        if not detectors and not observables:
            # A mechanism that flips nothing can never change the outcome.
            continue
        h_rows.extend(detectors)
        h_cols.extend([column] * len(detectors))
        o_rows.extend(observables)
        o_cols.extend([column] * len(observables))
        priors.append(probability)
        column += 1
    shape_h = (dem.num_detectors, column)
    shape_o = (dem.num_observables, column)
    H = sp.csr_matrix((np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
                      shape=shape_h)
    O = sp.csr_matrix((np.ones(len(o_rows), dtype=np.uint8), (o_rows, o_cols)),
                      shape=shape_o)
    return DetectorModel(H=H, O=O, priors=np.asarray(priors, dtype=np.float64))


def detector_model_from_circuit(circuit: stim.Circuit) -> DetectorModel:
    """Undecomposed DEM of ``circuit`` as a :class:`DetectorModel`."""
    return detector_model_from_dem(
        circuit.detector_error_model(decompose_errors=False,
                                     flatten_loops=True))


def rescale_circuit_noise(circuit: stim.Circuit, factor: float) -> stim.Circuit:
    """Multiply every noise probability in ``circuit`` by ``factor``.

    All circuit-noise mechanisms scale together, which is the usual
    single-parameter ``p`` family (CNOT depolarising ``p``, idle, measurement
    and reset flips all proportional to ``p``).  The returned circuit has the
    same detectors, observables and DEM mechanism ordering, so DEM columns
    line up across rates.
    """
    if factor <= 0.0:
        raise ValueError("factor must be positive")
    scaled = stim.Circuit()
    for instruction in circuit.flattened():
        if (instruction.name in _NOISE_INSTRUCTIONS and
                instruction.gate_args_copy()):
            args = [a * factor for a in instruction.gate_args_copy()]
            total = sum(args)
            limit = _MAX_DEPOLARIZE2 if instruction.name == "DEPOLARIZE2" else 1.0
            if total > limit:
                raise ValueError(
                    f"{instruction.name} probability {total:.4g} exceeds {limit}"
                    " after rescaling")
            scaled.append(instruction.name, instruction.targets_copy(), args)
        else:
            scaled.append(instruction)
    return scaled


def circuit_noise_strength(circuit: stim.Circuit) -> float:
    """Return the largest single noise argument in ``circuit``.

    For the vendored single-parameter circuits this is the physical rate
    ``p`` that every mechanism is proportional to.
    """
    strongest = 0.0
    for instruction in circuit.flattened():
        if instruction.name in _NOISE_INSTRUCTIONS:
            for arg in instruction.gate_args_copy():
                strongest = max(strongest, float(arg))
    return strongest


def priors_at_rates(circuit: stim.Circuit, reference_rate: float,
                    rates: Iterable[float]) -> dict[float, np.ndarray]:
    """Exact DEM priors of ``circuit`` rescaled to each physical rate.

    Mechanism ordering is identical across rates because only probabilities
    change; this is checked by comparing column supports.
    """
    reference = detector_model_from_circuit(circuit)
    result: dict[float, np.ndarray] = {}
    for rate in rates:
        model = detector_model_from_circuit(
            rescale_circuit_noise(circuit, rate / reference_rate))
        if (model.H != reference.H).nnz or (model.O != reference.O).nnz:
            raise RuntimeError("DEM structure changed under noise rescaling")
        result[float(rate)] = model.priors
    return result


def surface_code_circuit(distance: int,
                         rounds: int,
                         physical_rate: float,
                         basis: str = "z") -> stim.Circuit:
    """Rotated surface-code memory circuit with uniform circuit-level noise."""
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=physical_rate,
        before_round_data_depolarization=physical_rate,
        before_measure_flip_probability=physical_rate,
        after_reset_flip_probability=physical_rate,
    )


def repetition_code_circuit(distance: int, rounds: int,
                            physical_rate: float) -> stim.Circuit:
    """Repetition-code memory circuit; small enough for exact enumeration."""
    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=physical_rate,
        before_round_data_depolarization=physical_rate,
        before_measure_flip_probability=physical_rate,
        after_reset_flip_probability=physical_rate,
    )


def faults_from_indices(indices: Sequence[Sequence[int]],
                        num_mechanisms: int) -> np.ndarray:
    """Dense ``uint8`` fault rows from per-shot lists of active mechanisms."""
    faults = np.zeros((len(indices), num_mechanisms), dtype=np.uint8)
    for row, active in enumerate(indices):
        faults[row, list(active)] = 1
    return faults
