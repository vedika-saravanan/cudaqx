# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Decoder adapters and the failure oracle.

Every adapter implements :class:`ObservableDecoder`: given a batch of
syndromes it returns predicted observable flips and a convergence flag.  The
:class:`FailureOracle` turns that into failure labels against the simulator's
ground truth and counts every decoder call, which is the cost unit all
estimators report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp
import stim

from .dem import DetectorModel


@dataclass
class DecodeBatch:
    """Predicted observable flips ``(shots, num_observables)`` and per-shot
    convergence flags."""

    observables: np.ndarray
    converged: np.ndarray


class ObservableDecoder(Protocol):

    def decode_batch(self, syndromes: np.ndarray) -> DecodeBatch:
        ...


@dataclass
class FailureOracle:
    """Labels fault configurations as failing with the real decoder only.

    ``count_nonconvergence`` controls whether a non-converged decode counts
    as a failure regardless of the predicted observables.
    """

    model: DetectorModel
    decoder: ObservableDecoder
    count_nonconvergence: bool = True
    decoder_calls: int = field(default=0, init=False)
    wrong_predictions: int = field(default=0, init=False)
    nonconverged: int = field(default=0, init=False)

    def failures(self, faults: np.ndarray) -> np.ndarray:
        """Boolean failure label for each fault row."""
        faults = np.asarray(faults, dtype=np.uint8)
        if faults.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        truth = self.model.observables(faults)
        batch = self.decoder.decode_batch(self.model.syndromes(faults))
        predicted = np.asarray(batch.observables, dtype=np.uint8)
        if predicted.shape != truth.shape:
            raise ValueError("decoder returned observables of the wrong shape")
        wrong = np.any(predicted != truth, axis=1)
        converged = np.asarray(batch.converged, dtype=bool)
        self.decoder_calls += int(faults.shape[0])
        self.wrong_predictions += int(wrong.sum())
        self.nonconverged += int((~converged).sum())
        if self.count_nonconvergence:
            return wrong | ~converged
        return wrong

    def counters(self) -> dict[str, int]:
        return {
            "decoder_calls": self.decoder_calls,
            "wrong_predictions": self.wrong_predictions,
            "nonconverged": self.nonconverged,
        }


class PyMatchingDecoder:
    """Minimum-weight perfect matching through PyMatching.

    Built from the *decomposed* DEM of the same circuit so hyperedges are
    split into matchable edges while the sampling model stays undecomposed.
    """

    def __init__(self, circuit: stim.Circuit):
        import pymatching

        dem = circuit.detector_error_model(decompose_errors=True,
                                           flatten_loops=True)
        self._matching = pymatching.Matching.from_detector_error_model(dem)
        self.num_observables = dem.num_observables

    def decode_batch(self, syndromes: np.ndarray) -> DecodeBatch:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        predicted = self._matching.decode_batch(syndromes)
        predicted = np.asarray(predicted,
                               dtype=np.uint8).reshape(syndromes.shape[0], -1)
        return DecodeBatch(observables=predicted,
                           converged=np.ones(syndromes.shape[0], dtype=bool))


class BpOsdDecoder:
    """Belief propagation + ordered statistics decoding via the ``ldpc``
    package (CPU).  Supports both the ``ldpc`` v2 class API and the legacy
    ``bposd_decoder`` API."""

    def __init__(self,
                 model: DetectorModel,
                 max_iterations: int = 100,
                 osd_order: int = 10,
                 osd_method: str = "osd_cs"):
        self._model = model
        H = sp.csr_matrix(model.H, dtype=np.uint8)
        try:
            from ldpc import BpOsdDecoder as _Decoder  # ldpc >= 2
            self._decoder = _Decoder(H,
                                     channel_probs=list(model.priors),
                                     max_iter=max_iterations,
                                     bp_method="minimum_sum",
                                     osd_method=osd_method,
                                     osd_order=osd_order)
            self._legacy = False
        except ImportError:
            from ldpc import bposd_decoder as _Decoder  # ldpc 0.1.x
            self._decoder = _Decoder(H.toarray(),
                                     channel_probs=list(model.priors),
                                     max_iter=max_iterations,
                                     bp_method="msl",
                                     osd_method=osd_method,
                                     osd_order=osd_order)
            self._legacy = True

    def decode_batch(self, syndromes: np.ndarray) -> DecodeBatch:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        corrections = np.zeros((syndromes.shape[0], self._model.num_mechanisms),
                               dtype=np.uint8)
        bp_converged = np.ones(syndromes.shape[0], dtype=bool)
        for row, syndrome in enumerate(syndromes):
            corrections[row] = np.asarray(self._decoder.decode(syndrome),
                                          dtype=np.uint8)
            flag = getattr(self._decoder, "converge", None)
            if flag is not None:
                bp_converged[row] = bool(flag)
        # OSD is the fallback for BP non-convergence and always returns a
        # syndrome-consistent correction, so the combined decoder has
        # produced a valid answer for every shot; the BP flag is only
        # diagnostic and must not be scored as a decoder failure.
        self.last_bp_converged = bp_converged
        return DecodeBatch(observables=self._model.observables(corrections),
                           converged=np.ones(syndromes.shape[0], dtype=bool))


# Relay-BP configuration mirroring the CUDA-Q QEC gamma-ensemble benchmark
# (benchmarks/qec/relaybp_gamma_ensemble): disordered-memory min-sum legs
# with a fixed pre-iteration count and a first-convergence stopping rule.
RELAY_DEFAULTS = {
    "max_iterations": 50,
    "iter_per_check": 1,
    "use_sparsity": True,
    "bp_method": 3,
    "composition": 1,
    "gamma0": 0.125,
    "gamma_dist": [-0.24, 0.66],
    "clip_value": 200.0,
    "repeatable": True,
    "proc_float": "fp32",
    "srelay_config": {
        "pre_iter": 1,
        "num_sets": 600,
        "stopping_criterion": "FirstConv",
    },
}

DEFAULT_GPU_BATCH = 4096


class NvQldpcDecoder:
    """CUDA-Q QEC ``nv-qldpc-decoder`` (Relay-BP) returning observables.

    The decoder priors are frozen at ``model.priors`` (the reference rate):
    the object under test is one fixed decoder configuration.
    """

    def __init__(self,
                 model: DetectorModel,
                 batch_size: int = DEFAULT_GPU_BATCH,
                 gamma_ensemble_size: int = 1,
                 **overrides):
        import cudaq_qec as qec

        options = dict(RELAY_DEFAULTS)
        options.update(overrides)
        options["bp_batch_size"] = int(batch_size)
        options["gamma_ensemble_size"] = int(gamma_ensemble_size)
        self._batch_size = int(batch_size)
        self._model = model
        H = sp.csr_matrix(model.H, dtype=np.uint8)
        try:
            # cudaq-qec > 0.8.1: the decoder can emit observable flips.
            self._decoder = qec.get_decoder("nv-qldpc-decoder",
                                            H,
                                            O=model.dense_O(),
                                            output="observables",
                                            error_rate_vec=model.priors,
                                            **options)
            self._emits_observables = True
        except (ValueError, TypeError, RuntimeError):
            # cudaq-qec <= 0.8.1: error frames only; apply O ourselves.
            self._decoder = qec.get_decoder("nv-qldpc-decoder",
                                            H,
                                            error_rate_vec=model.priors,
                                            **options)
            self._emits_observables = False
        self.options = options

    def decode_batch(self, syndromes: np.ndarray) -> DecodeBatch:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        frames, converged = [], []
        # The batched relay path does not support gamma ensembles (0.8.1);
        # fall back to per-syndrome decode() for those configurations.
        batched = self.options.get("gamma_ensemble_size", 1) == 1
        for start in range(0, syndromes.shape[0], self._batch_size):
            chunk = syndromes[start:start + self._batch_size]
            if batched:
                results = self._decoder.decode_batch(chunk.tolist())
            else:
                results = [self._decoder.decode(row.tolist()) for row in chunk]
            for result in results:
                frames.append(np.asarray(result.result, dtype=np.float64) > 0.5)
                converged.append(bool(result.converged))
        frames_arr = np.asarray(frames, dtype=np.uint8)
        if self._emits_observables:
            observables = frames_arr
        else:
            observables = self._model.observables(frames_arr)
        return DecodeBatch(observables=observables,
                           converged=np.asarray(converged, dtype=bool))


def make_decoder(name: str, circuit: stim.Circuit, model: DetectorModel,
                 **options) -> ObservableDecoder:
    """Factory: ``pymatching``, ``bposd`` or ``nv-qldpc``."""
    if name == "pymatching":
        return PyMatchingDecoder(circuit)
    if name == "bposd":
        return BpOsdDecoder(model, **options)
    if name == "nv-qldpc":
        return NvQldpcDecoder(model, **options)
    raise ValueError(f"unknown decoder '{name}'")
