# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared command-line plumbing for the rare-event scripts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import stim

from rare_event import dem
from rare_event.decoders import FailureOracle, make_decoder

ASSET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "assets", "benchmarks"))


@dataclass
class Experiment:
    circuit: stim.Circuit
    reference_rate: float
    model: dem.DetectorModel
    oracle: FailureOracle
    decoder_name: str
    decoder_options: dict[str, Any]
    rng: np.random.Generator
    seed: int

    def priors_at(self, rate: float) -> np.ndarray:
        """Exact DEM priors of the circuit rescaled to ``rate``."""
        if abs(rate - self.reference_rate) <= 1e-15:
            return self.model.priors
        return dem.priors_at_rates(self.circuit, self.reference_rate,
                                   [rate])[rate]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--circuit",
        required=True,
        help="Path to a .stim file, 'surface:<d>:<rounds>' or "
        "'repetition:<d>:<rounds>'. Asset names such as 'bb72' resolve to "
        "assets/benchmarks/<name>_memory_Z.stim.")
    parser.add_argument(
        "--reference-rate",
        type=float,
        default=None,
        help="Physical rate of the supplied circuit (default: inferred from "
        "its strongest noise argument). Generated circuits use this rate.")
    parser.add_argument("--decoder",
                        default="pymatching",
                        choices=["pymatching", "bposd", "nv-qldpc"])
    parser.add_argument(
        "--decoder-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Decoder constructor override (JSON-parsed value), repeatable.")
    parser.add_argument("--count-nonconvergence-as-failure",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", required=True, help="Output directory.")


def parse_options(items: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        if not key or not raw:
            raise argparse.ArgumentTypeError(f"bad option '{item}'")
        try:
            options[key] = json.loads(raw)
        except json.JSONDecodeError:
            options[key] = raw
    return options


def load_circuit(spec: str,
                 reference_rate: float | None) -> tuple[stim.Circuit, float]:
    if spec.startswith("surface:") or spec.startswith("repetition:"):
        kind, distance, rounds = spec.split(":")
        if reference_rate is None:
            raise SystemExit(
                "--reference-rate is required for generated circuits")
        builder = (dem.surface_code_circuit
                   if kind == "surface" else dem.repetition_code_circuit)
        return builder(int(distance), int(rounds),
                       reference_rate), reference_rate
    path = spec
    if not os.path.exists(path):
        candidate = os.path.join(ASSET_DIR, f"{spec}_memory_Z.stim")
        if os.path.exists(candidate):
            path = candidate
        else:
            raise SystemExit(f"circuit '{spec}' not found")
    circuit = stim.Circuit.from_file(path)
    inferred = dem.circuit_noise_strength(circuit)
    if reference_rate is None:
        reference_rate = inferred
    elif abs(reference_rate - inferred) > 1e-12:
        circuit = dem.rescale_circuit_noise(circuit, reference_rate / inferred)
    return circuit, reference_rate


def build_experiment(args: argparse.Namespace) -> Experiment:
    circuit, reference_rate = load_circuit(args.circuit, args.reference_rate)
    model = dem.detector_model_from_circuit(circuit)
    options = parse_options(args.decoder_option)
    decoder = make_decoder(args.decoder, circuit, model, **options)
    oracle = FailureOracle(
        model,
        decoder,
        count_nonconvergence=args.count_nonconvergence_as_failure)
    return Experiment(circuit=circuit,
                      reference_rate=reference_rate,
                      model=model,
                      oracle=oracle,
                      decoder_name=args.decoder,
                      decoder_options=options,
                      rng=np.random.default_rng(args.seed),
                      seed=args.seed)


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "stim": stim.__version__,
        "numpy": np.__version__,
    }
    try:
        gpu = subprocess.run([
            "nvidia-smi", "--query-gpu=name,driver_version",
            "--format=csv,noheader"
        ],
                             capture_output=True,
                             text=True,
                             timeout=10,
                             check=False)
        record["gpu"] = gpu.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        record["gpu"] = None
    try:
        import cudaq_qec
        record["cudaq_qec"] = getattr(cudaq_qec, "__version__", "unknown")
    except ImportError:
        record["cudaq_qec"] = None
    return record


def model_record(experiment: Experiment) -> dict[str, Any]:
    model = experiment.model
    return {
        "circuit":
            experiment.circuit.num_qubits,
        "reference_rate":
            experiment.reference_rate,
        "detectors":
            model.num_detectors,
        "mechanisms":
            model.num_mechanisms,
        "observables":
            model.num_observables,
        "expected_fault_count":
            model.expected_fault_count,
        "decoder":
            experiment.decoder_name,
        "decoder_options":
            experiment.decoder_options,
        "count_nonconvergence_as_failure":
            experiment.oracle.count_nonconvergence,
        "seed":
            experiment.seed,
    }


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        raise TypeError(f"not serialisable: {type(value)}")

    def finite(value):
        # Standard JSON has no inf/nan; store them as null.
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(v) for v in value]
        if isinstance(value, np.ndarray):
            return finite(value.tolist())
        if isinstance(value, np.floating):
            return finite(float(value))
        return value

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(finite(payload), handle, indent=2, default=default)


def parse_counts(spec: str) -> list[int]:
    """Fault counts from a spec like ``3,4`` or ``3-8,10``."""
    counts: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            counts.extend(range(int(lo), int(hi) + 1))
        else:
            counts.append(int(part))
    return counts
