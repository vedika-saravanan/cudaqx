# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""IBM Fail Fast technique I: exact fault-count stratified spectrum.

Example (surface code d=5 with PyMatching, no GPU needed)::

    python run_spectrum.py --circuit surface:5:5 --reference-rate 0.003 \
        --counts 0-14 --shots-per-count 20000 --target-rates 0.003,0.001,0.0003 \
        --onset 3 --output results/surface5

Example (BB72 with the GPU Relay-BP decoder)::

    python run_spectrum.py --circuit bb72 --decoder nv-qldpc \
        --counts 0-24 --shots-per-count 20000 --target-rates 0.002,0.001,0.0005 \
        --onset 3 --direct-shots 200000 --output results/bb72_relay
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import common
from rare_event import spectrum, stats, strata
from rare_event.onset import minimal_failing_sets

# Keep the Poisson-binomial table until this little mass remains.
TAIL_MASS = 1e-12


def parse_counts(spec: str) -> list[int]:
    counts: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            counts.extend(range(int(lo), int(hi) + 1))
        else:
            counts.append(int(part))
    return sorted(set(counts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_common_arguments(parser)
    parser.add_argument(
        "--counts",
        default="0-16",
        help="Fault counts to measure, e.g. '0-16' or '3,4,6,8'.")
    parser.add_argument("--shots-per-count", type=int, default=10_000)
    parser.add_argument("--target-rates",
                        default="",
                        help="Comma-separated physical rates to project to.")
    parser.add_argument("--onset",
                        type=int,
                        default=None,
                        help="Hypothesised minimum failing weight for the "
                        "ansatz fit (default: smallest count with a failure).")
    parser.add_argument("--ansatz", default="f3", choices=["f2", "f3", "f5"])
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--direct-shots",
                        type=int,
                        default=0,
                        help="Independent direct Monte Carlo control shots at "
                        "every target rate (0 disables).")
    parser.add_argument("--reduce-failures",
                        type=int,
                        default=0,
                        help="Reduce up to this many observed failing sets to "
                        "minimal failing sets (onset upper bound).")
    args = parser.parse_args()

    experiment = common.build_experiment(args)
    model, oracle, rng = experiment.model, experiment.oracle, experiment.rng
    started = time.perf_counter()

    counts = parse_counts(args.counts)
    source_pmf_full = strata.poisson_binomial_pmf(model.priors)
    max_count = max(max(counts),
                    strata.truncation_count(source_pmf_full, TAIL_MASS))
    sampler = strata.ConditionalSampler(model.priors, max_count)
    measured = spectrum.measure_spectrum(
        oracle, sampler, {k: args.shots_per_count for k in counts}, rng)
    spectrum_seconds = time.perf_counter() - started

    counts_arr, trials, failures = measured.arrays()
    report = {
        "environment": common.environment_record(),
        "model": common.model_record(experiment),
        "spectrum": {
            "counts": counts_arr,
            "trials": trials,
            "failures": failures,
            "failure_fraction": failures / np.maximum(trials, 1),
            "source_pmf": sampler.pmf,
            "seconds": spectrum_seconds,
        },
        "projections": {},
    }

    onset = args.onset
    observed_min = int(counts_arr[failures > 0].min()) if np.any(
        failures > 0) else None
    if onset is None:
        onset = observed_min
    report["onset"] = {"hypothesis": onset, "observed_minimum": observed_min}

    rates = [experiment.reference_rate]
    rates += [float(r) for r in args.target_rates.split(",") if r]
    for rate in dict.fromkeys(rates):
        priors = experiment.priors_at(rate)
        pmf = strata.poisson_binomial_pmf(priors, max_count)
        entry: dict[str, object] = {
            "pmf_tail_mass": max(0.0, 1.0 - float(pmf.sum())),
            "expected_fault_count": float(priors.sum()),
        }
        if abs(rate - experiment.reference_rate) <= 1e-15:
            entry["exact_bounds"] = spectrum.logical_error_bounds(measured, pmf)
        else:
            entry["reweighted"] = spectrum.logical_error_from_reweighting(
                measured, priors, pmf)
        if onset is not None and np.any(failures > 0):
            try:
                fit = spectrum.fit_ansatz(measured, onset,
                                          model.num_observables, args.ansatz)
                entry["ansatz_fit"] = {
                    "model":
                        fit.model,
                    "amplitude":
                        fit.amplitude,
                    "gamma1":
                        fit.gamma1,
                    "gamma2":
                        fit.gamma2,
                    "crossover":
                        fit.crossover,
                    "converged":
                        fit.converged,
                    "projected_ler":
                        spectrum.logical_error_from_fit(fit, pmf),
                    "bootstrap":
                        spectrum.bootstrap_fit_projection(
                            measured, pmf, onset, model.num_observables,
                            args.ansatz, args.bootstrap, rng),
                    "note":
                        "model-dependent projection; not a confidence "
                        "interval for the true LER",
                }
            except ValueError as error:
                entry["ansatz_fit"] = {"error": str(error)}
        if args.direct_shots > 0:
            direct_started = time.perf_counter()
            calls_before = oracle.decoder_calls
            failed = 0
            remaining = args.direct_shots
            while remaining > 0:
                chunk = min(remaining, 8192)
                faults = (rng.random((chunk, priors.size))
                          < priors).astype(np.uint8)
                failed += int(oracle.failures(faults).sum())
                remaining -= chunk
            lo, hi = stats.clopper_pearson(failed, args.direct_shots)
            entry["direct_control"] = {
                "shots": args.direct_shots,
                "failures": failed,
                "estimate": failed / args.direct_shots,
                "interval_95": (lo, hi),
                "decoder_calls": oracle.decoder_calls - calls_before,
                "seconds": time.perf_counter() - direct_started,
            }
        report["projections"][f"{rate:.6g}"] = entry

    if args.reduce_failures > 0:
        failing_rows = [
            row for s in measured.strata.values() for block in s.failing_faults
            for row in block
        ]
        if failing_rows:
            chosen = np.stack(failing_rows[:args.reduce_failures])
            found = minimal_failing_sets(oracle, chosen, rng)
            report["minimal_failing_sets"] = {
                "onset_upper_bound": found["onset_upper_bound"],
                "distinct_minimal_sets": found["distinct_minimal_sets"],
                "weight_histogram": found["weight_histogram"],
            }

    report["cost"] = {
        **oracle.counters(),
        "wall_seconds": time.perf_counter() - started,
    }
    common.write_json(os.path.join(args.output, "spectrum.json"), report)
    print(f"wrote {os.path.join(args.output, 'spectrum.json')}")
    for rate, entry in report["projections"].items():
        print(f"p={rate}: {entry}")


if __name__ == "__main__":
    main()
