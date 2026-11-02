# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Direct Monte Carlo control of the logical error rate at a low physical rate.

The circuit is rescaled to ``--rate`` and sampled with Stim's circuit
sampler (an independent code path from the DEM-based conditional samplers
used elsewhere), while the decoder keeps the priors of the reference-rate
circuit it was built from, as in the fixed-decoder setting of the
stratified estimates.  A non-converged decode counts as a failure when
``--count-nonconvergence-as-failure`` is set, matching ``FailureOracle``.
The report is rewritten after every ``--checkpoint-batches`` batches so a
long run can be read while it is still going.

Usage::

    python run_direct_mc.py --circuit bb72 --decoder nv-qldpc \
        --decoder-option batch_size=1024 --rate 1e-4 --shots 240000000 \
        --output results/direct_mc_bb72_relay_1e-4
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import common
from rare_event import dem, stats

DEFAULT_BATCH = 65_536
DEFAULT_CHECKPOINT_BATCHES = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_common_arguments(parser)
    parser.add_argument("--rate",
                        type=float,
                        required=True,
                        help="Physical rate at which to sample the circuit.")
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--checkpoint-batches",
                        type=int,
                        default=DEFAULT_CHECKPOINT_BATCHES)
    args = parser.parse_args()

    experiment = common.build_experiment(args)
    sampled_circuit = dem.rescale_circuit_noise(
        experiment.circuit, args.rate / experiment.reference_rate)
    sampler = sampled_circuit.compile_detector_sampler(seed=args.seed)
    decoder = experiment.oracle.decoder
    count_nonconvergence = args.count_nonconvergence_as_failure

    report_path = os.path.join(args.output, "direct_mc.json")
    started = time.perf_counter()
    shots_done = 0
    failures = 0
    wrong = 0
    nonconverged = 0
    batches = 0

    def write_report(final: bool) -> None:
        lo, hi = stats.clopper_pearson(failures, max(shots_done, 1))
        estimate = failures / max(shots_done, 1)
        payload = {
            "environment": common.environment_record(),
            "model": common.model_record(experiment),
            "parameters": vars(args),
            "sampled_rate": args.rate,
            "decoder_prior_rate": experiment.reference_rate,
            "shots": shots_done,
            "failures": failures,
            "wrong_predictions": wrong,
            "nonconverged": nonconverged,
            "estimate": estimate,
            "relative_standard_error": (float(np.sqrt(
                (1 - estimate) / failures)) if failures > 0 else None),
            "interval_95": [lo, hi],
            "seconds": time.perf_counter() - started,
            "complete": final,
        }
        common.write_json(report_path, payload)

    while shots_done < args.shots:
        chunk = min(args.batch, args.shots - shots_done)
        detectors, observables = sampler.sample(chunk,
                                                separate_observables=True)
        batch = decoder.decode_batch(detectors.astype(np.uint8))
        predicted = np.asarray(batch.observables, dtype=np.uint8)
        truth = observables.astype(np.uint8)
        if predicted.shape != truth.shape:
            raise ValueError("decoder returned observables of the wrong shape")
        wrong_rows = np.any(predicted != truth, axis=1)
        converged = np.asarray(batch.converged, dtype=bool)
        failed_rows = wrong_rows | ~converged if count_nonconvergence else wrong_rows
        shots_done += chunk
        failures += int(failed_rows.sum())
        wrong += int(wrong_rows.sum())
        nonconverged += int((~converged).sum())
        batches += 1
        if batches % args.checkpoint_batches == 0:
            write_report(final=False)
            elapsed = time.perf_counter() - started
            print(
                f"{shots_done:>12d} shots, {failures:6d} failures, "
                f"LER {failures / shots_done:.3e}, {elapsed / 60:6.1f} min",
                flush=True)
    write_report(final=True)
    lo, hi = stats.clopper_pearson(failures, shots_done)
    print(f"direct MC at p={args.rate:g}: {failures} failures in {shots_done} "
          f"shots, LER {failures / shots_done:.3e} (95% [{lo:.2e}, {hi:.2e}]), "
          f"{(time.perf_counter() - started) / 60:.1f} min")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
