# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exact low-weight strata by exhaustive enumeration.

For fault counts ``k`` small enough that ``C(N, k)`` decodes are affordable
(weight 1 and 2 on BB72: 2,232 and 2.49 million), every configuration is
decoded once and the conditional failure probability

    f(k) = sum_{failing E, |E| = k} prod_{j in E} o_j / e_k(o)

is computed exactly from the mechanism odds ``o_j = p_j / (1 - p_j)`` and
the elementary symmetric polynomial ``e_k(o)`` (which equals
``P(K = k) / prod_j (1 - p_j)``).  Because every stratum below the sampled
range is then exact, the "mass below the measured counts" that the sampled
curves had to assume non-failing becomes a measured term.  The failing
configurations are saved as witnesses.

Usage::

    python run_exact_head.py --circuit bb72 --decoder nv-qldpc \
        --max-weight 2 --output results/exact_head_bb72_relay
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import time

import numpy as np

import common
from rare_event import strata

BATCH = 65536


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_common_arguments(parser)
    parser.add_argument("--max-weight", type=int, default=2)
    parser.add_argument("--target-rates",
                        default="",
                        help="Comma-separated rates at which to also report "
                        "f(k); exact for every rate once failures are known.")
    args = parser.parse_args()
    experiment = common.build_experiment(args)
    model, oracle = experiment.model, experiment.oracle
    n = model.num_mechanisms
    started = time.perf_counter()
    rates = [experiment.reference_rate
            ] + [float(r) for r in args.target_rates.split(",") if r]
    priors_by_rate = {rate: experiment.priors_at(rate) for rate in rates}

    report = {
        "environment": common.environment_record(),
        "model": common.model_record(experiment),
        "weights": {},
    }
    for k in range(1, args.max_weight + 1):
        total = math.comb(n, k)
        print(f"weight {k}: {total:,} configurations")
        t0 = time.perf_counter()
        failing_sets = []
        combos = itertools.combinations(range(n), k)
        done = 0
        while True:
            chunk = list(itertools.islice(combos, BATCH))
            if not chunk:
                break
            faults = np.zeros((len(chunk), n), dtype=np.uint8)
            rows = np.repeat(np.arange(len(chunk)), k)
            cols = np.fromiter(itertools.chain.from_iterable(chunk),
                               dtype=np.int64,
                               count=len(chunk) * k)
            faults[rows, cols] = 1
            failed = oracle.failures(faults)
            for idx in np.flatnonzero(failed):
                failing_sets.append(chunk[idx])
            done += len(chunk)
            if done % (BATCH * 8) == 0 or done == total:
                print(
                    f"  {done:,}/{total:,} decoded, {len(failing_sets)} failing"
                )
        seconds = time.perf_counter() - t0
        entry = {
            "configurations": total,
            "failing_configurations": len(failing_sets),
            "seconds": seconds,
            "failing_sets": [list(map(int, s)) for s in failing_sets],
            "f_by_rate": {},
        }
        for rate, priors in priors_by_rate.items():
            odds = priors / (1 - priors)
            pmf = strata.poisson_binomial_pmf(priors, k)
            e_k = pmf[k] / float(np.prod(1 - priors))
            mass = sum(float(np.prod(odds[list(s)])) for s in failing_sets)
            entry["f_by_rate"][f"{rate:.6g}"] = {
                "f": mass / e_k if e_k > 0 else 0.0,
                "stratum_mass": float(pmf[k]),
                "ler_contribution": float(pmf[k]) * (mass / e_k if e_k else 0),
            }
        report["weights"][k] = entry
        print(
            f"weight {k}: {len(failing_sets)} failing of {total:,} in "
            f"{seconds:.1f} s; f({k}) at reference rate = "
            f"{entry['f_by_rate'][f'{experiment.reference_rate:.6g}']['f']:.3e}"
        )
    report["cost"] = {
        **oracle.counters(), "wall_seconds": time.perf_counter() - started
    }
    common.write_json(os.path.join(args.output, "exact_head.json"), report)
    print(f"wrote {os.path.join(args.output, 'exact_head.json')}")


if __name__ == "__main__":
    main()
