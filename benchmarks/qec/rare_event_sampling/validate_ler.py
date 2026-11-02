# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Matched-precision check of the LER projection with completed LACE runs.

``run_ler_speedup.py`` projects the decoder calls LACE needs for a target
relative error at a target rate from gains measured at other budgets.  This
script takes completed ``run_lace.py`` reports, one per fault count, run at
the per-count budgets that projection implied, and assembles what they
actually deliver at the target rate:

    LER(p) = sum_k P_k(p) f_k  (+ exact low-weight head),

with the plug-in and certified variance of each count's estimate propagated
through the exact count masses, the plain conditional (count-strata)
estimate from the same reports at the same decode count, and the realised
decodes and wall time, so the projected cost can be read against a run
that paid for its pilot, allocation and sampling.  Counts not covered by a
report contribute their mass to a reported tail and nothing to the
variance.  Certified standard errors combine per-count bounds that each
hold at 95%; the union bound over the covered counts is reported too.

Usage::

    python validate_ler.py --circuit bb72 --rate 1e-4 --exact-head 2:3.18e-7 \\
        results/validate_1e4_bb72_relay_k3.json results/validate_1e4_bb72_relay_k4.json \\
        results/validate_1e4_bb72_relay_k5.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import common
from rare_event import dem, strata

UNION_BOUND_LEVEL = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--reference-rate", type=float, default=None)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--exact-head", default="")
    parser.add_argument("--direct",
                        default=None,
                        help="Optional run_direct_mc.py report at --rate.")
    args = parser.parse_args()

    circuit, reference_rate = common.load_circuit(args.circuit,
                                                  args.reference_rate)
    priors = dem.priors_at_rates(circuit, reference_rate,
                                 [args.rate])[args.rate]
    per_count: dict[int, list[dict]] = {}
    wall = 0.0
    for path in args.reports:
        report = json.load(open(path))
        wall += float(report["cost"]["wall_seconds"])
        for k, pc in report["per_count"].items():
            per_count[int(k)] = pc["repeats"]
    counts = sorted(per_count)
    pmf = strata.poisson_binomial_pmf(priors, max(counts))
    exact = {}
    for item in filter(None, args.exact_head.split(",")):
        k, value = item.split(":")
        exact[int(k)] = float(value)
    head = sum(float(pmf[k]) * v for k, v in exact.items())
    covered = sum(float(pmf[k]) for k in counts) + sum(
        float(pmf[k]) for k in exact)
    repeats = min(len(per_count[k]) for k in counts)
    # Union bound over the covered counts: each count's never-failing bound
    # at level alpha / K instead of alpha.
    union_events = math.log(len(counts) / UNION_BOUND_LEVEL)
    results = []
    for i in range(repeats):
        ler = head
        var_plug = var_cert = var_cert_union = var_plain = 0.0
        plain = head
        decodes = 0
        for k in counts:
            r = per_count[k][i]
            P = float(pmf[k])
            ler += P * r["estimate"]
            var_plug += P**2 * r["standard_error"]**2
            bound_term = r["certified_variance"] - r["standard_error"]**2
            var_cert += P**2 * r["certified_variance"]
            var_cert_union += P**2 * (r["standard_error"]**2 +
                                      bound_term * union_events / 3.0)
            f_plain = r["plain_estimate_same_budget"]
            plain += P * f_plain
            var_plain += P**2 * f_plain * (1 - f_plain) / r["plain_shots"]
            decodes += int(r["plain_shots"])
        results.append({
            "ler": ler,
            "relative_se_plug_in": math.sqrt(var_plug) / ler,
            "relative_se_certified": math.sqrt(var_cert) / ler,
            "relative_se_certified_union": math.sqrt(var_cert_union) / ler,
            "plain_ler": plain,
            "plain_relative_se": math.sqrt(var_plain) / plain,
            "decodes_per_arm": decodes,
        })
    direct = None
    if args.direct:
        d = json.load(open(args.direct))
        direct = (d["estimate"], d["interval_95"], d["shots"], d["seconds"])

    print(f"p = {args.rate:g}: counts {counts} cover {covered:.4f} of the mass "
          f"(exact head {head:.2e}); {repeats} complete repeats, "
          f"{wall / 60:.1f} min of wall time for all repeats and both arms")
    for i, r in enumerate(results):
        print(f"  repeat {i}: LACE LER {r['ler']:.3e} +- "
              f"{100 * r['relative_se_plug_in']:.1f}% plug-in, "
              f"{100 * r['relative_se_certified']:.1f}% certified "
              f"({100 * r['relative_se_certified_union']:.1f}% with the union "
              f"bound); count strata at the same {r['decodes_per_arm']} "
              f"decodes: {r['plain_ler']:.3e} +- "
              f"{100 * r['plain_relative_se']:.1f}%")
    mean = {key: float(np.mean([r[key] for r in results])) for key in results[0]}
    print(
        f"  mean: LACE {100 * mean['relative_se_plug_in']:.1f}% plug-in / "
        f"{100 * mean['relative_se_certified']:.1f}% certified vs count "
        f"strata {100 * mean['plain_relative_se']:.1f}% at "
        f"{mean['decodes_per_arm']:.0f} decodes per arm; variance gain "
        f"{(mean['plain_relative_se'] / mean['relative_se_plug_in'])**2:.2f}"
        f" plug-in, "
        f"{(mean['plain_relative_se'] / mean['relative_se_certified'])**2:.2f}"
        f" certified")
    if direct:
        est, (lo, hi), shots, seconds = direct
        print(f"  direct MC: {est:.3e} [{lo:.2e}, {hi:.2e}] from {shots} shots "
              f"in {seconds / 60:.0f} min")
    common.write_json(
        "results/validate_ler_summary.json", {
            "rate": args.rate,
            "counts": counts,
            "covered_mass": covered,
            "exact_head": exact,
            "repeats": results,
            "mean": mean,
            "wall_seconds_all": wall,
            "direct": direct,
        })


if __name__ == "__main__":
    main()
