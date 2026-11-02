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

with the plug-in, heuristic and worst-case variance of each count's
estimate propagated through the exact count masses, the plain conditional
(count-strata) estimate from the same reports at the same decode count,
and the realised decodes and wall time, so the projected cost can be read
against a run that paid for its pilot, allocation and sampling.  Counts
not covered by a report contribute their mass to a reported tail and
nothing to the variance.  The worst-case bound is recomputed from the
stored rounds at level ``alpha / K`` per count so it holds jointly over
the ``K`` covered counts.

Two approximations are deliberate and stated: the per-count estimates were
sampled at the reference physical rate and are combined with the target
rate's count masses (the first-order assumption of the cost model; the
spectrum's exact reweighting agrees with it within 2.5% on BB72), and the
wall time is the reports' total for both arms, not a per-arm measurement.
Against a direct Monte Carlo report the script prints both the actual
decoder-call ratio of the two runs and the ratio normalised to equal
relative standard error; they are different numbers and are labelled.

Usage::

    python validate_ler.py --circuit bb72 --rate 1e-4 --exact-head 2:3.18e-7 \\
        --direct results/direct_mc_bb72_relay_1e-4.json \\
        results/validate_1e4_bb72_relay_k3.json results/validate_1e4_bb72_relay_k4.json \\
        results/validate_1e4_bb72_relay_k5.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import common
from rare_event import allocation, dem, strata

UNION_BOUND_LEVEL = 0.05
TARGET_RELATIVE_ERROR = 0.10


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
    weights_of: dict[int, np.ndarray] = {}
    wall = 0.0
    for path in args.reports:
        report = json.load(open(path))
        wall += float(report["cost"]["wall_seconds"])
        for k, pc in report["per_count"].items():
            per_count[int(k)] = pc["repeats"]
            weights_of[int(k)] = np.array([g["weight"] for g in pc["groups"]])
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
    results = []
    for i in range(repeats):
        ler = head
        var_plug = var_heur = var_worst = var_plain = 0.0
        plain = head
        decodes = 0
        for k in counts:
            r = per_count[k][i]
            P = float(pmf[k])
            ler += P * r["estimate"]
            var_plug += P**2 * r["standard_error"]**2
            var_heur += P**2 * r["heuristic_variance"]
            # Worst case at level alpha / K so the K counts hold jointly.
            labels = [np.array(n, dtype=float) for n in r["round_labels"][1:]]
            fails = [np.array(f, dtype=float) for f in r["round_failures"][1:]]
            joint = allocation.never_failing_bounds(weights_of[k],
                                                    labels,
                                                    fails,
                                                    level=UNION_BOUND_LEVEL /
                                                    len(counts))
            var_worst += P**2 * joint["worst_case"]
            f_plain = r["plain_estimate_same_budget"]
            plain += P * f_plain
            var_plain += P**2 * f_plain * (1 - f_plain) / r["plain_shots"]
            decodes += int(r["plain_shots"])
        results.append({
            "ler": ler,
            "relative_se_plug_in": math.sqrt(var_plug) / ler,
            "relative_se_heuristic": math.sqrt(var_heur) / ler,
            "relative_se_worst_case": math.sqrt(var_worst) / ler,
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
          f"{wall / 60:.1f} min of wall time for all repeats and both arms "
          f"(not a per-arm measurement); per-count estimates sampled at "
          f"p = {reference_rate:g} and combined with the p = {args.rate:g} "
          f"count masses (first-order)")
    for i, r in enumerate(results):
        print(f"  repeat {i}: LACE LER {r['ler']:.3e} +- "
              f"{100 * r['relative_se_plug_in']:.1f}% plug-in, "
              f"{100 * r['relative_se_heuristic']:.1f}% heuristic, "
              f"{100 * r['relative_se_worst_case']:.1f}% worst case; count "
              f"strata at the same {r['decodes_per_arm']} decodes: "
              f"{r['plain_ler']:.3e} +- {100 * r['plain_relative_se']:.1f}%")
    mean = {key: float(np.mean([r[key] for r in results])) for key in results[0]}
    print(
        f"  mean: LACE {100 * mean['relative_se_plug_in']:.1f}% plug-in / "
        f"{100 * mean['relative_se_heuristic']:.1f}% heuristic / "
        f"{100 * mean['relative_se_worst_case']:.1f}% worst case vs count "
        f"strata {100 * mean['plain_relative_se']:.1f}% at "
        f"{mean['decodes_per_arm']:.0f} decodes per arm; variance gain "
        f"{(mean['plain_relative_se'] / mean['relative_se_plug_in'])**2:.2f}"
        f" plug-in, "
        f"{(mean['plain_relative_se'] / mean['relative_se_heuristic'])**2:.2f}"
        f" heuristic, "
        f"{(mean['plain_relative_se'] / mean['relative_se_worst_case'])**2:.2f}"
        f" worst case")
    ratios = None
    if direct:
        est, (lo, hi), shots, seconds = direct
        direct_se = math.sqrt((1 - est) / (est * shots))
        actual = shots / mean["decodes_per_arm"]
        # Each arm scaled to the target relative error by (se / target)^2.
        normalised = (
            shots * (direct_se / TARGET_RELATIVE_ERROR)**2 /
            (mean["decodes_per_arm"] *
             (mean["relative_se_plug_in"] / TARGET_RELATIVE_ERROR)**2))
        ratios = {"actual_calls": actual, "precision_normalised": normalised}
        print(f"  direct MC: {est:.3e} [{lo:.2e}, {hi:.2e}] from {shots} shots "
              f"in {seconds / 60:.0f} min")
        print(f"  decoder calls, direct / LACE arm: {actual:.0f}x actual; "
              f"{normalised:.0f}x precision-normalised to "
              f"{TARGET_RELATIVE_ERROR:.0%} relative SE")
    common.write_json(
        "results/validate_ler_summary.json", {
            "rate": args.rate,
            "reference_rate": reference_rate,
            "counts": counts,
            "covered_mass": covered,
            "exact_head": exact,
            "repeats": results,
            "mean": mean,
            "wall_seconds_all_arms_and_repeats": wall,
            "direct": direct,
            "direct_ratios": ratios,
        })


if __name__ == "__main__":
    main()
