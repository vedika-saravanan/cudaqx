# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Recompute the summary of ``run_lace.py`` reports from their stored
rounds, without new decodes.

Every repeat stores its per-round labels and failures, so the per-round
unbiased variance plug-in, the rule-of-three bound for never-failing
groups, the plain variance at the realised decode count and the plug-in
Neyman ceilings (with and without the pilot) can be rebuilt for reports
written before those fields existed.

Usage::

    python recompute_layer_strata.py results/layers_*.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from types import SimpleNamespace

import numpy as np

import common
from rare_event import allocation
from run_lace import summarise_count


def recompute(report: dict) -> dict:
    args = SimpleNamespace(**report["parameters"])
    for k, pc in report["per_count"].items():
        names = [
            tuple(g["group"]) if isinstance(g["group"], list) else g["group"]
            for g in pc["groups"]
        ]
        weights = np.array([g["weight"] for g in pc["groups"]])
        for rep in pc["repeats"]:
            if "round_labels" not in rep:
                raise ValueError("report predates multi-round allocation")
            labels = [np.array(n, dtype=float) for n in rep["round_labels"]]
            failures = [np.array(f, dtype=float) for f in rep["round_failures"]]
            estimate, variance = allocation.fixed_weight_estimate(
                weights, labels[1:], failures[1:])
            rep["estimate"] = estimate
            rep["standard_error"] = math.sqrt(variance)
            rep.setdefault("plain_shots", int(args.budget))
            certified = allocation.certified_variance(weights, labels[1:],
                                                      failures[1:])
            rep["certified_variance"] = certified["certified"]
            rep["certified_empty_mass"] = certified["empty_mass"]
            rep["certified_bound_group"] = certified["bound_group"]
            rep["certified_bound_rate"] = certified["bound_rate"]
            for stale in ("standard_error_bound", "certified_variance_mixture",
                          "certified_variance_worst", "certified_bound_rates",
                          "certification_labels", "certification_failures"):
                rep.pop(stale, None)
        report["per_count"][k] = summarise_count(pc["repeats"], names, weights,
                                                 pc["rest_mass"], args)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    for path in args.reports:
        report = json.load(open(path))
        try:
            report = recompute(report)
        except ValueError as error:
            print(f"{path}: skipped ({error})", file=sys.stderr)
            continue
        common.write_json(path, report)
        for k, pc in report["per_count"].items():
            print(f"{path} k={k}: f={pc['stratified_mean']:.3e} +- "
                  f"{pc['stratified_sd_analytic']:.1e} (certified "
                  f"{pc['stratified_sd_certified']:.1e}); gain "
                  f"x{pc['variance_gain_analytic']:.2f} [certified "
                  f"x{pc['variance_gain_certified']:.2f}]; plug-in Neyman "
                  f"x{pc['variance_gain_neyman_plugin']:.2f} "
                  f"(x{pc['variance_gain_neyman_plugin_with_pilot']:.2f} "
                  f"with pilot); never-failing mass "
                  f"{pc['never_failing_mass']:.2f}: break-even "
                  f"{pc['never_failing_break_even_rate']:.1e} vs bound "
                  f"{pc['never_failing_bound_rate']}")


if __name__ == "__main__":
    main()
