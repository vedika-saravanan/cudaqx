# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Coverage and repeat-to-repeat spread of a many-repeat layer-strata report.

Given one ``run_lace.py`` report with many repeats and a set of
reference reports for the same decoder and count, the script pools the
plain conditional samples of every repeat of every report into a
ground-truth rate, then checks how often each repeat's intervals cover it:
the plug-in interval (per-round unbiased variance), the certified interval
(plug-in plus the family rule-of-three terms) and, as a control, the plain
estimator's own binomial interval.  It also reports the realised spread of
the stratified estimates against the mean reported standard error (their
ratio should be 1 if the plug-in is right) and bootstrap intervals on the
plug-in and certified variance gains across repeats.

Usage::

    python summarise_coverage.py results/layers_bb72_relay_position_inround_eps015_coverage.json \
        --reference "results/layers_bb72_relay_position*.json" --count 3
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

Z_68 = 1.0
Z_95 = 1.96
BOOTSTRAP_RESAMPLES = 2000


def per_count(report: dict, count: int) -> dict | None:
    """The ``per_count`` entry for ``count`` (reports key it by string)."""
    return report["per_count"].get(str(count))


def pooled_truth(paths: list[str], count: int) -> tuple[float, int, int]:
    shots = failures = 0
    for path in paths:
        entry = per_count(json.load(open(path)), count)
        if entry is None:
            continue
        for repeat in entry["repeats"]:
            n = int(repeat["plain_shots"])
            shots += n
            failures += int(round(repeat["plain_estimate_same_budget"] * n))
    return failures / shots, failures, shots


def coverage(estimates: np.ndarray, sds: np.ndarray, truth: float,
             z: float) -> float:
    return float(np.mean(np.abs(estimates - truth) <= z * sds))


def bootstrap_gain(strat_est: np.ndarray, plain_est: np.ndarray,
                   rng: np.random.Generator) -> tuple[float, float, float]:
    """Realised variance gain var(plain)/var(stratified) across repeats
    with a percentile bootstrap interval."""
    n = strat_est.size
    point = float(np.var(plain_est, ddof=1) / np.var(strat_est, ddof=1))
    boots = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        vs = np.var(strat_est[idx], ddof=1)
        if vs > 0:
            boots.append(np.var(plain_est[idx], ddof=1) / vs)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report")
    parser.add_argument("--reference",
                        required=True,
                        help="Glob of reports whose plain samples form the "
                        "ground truth (the report itself is included).")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()

    report = json.load(open(args.report))
    entry = per_count(report, args.count)
    if entry is None:
        raise SystemExit(f"count {args.count} not in {args.report}")
    paths = sorted(set(glob.glob(args.reference)) | {args.report})
    truth, truth_failures, truth_shots = pooled_truth(paths, args.count)
    truth_rel_se = 1.0 / np.sqrt(truth_failures)

    repeats = entry["repeats"]
    est = np.array([r["estimate"] for r in repeats])
    sd_plugin = np.array([r["standard_error"] for r in repeats])
    sd_cert = np.sqrt(
        np.array([r["certified_variance_mixture"] for r in repeats]))
    plain_est = np.array([r["plain_estimate_same_budget"] for r in repeats])
    plain_shots = np.array([r["plain_shots"] for r in repeats], dtype=float)
    sd_plain = np.sqrt(plain_est * (1 - plain_est) / plain_shots)
    decodes = np.array([r["decoder_calls"] for r in repeats], dtype=float)

    rng = np.random.default_rng(args.seed)
    gain, gain_lo, gain_hi = bootstrap_gain(est, plain_est, rng)
    reported_plugin_gain = float(np.mean(sd_plain**2) / np.mean(sd_plugin**2))
    reported_cert_gain = float(np.mean(sd_plain**2) / np.mean(sd_cert**2))

    print(f"{len(repeats)} repeats, {decodes.mean():.0f} decodes each; "
          f"ground truth f = {truth:.4e} from {truth_failures} plain failures "
          f"in {truth_shots} shots ({100 * truth_rel_se:.1f}% rel. SE), "
          f"{len(paths)} reports")
    rel_se_mean = est.std(ddof=1) / np.sqrt(est.size) / truth
    z_mean = (est.mean() / truth - 1) / np.hypot(rel_se_mean, truth_rel_se)
    print(
        f"stratified mean {est.mean():.4e} ({100 * (est.mean() / truth - 1):+.1f}% "
        f"vs truth, SE of mean {100 * rel_se_mean:.1f}%, {z_mean:+.1f} sigma with "
        f"the truth's own SE); plain mean {plain_est.mean():.4e} "
        f"({100 * (plain_est.mean() / truth - 1):+.1f}%)")
    print(f"realised SD / mean reported SD: plug-in "
          f"{est.std(ddof=1) / sd_plugin.mean():.2f}, certified "
          f"{est.std(ddof=1) / sd_cert.mean():.2f}, plain control "
          f"{plain_est.std(ddof=1) / sd_plain.mean():.2f}")
    for label, sds in (("plug-in", sd_plugin), ("certified", sd_cert)):
        print(f"coverage {label:9s}: 68% nominal -> "
              f"{100 * coverage(est, sds, truth, Z_68):.0f}%, 95% nominal -> "
              f"{100 * coverage(est, sds, truth, Z_95):.0f}%")
    print(
        f"coverage plain    : 68% nominal -> "
        f"{100 * coverage(plain_est, sd_plain, truth, Z_68):.0f}%, 95% nominal -> "
        f"{100 * coverage(plain_est, sd_plain, truth, Z_95):.0f}%")
    print(
        f"variance gain across repeats (realised): {gain:.2f} "
        f"[95% bootstrap {gain_lo:.2f}, {gain_hi:.2f}]; reported plug-in "
        f"{reported_plugin_gain:.2f}, reported certified {reported_cert_gain:.2f}"
    )
    per_repeat_plugin = sd_plain**2 / sd_plugin**2
    per_repeat_cert = sd_plain**2 / sd_cert**2
    print(
        f"per-repeat plug-in gain: median {np.median(per_repeat_plugin):.2f}, "
        f"range [{per_repeat_plugin.min():.2f}, {per_repeat_plugin.max():.2f}]; "
        f"certified: median {np.median(per_repeat_cert):.2f}, range "
        f"[{per_repeat_cert.min():.2f}, {per_repeat_cert.max():.2f}]")


if __name__ == "__main__":
    main()
