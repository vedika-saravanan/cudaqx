# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Layer-composition stratification of fault-count strata, several counts.

For every requested fault count the stratum is split exactly by the
position statistic of its layer composition (first and last occupied
detection layers for compact sets, densest-window start and count for
spread sets; ``position_core`` adds the largest single-layer count).
Each group receives a mass-proportional pilot, the decoder labels it, and
the remaining budget is Neyman-allocated over several rounds, each round
from all counts so far with family shrinkage, the rounds combined with
fixed weights so the estimate stays unbiased.  Never-failing groups are
certified inside the rounds: they receive enough labels that their
rule-of-three bound adds at most an allowance to the variance, the
allowance fixed (``--certify-fraction 0.15``, the default) or chosen each
round from the data within the budget (``auto``).  The resulting ``f(k)``
is reported with its plug-in and certified standard errors and compared
with plain conditional sampling at the same decoder budget.

Usage::

    python run_lace.py --circuit bb72 --decoder nv-qldpc \
        --decoder-option batch_size=1024 --detectors-per-layer 36 \
        --counts 3,4,5 --budget 600000 --repeats 3 \
        --output results/layers_bb72_relay_position_inround_eps015
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

import common
from rare_event import allocation, layers, strata

# Groups with mass below this fraction of the stratum are pooled into one
# sampled 'rest' group; every group receives at least this many labels per
# allocation round so no group's rate rests on a handful of shots.
MIN_GROUP_MASS = 0.002
MIN_ROUND_LABELS = 100
# Every group keeps at least this fraction of its mass-proportional share
# in each round, so a group whose pilot drew no failure can cost at most
# 1 / MIN_ROUND_SHARE times its plain-sampling variance.
MIN_ROUND_SHARE = 0.2
# Pseudo-trials at the pooled rate per pilot shot: a group with no pilot
# failure is read at one sixth of the pooled rate, enough to keep a share
# of the next round without letting empty groups dominate the allocation.
PILOT_PSEUDO_FRACTION = 0.2
REST_GROUP = ("rest",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_common_arguments(parser)
    parser.add_argument("--counts", default="3,4")
    parser.add_argument("--detectors-per-layer", type=int, default=None)
    parser.add_argument("--layers-from-coordinates", action="store_true")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--grouping",
                        default="position",
                        choices=["position", "position_core"],
                        help="'position': exact (first, last) layers for "
                        "compact sets, densest-window start and count for "
                        "spread sets; 'position_core': position refined by "
                        "the largest single-layer count.")
    parser.add_argument("--max-span", type=int, default=2)
    parser.add_argument(
        "--pilot-fraction",
        type=float,
        default=0.1,
        help="Mass-proportional pilot of this fraction of the budget "
        "(floor MIN_ROUND_LABELS per group).")
    parser.add_argument(
        "--certify-fraction",
        default="0.15",
        help="Certify inside the rounds: give never-failing families enough "
        "labels that their rule-of-three bound term is at most this "
        "fraction of the plain-sampling variance at the planned budget "
        "(default 0.15, the validated setting; the floors may push the "
        "realised decodes above --budget); 'auto' picks the fraction each "
        "round that minimises the projected certified variance within the "
        "budget (allocation.optimal_certify_fraction), which keeps the "
        "budget but certifies less; 'none' disables.")
    parser.add_argument("--rounds",
                        type=int,
                        default=3,
                        help="Neyman-allocated rounds after the pilot.")
    parser.add_argument("--budget", type=int, default=600000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    certify_auto = args.certify_fraction == "auto"
    certify_fraction = (None if args.certify_fraction in (None, "none") or
                        certify_auto else float(args.certify_fraction))

    experiment = common.build_experiment(args)
    model, oracle, rng = experiment.model, experiment.oracle, experiment.rng
    layer_of = layers.mechanism_layers(
        model, args.detectors_per_layer,
        experiment.circuit if args.layers_from_coordinates else None)
    counts = common.parse_counts(args.counts)
    partition = layers.LayerPartition(layer_of, model.priors, max(counts))
    started = time.perf_counter()
    report = {
        "environment": common.environment_record(),
        "model": common.model_record(experiment),
        "parameters": vars(args),
        "num_layers": len([t for t in partition.layers if t >= 0]),
        "per_count": {},
    }
    for k in counts:
        if args.grouping == "position":
            statistic = lambda c: layers.position_group(
                c, partition.layers, args.max_span, args.window)
        else:
            statistic = lambda c: layers.position_core_group(
                c, partition.layers, args.max_span, args.window)
        comp = layers.CompositionStrata(partition, k, statistic)
        total = comp.total_mass
        check = float(strata.poisson_binomial_pmf(model.priors, k)[k])
        assert abs(total - check) <= 1e-9 * check, "composition masses"
        group_mass = comp.group_masses()
        # Pool tiny groups.
        big = {
            g: m for g, m in group_mass.items() if m / total >= MIN_GROUP_MASS
        }
        small = [g for g in group_mass if g not in big]
        names = sorted(big, key=str)
        rest_mass = sum(group_mass[g] for g in small) / total
        if small:
            # The pooled remainder is sampled as one group so the estimator
            # covers the whole stratum (dropping it would bias f(k) by up to
            # rest_mass, which can exceed f(k) itself).
            comp.groups[REST_GROUP] = [i for g in small for i in comp.groups[g]]
            big[REST_GROUP] = sum(group_mass[g] for g in small)
            names.append(REST_GROUP)
        weights = np.array([big[g] / total for g in names])
        # Family of a group is the first element of its key ('compact',
        # 'spread' or 'rest').
        # Groups of one family (same position-free statistic) share a
        # failure-rate prior in the allocation, so position-resolved groups
        # that never fail stop drawing labels once their family has enough
        # failure-free trials.
        bound_families = np.array([
            str(layers.position_family(g)) if isinstance(g, tuple) else str(g)
            for g in names
        ])
        families = bound_families
        plain_sampler = strata.ConditionalSampler(model.priors, k)
        print(f"k={k}: {len(comp.compositions)} compositions, {len(names)} "
              f"groups kept (pooled rest {rest_mass:.2e})")

        def draw(index: int, shots: int) -> np.ndarray:
            return comp.sample_group(names[index], shots, rng)

        repeats = []
        for rep in range(args.repeats):
            calls_before = oracle.decoder_calls
            # Round 0 is a uniform pilot; each later round is Neyman-allocated
            # from all counts so far and the rounds are combined with fixed
            # weights, which keeps the estimator unbiased (see
            # allocation.fixed_weight_estimate).  An unlucky pilot zero in a
            # heavy group is corrected by the next round instead of starving
            # the group for the whole budget.
            pilot = allocation.proportional_pilot(
                weights, int(args.pilot_fraction * args.budget),
                MIN_ROUND_LABELS)
            labels = [pilot]
            chosen_fractions = []
            failures = [
                np.array([
                    int(oracle.failures(draw(i, int(pilot[i]))).sum())
                    for i in range(len(names))
                ])
            ]
            round_budget = (args.budget - int(labels[0].sum())) // args.rounds
            pseudo = PILOT_PSEUDO_FRACTION * float(pilot.mean())
            for round_index in range(1, args.rounds + 1):
                failures_so_far = np.sum(failures, axis=0)
                labels_so_far = np.sum(labels, axis=0)
                floors = None
                if certify_auto or certify_fraction is not None:
                    # Rate estimate for the floor from the pilot and the
                    # rounds so far (allocation may use anything).
                    rate_estimate = float(
                        np.dot(
                            weights,
                            np.divide(failures_so_far,
                                      labels_so_far,
                                      out=np.zeros(len(names)),
                                      where=labels_so_far > 0)))
                    epsilon = certify_fraction
                    if certify_auto:
                        epsilon = allocation.optimal_certify_fraction(
                            weights, failures_so_far == 0,
                            allocation.stratum_sigmas(failures_so_far,
                                                      labels_so_far, pseudo,
                                                      weights, families),
                            rate_estimate, args.budget,
                            float(labels_so_far.sum()))
                        chosen_fractions.append(epsilon)
                    if epsilon is not None:
                        floors = allocation.certification_floor(
                            weights, failures_so_far == 0, rate_estimate,
                            args.budget, epsilon, args.rounds, round_index,
                            np.sum(labels[1:], axis=0)
                            if len(labels) > 1 else np.zeros(len(names)))
                n = allocation.round_labels(round_budget,
                                            weights,
                                            failures_so_far,
                                            labels_so_far,
                                            MIN_ROUND_LABELS,
                                            pseudo=pseudo,
                                            families=families,
                                            floor_share=MIN_ROUND_SHARE,
                                            min_labels=floors)
                labels.append(n)
                failures.append(
                    np.array([
                        int(oracle.failures(draw(i, int(n[i]))).sum())
                        for i in range(len(names))
                    ]))
            estimate, variance = allocation.fixed_weight_estimate(
                weights, labels[1:], failures[1:])
            certified = allocation.certified_variance(weights, labels[1:],
                                                      failures[1:])
            total_f = np.sum(failures, axis=0)
            total_n = np.sum(labels, axis=0)
            # Plain conditional sampling at the decode count the stratified
            # estimator actually used (the floors push it above the budget).
            plain_shots = int(total_n.sum())
            x = plain_sampler.sample(k, plain_shots, rng)
            plain = float(oracle.failures(x).mean())
            repeats.append({
                "estimate": estimate,
                "standard_error": math.sqrt(variance),
                "certified_variance": certified["certified"],
                "certified_empty_mass": certified["empty_mass"],
                "certified_bound_group": certified["bound_group"],
                "certified_bound_rate": certified["bound_rate"],
                "certify_fractions": chosen_fractions,
                "plain_shots": plain_shots,
                "rest_mass_pooled": rest_mass,
                "decoder_calls": oracle.decoder_calls - calls_before,
                "group_labels": total_n.tolist(),
                "group_failures": total_f.tolist(),
                "round_labels": [n.tolist() for n in labels],
                "round_failures": [f.tolist() for f in failures],
                "plain_estimate_same_budget": plain,
            })
            print(f"  repeat {rep}: stratified f({k}) = {estimate:.3e} +- "
                  f"{math.sqrt(variance):.1e} (certified "
                  f"{math.sqrt(certified['certified']):.1e}); plain "
                  f"{plain:.3e} +- "
                  f"{math.sqrt(plain*(1-plain)/plain_shots):.1e} at "
                  f"{plain_shots} decodes")
        report["per_count"][k] = summarise_count(repeats, names, weights,
                                                 rest_mass, args)
        pc = report["per_count"][k]
        certified = pc["variance_gain_certified"] or float("nan")
        print(f"k={k}: f = {pc['stratified_mean']:.3e}; variance gain over "
              f"plain at equal decodes x{pc['variance_gain_analytic']:.2f} "
              f"(certified x{certified:.2f}; "
              f"plug-in Neyman at "
              f"pooled rates x{pc['variance_gain_neyman_plugin']:.2f}, "
              f"x{pc['variance_gain_neyman_plugin_with_pilot']:.2f} with "
              f"the pilot); never-failing mass "
              f"{pc['never_failing_mass']:.2f}, break-even rate "
              f"{pc['never_failing_break_even_rate']:.1e}, bound "
              f"{pc['never_failing_bound_rate']}")
    report["cost"] = {
        **oracle.counters(), "wall_seconds": time.perf_counter() - started
    }
    common.write_json(os.path.join(args.output, "layer_strata.json"), report)
    print(f"wrote {os.path.join(args.output, 'layer_strata.json')}")


def summarise_count(repeats: list[dict], names: list, weights: np.ndarray,
                    rest_mass: float, args: argparse.Namespace) -> dict:
    """Per-count summary from the repeats' stored rounds (also used by
    ``recompute_layer_strata.py`` on earlier reports)."""
    pooled_f = np.sum([r["group_failures"] for r in repeats], axis=0)
    pooled_n = np.sum([r["group_labels"] for r in repeats], axis=0)
    pooled_rates = np.divide(pooled_f,
                             pooled_n,
                             out=np.zeros(len(names)),
                             where=pooled_n > 0)
    f_all = float(np.dot(weights, pooled_rates))
    neyman_sigma = float(
        np.dot(weights, np.sqrt(pooled_rates * (1 - pooled_rates))))
    plugin_gain = (f_all * (1 - f_all) /
                   neyman_sigma**2 if neyman_sigma > 0 else float("nan"))
    pilot_total = int(np.mean([sum(r["round_labels"][0]) for r in repeats]))
    estimates = np.array([r["estimate"] for r in repeats])
    plains = np.array([r["plain_estimate_same_budget"] for r in repeats])
    pooled = float(estimates.mean())
    shots = float(np.mean([sum(r["group_labels"]) for r in repeats]))
    plain_sd = math.sqrt(pooled * (1 - pooled) / shots)
    strat_sd = float(np.mean([r["standard_error"] for r in repeats]))
    certified_sd = float(
        np.mean([math.sqrt(r["certified_variance"]) for r in repeats]))

    def gain(sd: float) -> float:
        return (plain_sd / sd)**2 if sd > 0 else float("nan")

    # Rate of the never-failing groups at which the gain would vanish
    # (common-rate, first order) and the rate the certified bound actually
    # charges its binding group, averaged over repeats.
    break_even = []
    for r in repeats:
        labels = [np.array(n, dtype=float) for n in r["round_labels"][1:]]
        fails = [np.array(f, dtype=float) for f in r["round_failures"][1:]]
        break_even.append(
            allocation.zero_group_break_even(weights, labels, fails,
                                             plain_sd**2))
    bound_rates = [
        r["certified_bound_rate"]
        for r in repeats
        if r.get("certified_bound_rate") is not None
    ]
    bound_rate = float(np.mean(bound_rates)) if bound_rates else None
    never = pooled_f == 0

    return {
        "groups": [{
            "group": list(g) if isinstance(g, tuple) else g,
            "weight": float(w)
        } for g, w in zip(names, weights)],
        "rest_mass":
            rest_mass,
        "repeats":
            repeats,
        "stratified_mean":
            pooled,
        "stratified_sd_analytic":
            strat_sd,
        "stratified_sd_realised":
            float(estimates.std(ddof=1)) if len(estimates) > 1 else None,
        "plain_mean":
            float(plains.mean()),
        "plain_shots_mean":
            shots,
        "stratified_sd_certified":
            certified_sd,
        # Gain with the never-failing groups charged by the exposure-aware
        # bound (allocation.certified_variance).
        "variance_gain_certified":
            gain(certified_sd),
        "plain_sd_analytic":
            plain_sd,
        "plain_sd_realised":
            float(plains.std(ddof=1)) if len(plains) > 1 else None,
        # Per-round unbiased plug-in (zero-failure groups at rate 0).
        "variance_gain_analytic":
            gain(strat_sd),
        # Plug-in Neyman at the pooled rates (zero-failure groups at sigma
        # 0), without and with the uniform pilot charged.
        "variance_gain_neyman_plugin":
            plugin_gain,
        "variance_gain_neyman_plugin_with_pilot":
            plugin_gain * max(args.budget - pilot_total, 0) / args.budget,
        "never_failing_mass":
            float(weights[never].sum()),
        "never_failing_break_even_rate":
            float(np.mean(break_even)),
        "never_failing_bound_rate":
            bound_rate,
    }


if __name__ == "__main__":
    main()
