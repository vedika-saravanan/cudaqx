# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Logical-error-rate speedup and decoder cost of layer stratification.

Combines a fault-count spectrum (plain conditional sampling per count) with
the per-count layer-stratification measurements into the quantities the
study is about: for each target physical rate ``p``,

    LER(p) = sum_k P_k(p) f_k  (+ exact low-weight terms if given),

the decoder calls needed for a relative standard error ``eps`` on
``LER(p)`` under Neyman allocation across counts, and the direct Monte
Carlo cost ``(1 - LER) / (eps^2 LER)``.  Plain conditional sampling costs

    N_plain = (sum_k P_k sqrt(v_k))^2 / (eps LER)^2,   v_k = f_k (1 - f_k).

Layer stratification in count ``k`` was measured at one budget ``B_k``
with a uniform pilot of ``pilot_k`` decodes that is excluded from the
estimate, so its variance scales as ``v_k / (G'_k (n_k - pilot_k))`` with
the main-stage gain ``G'_k = G_k (B_k - pilot_k) / B_k`` (``G_k`` the gain
at equal total decodes).  Minimising the total cost gives

    N_layer = sum_k pilot_k + (sum_k P_k sqrt(v_k / G'_k))^2 / (eps LER)^2

where only the counts with a measured gain carry a pilot.  The speedup is
``N_plain / N_layer``.  ``f_k`` is the spectrum's value at the reference
rate (first-order assumption shared with Fail fast; on BB72 it agrees
with exact reweighting to <1% at the rates below); the gain range
``[G_bound, G_plugin]`` from the layer reports is propagated as a range.
Counts above the measured range contribute their mass to a reported tail
and nothing to the variance.

Usage::

    python run_ler_speedup.py --circuit bb72 --rates 2e-3,1e-3,5e-4,2e-4,1e-4 \
        --spectrum results/bb72_relay_spectrum_lowk_spectrum.json \
        --layers results/layers_bb72_relay_position.json --label "Relay-BP BB72" \
        --exact-head 2:3.18e-7 \
        --output results/ler_speedup_bb72_relay.json --figure figures/ler_speedup.png
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

import common
from rare_event import dem, strata

RELATIVE_ERROR = 0.10


def load_spectrum(path: str) -> dict[int, tuple[int, int]]:
    """``{k: (trials, failures)}`` from a ``run_spectrum.py`` report."""
    report = json.load(open(path))
    spectrum = report["spectrum"]
    return {
        int(k): (int(n), int(f)) for k, n, f in zip(
            spectrum["counts"], spectrum["trials"], spectrum["failures"])
    }


def load_gains(paths: str) -> dict[int, dict[str, float]]:
    """Per-count gains from one or more ``run_lace.py`` reports
    (comma-separated; later reports override earlier ones per count).

    ``gain`` is the per-round plug-in gain (never-failing groups at rate
    zero); ``gain_certified`` charges them at their family rule-of-three
    bound (the certified gain, equal to the plug-in when nothing is
    unmeasured).  The pilot is the stored first round of the first repeat,
    so uniform and mass-proportional pilots are both accounted for.
    """
    gains: dict[int, dict[str, float]] = {}
    for path in paths.split(","):
        report = json.load(open(path))
        params = report["parameters"]
        for k, pc in report["per_count"].items():
            gain = pc.get("variance_gain_analytic")
            if gain is None or not math.isfinite(gain):
                continue
            certified = pc.get("variance_gain_certified")
            if certified is None or not math.isfinite(certified):
                certified = pc.get("variance_gain_bound", gain)
            pilot = int(sum(pc["repeats"][0]["round_labels"][0]))
            budget = int(pc.get("plain_shots_mean") or params["budget"])
            gains[int(k)] = {
                "gain": float(gain),
                "gain_certified": float(certified),
                "f": float(pc["stratified_mean"]),
                "se": float(pc["stratified_sd_analytic"]),
                "budget": budget,
                "pilot": pilot,
                "source": os.path.basename(path),
            }
    return gains


def layer_cost(P: dict[int, float],
               f: dict[int, float],
               gains: dict[int, dict[str, float]],
               counts: list[int],
               eps: float,
               ler: float,
               key: str,
               fixed_pilot: bool = False) -> tuple[float, dict[int, float]]:
    """Decoder calls for relative error ``eps`` with layer stratification in
    the measured counts, and the implied decodes per count.

    ``fixed_pilot=False`` (constant pilot share): the gain ``G_k`` measured
    at equal total decodes is taken to hold at the implied per-count
    budget, i.e. the pilot shrinks with the budget; the implied budgets
    are returned so a reader can see when they fall below the budget the
    gain was measured at.  ``fixed_pilot=True``: the measured uniform
    pilot is charged as a fixed cost per count and only the main stage
    scales (pessimistic for targets far cheaper than the measured runs).
    """
    sigma = 0.0
    pilots = 0
    terms = {}
    for k in counts:
        v = f[k] * (1 - f[k])
        if k in gains:
            g = gains[k]
            gain = g[key]
            if fixed_pilot:
                gain *= (g["budget"] - g["pilot"]) / g["budget"]
                pilots += g["pilot"]
            terms[k] = P[k] * math.sqrt(v / gain)
        else:
            terms[k] = P[k] * math.sqrt(v)
        sigma += terms[k]
    variable = (sigma / (eps * ler))**2
    implied = {
        k: (terms[k] / sigma) * variable +
           (gains[k]["pilot"] if fixed_pilot and k in gains else 0)
        for k in counts
    }
    return pilots + variable, implied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--combine",
        default=None,
        help="Comma-separated reports of this script to plot together into "
        "--figure; all other arguments are then ignored.")
    parser.add_argument("--circuit", default=None)
    parser.add_argument("--reference-rate", type=float, default=None)
    parser.add_argument("--rates", default=None)
    parser.add_argument("--spectrum", default=None)
    parser.add_argument("--layers",
                        default=None,
                        help="Comma-separated run_lace.py reports; later ones "
                        "override earlier ones per count.")
    parser.add_argument("--label", default="")
    parser.add_argument("--relative-error", type=float, default=RELATIVE_ERROR)
    parser.add_argument(
        "--exact-head",
        default="",
        help="Exactly known low-weight terms 'k:f_k,...' (zero variance, "
        "zero decoder cost), e.g. the enumerated weight-2 Relay-BP failures.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--figure", default=None)
    args = parser.parse_args()
    if args.combine:
        plot(args.figure, [json.load(open(p)) for p in args.combine.split(",")])
        return
    for name in ("circuit", "rates", "spectrum", "layers", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required without --combine")

    circuit, reference_rate = common.load_circuit(args.circuit,
                                                  args.reference_rate)
    rates = [float(r) for r in args.rates.split(",")]
    priors = dem.priors_at_rates(circuit, reference_rate, rates)
    spectrum = load_spectrum(args.spectrum)
    gains = load_gains(args.layers)
    counts = sorted(spectrum)
    # f_k from the spectrum (unbiased, same source as the baseline tables);
    # the layer reports' f_k are listed beside them in the output.
    f = {k: spectrum[k][1] / spectrum[k][0] for k in counts}
    exact = {}
    for item in filter(None, args.exact_head.split(",")):
        k, value = item.split(":")
        exact[int(k)] = float(value)
    eps = args.relative_error
    onset = min(k for k in counts if f[k] > 0)
    rows = []
    for rate in rates:
        pmf = strata.poisson_binomial_pmf(priors[rate], max(counts))
        P = {k: float(pmf[k]) for k in counts}
        below = float(sum(pmf[k] for k in range(min(counts))))
        tail = 1.0 - sum(P.values()) - below
        ler_sampled = sum(P[k] * f[k] for k in counts)
        ler_exact = sum(float(pmf[k]) * fk for k, fk in exact.items())
        ler = ler_sampled + ler_exact
        plain_sigma = sum(P[k] * math.sqrt(f[k] * (1 - f[k])) for k in counts)
        plain_cost = (plain_sigma / (eps * ler))**2
        layer_plugin, implied = layer_cost(P, f, gains, counts, eps, ler,
                                           "gain")
        layer_bound, _ = layer_cost(P, f, gains, counts, eps, ler,
                                    "gain_certified")
        layer_fixed, _ = layer_cost(P,
                                    f,
                                    gains,
                                    counts,
                                    eps,
                                    ler,
                                    "gain",
                                    fixed_pilot=True)
        # Counts whose implied budget is below the one the gain was
        # measured at (the constant-share model extrapolates there).
        below_measured = {
            k: implied[k] for k in gains if implied[k] < gains[k]["budget"]
        }
        direct_cost = (1 - ler) / (eps**2 * ler)
        share = {
            k: P[k] * math.sqrt(f[k] * (1 - f[k])) / plain_sigma for k in counts
        }
        rows.append({
            "rate": rate,
            "ler": ler,
            "ler_exact_head": ler_exact,
            "mass_below_measured": below,
            "mass_above_measured": tail,
            "direct_cost": direct_cost,
            "plain_stratified_cost": plain_cost,
            "layer_stratified_cost": layer_plugin,
            "layer_stratified_cost_certified": layer_bound,
            "layer_stratified_cost_fixed_pilot": layer_fixed,
            "implied_decodes_by_count": implied,
            "counts_below_measured_budget": below_measured,
            "speedup_over_plain": plain_cost / layer_plugin,
            "speedup_over_plain_certified": plain_cost / layer_bound,
            "speedup_over_plain_fixed_pilot": plain_cost / layer_fixed,
            "speedup_over_direct": direct_cost / layer_plugin,
            "onset_count": onset,
            "plain_variance_share_by_count": share,
        })
        print(
            f"p={rate:.1e}: LER={ler:.2e}  decodes for {eps:.0%}: direct "
            f"{direct_cost:.2e}, count-stratified {plain_cost:.2e}, layer-"
            f"stratified {layer_plugin:.2e} (certified {layer_bound:.2e}, fixed "
            f"pilot {layer_fixed:.2e}); speedup x{plain_cost/layer_plugin:.2f}"
            f" [certified x{plain_cost/layer_bound:.2f}; fixed pilot "
            f"x{plain_cost/layer_fixed:.2f}]; onset k={onset} share of plain"
            f" variance {share[onset]:.2f}; implied k={onset} decodes "
            f"{implied[onset]:.1e}" +
            (f" (below measured budget for k={sorted(below_measured)})"
             if below_measured else ""))
    report = {
        "label": args.label,
        "circuit": args.circuit,
        "reference_rate": reference_rate,
        "relative_error": eps,
        "counts": counts,
        "f": {
            k: f[k] for k in counts
        },
        "exact_head": exact,
        "gains": gains,
        "rows": rows,
    }
    common.write_json(args.output, report)
    print(f"wrote {args.output}")
    if args.figure:
        plot(args.figure, [report])


def plot(path: str, reports: list[dict]) -> None:
    """Plug-in speedup (band: fixed pilot to constant pilot share),
    certified speedup, and decoder cost versus LER."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.4))
    for report in reports:
        rows = report["rows"]
        ler = [r["ler"] for r in rows]
        label = report["label"] or report["circuit"]
        line, = ax1.plot(ler, [r["speedup_over_plain"] for r in rows],
                         "o-",
                         label=label)
        ax1.fill_between(ler,
                         [r["speedup_over_plain_fixed_pilot"] for r in rows],
                         [r["speedup_over_plain"] for r in rows],
                         color=line.get_color(),
                         alpha=0.2)
        ax2.plot(ler, [r["speedup_over_plain_certified"] for r in rows],
                 "o-",
                 color=line.get_color(),
                 label=label)
        ax3.plot(ler, [r["direct_cost"] for r in rows], ":", color="gray")
        ax3.plot(ler, [r["plain_stratified_cost"] for r in rows],
                 "s--",
                 color=line.get_color(),
                 label=f"{label}: count strata")
        ax3.plot(ler, [r["layer_stratified_cost"] for r in rows],
                 "o-",
                 color=line.get_color(),
                 label=f"{label}: layer strata")
    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log")
        ax.set_xlabel("logical error rate")
        ax.invert_xaxis()
    ax1.set_ylabel("variance speedup over count stratification")
    ax1.axhline(1.0, color="gray", lw=0.8)
    ax1.legend(fontsize=8)
    ax1.set_title("Plug-in speedup\n(band: measured pilot as fixed cost, "
                  "to constant pilot share)")
    ax2.set_ylabel("certified speedup over count stratification")
    ax2.axhline(1.0, color="gray", lw=0.8)
    ax2.legend(fontsize=8)
    ax2.set_title("Certified speedup\n(never-failing groups at their "
                  "family bound)")
    ax3.set_yscale("log")
    ax3.set_ylabel(f"decoder calls for {int(reports[0]['relative_error']*100)}%"
                   " relative error")
    ax3.legend(fontsize=7)
    ax3.set_title("Decoder cost (dotted gray: direct Monte Carlo)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
