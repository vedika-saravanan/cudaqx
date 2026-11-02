# LACE: rare-event logical error rate estimation for a fixed decoder

Benchmark and reference implementation of a decoder-agnostic estimator of
the logical error rate (LER) of a fixed, black-box decoder at physical
error rates where the LER is far too small for direct Monte Carlo. Three
estimators are implemented and compared on the same circuits and decoders:

1. **Direct Monte Carlo** (`run_direct_mc.py`): sample the circuit at the
   target rate with Stim, decode every shot. The reference, affordable down
   to LER ~ 1e-7 only with hours of GPU time per rate.
2. **IBM "Fail fast" baseline** (`run_spectrum.py`, arXiv:2511.15177):
   condition on the number of faults `k`, sample each count stratum
   exactly, measure `f(k) = P(fail | k)`, and combine. The ansatz fit IBM
   uses to extrapolate from low `k` is implemented and measured; the
   unbiased variant (exact cross-rate reweighting of the measured `f(k)`)
   is the fair baseline.
3. **LACE, Layer-stratified Allocation with Calibrated Error bars**
   (`run_lace.py`), the method under test: inside each count stratum,
   stratify again exactly by the time structure of the faults (which
   detection layers they occupy), allocate decodes across groups with
   multi-round Neyman allocation and family shrinkage, and report a
   plug-in error bar whose coverage has been measured over 100 repeats.
   Groups that never fail are given labels until a heuristic rule-of-three
   charge for them stays under an allowance (`--allowance 0.15`); that
   charge is reported as a labelled heuristic, not a guarantee, beside a
   worst-case interval (plug-in uncertainty plus a simultaneous upper
   allowance for the never-failing groups; full-interval coverage is
   empirical) that shows a rigorous certificate is not achievable at these
   budgets.

`RESULTS.md` holds every number, labelled measured or projected, the
stress tests against direct Monte Carlo (direct controls at four rates,
100-repeat coverage, ten-repeat spreads, a matched-precision run at
p = 1e-4) and the conclusions. In short, on BB72 with Relay-BP (GPU) and
BP-OSD (CPU): the spectrum is unbiased against direct Monte Carlo at every
rate, LACE's plug-in error bars are calibrated, and measured per count
LACE realises 3.3-3.9x the variance reduction of the unbiased IBM baseline
for Relay-BP at the onset count (2.55x [1.71, 3.81] measured across 100
repeats at half the budget); for BP-OSD the plug-in estimate is 7x but
ten repeats give 3.5x [0.83, 14.7], which does not settle it. At the LER
level the cost model projects
2.9x over IBM and 930x over direct Monte Carlo for Relay-BP at p = 1e-4;
the one completed matched-precision check realised 2.4x over IBM and, in
actual decoder calls, 483x over the direct run (641x at equal precision),
landing inside the direct interval. No worst-case guarantee of a gain is
claimed: the valid allowance for never-failing groups is too wide to
certify anything here, the full worst-case interval's coverage is
empirical, and `RESULTS.md` section 1 records the two invalid attempts
that preceded it. For PyMatching on the surface code the layer structure
adds 1.4x at onset and about 3% above it, so count strata alone are the
right tool there.

## Layout

| Path | Role |
|---|---|
| `rare_event/dem.py` | Stim circuit -> detector error model -> sparse `H`, `O`, priors; noise rescaling; fault propagation by column gather and XOR reduction. |
| `rare_event/strata.py` | Exact fault-count stratification for unequal mechanism probabilities: Poisson-binomial masses, exact conditional subset sampler (positions by inverse transform on cumulative suffix tables, about 1 us per shot at any batch size), exact cross-rate importance weights. |
| `rare_event/spectrum.py` | Failure spectrum `f(k)` with exact binomial intervals, cross-rate reweighting and the IBM `f3`/`f5` ansatz fits (baseline). |
| `rare_event/layers.py` | Exact layer-composition strata of a count stratum (product of per-layer Poisson-binomial masses), position and position + core grouping, exact group sampling vectorised by layer and count. |
| `rare_event/allocation.py` | Hierarchical shrinkage of group rates, Neyman rounds with a proportional floor, fixed-weight combination of rounds (unbiased under adaptive allocation), per-round unbiased variance, the two never-failing charges of `never_failing_bounds` (exposure-aware heuristic, not a bound; simultaneous worst-case allowance for the never-failing groups by Bonferroni and Ville's inequality, added to the plug-in of the failing groups), the allowance floor and the data-driven allowance `optimal_allowance`. |
| `rare_event/onset.py` | Minimal failing sets by descent from observed failures (used by the baseline to test IBM's enclosure assumption). |
| `rare_event/decoders.py` | `FailureOracle` and adapters for PyMatching, `ldpc` BP-OSD and the CUDA-Q QEC `nv-qldpc-decoder` (Relay-BP), all behind one batch `decode` contract. |
| `rare_event/stats.py` | Clopper-Pearson intervals and error propagation. |
| `common.py` | Shared CLI arguments, circuit loading (`bb72`, `surface:d:rounds`, `.stim` files), experiment construction, JSON output. |
| `run_spectrum.py` | Baseline: count-stratified spectrum, exact reweighting, ansatz fits, direct control. |
| `run_direct_mc.py` | Direct Monte Carlo control at a rescaled physical rate with the same decoder instance; checkpointed. |
| `run_exact_head.py` | Exhaustive enumeration of weight-1 and weight-2 configurations (exact low-weight head of the spectrum). |
| `run_lace.py` | LACE, the method under test; one report per run with every round stored. |
| `recompute_layer_strata.py` | Rebuild report summaries from stored rounds without new decodes. |
| `run_ler_speedup.py` | Projected LER-level cost and speedup versus target rate from a spectrum plus the per-count gains (a cost model, labelled as such); `--combine` plots several decoders (`figures/ler_speedup.png`). |
| `summarise_coverage.py` | Coverage and repeat-to-repeat spread of a many-repeat report against a pooled ground truth for the plug-in, heuristic and worst-case intervals; bootstrap interval on the realised gain. |
| `validate_ler.py` | Matched-precision check of the LER projection: assembles LER(p) with plug-in, heuristic and worst-case errors from completed per-count LACE runs at the budgets the projection implied, beside the count-strata estimate at the same decodes and the direct control, reporting actual and precision-normalised call ratios separately. |
| `tests/` | Unit tests on toy models: exact laws of both samplers, mass identities, unbiasedness of the fixed-weight estimator, the never-failing charges (including the data-selection counterexample the heuristic fails and the worst-case allowance passes), the allowance rule, propagation. |
| `results/` | JSON reports behind every number in `RESULTS.md`. |

## Quick start

All commands run from this directory with `PYTHONPATH=.`. The BB72
circuit is `assets/benchmarks/bb72_memory_Z.stim` (6 rounds, 36 detectors
per layer); surface codes are generated by Stim.

Baseline spectrum and direct control (surface code d=5, CPU):

```bash
python run_spectrum.py --circuit surface:5:5 --reference-rate 0.003 \
    --counts 0-20 --shots-per-count 20000 --target-rates 0.003,0.001,0.0003 \
    --onset 3 --direct-shots 400000 --output results/surface5_pymatching_spectrum
```

Baseline spectrum on BB72 with Relay-BP (GPU):

```bash
python run_spectrum.py --circuit bb72 --decoder nv-qldpc \
    --decoder-option batch_size=1024 --counts 3-8 --shots-per-count 300000 \
    --target-rates 0.001,0.0005,0.0002,0.0001 --onset 3 \
    --output results/bb72_relay_spectrum_lowk
```

Direct Monte Carlo control:

```bash
python run_direct_mc.py --circuit bb72 --decoder nv-qldpc \
    --decoder-option batch_size=1024 --rate 1e-4 --shots 240000000 \
    --output results/direct_mc_bb72_relay_1e-4
```

LACE (Relay-BP, onset count 3, 600k decode budget, ten independent
repeats):

```bash
python run_lace.py --circuit bb72 --detectors-per-layer 36 \
    --decoder nv-qldpc --decoder-option batch_size=1024 \
    --grouping position --window 2 --max-span 2 --pilot-fraction 0.1 \
    --rounds 3 --counts 3 --budget 600000 --allowance 0.15 \
    --repeats 10 --seed 20261101 \
    --output results/layers_bb72_relay_position_inround_eps015_10rep
```

`--allowance` takes a fixed value (default `0.15`; the allowance floors
are added on top of the allocation, so the realised decode count exceeds
`--budget` by 1.1x at 600k and up to 2.5x at 200k; the budget is a floor,
not a cap), `auto` (chosen each round from the data; realises more
plug-in gain and leaves a larger heuristic charge at equal decodes, see
`RESULTS.md` section 6), or `none`. `--grouping position_core` adds the
largest single-layer count to the statistic (used for BP-OSD at k = 4).
For BP-OSD use `--decoder bposd`; for surface codes `--circuit surface:5:5
--layers-from-coordinates --decoder pymatching`.

Projected LER-level cost from a spectrum and the per-count reports (later
reports override earlier ones per count, so list the default run first):

```bash
python run_ler_speedup.py --circuit bb72 --spectrum results/bb72_relay_spectrum_lowk_spectrum.json \
    --layers results/layers_bb72_relay_position.json,results/layers_bb72_relay_position_inround_eps015.json,results/layers_bb72_relay_position_inround_eps03_k4.json \
    --exact-head 2:3.18e-7 --rates 0.002,0.001,0.0005,0.0002,0.0001 \
    --label "Relay-BP, BB72" --output results/ler_speedup_bb72_relay.json
python run_ler_speedup.py --combine results/ler_speedup_bb72_relay.json,results/ler_speedup_bb72_bposd.json,results/ler_speedup_surface5_pymatching.json \
    --figure figures/ler_speedup.png
```

Matched-precision check of one projected row with completed runs:

```bash
python validate_ler.py --circuit bb72 --rate 1e-4 --exact-head 2:3.18e-7 \
    --direct results/direct_mc_bb72_relay_1e-4.json \
    results/validate_1e4_bb72_relay_k3.json results/validate_1e4_bb72_relay_k4.json \
    results/validate_1e4_bb72_relay_k5.json
```

Coverage of a many-repeat report against the pooled plain samples of all
reports for the same decoder and count:

```bash
python summarise_coverage.py results/layers_bb72_relay_position_inround_eps015_coverage.json \
    --reference "results/layers_bb72_relay_position*.json" --count 3
```

Tests: `python -m pytest -q tests`.

## Environment used for the results

NVIDIA RTX PRO 6000 Blackwell Workstation Edition, WSL2 Ubuntu, Python
3.12, Stim 1.16.0, PyMatching 2.4.0, ldpc 2.4.1, cudaq-qec 0.8.1
(`nv-qldpc-decoder`, Relay-BP: min-sum DMem, gamma0 0.125, gamma in
[-0.24, 0.66], 1 pre-iteration, up to 600 legs, FirstConv, fp32,
repeatable). Decoder priors are always those of the reference-rate
circuit the decoder was built from: the object under test is one fixed
decoder configuration.
