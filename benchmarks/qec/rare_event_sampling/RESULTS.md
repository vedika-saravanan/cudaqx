# LACE results: layer-stratified allocation with certified error bars against direct Monte Carlo and the IBM baseline

All numbers come from the JSON reports in `results/` with the stated seeds;
nothing is tuned after the fact. Hardware and software are listed in
`README.md`. "Direct" always means an independent Monte Carlo of the same
circuit decoded by the same decoder instance, the ground truth wherever it
is affordable. Codes: BB72 (`[[72,12,6]]`, 6 rounds, reference rate
p = 2e-3) with Relay-BP (GPU, batch 1024) and with `ldpc` BP-OSD (CPU,
min-sum, 100 iterations, OSD-CS order 10); surface code d = 5, 5 rounds,
p = 3e-3, with PyMatching.

Two kinds of number appear below and are labelled throughout. *Measured*
numbers come from completed runs: per-count gains, repeats, coverage, the
direct controls and the matched-precision check of section 5. *Projected*
numbers come from the cost model of `run_ler_speedup.py`, which combines
measured per-count gains with exact count masses to estimate decoder calls
at target rates where no run was made; section 4 is projected and says how
far it extrapolates.

## 1. The three estimators

**Direct Monte Carlo.** Sample the circuit at the target rate p with Stim's
circuit sampler, decode every shot, count failures. Unbiased; the cost to a
relative error e is 1 / (e^2 LER), so 2.3e8 decodes for 10% at LER 4e-7,
and every rate needs its own run.

**IBM "Fail fast" (count strata).** Write LER(p) = sum_k P_k(p) f(k) with
P_k the exact Poisson-binomial mass of k faults and f(k) = P(fail | k)
measured by exact conditional sampling of each count stratum once. In
`run_spectrum.py` the stored failing configurations are reweighted exactly
to each target rate (the exact likelihood ratio of the two rates' odds is
applied per configuration; `rare_event/spectrum.py`). IBM's paper
additionally fits an ansatz to f(k) at low k and extrapolates; that fit is
implemented and measured in section 2 and is biased on BB72, so the
unbiased reweighted variant is the baseline used throughout.

**LACE, Layer-stratified Allocation with Certified Error bars (this work).**
Inside each count stratum the configurations are split exactly by the
composition of the k faults over the detection layers (the mass of each
composition is a product of per-layer Poisson-binomial terms, so every
group weight is exact), grouped by a position statistic: for compact sets
(span at most 2 layers) the exact first and last layer, for spread sets the
start and count of the densest 2-layer window; `position_core` refines by
the largest single-layer count. Groups below 0.2% of the stratum are
pooled into one sampled "rest" group, so nothing is dropped. Allocation:
a mass-proportional pilot of 10% of the budget, then three Neyman rounds
whose rates are shrunk hierarchically toward the pooled rate of the
group's position-free family; rounds are combined with weights fixed in
advance, which keeps the estimator unbiased under adaptive allocation, and
the variance is the per-round unbiased plug-in. Certification: a group
that never fails contributes zero to the plug-in variance, but zero
failures do not mean zero rate. If the never-failing groups have labels
n_g and true rates f_g, the observation has probability at most
exp(-sum_g n_g f_g), so sum_g n_g f_g <= 3 holds with probability at least
95%, path by path and therefore also under adaptive allocation; the
*certified* variance adds the largest variance those rates can produce
under that constraint, which is one group at rate 3 / n_g, the group with
the largest w_g^2 / n_g^2 (`allocation.certified_variance`). It is a bound
per count stratum at 95%; combining several counts needs a union bound,
which section 5 applies. Inside the rounds the never-failing groups
receive enough labels that this charge is at most an allowance epsilon
times the plain-sampling variance at the planned budget (every empty group
is pushed toward n_g = w_g sqrt(3 N / (epsilon f)); the allowance is
fixed at 0.15 by default or chosen each round by `auto`, section 6). The
gain is reported two ways: *plug-in* (variance of plain conditional
sampling at the same decode count over the estimator's plug-in variance;
what the estimator realises, confirmed by repeats) and *certified* (the
same with the certified variance; what one run can promise with no
assumption about the never-failing groups).

An earlier version of the certified variance pooled the labels of a
position-free family and charged every member group the pooled rate. That
is not a bound: a group with fewer labels than its share of the mass can
hide a higher rate than the pooled figure allows. It was replaced by the
exposure-aware bound above on 2026-11-02 and every report was rebuilt from
its stored rounds; the certified figures below are the rebuilt ones and
are higher than the pooled heuristic gave in every run, because that
heuristic charged all empty groups at once.

The same stored spectrum answers every target rate; LACE's per-count gains
enter the LER-level cost through the exact count masses.

## 2. The IBM baseline on its own terms (measured)

Surface code d = 5, PyMatching: 21 strata (k = 0..20), 20,000 exact
conditional samples each (420,000 decodes, 13.6 s). All 50 reduced failing
sets have weight 3, so IBM's enclosure model holds here.

| p | Exact stratified | Reweighted (unbiased) | f3 ansatz | f5 ansatz | Direct (400k shots) |
|---|---|---|---|---|---|
| 0.003 | 3.20e-3 [2.54e-3, 4.18e-3] | - | 4.66e-3 | 3.21e-3 | 3.25e-3 [3.08e-3, 3.43e-3] |
| 0.001 | - | 1.42e-4 +- 0.14e-4 | 2.80e-4 | 1.37e-4 | 1.13e-4 [0.82e-4, 1.51e-4] |
| 0.0003 | - | 4.4e-6 +- 0.6e-6 | 9.5e-6 | 4.0e-6 | 1.0e-5 (4 failures) [2.7e-6, 2.6e-5] |

BB72, Relay-BP: 25 strata (k = 0..24), 20,000 samples each (500,000
decodes, 48 s), then a second pass with IBM's allocation (k = 3..8,
300,000 samples each, 1.8M decodes, 128 s). Minimal failing sets found by
descent from 40 observed failures have weights 3 (19), 4 (11), 5 (9) and
6 (1): many failures are irreducible at weight 4-6, so the enclosure model
behind the ansatz is violated for Relay-BP on BB72.

| p | Reweighted (unbiased) | f3 ansatz | Direct (400k shots) |
|---|---|---|---|
| 0.002 | 9.56e-3 [8.14e-3, 1.13e-2] | 1.12e-2 | 9.34e-3 [9.04e-3, 9.64e-3] |
| 0.0015 | 3.29e-3 +- 0.10e-3 | 4.39e-3 | 3.32e-3 [3.15e-3, 3.51e-3] |
| 0.001 | 7.5e-4 +- 0.5e-4 | 1.24e-3 | 8.1e-4 [7.2e-4, 9.0e-4] |
| 0.0005 | 7.6e-5 +- 1.4e-5 | 1.56e-4 | 7.3e-5 [4.9e-5, 10.4e-5] |
| 0.0002 | 5.0e-6 +- 1.8e-6 | 1.08e-5 | 2.5e-6 (1 failure) [6e-8, 1.4e-5] |

Second pass (k = 3..8, 300k samples each): f(3) = 8.7e-5, f(4) = 4.3e-4,
f(5) = 1.45e-3, f(6) = 3.5e-3, f(7) = 7.3e-3, f(8) = 1.34e-2; k <= 2 are
non-failing by exhaustive enumeration of weight 1 and 2 for BP-OSD and
PyMatching and f(2) = 3.18e-7 exactly for Relay-BP (5 failing weight-2
pairs, `run_exact_head.py`).

| p | Reweighted from k = 3..8 (exact) | Relative SE | Direct shots for the same precision |
|---|---|---|---|
| 0.0005 | 7.00e-5 +- 0.29e-5 | 4% | 9e6 (1.8e6 used) |
| 0.0002 | 3.60e-6 +- 0.35e-6 | 10% | 2.8e7 |
| 0.0001 | 4.2e-7 +- 0.6e-7 | 14% | 1.3e8 |

The ansatz over-predicts by 1.2x at p = 2e-3 and 2.2x at p = 2e-4 and its
projection depends on allocation and model choice (f5 fits within 7 nats
of log-likelihood project 0.36x to 0.94x of the unbiased value at 2e-4)
far more than on its 2% bootstrap width. The exact reweighting is the
baseline; the ansatz is not used again.

## 3. LACE: gains per count (measured)

Each row is the estimator's gain over plain conditional sampling of the
same count stratum at the realised decode count. The certification floors
add to the nominal budget, so the realised count exceeds it (section 6
quantifies this); every comparison is at the realised count. "Never-
failing mass" is the share of the stratum whose groups produced no failure
and are charged by the certified bound.

| Decoder, k | configuration | decodes per arm | plug-in gain | certified gain | never-failing mass |
|---|---|---|---|---|---|
| Relay-BP, 3 | in-round, epsilon 0.3 | 0.93M | 3.70x | **2.00x** | 0.74 |
| Relay-BP, 3 | in-round, epsilon 0.15 | 1.06M | 3.28x | **2.22x** | 0.72 |
| Relay-BP, 3 | in-round, epsilon 0.15, 3x budget | 2.50M | 3.88x | **2.81x** | 0.65 |
| Relay-BP, 4 | in-round, epsilon 0.3 | 0.65M | 1.66x | **1.15x** | 0.31 |
| BP-OSD, 3 | in-round, epsilon 0.3 | 0.97M | 7.02x | **2.19x** | 0.93 |
| BP-OSD, 3 | in-round, epsilon 0.3, 2x budget | 1.99M | 7.13x | **2.54x** | 0.88 |
| BP-OSD, 3 | in-round, epsilon 0.15, 2x budget | 2.51M | 1.38x | 1.01x | 0.89 |
| BP-OSD, 4 | position + core, in-round, epsilon 0.3 | 1.63M | 4.77x | **2.90x** | 0.81 |
| PyMatching d=5, 3 | position | 0.60M | 1.44x | 1.18x | 0.05 |
| PyMatching d=5, 4 | position | 0.60M | 1.03x | 1.03x | 0.00 |

Reports: `results/layers_bb72_relay_position_inround_{eps015,eps03,eps015_1p8M,eps03_k4}.json`,
`results/layers_bb72_bposd_position_inround_{eps03,eps03_1p2M,eps015_1p2M}.json`,
`results/layers_bb72_bposd_position_core_inround_eps03_k4_1p2M.json`,
`results/layers_surface5_pymatching_position.json`.

The certified gain tracks the sqrt(N) scaling of the certification cost:
for Relay-BP at onset it is 2.0-2.2x at about 1M decodes and 2.8x at
2.5M, while the plug-in gain stays near 3.3-3.9x. BP-OSD has the largest
plug-in gain (7x) because 88-93% of its onset mass never fails and is
read at zero, and the same fact makes its certified gain depend on budget
(2.2x at 1M, 2.5x at 2M); the tight allowance 0.15 at 2.5M spent the
budget on certification and gained nothing. For matching on the surface
code the layer structure is worth 1.18x certified at onset and 1.03x
above it at d = 5, and nothing certifiable at d = 7 (f(4) = 6.7e-6 puts
the onset stratum behind a 1 / f wall; `results/layers_surface7_*`), so
count strata alone are the right tool there.

## 4. LER-level cost against both baselines (projected)

`run_ler_speedup.py` is a cost model. It takes the k <= 8 spectrum (one
5e5-decode measurement) and the per-count gains of section 3, holds f(k)
at its reference-rate value (a first-order assumption; the exactly
reweighted LER from the spectrum run is listed beside it where the run
stored one, and agrees within 2.5%), and assumes each count's measured
gain holds at the budget the target rate implies, which for p = 1e-4 is
2.4e5 decodes at k = 3 against the 1.06M the gain was measured at. The
output is the decoder calls each estimator would need for a 10% relative
error. Section 5 checks the p = 1e-4 row with completed runs. Per-count
gains used: Relay-BP k = 3 epsilon 0.15, k = 4 epsilon 0.3, k = 5 the
uncertified default run; BP-OSD k = 3 epsilon 0.3 at 2M, k = 4 position
+ core; PyMatching position runs. For Relay-BP the exact weight-2 head
3.18e-7 is added; its p = 2e-3 row is a lower bound (15% of the mass lies
above k = 8 there).

| Decoder / code | p | LER, first order | LER, exact reweighting | direct MC | IBM count strata | LACE, plug-in | LACE, certified | vs direct (plug-in) | vs IBM: plug-in / certified |
|---|---|---|---|---|---|---|---|---|---|
| Relay-BP, BB72 | 2e-3 | 3.2e-3 | - | 3.1e4 | 1.7e4 | 1.6e4 | 1.7e4 | 2x | 1.10x / 1.04x |
| Relay-BP, BB72 | 1e-3 | 6.7e-4 | - | 1.5e5 | 5.0e4 | 3.6e4 | 4.3e4 | 4x | 1.40x / 1.18x |
| Relay-BP, BB72 | 5e-4 | 7.0e-5 | 7.0e-5 | 1.4e6 | 1.8e5 | 9.1e4 | 1.2e5 | 16x | 1.92x / 1.44x |
| Relay-BP, BB72 | 2e-4 | 3.6e-6 | 3.6e-6 | 2.8e7 | 5.1e5 | 2.0e5 | 2.8e5 | 140x | 2.6x / 1.79x |
| Relay-BP, BB72 | 1e-4 | 4.3e-7 | 4.2e-7 | 2.3e8 | 7.3e5 | 2.5e5 | 3.7e5 | 930x | 2.9x / 1.97x |
| BP-OSD, BB72 | 2e-3 | 9.0e-4 | - | 1.1e5 | 6.3e4 | 5.7e4 | 5.9e4 | 2x | 1.10x / 1.07x |
| BP-OSD, BB72 | 1e-3 | 1.9e-4 | - | 5.4e5 | 1.8e5 | 1.1e5 | 1.3e5 | 5x | 1.62x / 1.38x |
| BP-OSD, BB72 | 5e-4 | 1.9e-5 | 1.9e-5 | 5.2e6 | 6.5e5 | 2.1e5 | 3.3e5 | 25x | 3.1x / 1.96x |
| BP-OSD, BB72 | 2e-4 | 1.0e-6 | 1.0e-6 | 9.9e7 | 1.9e6 | 3.4e5 | 7.6e5 | 290x | 5.4x / 2.44x |
| BP-OSD, BB72 | 1e-4 | 1.2e-7 | 1.2e-7 | 8.5e8 | 2.7e6 | 4.3e5 | 1.1e6 | 2000x | 6.4x / 2.53x |
| PyMatching, surface d=5 | 1e-3 | 1.4e-4 | 1.4e-4 | 7.1e5 | 3.6e4 | 2.8e4 | 3.2e4 | 25x | 1.3x / 1.13x |
| PyMatching, surface d=5 | 3e-4 | 4.4e-6 | 4.4e-6 | 2.3e7 | 5.2e4 | 3.7e4 | 4.4e4 | 610x | 1.4x / 1.17x |

(`results/ler_speedup_*.json`, `figures/ler_speedup.png`.) The projected
advantage over IBM grows toward low LER because the onset stratum's share
of the count-stratified variance rises from 2% at p = 2e-3 to 85% at
p = 1e-4 on BB72, and that is the stratum where the layer structure acts.

## 5. Stress tests against direct Monte Carlo (measured)

**Direct controls at every rate.** `run_direct_mc.py` with the same
Relay-BP instance (Stim circuit sampler, an independent code path from the
DEM-based conditional samplers); the pipeline column is the spectrum's
exactly reweighted LER where the spectrum run stored one and the
first-order value otherwise.

| p | direct MC LER (95% Clopper-Pearson) | failures / shots | wall | spectrum LER |
|---|---|---|---|---|
| 1e-3 | 7.20e-4 [5.91e-4, 8.69e-4] | 108 / 1.5e5 | 0.1 min | 6.67e-4 (first order) |
| 5e-4 | 7.14e-5 [5.81e-5, 8.69e-5] | 100 / 1.4e6 | 0.7 min | 7.00e-5 +- 0.29e-5 |
| 2e-4 | 3.71e-6 [3.03e-6, 4.50e-6] | 104 / 2.8e7 | 15 min | 3.60e-6 +- 0.35e-6 |
| 1e-4 | 4.58e-7 [3.77e-7, 5.52e-7] | 110 / 2.4e8 | 121 min | 4.2e-7 +- 0.6e-7 |

(`results/direct_mc_bb72_relay_*.json`.) The spectrum estimate is inside
the direct interval at every rate. This validates the count-stratified
spectrum and its exact reweighting, the core shared with the IBM baseline;
LACE's own per-count estimates are validated by the repeats and the
coverage test below, and the whole chain at one rate by the
matched-precision run.

**Matched-precision check of the p = 1e-4 projection.** Section 4 projects
2.5e5 decodes for 10% at p = 1e-4 with the Relay-BP gains, against 7.3e5
for count strata. `validate_ler.py` assembles LER(1e-4) from complete LACE
runs at the per-count budgets the projection implies (k = 3 at a 200k
nominal budget, k = 4 at 47k, k = 5 at 20k; epsilon 0.15, three repeats,
seed 20261103; `results/validate_1e4_bb72_relay_k{3,4,5}.json`,
`results/validate_ler_summary.json`), with the plug-in and certified
variances propagated through the exact count masses, the plain
count-strata estimate from the same reports at the same decode count, and
the realised cost including pilot, allocation and sampling.

| | LACE | count strata, same decodes | direct MC |
|---|---|---|---|
| LER(1e-4), three repeats | 5.0e-7, 3.8e-7, 4.5e-7 | 4.0e-7, 4.5e-7, 3.6e-7 | 4.58e-7 [3.77e-7, 5.52e-7] |
| relative SE, mean of repeats | 8.3% plug-in, 10.4% certified (11.2% with the union bound over 3 counts) | 12.9% | 9.5% |
| decodes per arm, realised | 5.0e5 (projected 2.5e5 for 10%) | 5.0e5 (projected 7.3e5 for 10%) | 2.4e8 |
| decodes for 10%, scaled from the realised SE | 3.4e5 | 8.3e5 | 2.4e8 |
| wall time | 15 s per repeat and arm | same | 121 min |

The realised variance gain at equal decodes is 2.43x plug-in and 1.55x
certified against the projected 2.88x and 1.97x, so the projection
overstates the k = 3 gain at this smaller budget by 20-25%, as its own
"below measured budget" flag warns. The realised decodes are twice the
nominal budget because of the certification floors (section 6), which is
why LACE needs 3.4e5 rather than 2.5e5 decodes for 10%. Against direct
Monte Carlo the realised advantage at p = 1e-4 is 690x in decoder calls
and about 480x in wall time. Every LACE estimate is inside the direct
interval. This is one rate and one decoder; the other rows of section 4
remain projections.

**Ten independent repeats, Relay-BP k = 3** (epsilon 0.15, 600k budget,
1.07M decodes per arm; `..._eps015_10rep.json`). Pooled truth from the
plain samples of all Relay-BP position reports: f(3) = 9.21e-5 from
12,150 failures (0.9%). Stratified mean +0.0% from truth; realised
variance gain across repeats 3.35 [95% bootstrap 1.67, 6.10] against a
mean reported plug-in gain of 3.05 and certified 1.97 (per-repeat
certified 1.32-2.70, none below 1).

**Ten independent repeats, BP-OSD k = 3** (epsilon 0.3, 1.2M budget,
2.1M decodes per arm; `..._eps03_1p2M_10rep.json`). Stratified mean +3.0%
from a 1060-failure pooled truth, 1.0 sigma with the truth's own error.
Every repeat certified above 1 (2.14-3.37, median 2.56); plug-in 5.5-14.8
(median 7.3). The realised gain across ten repeats, 3.5 [0.83, 14.7], is
too wide to pin the plug-in figure better than a factor 2.

**Coverage, 100 repeats, Relay-BP k = 3** (epsilon 0.15, 200k budget,
5.1e5 decodes per arm, about 50 failures each; `..._eps015_coverage.json`,
`summarise_coverage.py`). The plug-in interval covered the pooled truth
67% and 94% of the time at the 68% and 95% levels; the certified interval
78% and 98%; the plain binomial interval 71% and 94%. The realised standard
deviation equals the mean reported plug-in standard error (ratio 1.00) and
is 0.79x the certified one. Realised gain 2.55 [1.71, 3.81] against 2.70
reported (plug-in) and 1.69 (certified; per-repeat 0.76-2.90, 2 of 100
below 1 at this budget). Stratified mean 1.8% below truth, 1.3 sigma.

What the tests establish: the count-stratified spectrum is unbiased
against an independent control at every rate of the LER table and the
whole LACE chain at p = 1e-4; the plug-in standard error is calibrated at
50-100 failures per arm and conservative by 1.3x below that; the certified
interval is conservative (78% at nominal 68%) and its gains hold in every
repeat but two of 120, both at the smallest budget; the plug-in gains are
what the estimator realises, confirmed to 30% for Relay-BP and to a
factor 2 for BP-OSD; and the cost projection overstates the gain by about
a quarter when it extrapolates to a budget four times below the measured
one.

**Wall time.** Direct Monte Carlo ran at 30 us per shot end to end. The
stratified runs above cost 90-103 us per decode, all of the excess in the
conditional sampler, whose sequential (Chen-Liu) draw took one Python
step per mechanism, a fixed 13 ms per call on BB72's 2232 mechanisms that
the per-group draws of a few hundred to a few thousand shots turned into
30-145 us per shot. The sampler now draws positions in increasing order by
inverse transform on one cumulative table per remaining count (same law,
tested against the sequential scheme), and syndromes are formed by
gathering the fired mechanisms' columns and XOR-reducing per row. Measured
on the same machine: sampling 1 us per shot at every batch size, syndromes
4-7 us, Relay-BP decode 14-22 us, 19-28 us in total against 49-168 us
before; the same ten-repeat job fell from 37 min to 11 min and a
200k-budget run from 77.6 s to 31.8 s. Per decode the pipeline now runs
at the direct sampler's cost, so wall-time ratios follow decoder-call
ratios; the matched-precision check above is the measured instance.

## 6. The certification allowance and the budget (measured)

The allowance epsilon is the one tuning knob of LACE. Too tight and the
never-failing groups eat the budget; too loose and the bound dominates the
certified variance. Two facts about the mechanism matter for reading the
numbers. First, the certification floors and the per-group minimum labels
are added on top of the Neyman allocation rather than taken from it, so
the realised decodes exceed the nominal `--budget`: by 1.6-2.5x at a 200k
budget and about 1.1x at 600k for the fixed rule, and by 1.6x at 200k for
`auto`. The nominal budget is a floor on what the run spends, not a cap;
every gain in this document is quoted at the realised count, and the
realised count is what the cost projection should be compared with.
Second, `optimal_certify_fraction` (`--certify-fraction auto`) chooses
epsilon each round by minimising a projection of the certified variance,
A / (N - N_cert(epsilon)) + epsilon f / N, where A is the Neyman variance
constant from the shrunk rates and N_cert the labels the floor rule needs,
keeping the failing groups at least a quarter of the budget. The realised
bound matched the targeted allowance within 10% in every run.

Relay-BP k = 3, seed 20261102 (20261101 for the 600k pairs), ordered by
realised decodes per arm so that neighbouring rows compare at equal spend:

| allowance | nominal budget | decodes per arm | plug-in gain | certified gain | repeats certified below 1 | never-failing mass |
|---|---|---|---|---|---|---|
| auto (chose 0.5-2.5) | 200k | 317k | 2.43 | 0.99 | 17 of 30 | 0.46 |
| fixed 0.15 | 120k | 375k | 2.17 | **1.35** | 4 of 30 | 0.42 |
| fixed 0.15 | 200k | 513k | 2.72 | **1.69** | 2 of 100 | 0.29 |
| auto (chose 0.3-0.5) | 330k | 535k | 2.92 | 1.45 | 2 of 30 | 0.45 |
| auto (chose 0.25-0.5) | 600k | 948k | 3.49 | 2.00 | 0 of 10 | 0.72 |
| fixed 0.15 | 600k | 1.08M | 2.95 | **1.97** | 1 of 10 | 0.59 |

(`results/layers_bb72_relay_position_inround_{auto_200k_30rep,eps015_120k_30rep,eps015_coverage,auto_330k_30rep,auto_10rep,eps015_10rep_v2}.json`.)

At equal spend the auto rule realises 10-20% more plug-in gain and
certifies 15-35% less, with 57% of its repeats below 1 at the smallest
spend where the fixed rule has 13%; the two tie on the certified figure
near 1M decodes per arm. The reason is in the last column: the fixed rule
leaves 29-42% of the stratum mass never failing, the auto rule 45-72%.
The rule's projection treats the set of never-failing groups as given and
values certification labels only for shrinking their bound, whereas those
labels also turn near-empty groups into measured ones, which removes the
bound on them altogether; the rule undervalues that and loosens the
allowance too far. Since the certified figure is what one run can
promise, the fixed allowance of 0.15 stays the default (it is also the
setting behind the coverage test and both ten-repeat runs); `auto` is
kept as the option that maximises the realised gain when a guarantee is
not needed. Two improvements were not attempted: crediting the discovery
of failures in the projection, and renormalising the allocation after the
floors so that the nominal budget becomes a cap.

## 7. What was tried and dropped

Documented briefly so it is not repeated; the reports were removed from
this branch.

* **Spatial (tile) refinement of the time statistic.** On surface d = 5
  with PyMatching the best spacetime statistic adds at most 6% to position
  + core at any count; on d = 7 it adds 8-31% bias-corrected at 2.4-3.3x
  the number of groups and shrinks as failures accumulate. Not a lever.
* **A cheap decoder as a control variate for an expensive one.** On BB72
  the correlation of BP-OSD failure with truncated BP (10-20 iterations)
  is 0.02-0.09 and with GPU Relay-BP 0.27-0.34 (Relay-BP misses 40-50% of
  BP-OSD's failures and fails 3x more often at k = 3); the ceiling with a
  free cheap arm is 1.13x. BP-OSD failures are mostly BP converging to a
  wrong codeword (only 15-30% had a non-converged BP), so the in-decoder
  flag is uninformative too. Decoders do not proxy each other.
* **Metropolis splitting (IBM technique III)** agreed with the spectrum
  down to p = 5e-4 and ran 25% low at 2.2e-4 with rising R-hat; the
  between-set spread does not cover a chain stuck in one failing sector.
* **A learned labelling or proposal** did not pay off against the exact
  stratification and adds a certification burden.
* **A pooled-family certified bound** (section 1) was not a bound and was
  replaced.

## 8. Conclusion

Exact fault-count stratification with exact reweighting is the unbiased
core of IBM's method and already beats direct Monte Carlo by two to three
orders of magnitude at LER ~ 1e-7; IBM's ansatz on top of it is biased by
up to 2x on BB72 and should not be used. LACE keeps that core, adds no
model, and is verified: the spectrum against an independent direct control
at every rate, the per-count estimates by calibrated plug-in intervals over
100 repeats, the certified bound by an argument that holds under adaptive
allocation and by conservative empirical coverage, and the whole chain at
p = 1e-4 by a matched-precision run that lands inside the direct interval.
Measured per count against the unbiased IBM baseline it is worth 2.0-2.8x
certified and 3.3-3.9x realised for Relay-BP at onset, 2.2-2.9x certified
and 5-7x realised for BP-OSD, and about 1.2x for matching on the surface
code, where count strata alone suffice. At the LER level the cost model
projects 2.9x (plug-in) and 2.0x (certified) over IBM and 930x over direct
Monte Carlo for Relay-BP at p = 1e-4; the one completed check realised
2.4x, 1.55x and 690x, so the projections should be read with about a
quarter of their gain in reserve. The one tuning knob, the certification
allowance, is best left at the validated fixed value; the nominal budget
is a floor, not a cap. The natural next study is BB144 and other qLDPC
codes with BP-family decoders, where the value concentrates.
