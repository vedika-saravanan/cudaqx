# LACE results: layer-stratified allocation with calibrated error bars against direct Monte Carlo and the IBM baseline

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

Three kinds of error bar appear for LACE and are never mixed. The
*plug-in* standard error is the per-round unbiased variance; it is the
headline and its coverage is measured (section 5). The *heuristic* charge
adds an exposure-aware rule-of-three term for the groups that never failed;
it is not a bound (section 1 says why) and is reported as a labelled
heuristic whose empirical coverage is conservative. The *worst-case*
interval is the plug-in uncertainty of the groups that failed plus a
simultaneous upper allowance for the groups that never failed; the
allowance part is a valid bound, the plug-in part is an estimate, so the
full interval is not a rigorous confidence interval and its coverage,
like the others', is empirical. It is reported so that the reader can see
that even that allowance alone is too wide to certify a gain at these
budgets with any estimator that leaves groups unobserved.

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

**LACE, Layer-stratified Allocation with Calibrated Error bars (this
work).** Inside each count stratum the configurations are split exactly by
the composition of the k faults over the detection layers (the mass of
each composition is a product of per-layer Poisson-binomial terms, so
every group weight is exact), grouped by a position statistic: for compact
sets (span at most 2 layers) the exact first and last layer, for spread
sets the start and count of the densest 2-layer window; `position_core`
refines by the largest single-layer count. Groups below 0.2% of the
stratum are pooled into one sampled "rest" group, so nothing is dropped.
Allocation: a mass-proportional pilot of 10% of the budget, then three
Neyman rounds whose rates are shrunk hierarchically toward the pooled rate
of the group's position-free family; rounds are combined with weights
fixed in advance, which keeps the estimator unbiased under adaptive
allocation, and the variance is the per-round unbiased plug-in. Groups
that never fail receive enough labels inside the rounds that their
heuristic charge (below) stays under an allowance, 0.15 of the
plain-sampling variance at the planned budget by default (`--allowance`;
section 6).

*Why "calibrated" and not "certified".* A group that never fails
contributes zero to the plug-in variance, but zero failures do not mean
zero rate, and two attempts to turn that into a guarantee were wrong. The
first pooled the labels of a position-free family and charged every member
the pooled rule-of-three rate; a group with fewer labels than its share of
the mass can hide a higher rate than that allows. The second bounded the
label-weighted sum of the never-failing groups' rates by 3 and charged the
single worst group; that argument holds for a set fixed in advance, but the
never-failing set is selected by the data (with 100 equal groups at rate
0.001 and 1000 labels each, about 37 are empty and their true contribution
exceeds the charge every time). It is kept as the *heuristic* charge:
exposure-aware, empirically conservative on BB72 (78% coverage at nominal
68% over 100 repeats), useful as a second number, not a bound. The
*worst-case* allowance gives every group its own one-sided limit
ln(G / 0.05) over its labels, Bonferroni over the G pre-specified groups,
each limit holding under adaptive allocation by Ville's inequality on the
supermartingale exp(f sum of labels) times the no-failure indicator, and
sums the charges of all empty groups (`allocation.never_failing_bounds`).
That part is a valid bound on what the never-failing groups can
contribute. The groups that did fail keep their plug-in variance
estimate, which is not an upper bound, so the worst-case *interval* is
plug-in uncertainty plus a simultaneous allowance, not a rigorous
confidence interval, and its coverage is reported empirically like the
others' (an exact binomial check of the construction on a one-group
example gives 93.5% at nominal 95%). Section 3 shows what the allowance
alone would certify: nothing. At 1M decodes per stratum it must allow
twenty to forty empty groups to sit simultaneously at six over their
label counts, ten to twenty times any plausible rate. That is a property
of worst-case allowances for stratified estimators with unobserved
strata, not of this estimator. What LACE can honestly claim is a plug-in
error bar whose coverage has been measured (section 5), hence the name.

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

Each row is the estimator's variance gain over plain conditional sampling
of the same count stratum at the realised decode count, with the
never-failing groups read at zero (plug-in), charged by the heuristic, and
charged by the worst-case allowance. The allowance floors are added on
top of the nominal budget, so the realised count exceeds it (section 6);
every comparison is at the realised count.

| Decoder, k | configuration | decodes per arm | plug-in gain | heuristic gain | worst-case allowance gain | never-failing mass |
|---|---|---|---|---|---|---|
| Relay-BP, 3 | allowance 0.3 | 0.93M | **3.70x** | 2.00x | 0.21x | 0.74 |
| Relay-BP, 3 | allowance 0.15 | 1.06M | **3.28x** | 2.22x | 0.33x | 0.72 |
| Relay-BP, 3 | allowance 0.15, 3x budget | 2.50M | **3.88x** | 2.81x | 0.48x | 0.65 |
| Relay-BP, 4 | allowance 0.3 | 0.65M | **1.66x** | 1.15x | 0.32x | 0.31 |
| BP-OSD, 3 | allowance 0.3 | 0.97M | **7.02x** | 2.19x | 0.13x | 0.93 |
| BP-OSD, 3 | allowance 0.3, 2x budget | 1.99M | **7.13x** | 2.54x | 0.18x | 0.88 |
| BP-OSD, 3 | allowance 0.15, 2x budget | 2.51M | 1.38x | 1.01x | 0.15x | 0.89 |
| BP-OSD, 4 | position + core, allowance 0.3 | 1.63M | **4.77x** | 2.90x | 0.18x | 0.81 |
| PyMatching d=5, 3 | position | 0.60M | 1.44x | 1.18x | 0.48x | 0.05 |
| PyMatching d=5, 4 | position | 0.60M | 1.03x | 1.03x | 1.03x | 0.00 |

Reports: `results/layers_bb72_relay_position_inround_{eps015,eps03,eps015_1p8M,eps03_k4}.json`,
`results/layers_bb72_bposd_position_inround_{eps03,eps03_1p2M,eps015_1p2M}.json`,
`results/layers_bb72_bposd_position_core_inround_eps03_k4_1p2M.json`,
`results/layers_surface5_pymatching_position.json`.

The plug-in gain is the number the repeats confirm (section 5): 3.3-3.9x
for Relay-BP at onset, 1.7x one count above, a 7x plug-in estimate for
BP-OSD at onset whose repeats give 3.5x [0.83, 14.7] and so do not settle
it, 4.8x one count above, 1.4x and 1.03x for matching on the surface code
at d = 5,
and 1.15-1.18x at d = 7 where f(4) = 6.7e-6 puts the onset stratum behind
a 1 / f wall (`results/layers_surface7_*`). The heuristic gains sit at
2.0-2.9x for the BP decoders and track the sqrt(N) scaling of the
allowance cost. The worst-case column is below 1 everywhere except where
nothing is unobserved: a rigorous worst-case certificate of a gain is not
available at these budgets, and section 1 explains why that is structural.
BP-OSD's 7x plug-in gain comes with 88-93% of its onset mass never
failing, so it leans most on the calibration evidence, and its ten
repeats (3.5x [0.83, 14.7]) are too few to confirm it; it is a plug-in
estimate until a larger repeat set exists.

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
gains used: Relay-BP k = 3 allowance 0.15, k = 4 allowance 0.3, k = 5 the
default run; BP-OSD k = 3 allowance 0.3 at 2M, k = 4 position + core;
PyMatching position runs. For Relay-BP the exact weight-2 head 3.18e-7 is
added; its p = 2e-3 row is a lower bound (15% of the mass lies above
k = 8 there). The LACE columns use the plug-in gain; the last column gives
the heuristic and worst-case versions of the speedup over IBM.

| Decoder / code | p | LER, first order | LER, exact reweighting | direct MC | IBM count strata | LACE | vs direct | vs IBM: plug-in / heuristic / worst case |
|---|---|---|---|---|---|---|---|---|
| Relay-BP, BB72 | 2e-3 | 3.2e-3 | - | 3.1e4 | 1.7e4 | 1.6e4 | 2x | 1.10x / 1.04x / 0.78x |
| Relay-BP, BB72 | 1e-3 | 6.7e-4 | - | 1.5e5 | 5.0e4 | 3.6e4 | 4x | 1.40x / 1.18x / 0.51x |
| Relay-BP, BB72 | 5e-4 | 7.0e-5 | 7.0e-5 | 1.4e6 | 1.8e5 | 9.1e4 | 16x | 1.92x / 1.44x / 0.38x |
| Relay-BP, BB72 | 2e-4 | 3.6e-6 | 3.6e-6 | 2.8e7 | 5.1e5 | 2.0e5 | 140x | 2.6x / 1.79x / 0.33x |
| Relay-BP, BB72 | 1e-4 | 4.3e-7 | 4.2e-7 | 2.3e8 | 7.3e5 | 2.5e5 | 930x | 2.9x / 1.97x / 0.33x |
| BP-OSD, BB72 | 2e-3 | 9.0e-4 | - | 1.1e5 | 6.3e4 | 5.7e4 | 2x | 1.10x / 1.07x / 0.80x |
| BP-OSD, BB72 | 1e-3 | 1.9e-4 | - | 5.4e5 | 1.8e5 | 1.1e5 | 5x | 1.62x / 1.38x / 0.43x |
| BP-OSD, BB72 | 5e-4 | 1.9e-5 | 1.9e-5 | 5.2e6 | 6.5e5 | 2.1e5 | 25x | 3.1x / 1.96x / 0.25x |
| BP-OSD, BB72 | 2e-4 | 1.0e-6 | 1.0e-6 | 9.9e7 | 1.9e6 | 3.4e5 | 290x | 5.4x / 2.44x / 0.19x |
| BP-OSD, BB72 | 1e-4 | 1.2e-7 | 1.2e-7 | 8.5e8 | 2.7e6 | 4.3e5 | 2000x | 6.4x / 2.53x / 0.18x |
| PyMatching, surface d=5 | 1e-3 | 1.4e-4 | 1.4e-4 | 7.1e5 | 3.6e4 | 2.8e4 | 25x | 1.3x / 1.13x / 0.59x |
| PyMatching, surface d=5 | 3e-4 | 4.4e-6 | 4.4e-6 | 2.3e7 | 5.2e4 | 3.7e4 | 610x | 1.4x / 1.17x / 0.51x |

(`results/ler_speedup_*.json`, `figures/ler_speedup.png`.) The projected
advantage over IBM grows toward low LER because the onset stratum's share
of the count-stratified variance rises from 2% at p = 2e-3 to 85% at
p = 1e-4 on BB72, and that is the stratum where the layer structure acts.
The "vs direct" column is dominated by the count stratification that IBM
also has; LACE's own contribution is the "vs IBM" column.

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
nominal budget, k = 4 at 47k, k = 5 at 20k; allowance 0.15, three repeats,
seed 20261103; `results/validate_1e4_bb72_relay_k{3,4,5}.json`,
`results/validate_ler_summary.json`), with the three variances propagated
through the exact count masses, the plain count-strata estimate from the
same reports at the same decode count, and the realised cost including
pilot, allocation and sampling. Two approximations are deliberate: the
per-count estimates were sampled at the reference rate p = 2e-3 and are
combined with the p = 1e-4 count masses, the same first-order assumption
as the cost model (the spectrum's exact reweighting agrees with it within
2.5%); and the wall time is the reports' total for both arms and all
repeats, not a per-arm measurement.

| | LACE | count strata, same decodes | direct MC |
|---|---|---|---|
| LER(1e-4), three repeats | 5.0e-7, 3.8e-7, 4.5e-7 | 4.0e-7, 4.5e-7, 3.6e-7 | 4.58e-7 [3.77e-7, 5.52e-7] |
| relative SE, mean of repeats | 8.3% plug-in; 10.4% heuristic; 38.9% worst case (joint over 3 counts) | 12.9% | 9.5% |
| decodes per arm, realised | 5.0e5 (projected 2.5e5 for 10%) | 5.0e5 (projected 7.3e5 for 10%) | 2.4e8 |
| decodes for 10%, scaled from the realised plug-in SE | 3.4e5 | 8.3e5 | 2.4e8 |
| wall time | 1.5 min for 3 repeats and both arms | included | 121 min |

The realised variance gain at equal decodes is 2.43x plug-in and 1.55x
heuristic against the projected 2.88x and 1.97x, so the projection
overstates the k = 3 gain at this smaller budget by 20-25%, as its own
"below measured budget" flag warns. The realised decodes are twice the
nominal budget because of the allowance floors (section 6), which is why
LACE needs 3.4e5 rather than 2.5e5 decodes for 10%. Against direct Monte
Carlo the two runs differ by 483x in actual decoder calls (2.4e8 against
5.0e5 per LACE arm) and by 641x when both are normalised to a 10% relative
error; the first is what was run, the second is the efficiency figure, and
they are not the same number. Every LACE estimate is inside the direct
interval. This is one rate and one decoder; the other rows of section 4
remain projections.

**Ten independent repeats, Relay-BP k = 3** (allowance 0.15, 600k budget,
1.07M decodes per arm; `..._eps015_10rep.json`). Pooled truth from the
plain samples of all Relay-BP position reports: f(3) = 9.21e-5 from
12,150 failures (0.9%). Stratified mean +0.0% from truth; realised
variance gain across repeats 3.35 [95% bootstrap 1.67, 6.10] against a
mean reported plug-in gain of 3.05, heuristic 1.97 (per-repeat 1.32-2.70)
and worst case 0.25.

**Ten independent repeats, BP-OSD k = 3** (allowance 0.3, 1.2M budget,
2.1M decodes per arm; `..._eps03_1p2M_10rep.json`). Stratified mean +3.0%
from a 1060-failure pooled truth, 1.0 sigma with the truth's own error.
Reported plug-in gains 5.5-14.8 (median 7.3), heuristic 2.14-3.37 (median
2.56), worst case 0.13-0.18. The realised gain across ten repeats, 3.5
[0.83, 14.7], is too wide to pin the plug-in figure better than a factor
2.

**Coverage, 100 repeats, Relay-BP k = 3** (allowance 0.15, 200k budget,
5.1e5 decodes per arm, about 50 failures each; `..._eps015_coverage.json`,
`summarise_coverage.py`). Against the pooled truth the plug-in interval
covered 67% and 94% of the time at the 68% and 95% levels; the heuristic
interval 78% and 98%; the worst-case interval 99% and 100%; the plain
binomial interval 71% and 94%. The realised standard deviation equals the
mean reported plug-in standard error (ratio 1.00), is 0.79x the heuristic
one and 0.28x the worst-case one. Realised gain 2.55 [1.71, 3.81] against
2.70 reported (plug-in), 1.69 (heuristic; per-repeat 0.76-2.90) and 0.21
(worst case). Stratified mean 1.8% below truth, 1.3 sigma.

What the tests establish: the count-stratified spectrum is unbiased
against an independent control at every rate of the LER table and the
whole LACE chain at p = 1e-4; the plug-in standard error is calibrated at
50-100 failures per arm and conservative by 1.3x below that, which is what
"calibrated" in the name refers to; the heuristic interval is conservative
(78% at nominal 68%) but carries no guarantee; the worst-case interval is
a valid allowance on the never-failing part, far too wide to be useful,
and not a confidence interval for the whole estimate; the plug-in gains
are what the
estimator realises, confirmed to 30% for Relay-BP and to a factor 2 for
BP-OSD; and the cost projection overstates the gain by about a quarter
when it extrapolates to a budget four times below the measured one.

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
ratios.

## 6. The allowance and the budget (measured)

The allowance epsilon is the one tuning knob of LACE: the never-failing
groups receive labels until their heuristic charge is at most epsilon
times the plain-sampling variance at the planned budget. Too tight and
those groups eat the budget; too loose and the heuristic charge dominates.
Two facts about the mechanism matter for reading the numbers. First, the
allowance floors and the per-group minimum labels are added on top of the
Neyman allocation rather than taken from it, so the realised decodes
exceed the nominal `--budget`: by 1.6-2.5x at a 200k budget and about 1.1x
at 600k for the fixed rule, and by 1.6x at 200k for `auto`. The nominal
budget is a floor on what the run spends, not a cap; every gain in this
document is quoted at the realised count, and the realised count is what
the cost projection should be compared with. Second, `optimal_allowance`
(`--allowance auto`) chooses epsilon each round by minimising a projection
of the heuristic variance, A / (N - N_cert(epsilon)) + epsilon f / N,
where A is the Neyman variance constant from the shrunk rates and N_cert
the labels the floor rule needs, keeping the failing groups at least a
quarter of the budget. The realised heuristic charge matched the targeted
allowance within 10% in every run. Since the heuristic is not a bound, the
comparison below is between two heuristics; the plug-in column is the one
with a measured meaning.

Relay-BP k = 3, seed 20261102 (20261101 for the 600k pairs), ordered by
realised decodes per arm so that neighbouring rows compare at equal spend:

| allowance | nominal budget | decodes per arm | plug-in gain | heuristic gain | repeats with heuristic gain below 1 | never-failing mass |
|---|---|---|---|---|---|---|
| auto (chose 0.5-2.5) | 200k | 317k | 2.43 | 0.99 | 17 of 30 | 0.46 |
| fixed 0.15 | 120k | 375k | 2.17 | 1.35 | 4 of 30 | 0.42 |
| fixed 0.15 | 200k | 513k | 2.72 | 1.69 | 2 of 100 | 0.29 |
| auto (chose 0.3-0.5) | 330k | 535k | 2.92 | 1.45 | 2 of 30 | 0.45 |
| auto (chose 0.25-0.5) | 600k | 948k | 3.49 | 2.00 | 0 of 10 | 0.72 |
| fixed 0.15 | 600k | 1.08M | 2.95 | 1.97 | 1 of 10 | 0.59 |

(`results/layers_bb72_relay_position_inround_{auto_200k_30rep,eps015_120k_30rep,eps015_coverage,auto_330k_30rep,auto_10rep,eps015_10rep_v2}.json`.)

At equal spend the auto rule realises 10-20% more plug-in gain and leaves
a 15-35% larger heuristic charge, because it leaves 45-72% of the stratum
mass never failing against 29-42% for the fixed rule: its projection
values labels for empty groups only for shrinking their charge, not for
the failures they would discover. Which rule is "better" therefore depends
on whether the heuristic charge is given weight. The fixed allowance of
0.15 stays the default because it is the setting behind the coverage test
and both ten-repeat runs, so its plug-in calibration is the one that has
been measured; `auto` is kept as the option that maximises the realised
gain. Two improvements were not attempted: crediting the discovery of
failures in the projection, and renormalising the allocation after the
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
* **Two "certified" bounds for the never-failing groups** (section 1), a
  pooled-family one and a data-selected single-group one, were both
  invalid; the second survives as a labelled heuristic beside a valid but
  vacuous worst-case allowance for the never-failing part.

## 8. Conclusion

Exact fault-count stratification with exact reweighting is the unbiased
core of IBM's method and already beats direct Monte Carlo by two to three
orders of magnitude at LER ~ 1e-7; IBM's ansatz on top of it is biased by
up to 2x on BB72 and should not be used. LACE keeps that core, adds no
model, and is verified where it can be: the spectrum against an
independent direct control at every rate, the per-count estimates by
plug-in intervals with measured coverage over 100 repeats, and the whole
chain at p = 1e-4 by a matched-precision run that lands inside the direct
interval. Measured per count against the unbiased IBM baseline it realises
3.3-3.9x for Relay-BP at onset (2.55x [1.71, 3.81] over 100 repeats at
half the budget) and a 7x plug-in estimate for BP-OSD that ten repeats
(3.5x [0.83, 14.7]) leave unconfirmed, about 1.4x for matching on the
surface code. It does not come with a worst-case guarantee of a gain: the
valid allowance for groups that never failed is too wide to certify
anything at these budgets, the full interval's coverage is empirical, and
the two attempts to tighten it were wrong; what it offers instead is an
error bar whose calibration has been measured and a labelled heuristic
charge that has been conservative in every test. At the LER level the cost
model projects 2.9x over IBM and 930x over direct Monte Carlo for Relay-BP
at p = 1e-4; the one completed check realised 2.4x and, against the direct
run, 483x in actual decoder calls and 641x at equal precision, so the
projections should be read with about a quarter of their gain in reserve.
The nominal budget is a floor, not a cap. The natural next study is BB144
and other qLDPC codes with BP-family decoders, where the value
concentrates.
